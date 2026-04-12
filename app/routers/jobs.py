# app/routers/jobs.py
"""
Job Management API
------------------
GET  /api/jobs                   — List all jobs for the authenticated user
POST /api/jobs                   — Create a new job
PATCH /api/jobs/{job_id}         — Update job name or status
GET  /api/jobs/{job_id}/preflight — Cost estimate for a job in 'approval' status
POST /api/jobs/{job_id}/approve  — Deduct tokens, set 'processing', fire background inference
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid as _uuid_mod
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core import firestore_client
from app.core.config import settings
from app.core.inference import run_inference
from app.core.land_cover_inference import run_land_cover_inference
from app.middleware.auth import get_current_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# ── Path constants (mirrors inference router) ──────────────────────────────────
_UPLOAD_DIR  = Path("uploads");  _UPLOAD_DIR.mkdir(exist_ok=True)
_RESULTS_DIR = Path("results");  _RESULTS_DIR.mkdir(exist_ok=True)
_MODELS_DIR  = Path("models");   _MODELS_DIR.mkdir(exist_ok=True)
_DEFAULT_MODELS_DIR = (Path(__file__).resolve().parent.parent / "models" / "default")
_DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_PALM_MODEL_NAME      = "palmCounting-model.onnx"
_LANDCOVER_MODEL_NAME = "unet_swin.onnx"

_C_BASE = 50
_W_AREA = 10    # tokens per hectare
_W_SIZE = 200   # tokens per GB


# ── Billing helpers ────────────────────────────────────────────────────────────

def _calc_tokens(area_sqm: float, size_gb: float) -> int:
    return math.ceil(_C_BASE + (area_sqm / 10_000) * _W_AREA + size_gb * _W_SIZE)


def _get_area_sqm(tif_path: str) -> float:
    import rasterio
    with rasterio.open(tif_path) as src:
        t = src.transform
        pixel_area = abs(t.a * t.e)
        if src.crs and src.crs.is_geographic:
            pixel_area *= 111_000 ** 2
        return float(pixel_area * src.width * src.height)


def _resolve_model(name: str) -> Path:
    p = _MODELS_DIR / name
    if p.exists():
        return p
    p = _DEFAULT_MODELS_DIR / name
    if p.exists():
        return p
    return _MODELS_DIR / name


def _resolve_yaml() -> Path:
    for base in (_MODELS_DIR, _DEFAULT_MODELS_DIR):
        p = base / "data.yaml"
        if p.exists():
            return p
    return _MODELS_DIR / "data.yaml"


# ── Pydantic models ────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    job_name: str = Field(default="Untitled Job", max_length=128)
    task_type: Literal["palm_counting", "land_cover"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    file_uri: str = Field(default="")


class PatchJobRequest(BaseModel):
    job_name: Optional[str] = Field(default=None, max_length=128)
    status: Optional[Literal["uploading", "approval", "processing", "done", "failed"]] = None
    file_uri: Optional[str] = None
    result_uri: Optional[str] = None
    token_cost: Optional[int] = None
    parameters: Optional[dict[str, Any]] = None


# ── Thread-pool helper ─────────────────────────────────────────────────────────

async def _run(fn, *args, **kwargs):
    """Run a blocking call in a thread pool."""
    return await asyncio.to_thread(fn, *args, **kwargs)


# ── CRUD endpoints ─────────────────────────────────────────────────────────────

@router.get("")
async def list_jobs(user: dict = Depends(get_current_user)):
    """Return all jobs owned by the authenticated user, newest first."""
    jobs = await _run(firestore_client.get_jobs_for_user, user["email"])
    return {"jobs": jobs}


@router.post("", status_code=201)
async def create_job(
    body: CreateJobRequest,
    user: dict = Depends(get_current_user),
):
    """Create a new job record. Status starts as 'uploading'."""
    job = await _run(
        firestore_client.create_job,
        user["email"],
        body.job_name,
        body.task_type,
        body.parameters,
        body.file_uri,
    )
    return job


@router.patch("/{job_id}")
async def patch_job(
    job_id: str,
    body: PatchJobRequest,
    user: dict = Depends(get_current_user),
):
    """Update job_name and/or status (and optionally other fields)."""
    existing = await _run(firestore_client.get_job, job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Job not found")

    is_admin = user.get("role") in ("admin", "superadmin")
    if existing["user_email"] != user["email"] and not is_admin:
        raise HTTPException(status_code=403, detail="Not authorised to modify this job")

    updates: dict[str, Any] = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    updated = await _run(firestore_client.update_job, job_id, updates)
    return updated


# ── Approval workflow ──────────────────────────────────────────────────────────

@router.get("/{job_id}/preflight")
async def job_preflight(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Return token cost estimate for a job in 'approval' status.

    Reads file size from GCS blob metadata and raster area via the GDAL
    /vsigs/ virtual filesystem (header-only, no full download).
    """
    job = await _run(firestore_client.get_job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    is_admin = user.get("role") in ("admin", "superadmin")
    if job["user_email"] != user["email"] and not is_admin:
        raise HTTPException(403, "Not authorised")

    if job["status"] != "approval":
        raise HTTPException(422, f"Job is not in 'approval' status (current: {job['status']})")

    gcs_path = job.get("file_uri", "")
    if not gcs_path:
        raise HTTPException(422, "Job has no file_uri — upload the file first")

    user_data = await _run(firestore_client.get_user, user["sub"])
    if not user_data:
        raise HTTPException(401, "User record not found")
    token_balance = user_data.get("token_balance", 0)

    # File size from GCS blob metadata
    if settings.dev_mode:
        file_size_bytes = 10 * 1024 * 1024   # mock 10 MB in dev
        area_sqm        = 500 * 10_000        # mock 500 ha in dev
    else:
        try:
            from google.cloud import storage as _gcs
            gcs_client = _gcs.Client(project=settings.firestore_project_id)
            blob = gcs_client.bucket(settings.gcs_bucket_name).blob(gcs_path)
            await _run(blob.reload)
            file_size_bytes: int = blob.size
        except Exception as exc:
            raise HTTPException(500, f"Could not read GCS file metadata: {exc}")

        # Raster area via GDAL /vsigs/ (reads IFD only)
        vsigs_path = f"/vsigs/{settings.gcs_bucket_name}/{gcs_path}"
        try:
            area_sqm = await _run(_get_area_sqm, vsigs_path)
        except Exception as exc:
            log.warning("job_preflight: vsigs area read failed (%s) — defaulting to 0", exc)
            area_sqm = 0.0

    file_size_gb = file_size_bytes / (1024 ** 3)
    area_ha      = area_sqm / 10_000
    area_cost    = math.ceil((area_sqm / 10_000) * _W_AREA)
    size_cost    = math.ceil(file_size_gb * _W_SIZE)
    token_cost   = _C_BASE + area_cost + size_cost
    balance_after = token_balance - token_cost

    return {
        "job_id":          job_id,
        "job_name":        job["job_name"],
        "task_type":       job["task_type"],
        "parameters":      job["parameters"],
        "file_size_bytes": file_size_bytes,
        "file_size_mb":    round(file_size_bytes / (1024 ** 2), 2),
        "area_ha":         round(area_ha, 2),
        "base_cost":       _C_BASE,
        "area_cost":       area_cost,
        "size_cost":       size_cost,
        "token_cost":      token_cost,
        "token_balance":   token_balance,
        "balance_after":   balance_after,
        "can_afford":      balance_after >= 0,
    }


@router.post("/{job_id}/approve")
async def approve_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """Approve a job: deduct tokens, set status to 'processing', fire inference background task.

    Returns immediately — the API does not wait for inference to complete.
    """
    job = await _run(firestore_client.get_job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    is_admin = user.get("role") in ("admin", "superadmin")
    if job["user_email"] != user["email"] and not is_admin:
        raise HTTPException(403, "Not authorised")

    if job["status"] != "approval":
        raise HTTPException(422, f"Job is not in 'approval' status (current: {job['status']})")

    gcs_path = job.get("file_uri", "")
    if not gcs_path:
        raise HTTPException(422, "No file_uri on job — upload the file first")

    user_uid  = user["sub"]
    user_data = await _run(firestore_client.get_user, user_uid)
    if not user_data:
        raise HTTPException(401, "User record not found")

    token_balance = user_data.get("token_balance", 0)

    # Calculate cost
    if settings.dev_mode:
        file_size_bytes = 10 * 1024 * 1024
        area_sqm        = 500 * 10_000
    else:
        try:
            from google.cloud import storage as _gcs
            gcs_client = _gcs.Client(project=settings.firestore_project_id)
            blob = gcs_client.bucket(settings.gcs_bucket_name).blob(gcs_path)
            await _run(blob.reload)
            file_size_bytes: int = blob.size
        except Exception as exc:
            raise HTTPException(500, f"Could not read GCS file metadata: {exc}")

        vsigs_path = f"/vsigs/{settings.gcs_bucket_name}/{gcs_path}"
        try:
            area_sqm = await _run(_get_area_sqm, vsigs_path)
        except Exception:
            area_sqm = 0.0

    file_size_gb  = file_size_bytes / (1024 ** 3)
    area_cost     = math.ceil((area_sqm / 10_000) * _W_AREA)
    size_cost     = math.ceil(file_size_gb * _W_SIZE)
    token_cost    = _C_BASE + area_cost + size_cost

    if token_cost > token_balance:
        raise HTTPException(402, f"Insufficient tokens: have {token_balance}, need {token_cost}")

    # Atomically deduct tokens
    try:
        new_balance = await _run(firestore_client.deduct_tokens, user_uid, token_cost)
    except ValueError as exc:
        raise HTTPException(402, str(exc))

    # Advance job to processing
    await _run(firestore_client.update_job, job_id, {
        "status":     "processing",
        "token_cost": token_cost,
    })

    # Fire background inference — API returns immediately
    background_tasks.add_task(
        _run_job_inference,
        job_id=job_id,
        gcs_path=gcs_path,
        task_type=job["task_type"],
        params=job.get("parameters", {}),
        user_uid=user_uid,
    )

    return {
        "job_id":      job_id,
        "status":      "processing",
        "token_cost":  token_cost,
        "new_balance": new_balance,
    }


# ── Background inference task ──────────────────────────────────────────────────

async def _run_job_inference(
    job_id: str,
    gcs_path: str,
    task_type: str,
    params: dict[str, Any],
    user_uid: str,
) -> None:
    """Download GCS file, run inference, save result, update job status."""
    file_id     = str(_uuid_mod.uuid4())
    raster_path = _UPLOAD_DIR  / f"{file_id}.tif"
    result_tif  = _RESULTS_DIR / f"{file_id}_lc.tif"

    try:
        # Download from GCS
        if not settings.dev_mode:
            from google.cloud import storage as _gcs
            gcs_client = _gcs.Client(project=settings.firestore_project_id)
            blob = gcs_client.bucket(settings.gcs_bucket_name).blob(gcs_path)
            await asyncio.to_thread(blob.download_to_filename, str(raster_path))
        else:
            # Dev mode: write a tiny stub so inference can be mocked
            raster_path.write_bytes(b"")

        if task_type == "palm_counting":
            model_path = _resolve_model(_PALM_MODEL_NAME)
            yaml_path  = _resolve_yaml()
            if not model_path.exists():
                raise FileNotFoundError(f"Palm model not found: {model_path}")
            if not yaml_path.exists():
                raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

            geojson = await asyncio.to_thread(
                run_inference,
                input_tif_path=str(raster_path),
                model_path=str(model_path),
                yaml_path=str(yaml_path),
                tile_width=int(params.get("tile_width", 640)),
                tile_height=int(params.get("tile_height", 640)),
                min_distance=float(params.get("min_distance", 1.0)),
                conf_threshold=float(params.get("conf_threshold", 0.25)),
                nms_threshold=float(params.get("nms_threshold", 0.4)),
            )
            result_path = _RESULTS_DIR / f"{file_id}.geojson"
            result_path.write_text(json.dumps(geojson, indent=2))

        else:  # land_cover
            model_path = _resolve_model(_LANDCOVER_MODEL_NAME)
            if not model_path.exists():
                raise FileNotFoundError(f"Land cover model not found: {model_path}")

            geojson = await asyncio.to_thread(
                run_land_cover_inference,
                str(raster_path),
                str(model_path),
                in_channels=int(params.get("in_channels", 3)),
                tile_size=int(params.get("tile_size", 512)),
                overlap=int(params.get("overlap", 128)),
                use_filter=bool(params.get("use_filter", True)),
                min_noise_size=int(params.get("min_noise_size", 250)),
                result_tif_path=str(result_tif),
            )
            result_path = _RESULTS_DIR / f"{file_id}_lc.geojson"
            result_path.write_text(json.dumps(geojson, indent=2))

            # Upload classified TIF to GCS for cross-instance preview serving
            if not settings.dev_mode and result_tif.exists():
                try:
                    from google.cloud import storage as _gcs
                    gcs_client = _gcs.Client(project=settings.firestore_project_id)
                    lc_blob = gcs_client.bucket(settings.gcs_bucket_name).blob(
                        f"results/{file_id}_lc.tif"
                    )
                    await asyncio.to_thread(lc_blob.upload_from_filename, str(result_tif))
                except Exception as gcs_err:
                    log.warning("Could not upload LC TIF to GCS: %s", gcs_err)

        await asyncio.to_thread(
            firestore_client.update_job, job_id,
            {"status": "done", "result_uri": file_id},
        )
        log.info("Job %s completed — result file_id=%s", job_id, file_id)

    except Exception as exc:
        log.exception("Job %s failed during inference: %s", job_id, exc)
        await asyncio.to_thread(
            firestore_client.update_job, job_id,
            {"status": "failed"},
        )
    finally:
        raster_path.unlink(missing_ok=True)
        if result_tif.exists():
            result_tif.unlink(missing_ok=True)
