# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana
"""
/api/inference          — upload GeoTIFF → GeoJSON detections (+ timing).
/api/preview/<id>       — stretched-RGB PNG for the map overlay.
/api/download/<id>      — download result as a GeoJSON file.
/api/models             — list available ONNX models (GET) or upload one (POST).
/api/cleanup            — purge uploads & results older than MAX_AGE_HOURS (Cloud Scheduler hook).
"""

import asyncio
import io
import json
import logging
import math
import os
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import rasterio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from app.middleware.auth import get_current_user
from fastapi.responses import JSONResponse, Response
from PIL import Image
from rasterio.warp import transform_bounds

from google.cloud import storage as gcs
from pydantic import BaseModel

from app.core import firestore_client
from app.core.config import settings
from app.core.inference import run_inference

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["inference"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Maximum age (hours) before a file is considered stale and eligible for cleanup.
# Cloud Run instances are ephemeral, so this is defense-in-depth for long-lived instances.
MAX_AGE_HOURS = int(os.getenv("CLEANUP_MAX_AGE_HOURS", "24"))

# Optional shared secret for the /api/cleanup endpoint.
# Set CLEANUP_SECRET env var in Cloud Run to protect the endpoint.
CLEANUP_SECRET = os.getenv("CLEANUP_SECRET", "")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Default model baked into the Docker image — survives scale-to-zero
DEFAULT_MODELS_DIR = Path("app/models/default")
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

YAML_PATH = MODELS_DIR / "data.yaml"


def _resolve_model_path(model_name: str) -> Path:
    """Return model path, checking user-uploaded dir then baked-in default dir."""
    p = MODELS_DIR / model_name
    if p.exists():
        return p
    p = DEFAULT_MODELS_DIR / model_name
    if p.exists():
        return p
    return MODELS_DIR / model_name  # not found — caller raises 404


def _resolve_yaml_path() -> Path:
    """Return data.yaml path, preferring user models dir, then default bundle."""
    if YAML_PATH.exists():
        return YAML_PATH
    default_yaml = DEFAULT_MODELS_DIR / "data.yaml"
    if default_yaml.exists():
        return default_yaml
    return YAML_PATH  # not found — caller raises 500

# Directories managed by the cleanup routine
_CLEANUP_DIRS = [UPLOAD_DIR, RESULTS_DIR, OUTPUT_DIR]


def _validate_file_id(file_id: str) -> None:
    """Reject file IDs that are not UUID-shaped (hex + dashes only)."""
    if not all(c in "0123456789abcdef-" for c in file_id):
        raise HTTPException(400, "Invalid file ID.")


def _purge_old_files(max_age_hours: int = MAX_AGE_HOURS) -> dict:
    """Delete files older than *max_age_hours* from temporary directories.

    Returns a summary dict with deleted/skipped counts for each directory.
    """
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    summary: dict[str, dict] = {}

    for directory in _CLEANUP_DIRS:
        deleted, skipped, errors = 0, 0, 0
        if not directory.exists():
            summary[directory.name] = {"deleted": 0, "skipped": 0, "errors": 0}
            continue
        for item in directory.iterdir():
            # Never delete .gitkeep or hidden sentinel files
            if item.name.startswith("."):
                skipped += 1
                continue
            try:
                mtime = item.stat().st_mtime
                if mtime < cutoff:
                    item.unlink() if item.is_file() else shutil.rmtree(item)
                    deleted += 1
                else:
                    skipped += 1
            except Exception as exc:
                log.warning("Cleanup: could not remove %s — %s", item, exc)
                errors += 1
        summary[directory.name] = {"deleted": deleted, "skipped": skipped, "errors": errors}

    return summary


# ---------------------------------------------------------------------------
# Billing helpers
# ---------------------------------------------------------------------------

C_BASE = 50
W_AREA = 10    # per hectare
W_SIZE = 200   # per GB
FREE_TIER_MAX_BYTES = 30 * 1024 * 1024   # 30 MB
FREE_TIER_MAX_DAILY = 3


def calculate_tokens(l_sqm: float, s_gb: float) -> int:
    """Token cost: C_base + ((L_sqm / 10000) * W_area) + (S_gb * W_size)."""
    return math.ceil(C_BASE + (l_sqm / 10_000) * W_AREA + s_gb * W_SIZE)


def get_raster_area_sqm(tif_path: str) -> float:
    """Read raster metadata and return area in square metres."""
    with rasterio.open(tif_path) as src:
        transform = src.transform
        pixel_area = abs(transform.a * transform.e)
        total_pixels = src.width * src.height
        if src.crs and src.crs.is_geographic:
            pixel_area = pixel_area * (111_000 ** 2)
        return float(pixel_area * total_pixels)


async def asyncio_to_thread_get_user(uid: str):
    return await asyncio.to_thread(firestore_client.get_user, uid)


async def asyncio_to_thread_check_daily(uid: str):
    return await asyncio.to_thread(firestore_client.check_and_increment_daily_upload, uid)


async def asyncio_to_thread_deduct_tokens(uid: str, amount: int):
    return await asyncio.to_thread(firestore_client.deduct_tokens, uid, amount)


def _get_service_account_email() -> str | None:
    """Fetch the active service account email from the GCE metadata server.

    On Cloud Run / GCE this returns the real SA email (e.g.
    123-compute@developer.gserviceaccount.com). Returns None when running
    outside GCP (local dev), in which case the caller falls back to the
    credentials object.
    """
    import urllib.error
    import urllib.request

    url = (
        "http://metadata.google.internal"
        "/computeMetadata/v1/instance/service-accounts/default/email"
    )
    try:
        req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode().strip()
    except Exception:
        return None


def generate_signed_upload_url(user_uid: str, filename: str) -> tuple[str, str]:
    """Generate a GCS signed PUT URL valid for 1 hour. Returns (url, gcs_path).

    Uses the IAM Credentials API for signing so this works on Cloud Run where
    the default Compute Engine credentials contain only a token (no private key).
    The service account email is retrieved dynamically from the GCE metadata
    server so we never hard-code it.  Falls back to the credentials object when
    running locally with a service-account key file.
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())

    # Prefer the metadata server: it returns the real SA email on Cloud Run.
    # Fall back to credentials.service_account_email for local dev with SA keys.
    sa_email = _get_service_account_email() or credentials.service_account_email

    client = gcs.Client(project=settings.firestore_project_id, credentials=credentials)
    bucket = client.bucket(settings.gcs_bucket_name)
    safe_name = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
    gcs_path = f"uploads/{user_uid}/{uuid.uuid4().hex}_{safe_name}"
    blob = bucket.blob(gcs_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=1),
        method="PUT",
        content_type="image/tiff",
        service_account_email=sa_email,
        access_token=credentials.token,
    )
    return url, gcs_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@router.post("/inference")
async def infer(
    file: UploadFile = File(...),
    model_name: str = Form("best_1.onnx"),
    tile_width: int = Form(640),
    tile_height: int = Form(640),
    min_distance: float = Form(1.0),
    conf_threshold: float = Form(0.25),
    nms_threshold: float = Form(0.4),
    current_user: dict = Depends(get_current_user),
):
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files (.tif / .tiff) are accepted.")

    # Read file bytes early — needed for size check before saving
    file_bytes = await file.read()
    file_size_bytes = len(file_bytes)
    file_size_gb = file_size_bytes / (1024 ** 3)

    # ── Fetch user for tier decision ─────────────────────────────────────
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data:
        raise HTTPException(401, "User record not found")

    token_balance = user_data.get("token_balance", 0)

    # ── Free tier pre-checks (fast-fail from cached user data) ───────────
    if token_balance == 0:
        if file_size_bytes > FREE_TIER_MAX_BYTES:
            raise HTTPException(
                403,
                f"Quota exceeded: free tier limit is 30 MB. Your file is "
                f"{file_size_bytes / 1024 / 1024:.1f} MB. Add tokens to process larger files.",
            )
        today = __import__("datetime").date.today().isoformat()
        last_date = user_data.get("last_upload_date", "")
        count = user_data.get("daily_upload_count", 0) if last_date == today else 0
        if count >= FREE_TIER_MAX_DAILY:
            raise HTTPException(
                403,
                "Quota exceeded: free tier allows 3 uploads/day. Add tokens for unlimited access.",
            )

    # ── Model / YAML checks ──────────────────────────────────────────────
    model_path = _resolve_model_path(model_name)
    if not model_path.exists():
        raise HTTPException(404, f"Model '{model_name}' not found. Upload it first.")
    yaml_path = _resolve_yaml_path()
    if not yaml_path.exists():
        raise HTTPException(500, f"data.yaml not found at {yaml_path}")

    # ── Save upload ──────────────────────────────────────────────────────
    file_id = str(uuid.uuid4())
    raster_path = UPLOAD_DIR / f"{file_id}.tif"
    raster_path.write_bytes(file_bytes)

    # ── Commercial tier: calculate + deduct tokens ───────────────────────
    tokens_deducted = 0
    if token_balance > 0:
        l_sqm = await asyncio.to_thread(get_raster_area_sqm, str(raster_path))
        cost = calculate_tokens(l_sqm=l_sqm, s_gb=file_size_gb)
        if cost > token_balance:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(
                402,
                f"Insufficient tokens: have {token_balance}, need {cost}",
            )
        try:
            await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
            tokens_deducted = cost
        except ValueError as e:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(402, str(e))
    else:
        # ── Free tier: atomic daily counter increment ────────────────────
        try:
            await asyncio_to_thread_check_daily(current_user["sub"])
        except ValueError:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(429, "Free tier daily limit reached.")

    # ── Inference (unchanged) ────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        geojson = run_inference(
            input_tif_path=str(raster_path),
            model_path=str(model_path),
            yaml_path=str(yaml_path),
            tile_width=tile_width,
            tile_height=tile_height,
            min_distance=min_distance,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
    except Exception as exc:
        raster_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    duration = round(time.perf_counter() - t0, 2)
    geojson["metadata"]["duration_seconds"] = duration

    # Persist result for later download
    result_path = RESULTS_DIR / f"{file_id}.geojson"
    result_path.write_text(json.dumps(geojson, indent=2))

    return JSONResponse({
        "file_id": file_id,
        "duration_seconds": duration,
        "tokens_deducted": tokens_deducted,
        "geojson": geojson,
    })


# ---------------------------------------------------------------------------
# Download result
# ---------------------------------------------------------------------------

@router.get("/download/{file_id}")
def download_result(file_id: str):
    _validate_file_id(file_id)
    path = RESULTS_DIR / f"{file_id}.geojson"
    if not path.exists():
        raise HTTPException(404, "Result not found. Run inference first.")
    return Response(
        content=path.read_bytes(),
        media_type="application/geo+json",
        headers={"Content-Disposition": f'attachment; filename="palm_detections_{file_id[:8]}.geojson"'},
    )


# ---------------------------------------------------------------------------
# Raster preview PNG
# ---------------------------------------------------------------------------

@router.get("/preview/{file_id}")
def get_preview(file_id: str):
    _validate_file_id(file_id)
    path = UPLOAD_DIR / f"{file_id}.tif"
    if not path.exists():
        raise HTTPException(404, "Raster not found.")

    with rasterio.open(path) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        data = src.read([1, 2, 3]) if src.count >= 3 else np.stack([src.read(1)] * 3)
        nodata = src.nodata
        rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            band = data[i].astype(float)
            mask = (band != nodata) if nodata is not None else np.ones_like(band, dtype=bool)
            valid = band[mask]
            if valid.size == 0:
                valid = band.flatten()
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 == p2:
                p98 = p2 + 1
            rgb[:, :, i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    if max(img.size) > 2048:
        ratio = 2048 / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    w, s, e, n = bounds_wgs84
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Raster-West":  str(w), "X-Raster-South": str(s),
            "X-Raster-East":  str(e), "X-Raster-North": str(n),
            "Access-Control-Expose-Headers":
                "X-Raster-West,X-Raster-South,X-Raster-East,X-Raster-North",
        },
    )


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@router.get("/models")
def list_models(current_user: dict = Depends(get_current_user)):
    """Return all .onnx models: baked-in defaults + user-uploaded custom models."""
    defaults = {p.name for p in DEFAULT_MODELS_DIR.glob("*.onnx")}
    custom   = {p.name for p in MODELS_DIR.glob("*.onnx")}
    return JSONResponse({"models": sorted(defaults | custom)})


@router.post("/models")
async def upload_model(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Upload a new ONNX model file to the models directory."""
    if not file.filename.lower().endswith(".onnx"):
        raise HTTPException(400, "Only .onnx model files are accepted.")

    # Sanitise filename: keep only safe chars
    safe_name = "".join(
        c for c in Path(file.filename).name if c.isalnum() or c in ("_", "-", ".")
    )
    if not safe_name:
        safe_name = f"model_{uuid.uuid4().hex[:8]}.onnx"

    dest = MODELS_DIR / safe_name
    dest.write_bytes(await file.read())
    return JSONResponse({"message": f"Model '{safe_name}' uploaded.", "model_name": safe_name})

# ---------------------------------------------------------------------------
# GCS presign + GPU submit (commercial tier)
# ---------------------------------------------------------------------------

class PresignRequest(BaseModel):
    filename: str
    file_size_bytes: int


class SubmitRequest(BaseModel):
    gcs_path: str
    model_name: str = "best_1.onnx"
    tile_width: int = 640
    tile_height: int = 640
    min_distance: float = 1.0
    conf_threshold: float = 0.25
    nms_threshold: float = 0.4


@router.post("/upload/signed-url")
async def get_signed_upload_url(
    body: PresignRequest,
    current_user: dict = Depends(get_current_user),
):
    """Generate a GCS signed PUT URL for direct browser-to-GCS upload.

    Available to all authenticated users. Billing is enforced at inference time.
    """
    if not settings.gcs_bucket_name:
        raise HTTPException(503, "GCS upload is not configured on this server.")
    try:
        upload_url, gcs_path = await asyncio.to_thread(
            generate_signed_upload_url, current_user["sub"], body.filename
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate signed URL: {e}")
    return {"upload_url": upload_url, "gcs_path": gcs_path}


@router.post("/inference/presign")
async def presign_upload(
    body: PresignRequest,
    current_user: dict = Depends(get_current_user),
):
    """Commercial tier only: get a GCS signed URL for direct large-file upload."""
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data or user_data.get("token_balance", 0) <= 0:
        raise HTTPException(403, "Presigned upload is for commercial tier only (requires token balance > 0)")

    try:
        upload_url, gcs_path = await asyncio.to_thread(
            generate_signed_upload_url, current_user["sub"], body.filename
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate signed URL: {e}")

    return {"upload_url": upload_url, "gcs_path": gcs_path}


@router.post("/inference/submit")
async def submit_gcs_inference(
    body: SubmitRequest,
    current_user: dict = Depends(get_current_user),
):
    """Commercial tier: run inference on a file already uploaded to GCS."""
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data or user_data.get("token_balance", 0) <= 0:
        raise HTTPException(403, "Submit endpoint is for commercial tier only")

    # Validate path ownership — must start with uploads/{uid}/
    expected_prefix = f"uploads/{current_user['sub']}/"
    if not body.gcs_path.startswith(expected_prefix):
        raise HTTPException(403, "gcs_path does not belong to the authenticated user")

    # Pre-check: ensure minimum balance to reduce result-loss on token deduction failure
    C_BASE = 50  # minimum possible token cost
    if user_data.get("token_balance", 0) < C_BASE:
        raise HTTPException(402, "Insufficient token balance to initiate GPU inference")

    # Forward to GPU worker
    if not settings.gpu_worker_url:
        raise HTTPException(503, "GPU worker not configured")

    async with httpx.AsyncClient(timeout=3600) as http:
        try:
            gpu_resp = await http.post(
                f"{settings.gpu_worker_url}/api/inference/internal",
                json={
                    "gcs_path": body.gcs_path,
                    "model_name": body.model_name,
                    "tile_width": body.tile_width,
                    "tile_height": body.tile_height,
                    "min_distance": body.min_distance,
                    "conf_threshold": body.conf_threshold,
                    "nms_threshold": body.nms_threshold,
                    "user_uid": current_user["sub"],
                },
                headers={"X-Internal-Secret": settings.cleanup_secret},
            )
        except Exception as e:
            raise HTTPException(502, f"GPU worker unreachable: {e}")

    if gpu_resp.status_code != 200:
        raise HTTPException(gpu_resp.status_code, f"GPU worker error: {gpu_resp.text}")

    result = gpu_resp.json()

    # Deduct tokens based on actual file stats returned by GPU worker
    l_sqm = result.get("area_sqm", 0.0)
    s_gb = result.get("file_size_gb", 0.0)
    cost = calculate_tokens(l_sqm=l_sqm, s_gb=s_gb)
    try:
        await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
    except ValueError as e:
        raise HTTPException(402, str(e))

    return JSONResponse({**result, "tokens_deducted": cost})


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

@router.post("/cleanup")
@router.get("/cleanup")
def cleanup_temporary_files(request: Request):
    """
    Purge old files from uploads, results, and output directories.
    Can be triggered via Google Cloud Scheduler.
    Example: GET /api/cleanup?secret=YOUR_SECRET
    """
    # If a secret is configured, require it via query param or header
    if CLEANUP_SECRET:
        provided_secret = request.query_params.get("secret") or request.headers.get("x-cleanup-secret")
        if provided_secret != CLEANUP_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

    summary = _purge_old_files(MAX_AGE_HOURS)
    
    total_deleted = sum(d["deleted"] for d in summary.values())
    total_skipped = sum(d["skipped"] for d in summary.values())
    total_errors = sum(d["errors"] for d in summary.values())

    return JSONResponse({
        "message": f"Cleanup executed. Deleted {total_deleted} files.",
        "details": summary,
        "max_age_hours": MAX_AGE_HOURS
    })
