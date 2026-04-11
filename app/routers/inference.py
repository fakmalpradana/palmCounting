# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana
from __future__ import annotations  # allows X | Y union hints on Python 3.9
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
from app.core.firestore_client import mark_landcover_free_used, mark_palm_free_used
from app.core.inference import run_inference
from app.core.land_cover_inference import LC_PALETTE, run_land_cover_inference

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

# Default model baked into the Docker image — survives scale-to-zero.
# Use an absolute path derived from __file__ so this resolves correctly
# regardless of the working directory (local dev vs Docker WORKDIR /app).
DEFAULT_MODELS_DIR = (Path(__file__).resolve().parent.parent / "models" / "default")
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
# Cloud Run's hard request-body limit is 32 MB and cannot be increased.
# Free-tier users upload directly (no GCS bypass), so the practical cap
# must stay below that ceiling.  25 MB gives comfortable headroom for
# multipart overhead.  Commercial users have no limit — they upload via
# GCS signed URLs which bypass Cloud Run entirely.
FREE_TIER_MAX_BYTES = 25 * 1024 * 1024   # 25 MB — Cloud Run direct-upload limit

# Model names are fixed per task — clients cannot override them.
# This prevents cross-contamination between the YOLO palm model and the
# SwinUnet land-cover model (e.g. running unet_swin.onnx against the YOLO
# inference pipeline would produce an ONNX shape-mismatch crash).
PALM_MODEL_NAME = "palmCounting-model.onnx"
LANDCOVER_MODEL_NAME = "unet_swin.onnx"


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
# Pre-flight cost estimate (no inference, no token deduction)
# ---------------------------------------------------------------------------

@router.post("/inference/preflight")
async def preflight_check(
    file: UploadFile = File(...),
    task: str = Form("palm"),  # "palm" | "land_cover"
    current_user: dict = Depends(get_current_user),
):
    """Read raster metadata and return a token cost estimate.

    Accepts the GeoTIFF, reads its header with rasterio, computes the cost
    breakdown, then discards the file.  No inference is run and no tokens are
    deducted.  The frontend shows this breakdown in the confirmation modal
    before the user commits to a full inference run.
    """
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files are accepted.")

    file_bytes = await file.read()
    file_size_bytes = len(file_bytes)
    file_size_gb = file_size_bytes / (1024 ** 3)

    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data:
        raise HTTPException(401, "User record not found")

    token_balance = user_data.get("token_balance", 0)

    # Write to a temp path just long enough to read raster metadata.
    import tempfile
    tmp_id = uuid.uuid4().hex
    tmp_path = UPLOAD_DIR / f"_preflight_{tmp_id}.tif"
    try:
        tmp_path.write_bytes(file_bytes)
        area_sqm = await asyncio.to_thread(get_raster_area_sqm, str(tmp_path))
    except Exception as exc:
        raise HTTPException(422, f"Could not read raster metadata: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    area_ha = area_sqm / 10_000
    area_cost = math.ceil((area_sqm / 10_000) * W_AREA)
    size_cost = math.ceil(file_size_gb * W_SIZE)
    total_cost = math.ceil(C_BASE + area_cost + size_cost)

    # Free-tier users: no token math needed; surface trial status instead.
    if token_balance == 0:
        trial_field = "free_palm_used" if task == "palm" else "free_landcover_used"
        trial_used = user_data.get(trial_field, False)
        over_size = file_size_bytes > FREE_TIER_MAX_BYTES
        return JSONResponse({
            "tier": "free",
            "task": task,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "area_ha": round(area_ha, 2),
            "trial_used": trial_used,
            "over_size_limit": over_size,
            "free_tier_max_mb": FREE_TIER_MAX_BYTES // (1024 * 1024),
        })

    # Commercial tier: full cost breakdown.
    balance_after = token_balance - total_cost
    return JSONResponse({
        "tier": "commercial",
        "task": task,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "file_size_gb": round(file_size_gb, 4),
        "area_sqm": round(area_sqm, 2),
        "area_ha": round(area_ha, 2),
        "base_cost": C_BASE,
        "area_cost": area_cost,
        "size_cost": size_cost,
        "total_cost": total_cost,
        "token_balance": token_balance,
        "balance_after": balance_after,
        "can_afford": balance_after >= 0,
    })


# ---------------------------------------------------------------------------
# Pre-flight for large files already uploaded to GCS
# ---------------------------------------------------------------------------

class PreflightGcsRequest(BaseModel):
    gcs_path: str
    task: str = "palm"  # "palm" | "land_cover"


@router.post("/inference/preflight-gcs")
async def preflight_check_gcs(
    body: PreflightGcsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Cost estimate for a file that is already in GCS.

    Used when the file is too large for Cloud Run's 32 MB request limit.
    File size is read from GCS blob metadata; raster area is read via
    rasterio's /vsigs/ GDAL driver (header-only, no full download).
    Returns the same JSON envelope as /api/inference/preflight.
    """
    expected_prefix = f"uploads/{current_user['sub']}/"
    if not body.gcs_path.startswith(expected_prefix):
        raise HTTPException(403, "gcs_path does not belong to the authenticated user")

    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data:
        raise HTTPException(401, "User record not found")

    token_balance = user_data.get("token_balance", 0)
    if token_balance == 0:
        raise HTTPException(403, "Large-file preflight requires a commercial tier account")

    # File size from GCS blob metadata — fast, no download
    try:
        gcs_client = gcs.Client(project=settings.firestore_project_id)
        blob = gcs_client.bucket(settings.gcs_bucket_name).blob(body.gcs_path)
        await asyncio.to_thread(blob.reload)
        file_size_bytes: int = blob.size  # type: ignore[assignment]
    except Exception as exc:
        raise HTTPException(500, f"Could not read GCS file metadata: {exc}")

    file_size_gb = file_size_bytes / (1024 ** 3)

    # Raster area via GDAL /vsigs/ — reads only the TIFF IFD, not pixel data
    vsigs_path = f"/vsigs/{settings.gcs_bucket_name}/{body.gcs_path}"
    try:
        area_sqm = await asyncio.to_thread(get_raster_area_sqm, vsigs_path)
    except Exception as exc:
        log.warning("preflight-gcs: vsigs area read failed (%s) — falling back to 0", exc)
        area_sqm = 0.0

    area_ha   = area_sqm / 10_000
    area_cost = math.ceil((area_sqm / 10_000) * W_AREA)
    size_cost = math.ceil(file_size_gb * W_SIZE)
    total_cost = math.ceil(C_BASE + area_cost + size_cost)
    balance_after = token_balance - total_cost

    return JSONResponse({
        "tier":           "commercial",
        "task":           body.task,
        "file_size_bytes": file_size_bytes,
        "file_size_mb":   round(file_size_bytes / (1024 * 1024), 2),
        "file_size_gb":   round(file_size_gb, 4),
        "area_sqm":       round(area_sqm, 2),
        "area_ha":        round(area_ha, 2),
        "base_cost":      C_BASE,
        "area_cost":      area_cost,
        "size_cost":      size_cost,
        "total_cost":     total_cost,
        "token_balance":  token_balance,
        "balance_after":  balance_after,
        "can_afford":     balance_after >= 0,
        "gcs_path":       body.gcs_path,   # echoed back so frontend can reuse it
    })


# ---------------------------------------------------------------------------
# GCS-backed land cover submit (commercial tier, large files)
# ---------------------------------------------------------------------------

class SubmitLandCoverRequest(BaseModel):
    gcs_path: str
    tile_size: int = 512
    overlap: int = 128
    use_filter: bool = True
    min_noise_size: int = 250
    in_channels: int = 3


@router.post("/inference/submit-land-cover")
async def submit_gcs_land_cover(
    body: SubmitLandCoverRequest,
    current_user: dict = Depends(get_current_user),
):
    """Commercial tier: run land cover inference on a file already in GCS.

    Downloads the raster from GCS, runs the SwinUnet pipeline, deducts
    tokens, and returns the polygon GeoJSON.  Mirrors /api/inference/submit
    but for the land-cover task.
    """
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data or user_data.get("token_balance", 0) <= 0:
        raise HTTPException(403, "Submit endpoint is for commercial tier only")

    expected_prefix = f"uploads/{current_user['sub']}/"
    if not body.gcs_path.startswith(expected_prefix):
        raise HTTPException(403, "gcs_path does not belong to the authenticated user")

    token_balance = user_data.get("token_balance", 0)
    if token_balance < C_BASE:
        raise HTTPException(402, "Insufficient token balance to initiate inference")

    # Download from GCS
    gcs_client = gcs.Client(project=settings.firestore_project_id)
    blob = gcs_client.bucket(settings.gcs_bucket_name).blob(body.gcs_path)

    file_id      = str(uuid.uuid4())
    raster_path  = UPLOAD_DIR / f"{file_id}.tif"
    result_tif   = RESULTS_DIR / f"{file_id}_lc.tif"
    result_geojson = RESULTS_DIR / f"{file_id}_lc.geojson"

    try:
        await asyncio.to_thread(blob.download_to_filename, str(raster_path))
    except Exception as exc:
        raise HTTPException(500, f"Failed to download file from GCS: {exc}")

    file_size_gb = raster_path.stat().st_size / (1024 ** 3)

    # Check model
    model_path = _resolve_model_path(LANDCOVER_MODEL_NAME)
    if not model_path.exists():
        raster_path.unlink(missing_ok=True)
        raise HTTPException(404, f"Land cover model '{LANDCOVER_MODEL_NAME}' not found.")

    # Billing pre-check
    l_sqm = await asyncio.to_thread(get_raster_area_sqm, str(raster_path))
    cost  = calculate_tokens(l_sqm=l_sqm, s_gb=file_size_gb)
    if cost > token_balance:
        raster_path.unlink(missing_ok=True)
        raise HTTPException(402, f"Insufficient tokens: have {token_balance}, need {cost}")

    # Inference
    t0 = time.perf_counter()
    try:
        geojson = await asyncio.to_thread(
            run_land_cover_inference,
            str(raster_path),
            str(model_path),
            in_channels=body.in_channels,
            tile_size=body.tile_size,
            overlap=body.overlap,
            use_filter=body.use_filter,
            min_noise_size=body.min_noise_size,
            result_tif_path=str(result_tif),
        )
    except Exception as exc:
        raster_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Land cover inference failed: {exc}") from exc

    duration = round(time.perf_counter() - t0, 2)
    geojson["metadata"]["duration_seconds"] = duration
    result_geojson.write_text(json.dumps(geojson, indent=2))

    # Deduct tokens atomically
    try:
        await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
    except ValueError as exc:
        raise HTTPException(402, str(exc))

    return JSONResponse({
        "file_id":          file_id,
        "task_type":        "land_cover",
        "duration_seconds": duration,
        "tokens_deducted":  cost,
        "geojson":          geojson,
    })


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@router.post("/inference")
async def infer(
    file: UploadFile = File(...),
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
                f"Quota exceeded: free tier file size limit is 25 MB. Your file is "
                f"{file_size_bytes / 1024 / 1024:.1f} MB. Add tokens to process larger files.",
            )
        if user_data.get("free_palm_used", False):
            raise HTTPException(
                403,
                "Your one-time free Palm Counting trial has already been used. "
                "Add tokens to continue.",
            )

    # ── Model / YAML checks ──────────────────────────────────────────────
    model_path = _resolve_model_path(PALM_MODEL_NAME)
    if not model_path.exists():
        raise HTTPException(404, f"Palm counting model '{PALM_MODEL_NAME}' not found. Upload it first.")
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
        # ── Free tier: atomically claim the one-time palm trial ──────────
        try:
            await asyncio.to_thread(mark_palm_free_used, current_user["sub"])
        except ValueError:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(
                403,
                "Your one-time free Palm Counting trial has already been used. "
                "Add tokens to continue.",
            )

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
# Land Cover inference
# ---------------------------------------------------------------------------

@router.post("/inference/land-cover")
async def infer_land_cover(
    file: UploadFile = File(...),
    in_channels: int = Form(3),
    tile_size: int = Form(512),
    overlap: int = Form(128),
    use_filter: bool = Form(True),
    min_noise_size: int = Form(250),
    current_user: dict = Depends(get_current_user),
):
    """
    Land cover classification: GeoTIFF in → polygon GeoJSON + classified raster out.

    Returns the same billing envelope as /api/inference, with an additional
    ``task_type: "land_cover"`` field and a ``class_summary`` in the metadata.
    The classified GeoTIFF is stored server-side so /api/preview/land-cover/<id>
    can render it with the class colour palette.
    """
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files (.tif / .tiff) are accepted.")

    file_bytes      = await file.read()
    file_size_bytes = len(file_bytes)
    file_size_gb    = file_size_bytes / (1024 ** 3)

    # ── Fetch user for tier decision ─────────────────────────────────────
    user_data = await asyncio_to_thread_get_user(current_user["sub"])
    if not user_data:
        raise HTTPException(401, "User record not found")

    token_balance = user_data.get("token_balance", 0)

    # ── Free tier pre-checks ─────────────────────────────────────────────
    if token_balance == 0:
        if file_size_bytes > FREE_TIER_MAX_BYTES:
            raise HTTPException(
                403,
                f"Quota exceeded: free tier file size limit is 25 MB. "
                f"Your file is {file_size_bytes / 1024 / 1024:.1f} MB. "
                f"Add tokens to process larger files.",
            )
        if user_data.get("free_landcover_used", False):
            raise HTTPException(
                403,
                "Your one-time free Land Cover Analysis trial has already been used. "
                "Add tokens to continue.",
            )

    # ── Model check ──────────────────────────────────────────────────────
    model_path = _resolve_model_path(LANDCOVER_MODEL_NAME)
    if not model_path.exists():
        raise HTTPException(404, f"Land cover model '{LANDCOVER_MODEL_NAME}' not found. Upload it first.")

    # ── Save upload ──────────────────────────────────────────────────────
    file_id      = str(uuid.uuid4())
    raster_path  = UPLOAD_DIR / f"{file_id}.tif"
    raster_path.write_bytes(file_bytes)

    # ── Billing ──────────────────────────────────────────────────────────
    tokens_deducted = 0
    if token_balance > 0:
        l_sqm = await asyncio.to_thread(get_raster_area_sqm, str(raster_path))
        cost  = calculate_tokens(l_sqm=l_sqm, s_gb=file_size_gb)
        if cost > token_balance:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(402, f"Insufficient tokens: have {token_balance}, need {cost}")
        try:
            await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
            tokens_deducted = cost
        except ValueError as e:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(402, str(e))
    else:
        # ── Free tier: atomically claim the one-time land cover trial ────
        try:
            await asyncio.to_thread(mark_landcover_free_used, current_user["sub"])
        except ValueError:
            raster_path.unlink(missing_ok=True)
            raise HTTPException(
                403,
                "Your one-time free Land Cover Analysis trial has already been used. "
                "Add tokens to continue.",
            )

    # ── Inference ────────────────────────────────────────────────────────
    result_tif     = RESULTS_DIR / f"{file_id}_lc.tif"
    result_geojson = RESULTS_DIR / f"{file_id}_lc.geojson"

    t0 = time.perf_counter()
    try:
        geojson = await asyncio.to_thread(
            run_land_cover_inference,
            str(raster_path),
            str(model_path),
            in_channels=in_channels,
            tile_size=tile_size,
            overlap=overlap,
            use_filter=use_filter,
            min_noise_size=min_noise_size,
            result_tif_path=str(result_tif),
        )
    except Exception as exc:
        raster_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Land cover inference failed: {exc}") from exc

    duration = round(time.perf_counter() - t0, 2)
    geojson["metadata"]["duration_seconds"] = duration

    result_geojson.write_text(json.dumps(geojson, indent=2))

    return JSONResponse({
        "file_id":          file_id,
        "task_type":        "land_cover",
        "duration_seconds": duration,
        "tokens_deducted":  tokens_deducted,
        "geojson":          geojson,
    })


# ---------------------------------------------------------------------------
# Land Cover preview (palette-coloured PNG)
# ---------------------------------------------------------------------------

@router.get("/preview/land-cover/{file_id}")
def get_land_cover_preview(file_id: str):
    """
    Return a palette-coloured PNG of the classified raster stored at
    results/{file_id}_lc.tif, with WGS84 bounds in response headers.
    """
    _validate_file_id(file_id)
    path = RESULTS_DIR / f"{file_id}_lc.tif"
    if not path.exists():
        raise HTTPException(404, "Land cover result not found. Run inference first.")

    with rasterio.open(path) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        data         = src.read(1)   # single-band class indices (uint8)

    # Map class indices to RGB using the palette
    h, w = data.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in LC_PALETTE.items():
        rgb[data == cls_id] = color

    img = Image.fromarray(rgb)
    if max(img.size) > 2048:
        ratio = 2048 / max(img.size)
        img = img.resize(
            (int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    bw, bs, be, bn = bounds_wgs84
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Raster-West":  str(bw), "X-Raster-South": str(bs),
            "X-Raster-East":  str(be), "X-Raster-North": str(bn),
            "Access-Control-Expose-Headers":
                "X-Raster-West,X-Raster-South,X-Raster-East,X-Raster-North",
        },
    )


# ---------------------------------------------------------------------------
# Land Cover GeoJSON download
# ---------------------------------------------------------------------------

@router.get("/download/land-cover/{file_id}")
def download_land_cover_result(file_id: str):
    """Download the land-cover polygon GeoJSON for a completed inference run."""
    _validate_file_id(file_id)
    path = RESULTS_DIR / f"{file_id}_lc.geojson"
    if not path.exists():
        raise HTTPException(404, "Land cover result not found. Run inference first.")
    return Response(
        content=path.read_bytes(),
        media_type="application/geo+json",
        headers={
            "Content-Disposition":
                f'attachment; filename="land_cover_{file_id[:8]}.geojson"'
        },
    )


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
    log.info(
        "list_models — DEFAULT_MODELS_DIR=%s defaults=%s custom=%s",
        DEFAULT_MODELS_DIR, sorted(defaults), sorted(custom),
    )
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
        raise HTTPException(402, "Insufficient token balance to initiate inference")

    # ── GPU worker path (when configured) ───────────────────────────────
    if settings.gpu_worker_url:
        async with httpx.AsyncClient(timeout=3600) as http:
            try:
                gpu_resp = await http.post(
                    f"{settings.gpu_worker_url}/api/inference/internal",
                    json={
                        "gcs_path": body.gcs_path,
                        "model_name": PALM_MODEL_NAME,
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
        l_sqm = result.get("area_sqm", 0.0)
        s_gb = result.get("file_size_gb", 0.0)
        cost = calculate_tokens(l_sqm=l_sqm, s_gb=s_gb)
        try:
            await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
        except ValueError as e:
            raise HTTPException(402, str(e))
        return JSONResponse({**result, "tokens_deducted": cost})

    # ── CPU fallback path (no GPU worker configured) ─────────────────────
    # Download the GCS file, run local inference, then bill based on actual stats.
    log.info("GPU worker not configured — falling back to CPU inference for gcs_path=%s", body.gcs_path)

    gcs_client = gcs.Client(project=settings.firestore_project_id)
    bucket = gcs_client.bucket(settings.gcs_bucket_name)
    blob = bucket.blob(body.gcs_path)

    file_id = str(uuid.uuid4())
    raster_path = UPLOAD_DIR / f"{file_id}.tif"
    try:
        await asyncio.to_thread(blob.download_to_filename, str(raster_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to download file from GCS: {e}")

    file_size_bytes = raster_path.stat().st_size
    file_size_gb = file_size_bytes / (1024 ** 3)

    model_path = _resolve_model_path(PALM_MODEL_NAME)
    if not model_path.exists():
        raster_path.unlink(missing_ok=True)
        raise HTTPException(404, f"Palm counting model '{PALM_MODEL_NAME}' not found.")
    yaml_path = _resolve_yaml_path()
    if not yaml_path.exists():
        raster_path.unlink(missing_ok=True)
        raise HTTPException(500, "data.yaml not found.")

    l_sqm = await asyncio.to_thread(get_raster_area_sqm, str(raster_path))
    cost = calculate_tokens(l_sqm=l_sqm, s_gb=file_size_gb)
    token_balance = user_data.get("token_balance", 0)
    if cost > token_balance:
        raster_path.unlink(missing_ok=True)
        raise HTTPException(402, f"Insufficient tokens: have {token_balance}, need {cost}")

    t0 = time.perf_counter()
    try:
        geojson = run_inference(
            input_tif_path=str(raster_path),
            model_path=str(model_path),
            yaml_path=str(yaml_path),
            tile_width=body.tile_width,
            tile_height=body.tile_height,
            min_distance=body.min_distance,
            conf_threshold=body.conf_threshold,
            nms_threshold=body.nms_threshold,
        )
    except Exception as exc:
        raster_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    duration = round(time.perf_counter() - t0, 2)
    geojson["metadata"]["duration_seconds"] = duration

    result_path = RESULTS_DIR / f"{file_id}.geojson"
    result_path.write_text(json.dumps(geojson, indent=2))

    try:
        await asyncio_to_thread_deduct_tokens(current_user["sub"], cost)
    except ValueError as e:
        raise HTTPException(402, str(e))

    return JSONResponse({
        "file_id": file_id,
        "duration_seconds": duration,
        "tokens_deducted": cost,
        "geojson": geojson,
    })


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
