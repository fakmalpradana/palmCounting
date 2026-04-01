"""
/api/inference          — upload GeoTIFF → GeoJSON detections (+ timing).
/api/preview/<id>       — stretched-RGB PNG for the map overlay.
/api/download/<id>      — download result as a GeoJSON file.
/api/models             — list available ONNX models (GET) or upload one (POST).
"""

import io
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import rasterio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from rasterio.warp import transform_bounds

from app.core.inference import run_inference

router = APIRouter(prefix="/api", tags=["inference"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

YAML_PATH = MODELS_DIR / "data.yaml"


def _validate_file_id(file_id: str) -> None:
    """Reject file IDs that are not UUID-shaped (hex + dashes only)."""
    if not all(c in "0123456789abcdef-" for c in file_id):
        raise HTTPException(400, "Invalid file ID.")


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
):
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files (.tif / .tiff) are accepted.")

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise HTTPException(404, f"Model '{model_name}' not found. Upload it first.")
    if not YAML_PATH.exists():
        raise HTTPException(500, f"data.yaml not found at {YAML_PATH}")

    # Persist upload
    file_id = str(uuid.uuid4())
    raster_path = UPLOAD_DIR / f"{file_id}.tif"
    raster_path.write_bytes(await file.read())

    # Timed inference
    t0 = time.perf_counter()
    try:
        geojson = run_inference(
            input_tif_path=str(raster_path),
            model_path=str(model_path),
            yaml_path=str(YAML_PATH),
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
def list_models():
    """Return all .onnx files present in the models directory."""
    models = sorted(p.name for p in MODELS_DIR.glob("*.onnx"))
    return JSONResponse({"models": models})


@router.post("/models")
async def upload_model(file: UploadFile = File(...)):
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
