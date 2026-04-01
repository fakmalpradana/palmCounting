"""
/api/inference      — upload a GeoTIFF, get back GeoJSON detections.
/api/preview/<id>   — stretched-RGB PNG + WGS84 bounds for the map overlay.
"""

import io
import os
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

MODEL_PATH = Path("models/best_1.onnx")
YAML_PATH = Path("models/data.yaml")


@router.post("/inference")
async def infer(
    file: UploadFile = File(...),
    tile_width: int = Form(640),
    tile_height: int = Form(640),
    min_distance: float = Form(1.0),
    conf_threshold: float = Form(0.25),
    nms_threshold: float = Form(0.4),
):
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files (.tif / .tiff) are accepted.")

    if not MODEL_PATH.exists():
        raise HTTPException(500, f"Model not found at {MODEL_PATH}")
    if not YAML_PATH.exists():
        raise HTTPException(500, f"YAML not found at {YAML_PATH}")

    # Save upload with a stable ID so the frontend can fetch it back
    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}.tif"
    content = await file.read()
    save_path.write_bytes(content)

    try:
        geojson = run_inference(
            input_tif_path=str(save_path),
            model_path=str(MODEL_PATH),
            yaml_path=str(YAML_PATH),
            tile_width=tile_width,
            tile_height=tile_height,
            min_distance=min_distance,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    return JSONResponse({"file_id": file_id, "geojson": geojson})


@router.get("/preview/{file_id}")
def get_preview(file_id: str):
    """Return a web-displayable PNG with raster bounds in the response body."""
    if not all(c in "0123456789abcdef-" for c in file_id):
        raise HTTPException(400, "Invalid file ID.")
    path = UPLOAD_DIR / f"{file_id}.tif"
    if not path.exists():
        raise HTTPException(404, "Raster not found.")

    with rasterio.open(path) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        n_bands = src.count

        # Pick RGB or grayscale bands
        if n_bands >= 3:
            data = src.read([1, 2, 3])
        else:
            data = np.stack([src.read(1)] * 3)

        # Percentile stretch per band, handle nodata
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
            stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            rgb[:, :, i] = stretched

    img = Image.fromarray(rgb)
    # Downsample large rasters for web delivery
    max_dim = 2048
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    w, s, e, n = bounds_wgs84
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Raster-West": str(w),
            "X-Raster-South": str(s),
            "X-Raster-East": str(e),
            "X-Raster-North": str(n),
            "Access-Control-Expose-Headers": "X-Raster-West,X-Raster-South,X-Raster-East,X-Raster-North",
        },
    )
