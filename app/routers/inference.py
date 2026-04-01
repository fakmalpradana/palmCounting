"""
/api/inference  — upload a GeoTIFF, get back GeoJSON detections.
/api/raster/<id> — serve the uploaded raster back to the frontend.
"""

import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

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


@router.get("/raster/{file_id}")
def get_raster(file_id: str):
    # Sanitise: only hex + dashes (UUID format)
    if not all(c in "0123456789abcdef-" for c in file_id):
        raise HTTPException(400, "Invalid file ID.")
    path = UPLOAD_DIR / f"{file_id}.tif"
    if not path.exists():
        raise HTTPException(404, "Raster not found.")
    return FileResponse(str(path), media_type="image/tiff")
