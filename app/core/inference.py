"""
Core YOLO ONNX inference logic — extracted from main.py.
All geo-coordinate math and overlap filtering lives here.
"""

import math
import os
import shutil
import tempfile

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import tifffile as tiff
import yaml
from PIL import Image
from rtree import index
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_labels(yaml_path: str) -> list[str]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["names"]


def load_yolo_model(model_path: str):
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_image_with_overlap(
    input_path: str, output_folder: str, tile_width: int, tile_height: int
) -> None:
    image = tiff.imread(input_path)
    img = Image.fromarray(image)
    img_width, img_height = img.size

    num_tiles_x = math.ceil(img_width / tile_width)
    num_tiles_y = math.ceil(img_height / tile_height)

    overlap_x = (
        math.ceil((num_tiles_x * tile_width - img_width) / (num_tiles_x - 1))
        if num_tiles_x > 1
        else 0
    )
    overlap_y = (
        math.ceil((num_tiles_y * tile_height - img_height) / (num_tiles_y - 1))
        if num_tiles_y > 1
        else 0
    )

    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y
    count = 0

    with rasterio.open(input_path) as src:
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                left = i * step_x
                upper = j * step_y

                if i == num_tiles_x - 1:
                    left = img_width - tile_width
                if j == num_tiles_y - 1:
                    upper = img_height - tile_height

                right = left + tile_width
                lower = upper + tile_height

                tile = img.crop((left, upper, right, lower))
                if tile.size != (tile_width, tile_height):
                    padded = Image.new(img.mode, (tile_width, tile_height), 0)
                    padded.paste(tile, (0, 0))
                    tile = padded

                tile_transform = rasterio.transform.from_bounds(
                    src.bounds.left + left * src.transform[0],
                    src.bounds.top - lower * abs(src.transform[4]),
                    src.bounds.left + right * src.transform[0],
                    src.bounds.top - upper * abs(src.transform[4]),
                    tile_width,
                    tile_height,
                )

                tile_filename = os.path.join(output_folder, f"tile_{count}.tif")
                with rasterio.open(
                    tile_filename,
                    "w",
                    driver="GTiff",
                    height=tile_height,
                    width=tile_width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=tile_transform,
                ) as dst:
                    tile_array = np.array(tile)
                    if len(tile_array.shape) == 2:
                        dst.write(tile_array, 1)
                    else:
                        for band in range(src.count):
                            dst.write(tile_array[:, :, band], band + 1)

                count += 1


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _get_gsd(raster_path: str) -> tuple[float, float]:
    with rasterio.open(raster_path) as src:
        return abs(src.transform[0]), abs(src.transform[4])


def _detect_on_tile(
    image: np.ndarray,
    net,
    labels: list[str],
    input_size: int = 640,
    conf_threshold: float = 0.1,
    nms_threshold: float = 0.25,
) -> list[dict]:
    row, col = image.shape[:2]
    max_rc = max(row, col)
    canvas = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    canvas[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(
        canvas, 1 / 255, (input_size, input_size), swapRB=True, crop=False
    )
    net.setInput(blob)
    preds = net.forward()[0]

    x_factor = max_rc / input_size
    y_factor = max_rc / input_size

    boxes, confidences, classes = [], [], []
    for det in preds:
        conf = det[4]
        if conf <= conf_threshold:
            continue
        class_score = det[5:].max()
        class_id = int(det[5:].argmax())
        if class_score <= 0.25:
            continue
        cx, cy, w, h = det[0:4]
        left = int((cx - 0.5 * w) * x_factor)
        top = int((cy - 0.5 * h) * y_factor)
        boxes.append([left, top, int(w * x_factor), int(h * y_factor)])
        confidences.append(float(conf))
        classes.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    if len(indices) > 0:
        flat = indices.flatten() if isinstance(indices, np.ndarray) else indices
        for i in flat:
            idx = int(i)
            b = boxes[idx]
            results.append(
                {
                    "class_name": labels[classes[idx]],
                    "confidence": confidences[idx],
                    "x": b[0],
                    "y": b[1],
                    "width": b[2],
                    "height": b[3],
                }
            )
    return results


def _filter_overlap(gdf: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    if len(gdf) == 0:
        return gdf

    spatial_idx = index.Index()
    points = []
    for i, row in gdf.iterrows():
        points.append((i, row.geometry))
        spatial_idx.insert(i, row.geometry.bounds)

    unique, seen = [], set()
    for i, pt in points:
        if i in seen:
            continue
        neighbours = list(spatial_idx.intersection(pt.buffer(distance).bounds))
        if neighbours:
            merged = unary_union([points[n][1] for n in neighbours]).centroid
            unique.append({"class_name": gdf.iloc[i]["class_name"],
                           "confidence": gdf.iloc[i]["confidence"],
                           "geometry": merged})
            seen.update(neighbours)

    return gpd.GeoDataFrame(unique, crs=gdf.crs)


# ---------------------------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------------------------

def run_inference(
    input_tif_path: str,
    model_path: str,
    yaml_path: str,
    tile_width: int = 640,
    tile_height: int = 640,
    min_distance: float = 1.0,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.4,
) -> dict:
    """
    Run the full palm-detection pipeline on a single GeoTIFF.

    Returns a GeoJSON-compatible dict (FeatureCollection) with detections,
    and metadata: total count, CRS string, and bounds.
    """
    labels = load_labels(yaml_path)
    net = load_yolo_model(model_path)

    temp_dir = tempfile.mkdtemp(prefix="palm_tiles_")
    try:
        # --- tile ---
        tile_image_with_overlap(input_tif_path, temp_dir, tile_width, tile_height)

        # --- detect ---
        all_results = []
        first_crs = None
        auto_gsd_x = auto_gsd_y = None

        tile_files = [
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if f.endswith(".tif")
        ]

        for tile_path in tqdm(tile_files, desc="Detecting", unit="tile"):
            img = cv2.imread(tile_path)
            if img is None:
                continue

            with rasterio.open(tile_path) as src:
                bounds = src.bounds
                if first_crs is None:
                    first_crs = src.crs
                if auto_gsd_x is None:
                    auto_gsd_x, auto_gsd_y = _get_gsd(tile_path)

            gsd_x = auto_gsd_x
            gsd_y = auto_gsd_y

            dets = _detect_on_tile(img, net, labels, 640, conf_threshold, nms_threshold)
            for d in dets:
                coord_x = (d["x"] + d["width"] / 2) * gsd_x + bounds.left
                coord_y = bounds.top - (d["y"] + d["height"] / 2) * gsd_y
                all_results.append(
                    {
                        "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 4),
                        "geometry": Point(coord_x, coord_y),
                    }
                )

        if not all_results:
            return {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {"count": 0, "crs": str(first_crs), "bounds": None},
            }

        gdf = gpd.GeoDataFrame(all_results, crs=first_crs or "EPSG:4326")
        gdf = _filter_overlap(gdf, min_distance)

        # Re-project to WGS84 for GeoJSON output
        gdf_wgs84 = gdf.to_crs("EPSG:4326")

        geojson = gdf_wgs84.__geo_interface__
        geojson["metadata"] = {
            "count": len(gdf_wgs84),
            "crs": str(first_crs),
            "bounds": list(gdf_wgs84.total_bounds),
        }
        return geojson

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
