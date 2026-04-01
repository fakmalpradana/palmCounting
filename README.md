# 🌴 Palm Counter

**YOLO ONNX geospatial palm-tree detection — GeoTIFF in, GeoJSON out.**

A full-stack WebGIS application that tiles large GeoTIFF rasters, runs YOLO ONNX inference on each tile, converts pixel detections back to geographic coordinates, and delivers the results as a GeoJSON FeatureCollection rendered on an interactive Leaflet map.

> **Copyright © 2026 Geo AI Twinverse.**
> Contributors: Fikri Kurniawan, Fairuz Akmal Pradana

---

## Features

| Feature | Details |
|---|---|
| Geospatial tiling | Overlapping 640 × 640 tile grid from any CRS GeoTIFF |
| ONNX inference | ONNX Runtime (CPU / CUDA) with cv2.dnn fallback |
| Overlap filtering | R-tree spatial index deduplication across tile boundaries |
| GeoJSON output | WGS 84 FeatureCollection with class name & confidence |
| REST API | FastAPI with Swagger UI at `/docs` |
| WebGIS viewer | Leaflet + OSM basemap, raster + vector layer toggles |
| Model management | Upload & switch ONNX models without restarting |
| Download | One-click GeoJSON export after inference |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.11** (3.10+ works) |
| Miniconda / Anaconda | Any recent version |
| GDAL (via rasterio) | Bundled via conda-forge |
| NVIDIA CUDA Toolkit *(optional)* | 11.8 or 12.x |
| Node.js | **Not required** — frontend is plain HTML/JS |

---

## Project Structure

```
palmCounting/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── core/
│   │   └── inference.py        # Tiling, YOLO detection, geo math
│   └── routers/
│       └── inference.py        # API route handlers
├── models/
│   ├── best_1.onnx             # Default YOLO ONNX model
│   └── data.yaml               # Class labels
├── static/
│   └── index.html              # Single-page WebGIS frontend
├── uploads/                    # Uploaded GeoTIFFs (auto-created)
├── results/                    # GeoJSON outputs (auto-created)
├── main.py                     # Original standalone CLI script
├── requirements.txt            # CPU pip dependencies
├── requirements-cuda.txt       # CUDA pip dependencies
├── environment.yml             # Conda CPU environment
├── environment-cuda.yml        # Conda CUDA environment
├── install_cpu.sh              # macOS/Linux CPU installer
├── install_cuda.sh             # Linux/WSL2 CUDA installer
├── install_cpu_windows.bat     # Windows CPU installer
├── install_cuda_windows.bat    # Windows CUDA installer
└── run.sh                      # Quick-start server script
```

---

## Installation

### macOS / Linux — CPU

```bash
# Option A: one-command installer
bash install_cpu.sh

# Option B: manual
conda env create -f environment.yml
conda activate palmcounting
```

### Windows — CPU

```bat
install_cpu_windows.bat
```
Or manually in Anaconda Prompt:
```bat
conda env create -f environment.yml
conda activate palmcounting
```

### Windows / Linux — CUDA (NVIDIA GPU)

> Requires CUDA Toolkit 11.8 or 12.x and matching cuDNN installed system-wide.

**Linux / WSL2:**
```bash
bash install_cuda.sh
conda activate palmcounting-cuda
```

**Windows:**
```bat
install_cuda_windows.bat
conda activate palmcounting-cuda
```

### Manual pip install (any platform)

```bash
# CPU
pip install -r requirements.txt

# CUDA
pip install -r requirements-cuda.txt
```

---

## Usage

### 1. Add your ONNX model

Place your YOLO ONNX model in the `models/` directory:

```
models/
├── your_model.onnx
└── data.yaml          # must list class names under the "names" key
```

`data.yaml` format:
```yaml
nc: 1
names: ['Nyawit']
```

### 2. Start the server

```bash
# Activate environment first
conda activate palmcounting        # or palmcounting-cuda

# Start API + frontend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the convenience script:
```bash
bash run.sh
```

### 3. Open the web UI

Navigate to **http://localhost:8000** in your browser.

#### UX Flow

| Step | Action |
|---|---|
| **Upload** | Drop or select a `.tif` / `.tiff` GeoTIFF file |
| **Select model** | Choose from the dropdown (or upload a new `.onnx` model) |
| **Configure** | Expand *Advanced Parameters* to tune confidence, NMS, tile size, min-distance |
| **Run** | Click **RUN DETECTION** — a scan animation plays while the backend processes |
| **View** | Results appear on a Leaflet map with raster + vector layer toggles |
| **Download** | Click **⬇ Download** to save the GeoJSON |

### 4. Interactive API docs

Swagger UI: **http://localhost:8000/docs**
ReDoc:       **http://localhost:8000/redoc**

---

## API Reference

### `POST /api/inference`

Run palm detection on an uploaded GeoTIFF.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | — | GeoTIFF raster (`.tif` / `.tiff`) |
| `model_name` | string | `best_1.onnx` | Model filename in `models/` |
| `tile_width` | int | `640` | Tile width in pixels |
| `tile_height` | int | `640` | Tile height in pixels |
| `min_distance` | float | `1.0` | Minimum inter-detection distance (CRS units) |
| `conf_threshold` | float | `0.25` | Confidence threshold `[0,1]` |
| `nms_threshold` | float | `0.40` | NMS IoU threshold `[0,1]` |

**Response:**
```json
{
  "file_id": "uuid",
  "duration_seconds": 12.4,
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": { "type": "Point", "coordinates": [lon, lat] },
        "properties": { "class_name": "Nyawit", "confidence": 0.87 }
      }
    ],
    "metadata": {
      "count": 142,
      "crs": "EPSG:32650",
      "raster_bounds": [west, south, east, north],
      "detection_bounds": [west, south, east, north],
      "duration_seconds": 12.4
    }
  }
}
```

---

### `GET /api/download/{file_id}`

Download the GeoJSON result as a file attachment.

```
GET /api/download/3f2a1b4c-...
→ palm_detections_3f2a1b4c.geojson
```

---

### `GET /api/preview/{file_id}`

Return a percentile-stretched RGB PNG of the uploaded raster for map overlay.
Bounds are returned in response headers: `X-Raster-West/South/East/North`.

---

### `GET /api/models`

List all available ONNX models.

```json
{ "models": ["best_1.onnx", "custom_v2.onnx"] }
```

---

### `POST /api/models`

Upload a new ONNX model file.

**Form fields:** `file` (`.onnx`)

```json
{ "message": "Model 'custom_v2.onnx' uploaded.", "model_name": "custom_v2.onnx" }
```

---

## CUDA Notes

The inference backend is auto-selected at startup:

```
Priority 1 → onnxruntime-gpu  (if installed + CUDA device visible)
Priority 2 → onnxruntime       (CPU, all platforms)
Priority 3 → cv2.dnn           (OpenCV fallback, CPU only)
```

To confirm which provider is active, check the server logs on startup.

---

## License

Copyright © 2026 **Geo AI Twinverse**.
Contributors: **Fikri Kurniawan**, **Fairuz Akmal Pradana**
