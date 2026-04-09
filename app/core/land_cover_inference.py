# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana
"""
Land Cover Classification inference pipeline for the palmCounting API.

Self-contained module adapted from inference_all/.  Uses a SwinUnet ONNX model
with sliding-window tiling, CLAHE preprocessing, Hanning-weighted blending,
sieve noise removal, and in-memory vectorisation to GeoJSON.

Public entry point:
    run_land_cover_inference(input_tif_path, model_path, **kwargs) -> dict
"""

import logging
import time

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes, sieve
from rasterio.warp import transform_bounds
from shapely.geometry import shape

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class definitions (mirrors inference_all/config/settings.py)
# ---------------------------------------------------------------------------

LC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0,   0,   0),   # background
    1: (0,   0, 127),   # ladang merica
    2: (0,   0, 255),   # low vegetation
    3: (0, 127,   0),   # high vegetation
    4: (0, 127, 127),   # tanah terbuka
    5: (0, 127, 255),   # jalan
    6: (0, 255,   0),   # bangunan
    7: (0, 255, 127),   # badan air
}

LC_CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "ladang merica",
    2: "low vegetation",
    3: "high vegetation",
    4: "tanah terbuka",
    5: "jalan",
    6: "bangunan",
    7: "badan air",
}

_EXCLUDE_CLASSES = [5, 6]   # road + building — tighter sieve threshold
_TILE_SIZE       = 512
_OVERLAP         = 128
_IN_CHANNELS     = 3
_MIN_NOISE_SIZE  = 250
_TARGET_RES_M    = 0.5
_CLAHE_CLIP      = 2.0
_CLAHE_GRID      = 8
_SIEVE_CONNECT   = 4
_SIMPLIFY_TOL    = 0.4

# ---------------------------------------------------------------------------
# Optional topology-preserving simplification (topojson)
# ---------------------------------------------------------------------------

try:
    import topojson as _tp
    _HAS_TOPOJSON = True
except ImportError:
    _HAS_TOPOJSON = False

# ---------------------------------------------------------------------------
# ONNX Runtime backend (auto-selects CUDA when available)
# ---------------------------------------------------------------------------

_ORT_SESSION_CLASS = None
_ORT_PROVIDERS: list[str] = []

try:
    import onnxruntime as _ort

    _avail = _ort.get_available_providers()
    if "CUDAExecutionProvider" in _avail:
        _ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        log.info("Land-cover ONNX backend: CUDA")
    else:
        _ORT_PROVIDERS = ["CPUExecutionProvider"]
        log.info("Land-cover ONNX backend: CPU")
    _ORT_SESSION_CLASS = _ort.InferenceSession
except ImportError:
    log.warning("onnxruntime not installed — land-cover inference unavailable")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_session(model_path: str):
    """Load the SwinUnet ONNX model. Raises RuntimeError if ort is missing."""
    if _ORT_SESSION_CLASS is None:
        raise RuntimeError(
            "onnxruntime is required for land-cover inference. "
            "Install it with: pip install onnxruntime"
        )
    session = _ORT_SESSION_CLASS(str(model_path), providers=_ORT_PROVIDERS)
    log.info("Loaded land-cover ONNX model: %s", model_path)
    return session


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """Per-band CLAHE contrast enhancement.  img shape: (C, H, W)."""
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=(_CLAHE_GRID, _CLAHE_GRID))
    bands = []
    for band in img:
        dtype = band.dtype
        b8 = band if dtype == np.uint8 else cv2.normalize(
            band, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        e8 = clahe.apply(b8)
        if dtype != np.uint8:
            max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
            e = cv2.normalize(e8.astype(np.float32), None, 0, max_val,
                              cv2.NORM_MINMAX).astype(dtype)
        else:
            e = e8
        bands.append(e)
    return np.stack(bands, axis=0)


def _weight_mask(tile_size: int, overlap: int) -> np.ndarray:
    """2-D Hanning blending mask with linear fade-in/out in overlap regions."""
    y, x = np.hanning(tile_size), np.hanning(tile_size)
    m = np.outer(y, x)
    m /= m.max()
    h = overlap // 2
    m[:h,  :] *= np.linspace(0, 1, h)[:, None]
    m[-h:, :] *= np.linspace(1, 0, h)[:, None]
    m[:,  :h] *= np.linspace(0, 1, h)[None, :]
    m[:, -h:] *= np.linspace(1, 0, h)[None, :]
    return m.astype(np.float32)


def _sliding_window(
    session,
    img: np.ndarray,
    n_classes: int,
    tile_size: int,
    overlap: int,
    in_channels: int,
) -> np.ndarray:
    """
    Weighted sliding-window inference over a full-scene raster.

    Returns ``pred_full`` (H, W) uint8 with class indices 0–n_classes.
    """
    H, W   = img.shape[1], img.shape[2]
    step   = tile_size - overlap
    wm     = _weight_mask(tile_size, overlap)
    prob   = np.zeros((n_classes, H, W), dtype=np.float32)
    cnt    = np.zeros((H, W), dtype=np.float32)
    inp_nm = session.get_inputs()[0].name

    n_tiles = len(range(0, H, step)) * len(range(0, W, step))
    log.info("Land-cover: processing %d tiles (%dx%d px image)…", n_tiles, W, H)

    for y in range(0, H, step):
        for x in range(0, W, step):
            ye, xe = min(y + tile_size, H), min(x + tile_size, W)
            patch  = img[:, y:ye, x:xe].astype(np.float32).transpose(1, 2, 0)
            ph, pw = patch.shape[:2]

            if patch.max() > 1.0:
                patch /= 255.0

            # channel count adjustment
            nc = patch.shape[2]
            if nc < in_channels:
                patch = np.concatenate(
                    [patch] + [patch[:, :, -1:]] * (in_channels - nc), axis=-1)
            elif nc > in_channels:
                patch = patch[:, :, :in_channels]

            # zero-pad edge tiles to tile_size
            if ph < tile_size or pw < tile_size:
                patch = np.pad(patch, ((0, tile_size - ph), (0, tile_size - pw), (0, 0)))

            tensor = patch.transpose(2, 0, 1)[None].astype(np.float32)
            probs  = session.run(None, {inp_nm: tensor})[0][0]   # (C, T, T)

            if probs.shape[0] > n_classes:
                probs = probs[-n_classes:]

            probs    = probs[:, :ph, :pw]
            wm_crop  = wm[:ph, :pw]
            for c in range(n_classes):
                prob[c, y:ye, x:xe] += probs[c] * wm_crop
            cnt[y:ye, x:xe] += wm_crop

    # average overlapping predictions
    safe = np.where(cnt == 0, 1.0, cnt)
    prob /= safe[None]

    # argmax in row chunks to stay within memory
    pred = np.zeros((H, W), dtype=np.uint8)
    for y0 in range(0, H, 2048):
        y1 = min(y0 + 2048, H)
        pred[y0:y1] = np.argmax(prob[:, y0:y1].astype(np.float32), axis=0)

    return pred


def _remove_noise(data: np.ndarray, min_size: int) -> np.ndarray:
    """
    Two-pass sieve filter (global pass then per-class pass) directly
    on a numpy array — no temp files required.
    """
    if data.dtype not in (np.uint8, np.uint16, np.int16, np.int32):
        data = data.astype(np.uint16)

    # global pass removes large speckle
    out = sieve(data, size=min_size * 4, connectivity=_SIEVE_CONNECT,
                out=np.zeros_like(data))

    # per-class pass with individual thresholds
    for cls in np.unique(data):
        if cls == 0:
            continue
        thr = max(50, int(min_size * 0.4)) if cls in _EXCLUDE_CLASSES else min_size
        cls_r  = np.where(data == cls, data, np.zeros_like(data))
        sieved = sieve(cls_r, size=thr, connectivity=_SIEVE_CONNECT,
                       out=np.zeros_like(cls_r))
        out = np.where(sieved != 0, sieved, out)

    return out


def _downsample(
    data: np.ndarray, meta: dict, target_m: float
) -> tuple[np.ndarray, dict]:
    """
    Downsample *data* to *target_m* metres/pixel if the current resolution is finer.
    Uses nearest-neighbour resampling (preserves discrete class labels).
    Returns (resampled_array, updated_meta).
    """
    res = (abs(meta["transform"].a) + abs(meta["transform"].e)) / 2
    if res >= target_m:
        return data, meta

    scale = target_m / res
    new_w = max(1, int(meta["width"] / scale))
    new_h = max(1, int(meta["height"] / scale))
    down  = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    new_tf = meta["transform"] * meta["transform"].scale(
        meta["width"] / new_w, meta["height"] / new_h)
    return down, {**meta, "width": new_w, "height": new_h, "transform": new_tf}


def _upsample(data: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Nearest-neighbour upsample back to original (orig_h × orig_w) dimensions."""
    return cv2.resize(data, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def _vectorize(pred: np.ndarray, meta: dict, tol: float) -> gpd.GeoDataFrame:
    """
    Convert the classified raster to a GeoDataFrame of polygons.
    Applies topology-preserving simplification (topojson if available,
    otherwise shapely's Douglas-Peucker).
    """
    crs = meta["crs"]
    feats = [
        (shape(g), int(v))
        for g, v in shapes(pred.astype(np.int16), transform=meta["transform"])
        if v != 0
    ]
    if not feats:
        return gpd.GeoDataFrame(
            columns=["nmr_obj", "nam_kls", "luas_ha", "geometry"], crs=crs)

    geoms, cids = zip(*feats)
    gdf = gpd.GeoDataFrame(
        {
            "nmr_obj": list(cids),
            "nam_kls": [LC_CLASS_NAMES.get(c, f"class_{c}") for c in cids],
            "geometry": list(geoms),
        },
        crs=crs,
    )
    gdf["luas_ha"] = gdf.geometry.area.abs() / 10_000

    if _HAS_TOPOJSON:
        topo = _tp.Topology(gdf, prequantize=False)
        gdf  = topo.toposimplify(
            tol, prevent_oversimplify=True, simplify_algorithm="dp"
        ).to_gdf()
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
    else:
        gdf.geometry = gdf.geometry.simplify(tol, preserve_topology=True)

    return gdf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_land_cover_inference(
    input_tif_path: str,
    model_path: str,
    *,
    in_channels:        int   = _IN_CHANNELS,
    tile_size:          int   = _TILE_SIZE,
    overlap:            int   = _OVERLAP,
    use_filter:         bool  = True,
    min_noise_size:     int   = _MIN_NOISE_SIZE,
    target_resolution:  float = _TARGET_RES_M,
    simplify_tolerance: float = _SIMPLIFY_TOL,
    result_tif_path:    str | None = None,
) -> dict:
    """
    Run the full land-cover classification pipeline on a single GeoTIFF.

    Parameters
    ----------
    input_tif_path    : path to the input multi-band GeoTIFF
    model_path        : path to the SwinUnet ONNX weights file
    in_channels       : number of input bands the model expects (default 3)
    tile_size         : sliding-window tile size in pixels (default 512)
    overlap           : overlap between adjacent tiles (default 128)
    use_filter        : apply sieve noise-removal (default True)
    min_noise_size    : minimum connected-component size in pixels (default 250)
    target_resolution : downsample resolution for noise removal in metres (default 0.5)
    simplify_tolerance: polygon simplification tolerance (default 0.4)
    result_tif_path   : if given, the classified single-band GeoTIFF is saved here
                        (used by the /api/preview/land-cover/{file_id} endpoint)

    Returns
    -------
    GeoJSON-compatible dict (FeatureCollection) with polygon features and
    ``metadata`` containing:
        count          — total polygon count
        crs            — original CRS string
        raster_bounds  — [W, S, E, N] in WGS84
        class_summary  — { class_name: {count, area_ha} }
        duration_seconds
    """
    t0 = time.perf_counter()

    # 1. Load ONNX model
    session       = _load_session(model_path)
    n_classes_out = session.get_outputs()[0].shape[1]
    n_classes     = min(n_classes_out, len(LC_PALETTE))
    log.info("Land-cover: %d model classes, %d palette classes", n_classes_out, n_classes)

    # 2. Read input raster
    with rasterio.open(input_tif_path) as src:
        img          = src.read()
        meta         = src.meta.copy()
        bounds_wgs84 = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
        orig_crs     = src.crs
        orig_h       = src.height
        orig_w       = src.width
        orig_tf      = src.transform

    # 3. CLAHE contrast enhancement (per-band)
    img = _apply_clahe(img)

    # 4. Sliding-window inference
    pred = _sliding_window(session, img, n_classes, tile_size, overlap, in_channels)

    # 5. Post-processing — downsample → sieve → upsample
    if use_filter:
        log.info("Land-cover: applying sieve noise removal…")
        down, down_meta = _downsample(pred, meta, target_resolution)
        cleaned         = _remove_noise(down, min_noise_size)
        pred            = _upsample(cleaned, orig_h, orig_w)

    # 6. Persist classified GeoTIFF for the preview endpoint
    if result_tif_path:
        lc_meta = meta.copy()
        lc_meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
        with rasterio.open(result_tif_path, "w", **lc_meta) as dst:
            dst.write(pred.astype("uint8"), 1)
        log.info("Classified GeoTIFF saved: %s", result_tif_path)

    # 7. Vectorise — raster class indices → polygon GeoDataFrame
    log.info("Land-cover: vectorising classified raster…")
    vec_meta = {**meta, "transform": orig_tf, "height": orig_h, "width": orig_w}
    gdf      = _vectorize(pred, vec_meta, simplify_tolerance)

    # 8. Reproject polygons to WGS84 for GeoJSON output
    if not gdf.empty:
        gdf = gdf.to_crs("EPSG:4326")

    # 9. Build per-class summary
    summary: dict[str, dict] = {}
    if not gdf.empty and "nam_kls" in gdf.columns:
        for cls_name, grp in gdf.groupby("nam_kls"):
            summary[str(cls_name)] = {
                "count":   int(len(grp)),
                "area_ha": round(float(grp["luas_ha"].sum()), 4),
            }

    # 10. Assemble GeoJSON output
    geojson: dict = (
        gdf.__geo_interface__ if not gdf.empty
        else {"type": "FeatureCollection", "features": []}
    )
    geojson["metadata"] = {
        "count":            len(gdf),
        "crs":              str(orig_crs),
        "raster_bounds":    bounds_wgs84,    # [W, S, E, N]
        "class_summary":    summary,
        "duration_seconds": round(time.perf_counter() - t0, 2),
    }
    return geojson
