# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana
from __future__ import annotations  # allows X | Y union hints on Python 3.9
"""
Land Cover Classification inference pipeline for the palmCounting API.

Self-contained module adapted from inference_all/.  Uses a SwinUnet ONNX model
with sliding-window tiling, CLAHE preprocessing, Hanning-weighted blending,
sieve noise removal, and in-memory vectorisation to GeoJSON.

Public entry point:
    run_land_cover_inference(input_tif_path, model_path, **kwargs) -> dict
"""

import gc
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows as rwin
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
# Strip height for memory-bounded sliding-window inference.
# The probability accumulator is only allocated for STRIP_HEIGHT + 2*tile_size
# rows at a time instead of the full raster, capping peak RAM to ~1 GB per
# strip for a 10 000-column, 8-class image (vs. 9+ GB for full allocation).
_STRIP_HEIGHT    = 2048
# Pixel-count threshold above which in-memory sieve is skipped to avoid OOM.
# Below the threshold: pred (uint8) + downsampled buffers fit well within 16 GiB.
# Above the threshold: windowed inference already gives clean class masks;
# noise removal is skipped and a warning is logged instead.
_OOM_SIEVE_THRESHOLD = 2_000_000_000   # 2 billion pixels ≈ 44 800 × 44 800
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

    Processes the image in horizontal strips of ``_STRIP_HEIGHT`` rows so
    that the float32 probability accumulator never exceeds:

        n_classes × (STRIP_HEIGHT + 2 × tile_size) × W × 4 bytes

    For an 8-class model, W=10 000 and the default constants this is ~1 GB
    per strip instead of 9+ GB for a naïve full-image allocation.

    Each strip includes ``tile_size`` rows of context padding on each side
    so tiles that straddle a strip boundary have full receptive-field context
    and Hanning weights are correctly applied; only the "core" rows
    (without padding) are written to the output.

    Returns ``pred_full`` (H, W) uint8 with class indices 0–n_classes.
    """
    H, W   = img.shape[1], img.shape[2]
    step   = tile_size - overlap
    wm     = _weight_mask(tile_size, overlap)
    inp_nm = session.get_inputs()[0].name

    n_strips = max(1, -(-H // _STRIP_HEIGHT))   # ceiling division
    n_tiles  = len(range(0, H, step)) * len(range(0, W, step))
    log.info(
        "Land-cover: %d tiles (%dx%d px) processed in %d strip(s) of %d rows",
        n_tiles, W, H, n_strips, _STRIP_HEIGHT,
    )

    pred_full = np.zeros((H, W), dtype=np.uint8)

    # Context padding: one full tile_size on each side prevents boundary
    # artefacts where strip-edge tiles would otherwise be truncated.
    pad = tile_size

    row = 0
    while row < H:
        # Extended strip bounds (with context padding, clamped to image)
        ext_y0 = max(0, row - pad)
        ext_y1 = min(H, row + _STRIP_HEIGHT + pad)
        strip  = img[:, ext_y0:ext_y1, :]
        sh     = strip.shape[1]

        # Per-strip probability accumulator — freed at end of each iteration
        prob = np.zeros((n_classes, sh, W), dtype=np.float32)
        cnt  = np.zeros((sh, W), dtype=np.float32)

        for y in range(0, sh, step):
            for x in range(0, W, step):
                ye, xe = min(y + tile_size, sh), min(x + tile_size, W)
                patch  = strip[:, y:ye, x:xe].astype(np.float32).transpose(1, 2, 0)
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
                    patch = np.pad(
                        patch, ((0, tile_size - ph), (0, tile_size - pw), (0, 0)))

                tensor = patch.transpose(2, 0, 1)[None].astype(np.float32)
                probs  = session.run(None, {inp_nm: tensor})[0][0]   # (C, T, T)

                if probs.shape[0] > n_classes:
                    probs = probs[-n_classes:]

                probs   = probs[:, :ph, :pw]
                wm_crop = wm[:ph, :pw]
                for c in range(n_classes):
                    prob[c, y:ye, x:xe] += probs[c] * wm_crop
                cnt[y:ye, x:xe] += wm_crop

        # Average overlapping predictions within this strip
        safe  = np.where(cnt == 0, 1.0, cnt)
        prob /= safe[None]

        # Map strip-local coordinates back to global image coordinates.
        # Only the "core" rows (excluding context padding) go into the output.
        core_y0 = row - ext_y0                    # start of core within strip
        core_y1 = min(row + _STRIP_HEIGHT, H) - ext_y0  # end of core within strip
        core_h  = core_y1 - core_y0

        pred_full[row: row + core_h] = np.argmax(
            prob[:, core_y0:core_y1].astype(np.float32), axis=0
        ).astype(np.uint8)

        del prob, cnt   # release strip memory before allocating the next strip
        row += _STRIP_HEIGHT

    return pred_full


def _sliding_window_windowed(
    session,
    src,          # open rasterio.DatasetReader — pixel data read on demand
    n_classes: int,
    tile_size: int,
    overlap: int,
    in_channels: int,
    dst,          # open rasterio.DatasetWriter — predictions written on demand
) -> None:
    """
    True windowed I/O sliding-window inference.

    The full raster is **never loaded into RAM**.  For each horizontal strip of
    ``_STRIP_HEIGHT`` rows:

    1. Read the strip (plus ``tile_size`` rows of context padding) from *src*
       using ``rasterio.windows``.
    2. Apply CLAHE contrast enhancement to the strip.
    3. Run the Hanning-blended tile loop (same algorithm as ``_sliding_window``).
    4. Write the strip's argmax predictions immediately to *dst*.
    5. Delete all temporary buffers and call ``gc.collect()`` before the next
       strip — this keeps VRAM/RAM usage bounded regardless of raster size.

    Peak RAM per strip ≈ n_classes × (STRIP_HEIGHT + 2×tile_size) × W × 4 bytes.
    For an 8-class model on a 45 000-column raster: ~4.6 GB/strip with 16 GiB
    available — well within budget.
    """
    H, W   = src.height, src.width
    step   = tile_size - overlap
    wm     = _weight_mask(tile_size, overlap)
    inp_nm = session.get_inputs()[0].name
    pad    = tile_size   # context rows added on each side of every strip

    n_strips = max(1, -(-H // _STRIP_HEIGHT))   # ceiling division
    log.info(
        "Land-cover windowed I/O: %dx%d px processed in %d strip(s) of %d rows",
        W, H, n_strips, _STRIP_HEIGHT,
    )

    row = 0
    while row < H:
        ext_y0 = max(0, row - pad)
        ext_y1 = min(H, row + _STRIP_HEIGHT + pad)

        # ── 1. Windowed read from disk ──────────────────────────────────────
        read_win = rwin.Window(
            col_off=0, row_off=ext_y0,
            width=W,   height=ext_y1 - ext_y0,
        )
        strip = src.read(window=read_win)   # shape: (C, sh, W)
        sh    = strip.shape[1]

        # ── 2. CLAHE contrast enhancement for this strip ────────────────────
        strip = _apply_clahe(strip)

        # ── 3. Per-strip probability accumulator ────────────────────────────
        prob = np.zeros((n_classes, sh, W), dtype=np.float32)
        cnt  = np.zeros((sh, W),           dtype=np.float32)

        for y in range(0, sh, step):
            for x in range(0, W, step):
                ye = min(y + tile_size, sh)
                xe = min(x + tile_size, W)
                patch = strip[:, y:ye, x:xe].astype(np.float32).transpose(1, 2, 0)
                ph, pw = patch.shape[:2]

                if patch.max() > 1.0:
                    patch /= 255.0

                nc = patch.shape[2]
                if nc < in_channels:
                    patch = np.concatenate(
                        [patch] + [patch[:, :, -1:]] * (in_channels - nc), axis=-1)
                elif nc > in_channels:
                    patch = patch[:, :, :in_channels]

                if ph < tile_size or pw < tile_size:
                    patch = np.pad(
                        patch,
                        ((0, tile_size - ph), (0, tile_size - pw), (0, 0)),
                    )

                tensor = patch.transpose(2, 0, 1)[None].astype(np.float32)
                probs  = session.run(None, {inp_nm: tensor})[0][0]  # (C, T, T)

                if probs.shape[0] > n_classes:
                    probs = probs[-n_classes:]

                probs   = probs[:, :ph, :pw]
                wm_crop = wm[:ph, :pw]
                for c in range(n_classes):
                    prob[c, y:ye, x:xe] += probs[c] * wm_crop
                cnt[y:ye, x:xe] += wm_crop

        # Average overlapping predictions within this strip
        safe  = np.where(cnt == 0, 1.0, cnt)
        prob /= safe[None]

        # Core rows (excluding context padding)
        core_y0 = row - ext_y0
        core_y1 = min(row + _STRIP_HEIGHT, H) - ext_y0
        core_h  = core_y1 - core_y0

        core_pred = np.argmax(
            prob[:, core_y0:core_y1].astype(np.float32), axis=0,
        ).astype(np.uint8)

        # ── 4. Windowed write — persist predictions immediately ─────────────
        write_win = rwin.Window(col_off=0, row_off=row, width=W, height=core_h)
        dst.write(core_pred[np.newaxis], window=write_win)

        # ── 5. Aggressive memory release before next strip ──────────────────
        del prob, cnt, strip, core_pred
        gc.collect()

        row += _STRIP_HEIGHT


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

    # 2. Read raster HEADER ONLY — never call src.read() without a window.
    #    For a 2 GB compressed TIF, a full read would expand to 10–20 GB in RAM
    #    and OOM the container before inference even starts.
    with rasterio.open(input_tif_path) as src:
        meta         = src.meta.copy()
        bounds_wgs84 = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
        orig_crs     = src.crs
        orig_h       = src.height
        orig_w       = src.width
        orig_tf      = src.transform

    total_pixels = orig_h * orig_w
    log.info(
        "Land-cover: raster %d×%d = %d MP (%s)",
        orig_w, orig_h, total_pixels // 1_000_000,
        "windowed I/O" if total_pixels > 0 else "empty",
    )

    # Always write predictions to a TIF — windowed writing is the only way to
    # process massive rasters without keeping the full pred array in memory.
    # If the caller supplied result_tif_path, use it directly; otherwise use
    # a temp file that is cleaned up after vectorisation.
    _own_tif = (result_tif_path is None)
    if _own_tif:
        tmp_fd, _tmp = tempfile.mkstemp(suffix="_lc_tmp.tif")
        os.close(tmp_fd)
        _tif_path = Path(_tmp)
    else:
        _tif_path = Path(result_tif_path)

    # 3+4. CLAHE + sliding-window inference — read strip-by-strip from disk,
    #      write predictions strip-by-strip to _tif_path.  Peak RAM is bounded
    #      to ~one strip's accumulator regardless of total raster size.
    lc_meta = meta.copy()
    lc_meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
    with rasterio.open(input_tif_path) as src, \
         rasterio.open(str(_tif_path), "w", **lc_meta) as dst:
        _sliding_window_windowed(
            session, src, n_classes, tile_size, overlap, in_channels, dst,
        )
    log.info("Land-cover: windowed inference complete → %s", _tif_path)
    gc.collect()

    # 5. Post-processing — sieve noise removal.
    #    The pred array is uint8 (1 byte/px) — much cheaper than the raw input.
    #    For rasters above _OOM_SIEVE_THRESHOLD pixels we skip sieve to avoid
    #    OOM; the blended strip predictions are already clean.
    if use_filter:
        if total_pixels <= _OOM_SIEVE_THRESHOLD:
            log.info("Land-cover: applying sieve noise removal (%d MP)…",
                     total_pixels // 1_000_000)
            with rasterio.open(str(_tif_path)) as tsrc:
                pred = tsrc.read(1)            # uint8, 1 byte/px
            down, _ = _downsample(pred, {**meta, "width": orig_w,
                                          "height": orig_h, "transform": orig_tf},
                                  target_resolution)
            cleaned  = _remove_noise(down, min_noise_size)
            pred     = _upsample(cleaned, orig_h, orig_w)
            del down, cleaned
            gc.collect()
            # Write sieved result back to the TIF
            with rasterio.open(str(_tif_path), "r+") as tdst:
                tdst.write(pred.astype("uint8"), 1)
        else:
            log.warning(
                "Land-cover: sieve noise removal skipped — raster is %d MP "
                "(> %d MP threshold); increase _OOM_SIEVE_THRESHOLD or RAM to enable.",
                total_pixels // 1_000_000,
                _OOM_SIEVE_THRESHOLD // 1_000_000,
            )
            with rasterio.open(str(_tif_path)) as tsrc:
                pred = tsrc.read(1)
    else:
        with rasterio.open(str(_tif_path)) as tsrc:
            pred = tsrc.read(1)

    # 6. Classified GeoTIFF — already written incrementally during step 3+4.
    if result_tif_path:
        log.info("Classified GeoTIFF saved: %s", result_tif_path)

    # 7. Vectorise — raster class indices → polygon GeoDataFrame
    log.info("Land-cover: vectorising classified raster…")
    vec_meta = {**meta, "transform": orig_tf, "height": orig_h, "width": orig_w}
    gdf      = _vectorize(pred, vec_meta, simplify_tolerance)
    del pred
    gc.collect()

    # Clean up temp TIF (only created when result_tif_path was None)
    if _own_tif:
        try:
            _tif_path.unlink(missing_ok=True)
        except Exception:
            pass

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
