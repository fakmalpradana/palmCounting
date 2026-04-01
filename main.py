import os
import shutil
import tempfile
from PIL import Image
import tifffile as tiff
import math
import cv2
import yaml
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from shapely.ops import unary_union
from rtree import index
from tqdm import tqdm

# ==================== FUNGSI TILING ====================

def tile_image_with_overlap(input_path, output_folder, tile_width, tile_height):
    image = tiff.imread(input_path)
    img = Image.fromarray(image)
    
    img_width, img_height = img.size
    
    # Hitung jumlah tiles yang dibutuhkan
    num_tiles_x = math.ceil(img_width / tile_width)
    num_tiles_y = math.ceil(img_height / tile_height)
    
    # Hitung overlap yang dibutuhkan
    if num_tiles_x > 1:
        overlap_x = math.ceil((num_tiles_x * tile_width - img_width) / (num_tiles_x - 1))
    else:
        overlap_x = 0
    
    if num_tiles_y > 1:
        overlap_y = math.ceil((num_tiles_y * tile_height - img_height) / (num_tiles_y - 1))
    else:
        overlap_y = 0
    
    total_tiles = num_tiles_x * num_tiles_y
    
    count = 0
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    with tqdm(total=total_tiles, desc=f"Tiling {os.path.basename(input_path)}", unit="tile") as pbar:
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Hitung posisi tile
                left = i * step_x
                upper = j * step_y
                
                # Pastikan tile terakhir tidak melebihi batas gambar
                if i == num_tiles_x - 1:
                    left = img_width - tile_width
                if j == num_tiles_y - 1:
                    upper = img_height - tile_height
                
                right = left + tile_width
                lower = upper + tile_height
                
                # Crop tile
                tile = img.crop((left, upper, right, lower))
                
                # Pastikan dimensi tile sesuai
                if tile.size != (tile_width, tile_height):
                    padded_tile = Image.new(img.mode, (tile_width, tile_height), 0)
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile
                
                tile_filename = os.path.join(output_folder, f"tile_{count}.tif")
                
                # Simpan sebagai GeoTIFF dengan georeferencing
                with rasterio.open(input_path) as src:
                    # Hitung transform untuk tile ini
                    tile_transform = rasterio.transform.from_bounds(
                        src.bounds.left + left * src.transform[0],
                        src.bounds.top - lower * abs(src.transform[4]),
                        src.bounds.left + right * src.transform[0],
                        src.bounds.top - upper * abs(src.transform[4]),
                        tile_width,
                        tile_height
                    )
                    
                    # Simpan tile sebagai GeoTIFF
                    with rasterio.open(
                        tile_filename,
                        'w',
                        driver='GTiff',
                        height=tile_height,
                        width=tile_width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=tile_transform
                    ) as dst:
                        tile_array = np.array(tile)
                        if len(tile_array.shape) == 2:
                            dst.write(tile_array, 1)
                        else:
                            for band in range(src.count):
                                dst.write(tile_array[:, :, band], band + 1)
                
                count += 1
                pbar.update(1)

def process_tiling(input_folder, output_folder, tile_width, tile_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    tif_files = [f for f in os.listdir(input_folder) 
                 if f.endswith(".tif") or f.endswith(".tiff")]
    
    if not tif_files:
        print("❌ Tidak ada file TIF ditemukan di folder input!")
        return
    
    print(f"\n📁 Ditemukan {len(tif_files)} file TIF untuk diproses")
    print("="*60)
    
    for filename in tif_files:
        input_path = os.path.join(input_folder, filename)
        
        file_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        if not os.path.exists(file_output_folder):
            os.makedirs(file_output_folder)
        
        tile_image_with_overlap(input_path, file_output_folder, tile_width, tile_height)

# ==================== FUNGSI DETEKSI YOLO ====================

def load_labels(yaml_path):
    with open(yaml_path, mode='r') as f:
        data_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    return data_yaml['names']

def load_yolo_model(model_path):
    yolo = cv2.dnn.readNetFromONNX(model_path)
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return yolo

def get_gsd_from_raster(raster_path):
    with rasterio.open(raster_path) as src:
        transform = src.transform
        gsd_x = abs(transform[0])
        gsd_y = abs(transform[4])
    return gsd_x, gsd_y

def perform_yolo_detection(image, yolo, labels, input_size=640, conf_threshold=0.1, nms_threshold=0.25):
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (input_size, input_size), swapRB=True, crop=False)
    yolo.setInput(blob)
    preds = yolo.forward()
    detections = preds[0]
    
    boxes = []
    confidences = []
    classes = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / input_size
    y_factor = image_h / input_size

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > conf_threshold:
            class_score = row[5:].max()
            class_id = row[5:].argmax()

            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = [left, top, width, height]
                confidences.append(float(confidence))
                boxes.append(box)
                classes.append(class_id)

    # Non-maximum suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    
    if len(indices) > 0:
        if isinstance(indices, np.ndarray):
            if indices.ndim == 2:
                indices = indices.flatten()
        
        for i in indices:
            idx = int(i) if isinstance(i, (np.integer, np.ndarray)) else i
            
            box = boxes[idx]
            class_name = labels[classes[idx]]
            confidence = confidences[idx]
            results.append({
                'class_name': class_name,
                'confidence': confidence,
                'x': box[0],
                'y': box[1],
                'width': box[2],
                'height': box[3]
            })

    return results

def filter_overlap(gdf, distance):
    if len(gdf) == 0:
        return gdf
    
    spatial_index = index.Index()
    points = []

    for idx, row in gdf.iterrows():
        point = row.geometry
        points.append((idx, point))
        spatial_index.insert(idx, point.bounds)

    unique_points = []
    seen = set()

    for idx, point in points:
        if idx in seen:
            continue
        overlap_indices = list(spatial_index.intersection(point.buffer(distance).bounds))
        if overlap_indices:
            overlapping_points = [points[i][1] for i in overlap_indices]
            centroid = unary_union(overlapping_points).centroid
            unique_points.append((idx, centroid))
            seen.update(overlap_indices)

    filtered_data = []
    for idx, centroid in unique_points:
        filtered_data.append({
            'class_name': gdf.iloc[idx]['class_name'],
            'confidence': gdf.iloc[idx]['confidence'],
            'geometry': centroid
        })

    filtered_gdf = gpd.GeoDataFrame(filtered_data, crs=gdf.crs)
    return filtered_gdf

def process_detection(folder_path, model_path, yaml_path, output_shp_path, 
                      min_distance=1.0, gsd_x=None, gsd_y=None, 
                      conf_threshold=0.1, nms_threshold=0.25):  # ← Tambahkan parameter
    labels = load_labels(yaml_path)
    yolo = load_yolo_model(model_path)
    all_results = []
    
    first_crs = None
    auto_gsd_x = None
    auto_gsd_y = None

    # Kumpulkan semua file TIF
    all_tif_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                all_tif_files.append(os.path.join(root, filename))
    
    if not all_tif_files:
        print("❌ Tidak ada file TIF ditemukan untuk deteksi!")
        return

    # Proses dengan progress bar
    with tqdm(total=len(all_tif_files), desc="🔍 Deteksi YOLO", unit="tile") as pbar:
        for img_path in all_tif_files:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                pbar.set_postfix_str(f"❌ {filename}")
                pbar.update(1)
                continue
            
            with rasterio.open(img_path) as src:
                bounds = src.bounds
                if first_crs is None:
                    first_crs = src.crs
                
                if auto_gsd_x is None and (gsd_x is None or gsd_y is None):
                    auto_gsd_x, auto_gsd_y = get_gsd_from_raster(img_path)
                    
            current_gsd_x = gsd_x if gsd_x is not None else auto_gsd_x
            current_gsd_y = gsd_y if gsd_y is not None else auto_gsd_y
                    
            detections = perform_yolo_detection(img, yolo, labels, 
                                       conf_threshold=conf_threshold,  # ← Gunakan parameter
                                       nms_threshold=nms_threshold)

            if detections:
                for det in detections:
                    koorX = (det['x'] + (det['width'] / 2)) * current_gsd_x + bounds.left
                    koorY = bounds.top - (det['y'] + (det['height'] / 2)) * current_gsd_y
                    
                    all_results.append({
                        'class_name': det['class_name'],
                        'confidence': det['confidence'],
                        'geometry': Point(koorX, koorY)
                    })
                
                pbar.set_postfix_str(f"✅ {len(detections)} obj")
            else:
                pbar.set_postfix_str("⚪ 0 obj")
            
            pbar.update(1)

    if not all_results:
        print("\n❌ Tidak ada deteksi ditemukan!")
        return

    print(f"\n📊 Total deteksi sebelum filtering: {len(all_results)}")
    
    gdf = gpd.GeoDataFrame(all_results, crs=first_crs if first_crs else 'EPSG:4326')
    
    print("🔄 Filtering overlap...")
    filtered_gdf = filter_overlap(gdf, min_distance)
    print(f"✅ Total deteksi setelah filtering: {len(filtered_gdf)}")

    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
    filtered_gdf.to_file(output_shp_path)
    print(f"\n💾 Shapefile disimpan: {output_shp_path}")
    print(f"🗺️  CRS: {filtered_gdf.crs}")

# ==================== FUNGSI UTAMA ====================

def process_tif_to_shapefile(input_folder, model_path, yaml_path, output_shp_path, 
                              tile_width=640, tile_height=640, min_distance=1.0, 
                              gsd_x=None, gsd_y=None,
                              conf_threshold=0.25, nms_threshold=0.4):
    """
    Proses lengkap dari tiling hingga deteksi YOLO dan menghasilkan shapefile
    
    Parameters:
    - input_folder: Folder berisi file TIF asli
    - model_path: Path ke model YOLO ONNX
    - yaml_path: Path ke file data.yaml
    - output_shp_path: Path output shapefile
    - tile_width: Lebar tile (default: 640)
    - tile_height: Tinggi tile (default: 640)
    - min_distance: Jarak minimum untuk filtering overlap (dalam unit CRS)
    - gsd_x: Ground Sample Distance X (opsional, auto-detect jika None)
    - gsd_y: Ground Sample Distance Y (opsional, auto-detect jika None)
    """
    
    # Buat folder temporary
    temp_dir = tempfile.mkdtemp(prefix="tiles_temp_")
    print(f"\n{'='*60}")
    print(f"📂 Folder temporary: {temp_dir}")
    print(f"{'='*60}\n")
    
    try:
        # STEP 1: Tiling
        print("="*60)
        print("🔷 STEP 1: PROSES TILING")
        print("="*60)
        process_tiling(input_folder, temp_dir, tile_width, tile_height)
        
        # STEP 2: Deteksi YOLO
        print("\n" + "="*60)
        print("🔷 STEP 2: PROSES DETEKSI YOLO")
        print("="*60)
        process_detection(temp_dir, model_path, yaml_path, output_shp_path, 
                     min_distance, gsd_x, gsd_y,
                     conf_threshold, nms_threshold) 
        
        print("\n" + "="*60)
        print("✅ PROSES SELESAI!")
        print("="*60)
        
    finally:
        # Hapus folder temporary
        print(f"\n🗑️  Menghapus folder temporary...")
        shutil.rmtree(temp_dir)
        print("✅ Folder temporary berhasil dihapus!")

# ==================== CONTOH PENGGUNAAN ====================

if __name__ == "__main__":
    # Konfigurasi
    input_folder = r"input_tif\25_PT MUSIM MAS"              # Folder berisi file TIF asli
    model_path = r"model_dir\model_best\weights\best.onnx"    # Path ke model YOLO
    yaml_path = r"model_dir\data.yaml"      # Path ke data.yaml
    output_shp_path = "./output/25_PT MUSIM MAS.shp"  # Output shapefile
    
    # Parameter tiling
    tile_width = 640
    tile_height = 640
    
    # Parameter deteksi
    min_distance = 3  # Jarak minimum untuk filtering (dalam unit CRS, misal: meter)
    
    # Jalankan proses lengkap
    process_tif_to_shapefile(
        input_folder=input_folder,
        model_path=model_path,
        yaml_path=yaml_path,
        output_shp_path=output_shp_path,
        tile_width=640,
        tile_height=640,
        min_distance=3.0,
        conf_threshold=0.3,   # ← Atur confidence threshold (0.0 - 1.0)
        nms_threshold=0.3     # ← Atur NMS threshold (0.0 - 1.0)
    )
