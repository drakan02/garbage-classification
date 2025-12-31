import os
from ultralytics import YOLO
from tqdm import tqdm
import torch

# --- CẤU HÌNH ---
INPUT_DIR = 'data'

# Cấu hình folder (Giữ nguyên như cũ)
FOLDER_CONFIG = {
    'battery': { 'id': 0, 'keywords': ["battery", "alkaline battery", "AA battery", "AAA battery", "lithium battery", "button cell", "dry cell"] },
    'biological': { 'id': 1, 'keywords': ["food waste", "fruit", "vegetable", "banana peel", "apple core", "organic waste", "leftover food", "rotten food"] },
    'cardboard': { 'id': 2, 'keywords': ["cardboard box", "carton", "pizza box", "shipping box", "corrugated box", "brown box"] },
    'clothes': { 'id': 3, 'keywords': ["clothes", "shirt", "t-shirt", "pants", "jacket", "dress", "clothing", "fabric", "textile", "jeans"] },
    'glass': { 'id': 4, 'keywords': ["glass bottle", "glass jar", "wine bottle", "beer bottle", "broken glass", "glass container"] },
    'metal': { 'id': 5, 'keywords': ["metal can", "aluminum can", "soda can", "tin can", "food can", "scrap metal", "beverage can"] },
    'paper': { 'id': 6, 'keywords': ["paper", "newspaper", "crumpled paper", "magazine", "flyer", "document", "sheet of paper", "waste paper"] },
    'plastic': { 'id': 7, 'keywords': ["plastic bag", "plastic bottle", "water bottle", "plastic cup", "snack wrapper", "plastic container", "straw", "plastic tub"] },
    'shoe': { 'id': 8, 'keywords': ["shoe", "sneaker", "boot", "sandal", "footwear", "running shoe", "leather shoe"] },
    'trash': { 'id': 9, 'keywords': ["trash", "garbage", "rubbish", "waste", "plastic bag", "bottle", "can", "paper", "box", "food waste", "face mask", "medical mask", "surgical mask", "toothbrush", "plastic toothbrush", "diaper", "nappy", "baby diaper"] }
}

def auto_label_multi_object_fixed():
    print("🚀 Chế độ: Quét sạch sành sanh (Multi-Object) - Đã vá lỗi Device Mismatch...")
    
    # Kiểm tra thiết bị
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"ℹ️ Đang chạy trên: {device.upper()}")

    # 1. Tải model MỘT LẦN ở ngoài vòng lặp
    model = YOLO('yolov8l-world.pt')
    
    # Ép model lên đúng thiết bị ngay từ đầu
    model.to(device)

    total_files = 0
    total_labels = 0

    for folder_name, config in FOLDER_CONFIG.items():
        folder_path = os.path.join(INPUT_DIR, folder_name)
        if not os.path.exists(folder_path): 
            continue
            
        print(f"\n📂 Đang xử lý: {folder_name.upper()} (ID: {config['id']})")
        
        target_id = config['id']
        keywords = config['keywords']
        
        # --- KHẮC PHỤC LỖI TẠI ĐÂY ---
        # Thay vì set_classes trực tiếp, ta dùng trick: Load lại trọng số nhẹ hoặc clear cache nếu cần.
        # Nhưng cách đơn giản nhất là set_classes và đảm bảo model vẫn ở trên GPU.
        try:
            model.set_classes(keywords)
        except Exception as e:
            print(f"⚠️ Lỗi khi set_classes cho {folder_name}: {e}")
            print("🔄 Đang thử reset lại model cho folder này...")
            # Nếu lỗi, load lại model mới hoàn toàn cho folder này (chậm hơn xíu nhưng chắc chắn chạy)
            model = YOLO('yolov8l-world.pt')
            model.to(device)
            model.set_classes(keywords)

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total_files += len(files)
        
        # Chạy batch nhỏ hoặc từng ảnh
        for filename in tqdm(files):
            img_path = os.path.join(folder_path, filename)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            
            # Nếu đã có file txt (ví dụ chạy lần trước bị lỗi), có thể bỏ qua hoặc ghi đè
            # Ở đây ta chọn ghi đè để đảm bảo chính xác
            
            all_detections = []
            
            try:
                # Predict
                results = model.predict(img_path, conf=0.01, iou=0.5, verbose=False, max_det=100, device=device)
                
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        # Quan trọng: Chuyển box về CPU trước khi xử lý list
                        xywh = box.xywhn[0].cpu().tolist()
                        all_detections.append((target_id, xywh))
            except RuntimeError as e:
                # Nếu gặp lỗi CUDA OOM (hết bộ nhớ) hoặc lỗi device khác
                print(f"❌ Lỗi ảnh {filename}: {e}")
                continue

            # Ghi file
            if len(all_detections) > 0:
                with open(txt_path, 'w') as f:
                    for det in all_detections:
                        cls_id, (x, y, w, h) = det
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                total_labels += len(all_detections)

    print("\n✅ HOÀN TẤT TOÀN BỘ!")
    print(f"📊 Tổng ảnh: {total_files} | Tổng box: {total_labels}")

if __name__ == "__main__":
    # Để tránh lỗi multiprocessing trên Windows nếu có
    torch.multiprocessing.freeze_support()
    auto_label_multi_object_fixed()