import os
import shutil
import random
import cv2
import albumentations as A
from tqdm import tqdm

# --- CẤU HÌNH ---
SOURCE_DIR = 'data_approved'  # Folder chứa dữ liệu gốc sạch (đã duyệt)
DEST_DIR = 'detection-dataset' # Folder đích để train

# Số lượng ảnh MỤC TIÊU cho tập TRAIN mỗi lớp
TARGET_TRAIN_COUNT = 700 

# Tỷ lệ chia
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Các lớp cần xử lý
CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
           'metal', 'paper', 'plastic', 'shoes', 'trash']

# --- ĐỊNH NGHĨA BIẾN ĐỔI (ALBUMENTATIONS) ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.RandomScale(scale_limit=0.1, p=0.2),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

def create_dirs():
    for split in ['train', 'val', 'test']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_DIR, split, dtype), exist_ok=True)

def read_yolo_label(txt_path):
    bboxes = []
    labels = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:])
                    bboxes.append([x, y, w, h])
                    labels.append(class_id)
    return bboxes, labels

def save_yolo_label(txt_path, bboxes, labels):
    with open(txt_path, 'w') as f:
        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            f.write(f"{int(label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def copy_files(files, split_type):
    """Copy file gốc vào thư mục đích"""
    dest_img_dir = os.path.join(DEST_DIR, split_type, 'images')
    dest_lbl_dir = os.path.join(DEST_DIR, split_type, 'labels')
    
    for img_src, txt_src in files:
        shutil.copy(img_src, os.path.join(dest_img_dir, os.path.basename(img_src)))
        shutil.copy(txt_src, os.path.join(dest_lbl_dir, os.path.basename(txt_src)))

def augment_class_in_train(class_id, class_name):
    """Chỉ tăng cường dữ liệu trong folder TRAIN"""
    train_img_dir = os.path.join(DEST_DIR, 'train', 'images')
    train_lbl_dir = os.path.join(DEST_DIR, 'train', 'labels')

    # Lấy danh sách ảnh hiện có trong tập Train thuộc class này
    existing_files = []
    
    # Quét folder labels để tìm file thuộc class hiện tại
    all_txts = [f for f in os.listdir(train_lbl_dir) if f.endswith('.txt')]
    
    for txt_name in all_txts:
        txt_path = os.path.join(train_lbl_dir, txt_name)
        _, labels = read_yolo_label(txt_path)
        if class_id in labels:
            img_name = os.path.splitext(txt_name)[0] + ".jpg" # Giả định đuôi jpg
            # Check các đuôi khác
            if not os.path.exists(os.path.join(train_img_dir, img_name)):
                for ext in ['.jpeg', '.png']:
                    temp_name = os.path.splitext(txt_name)[0] + ext
                    if os.path.exists(os.path.join(train_img_dir, temp_name)):
                        img_name = temp_name
                        break
            
            existing_files.append((os.path.join(train_img_dir, img_name), txt_path))

    current_count = len(existing_files)
    needed = TARGET_TRAIN_COUNT - current_count
    
    print(f"   + Lớp '{class_name}' (ID {class_id}): Có {current_count} ảnh gốc trong Train -> Cần thêm {needed}")

    if needed <= 0 or current_count == 0:
        return

    pbar = tqdm(total=needed, desc=f"Augmenting {class_name}", leave=False)
    count_generated = 0
    
    while count_generated < needed:
        img_src, txt_src = random.choice(existing_files)
        
        image = cv2.imread(img_src)
        bboxes, labels = read_yolo_label(txt_src)
        
        if not bboxes: continue

        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']
            
            if not aug_bboxes: continue
            
            # Tạo tên file mới
            base_name = os.path.splitext(os.path.basename(img_src))[0]
            new_name = f"aug_{class_name}_{count_generated}_{base_name}"
            
            cv2.imwrite(os.path.join(train_img_dir, new_name + ".jpg"), aug_img)
            save_yolo_label(os.path.join(train_lbl_dir, new_name + ".txt"), aug_bboxes, aug_labels)
            
            count_generated += 1
            pbar.update(1)
        except:
            continue
    pbar.close()

def main():
    if not os.path.exists(SOURCE_DIR):
        print("Lỗi: Không tìm thấy folder nguồn!")
        return

    create_dirs()
    
    # BƯỚC 1: QUÉT VÀ CHIA DATA GỐC (SPLIT)
    print("\n--- BƯỚC 1: CHIA DỮ LIỆU GỐC (SPLIT) ---")
    
    # Giả định folder nguồn có dạng: data_approved/battery, data_approved/trash ...
    for idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
            
        # Lấy cặp ảnh/txt
        files = []
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, f)
                txt_path = os.path.join(class_dir, os.path.splitext(f)[0] + ".txt")
                if os.path.exists(txt_path):
                    files.append((img_path, txt_path))
        
        random.shuffle(files)
        total = len(files)
        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)
        
        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val:]
        
        print(f"Lớp {class_name}: Tổng {total} -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
        
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

    # BƯỚC 2: TĂNG CƯỜNG DỮ LIỆU (CHỈ TRAIN)
    print("\n--- BƯỚC 2: TĂNG CƯỜNG TẬP TRAIN (AUGMENT) ---")
    
    for idx, class_name in enumerate(CLASSES):
        augment_class_in_train(idx, class_name)
        
    print("\n=== HOÀN TẤT ===")
    print(f"Dữ liệu mới nằm tại: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    main()