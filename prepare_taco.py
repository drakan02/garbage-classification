import json
import os
import shutil
import random
from tqdm import tqdm

# --- CẤU HÌNH ---
json_file = 'data/annotations.json'
data_root = 'data'
output_dir = 'taco_yolo'
train_ratio = 0.8

# --- ĐỊNH NGHĨA 5 NHÓM CLASS ---
# 0: Plastic
# 1: Metal
# 2: Glass
# 3: Paper
# 4: Other

# --- LOGIC CÂN BẰNG CHẤT LIỆU (5 CLASSES) ---
# 0: Soft Plastic (Túi, Bao bì, Rác nhựa vụn)
# 1: Hard Plastic (Chai nhựa, Cốc, Hộp)
# 2: Paper (Giấy, Bìa)
# 3: Metal & Glass (Lon, Chai thủy tinh, Giấy bạc)
# 4: Cigarette (Đầu lọc thuốc lá)

def get_new_class_id(cat_name):
    name = cat_name.lower()
    
    # --- NHÓM 4: CIGARETTE (Đặc biệt: Nhỏ nhưng số lượng nhiều) ---
    if 'cigarette' in name:
        return 4

    # --- NHÓM 3: METAL & GLASS (Vật liệu cứng không phải nhựa) ---
    # Gộp chung để tăng số lượng cho nhóm này (vì Glass quá ít)
    if any(x in name for x in ['metal', 'aluminium', 'foil', 'can', 'tin', 'aerosol', 'pop tab', 'glass', 'jar']):
        return 3

    # --- NHÓM 2: PAPER (Giấy) ---
    if any(x in name for x in ['paper', 'carton', 'box', 'cardboard', 'receipt', 'tissue', 'tube']):
        return 2

    # --- NHÓM 1: HARD PLASTIC (Nhựa Cứng - Có hình khối rõ ràng) ---
    # Chai, Cốc, Hộp, Nắp, Ống hút, Thìa dĩa
    if any(x in name for x in ['bottle', 'cup', 'lid', 'straw', 'utensil', 'cutlery', 'container', 'tub', 'bucket', 'detergent', 'polystyrene', 'styrofoam']):
        return 1

    # --- NHÓM 0: SOFT PLASTIC (Nhựa Mềm & Rác vụn) ---
    # Túi, Vỏ kẹo, Màng bọc. 
    # Mẹo: Gán cả "Unlabeled" và "Fragment" vào đây vì đa số rác vô danh là mảnh nilon/nhựa vỡ.
    if any(x in name for x in ['bag', 'wrapper', 'packet', 'sachet', 'film', 'pouch', 'blister', 'unlabeled', 'litter', 'plastic', 'fragment', 'debris', 'rope', 'shoe', 'clothing']):
        return 0

    return -1

# --- BẮT ĐẦU XỬ LÝ  ---
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

dirs = {
    'train_imgs': os.path.join(output_dir, 'images', 'train'),
    'val_imgs':   os.path.join(output_dir, 'images', 'val'),
    'train_lbls': os.path.join(output_dir, 'labels', 'train'),
    'val_lbls':   os.path.join(output_dir, 'labels', 'val')
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return x, y, w, h

print("Đang đọc dữ liệu JSON...")
with open(json_file, 'r') as f:
    data = json.load(f)

cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

img_anns = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in img_anns:
        img_anns[img_id] = []
    img_anns[img_id].append(ann)

images = data['images']
random.shuffle(images)

print("Đang phân loại vào 5 nhóm tối ưu...")
for i, img in tqdm(enumerate(images), total=len(images)):
    img_id = img['id']
    file_name = img['file_name']
    src_path = os.path.join(data_root, file_name)
    
    if not os.path.exists(src_path):
        continue

    # Chia train/val
    if i < len(images) * train_ratio:
        save_img_dir = dirs['train_imgs']
        save_lbl_dir = dirs['train_lbls']
    else:
        save_img_dir = dirs['val_imgs']
        save_lbl_dir = dirs['val_lbls']

    new_name = file_name.replace('/', '_')
    shutil.copy(src_path, os.path.join(save_img_dir, new_name))

    if img_id in img_anns:
        txt_name = os.path.splitext(new_name)[0] + ".txt"
        with open(os.path.join(save_lbl_dir, txt_name), 'w') as f:
            for ann in img_anns[img_id]:
                original_name = cat_id_to_name[ann['category_id']]
                new_id = get_new_class_id(original_name)
                bbox = convert_bbox((img['width'], img['height']), ann['bbox'])
                f.write(f"{new_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

print("\nHoàn tất! Đã gom gọn thành 5 nhóm.")