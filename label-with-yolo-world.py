import os
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ================== CONFIG ==================
DATA_DIR = "data"
CONF_THRESHOLD = 0.15          
IOU_THRESHOLD = 0.5
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Class + keyword 
FOLDER_CONFIG = {
    "battery":    ["battery", "alkaline battery", "AA battery", "AAA battery", "lithium battery", "button cell"],
    "biological": ["food waste", "fruit", "vegetable", "organic waste", "banana peel", "apple core"],
    "cardboard":  ["cardboard box", "carton", "pizza box", "shipping box"],
    "clothes":    ["clothes", "shirt", "t-shirt", "pants", "jacket", "fabric"],
    "glass":      ["glass bottle", "glass jar", "broken glass"],
    "metal":      ["metal can", "aluminum can", "tin can"],
    "paper":      ["paper", "newspaper", "magazine", "document"],
    "plastic":    ["plastic bottle", "plastic bag", "plastic container", "wrapper"],
    "shoes":      ["shoe", "sneaker", "boot", "sandal"],
    "trash":      ["garbage", "trash", "waste", "rubbish"]
}

# ================== MAIN ==================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    model = YOLO("yolov8l-world.pt").to(device)

    class_names = list(FOLDER_CONFIG.keys())
    class_id_map = {name: idx for idx, name in enumerate(class_names)}

    total_images = 0
    total_labeled = 0

    for class_name, keywords in FOLDER_CONFIG.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\nLabeling folder: {class_name}")
        model.set_classes(keywords)

        images = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTS)]

        for img_name in tqdm(images, desc=class_name):
            img_path = os.path.join(class_dir, img_name)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            total_images += 1

            results = model.predict(
                img_path,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
                verbose=False,
                max_det=50
            )

            if not results or len(results[0].boxes) == 0:
                # Không detect được → KHÔNG ghi label
                continue

            with open(txt_path, "w") as f:
                for box in results[0].boxes:
                    x, y, w, h = box.xywhn[0].cpu().tolist()
                    cls_id = class_id_map[class_name]
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            total_labeled += 1

    print("\n=== DONE ===")
    print(f"Tổng ảnh quét: {total_images}")
    print(f"Ảnh có label: {total_labeled}")

if __name__ == "__main__":
    main()
