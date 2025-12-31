import cv2
import os
import matplotlib.pyplot as plt
import glob
import random

# --- CẤU HÌNH ---
INPUT_DIR = 'data'
NUM_SAMPLES = 6  # Số lượng ảnh muốn kiểm tra mỗi lần chạy

# Mapping ID -> Tên Class (Khớp với config bạn vừa chạy)
CLASS_NAMES = {
    0: 'Battery',
    1: 'Biological',
    2: 'Cardboard',
    3: 'Clothes',
    4: 'Glass',
    5: 'Metal',
    6: 'Paper',
    7: 'Plastic',
    8: 'Shoe',
    9: 'Trash'
}

def visualize_labels():
    # 1. Tìm tất cả các file ảnh trong thư mục data (bao gồm cả thư mục con)
    # Tìm đuôi jpg, png, jpeg
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))
    
    if not image_paths:
        print("❌ Không tìm thấy ảnh nào trong thư mục data!")
        return

    print(f"🔍 Tìm thấy tổng cộng {len(image_paths)} ảnh. Đang chọn ngẫu nhiên {NUM_SAMPLES} ảnh để hiển thị...")

    # 2. Chọn ngẫu nhiên ảnh để check
    samples = random.sample(image_paths, min(NUM_SAMPLES, len(image_paths)))

    plt.figure(figsize=(20, 10))

    for i, img_path in enumerate(samples):
        # Đọc ảnh
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển sang RGB để hiển thị đúng màu
        h, w, _ = img.shape

        # Tìm file label tương ứng
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # Vẽ từng box lên ảnh
            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:])

                # Chuyển đổi tọa độ YOLO (0-1) sang Pixel
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                # Chọn màu (mỗi class 1 màu cho dễ nhìn)
                color = plt.cm.tab10(cls_id % 10)
                color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

                # Vẽ hình chữ nhật
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Viết tên class
                label_text = CLASS_NAMES.get(cls_id, str(cls_id))
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            print(f"⚠️ Ảnh {os.path.basename(img_path)} chưa có file label .txt!")

        # Hiển thị lên subplot
        plt.subplot(2, 3, i + 1) # Hiển thị dạng lưới 2 hàng 3 cột
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(os.path.dirname(img_path)) + "/" + os.path.basename(img_path))

    plt.tight_layout()
    plt.show()

# Chạy hàm
visualize_labels()