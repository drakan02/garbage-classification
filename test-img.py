import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os

# ===== CẤU HÌNH (CONFIG) =====
DETECT_MODEL_PATH = "yolov8s-worldv2.pt"
CLS_MODEL_PATH = "best.onnx"
CONF_DET = 0.1
CONF_CLS = 0.3
IMG_SIZE = 224

class_names = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

# ===== HÀM HỖ TRỢ (MỚI: TẠO VIỀN) =====
def resize_with_white_padding(image, target_size):
    """
    Resize ảnh giữ nguyên tỷ lệ, thêm viền TRẮNG (padding)
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Tạo nền trắng (255)
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    
    # Căn giữa
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    return canvas

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ===== 1. TẢI MÔ HÌNH =====
print("--> Đang khởi tạo mô hình...")
try:
    det_model = YOLO(DETECT_MODEL_PATH)
    
    # Định nghĩa các class cho YOLO-World
    det_model.set_classes([
        "battery",
        "food waste",
        "cardboard box",
        "clothes",
        "glass bottle",
        "metal can",
        "paper",
        "plastic bottle",
        "shoe",
        "trash"])
    
    ort_sess = ort.InferenceSession(CLS_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = ort_sess.get_inputs()[0].name
    
    print("--> Tải xong! Đang chờ chọn ảnh...")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}")
    exit()

# ===== 2. CHỌN ẢNH =====
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(
    title="Chọn ảnh để kiểm tra",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")]
)

if not img_path:
    print("Bạn chưa chọn ảnh nào.")
    exit()

# ===== 3. XỬ LÝ ẢNH =====
img_array = np.fromfile(img_path, np.uint8)
frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if frame is None:
    print("Lỗi đọc file ảnh.")
    exit()

h, w, _ = frame.shape
print(f"--> Đang phân tích: {os.path.basename(img_path)}")

# --- DETECT ---
results = det_model(frame, conf=CONF_DET, iou=0.4, agnostic_nms=True, verbose=False)[0]
object_count = 0

if results.boxes is not None:
    for box in results.boxes:
        object_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        # --- PREPROCESS (THAY ĐỔI Ở ĐÂY) ---
        # Dùng hàm tạo viền thay vì resize thô
        img = resize_with_white_padding(crop, IMG_SIZE)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]

        # --- CLASSIFY ---
        logits = ort_sess.run(None, {input_name: img})[0][0]

        if np.sum(logits) > 0.9 and np.sum(logits) < 1.1: probs = logits
        else: probs = softmax(logits)

        cls_id = int(np.argmax(probs))
        score = float(probs[cls_id])
        label = class_names[cls_id] if score >= CONF_CLS else "unknown"

        # --- DRAW ---
        print(f"   + Box [{x1},{y1}]: {label} ({score:.2f})")
        
        color = (0, 255, 0)
        if label == "unknown": color = (0, 0, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{label} {score:.2f}"
        (t_w, t_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_w, y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

if object_count == 0:
    print("--> Không tìm thấy vật thể nào!")

# ===== 4. HIỂN THỊ =====
display_h, display_w = h, w
MAX_W = 1200
if w > MAX_W:
    scale = MAX_W / w
    display_w = int(w * scale)
    display_h = int(h * scale)
    frame_show = cv2.resize(frame, (display_w, display_h))
else:
    frame_show = frame

cv2.imshow("Ket qua (Nhan phim bat ky de thoat)", frame_show)
cv2.waitKey(0)
cv2.destroyAllWindows()