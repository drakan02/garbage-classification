import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from collections import deque, Counter

# ===== CONFIG =====
DETECT_MODEL_PATH = "yolov8s-worldv2.pt" # Nên dùng bản FP16 hoặc v2
CLS_MODEL_PATH = "best.onnx"          # Model ONNX của bạn
CONF_DET = 0.3    # <--- TĂNG LÊN để giảm nhiễu
CONF_CLS = 0.5    # <--- TĂNG LÊN để chắc chắn hơn
CAM_ID = 0
IMG_SIZE = 224

# Bộ nhớ đệm để khử nhiễu (Lưu 10 kết quả gần nhất cho mỗi ID)
# Càng tăng số này (ví dụ 15, 20) thì càng mượt nhưng sẽ bị trễ (delay) một chút
SMOOTHING_BUFFER_LEN = 10 
track_history = {} # Dictionary lưu lịch sử: {track_id: deque([label1, label2...])}

class_names = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

# ... (Giữ nguyên hàm resize_with_white_padding và softmax ở các bước trước) ...
def resize_with_white_padding(image, target_size):
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    x_offset, y_offset = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    return canvas

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ===== LOAD MODELS =====
print("--> Đang tải mô hình...")
try:
    det_model = YOLO(DETECT_MODEL_PATH)
    # ... (Set classes cho YOLO-World như cũ) ...
    det_model.set_classes(["battery", "alkaline battery", "AA battery", "AAA battery", "lithium battery", "button cell", "dry cell",
                       "food waste", "fruit", "vegetable", "banana peel", "apple core", "organic waste", "leftover food", "rotten food",
                       "cardboard box", "carton", "pizza box", "shipping box", "corrugated box", "brown box",
                       "clothes", "shirt", "t-shirt", "pants", "jacket", "dress", "clothing", "fabric", "textile", "jeans",
                       "glass bottle", "glass jar", "wine bottle", "beer bottle", "broken glass", "glass container",
                       "metal can", "aluminum can", "soda can", "tin can", "food can", "scrap metal", "beverage can",
                       "paper", "newspaper", "crumpled paper", "magazine", "flyer", "document", "sheet of paper", "waste paper",
                       "plastic bag", "plastic bottle", "water bottle", "plastic cup", "snack wrapper", "plastic container", "straw", "plastic tub",
                       "shoe", "sneaker", "boot", "sandal", "footwear", "running shoe", "leather shoe",
                       "trash", "garbage", "rubbish", "waste", "plastic bag", "bottle", "can", "paper", "box", "food waste",
                       "face mask", "medical mask", "surgical mask", "toothbrush", "plastic toothbrush", "diaper", "nappy", "baby diaper"])
    
    ort_sess = ort.InferenceSession(CLS_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = ort_sess.get_inputs()[0].name
except Exception as e:
    print(f"Lỗi: {e}")
    exit()

# ===== MAIN LOOP =====
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    # --- 1. TRACKING (Thay vì Detect thường) ---
    # persist=True: Giúp model nhớ ID vật thể qua các frame
    # tracker="bytetrack.yaml": Thuật toán theo dõi nhẹ và nhanh
    results = det_model.track(frame, conf=CONF_DET, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    if results.boxes is not None and results.boxes.id is not None:
        # Lấy danh sách boxes và IDs
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Cắt ảnh
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            # --- 2. CLASSIFY ---
            img = resize_with_white_padding(crop, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None]

            logits = ort_sess.run(None, {input_name: img})[0][0]
            
            # Kiểm tra logits hay probs
            if np.sum(logits) > 0.9 and np.sum(logits) < 1.1: probs = logits
            else: probs = softmax(logits)

            current_cls_id = int(np.argmax(probs))
            current_score = float(probs[current_cls_id])
            
            # Lấy nhãn hiện tại (hoặc unknown)
            raw_label = class_names[current_cls_id] if current_score >= CONF_CLS else "unknown"

            # --- 3. SMOOTHING (BỎ PHIẾU) ---
            # Tạo lịch sử cho ID này nếu chưa có
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=SMOOTHING_BUFFER_LEN)
            
            # Thêm nhãn mới vào hàng đợi
            track_history[track_id].append(raw_label)

            # Tìm nhãn xuất hiện nhiều nhất trong lịch sử (Vote)
            # most_common(1) trả về [(label, count)]
            final_label, count = Counter(track_history[track_id]).most_common(1)[0]

            # --- DRAW ---
            # Vẽ màu xanh lá nếu ổn định, màu vàng nếu đang lưỡng lự (vote chưa áp đảo)
            color = (0, 255, 0)
            if final_label == "unknown":
                color = (0, 0, 255) # Đỏ
            elif count < (SMOOTHING_BUFFER_LEN // 2): 
                color = (0, 255, 255) # Vàng (Chưa chắc chắn lắm)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Hiển thị ID và Label chốt
            cv2.putText(frame, f"ID:{track_id} {final_label}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Dọn dẹp bộ nhớ: Xóa các ID không còn xuất hiện quá lâu (để tránh tràn RAM dictionary)
    # (Phần này nâng cao, bạn có thể tạm bỏ qua nếu chạy ngắn)

    cv2.imshow("Smart Waste Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()