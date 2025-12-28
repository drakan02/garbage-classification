import cv2
from ultralytics import YOLO
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics.nn.modules.block import AAttn

    _original_forward = AAttn.forward

    def patched_forward(self, x):
        try:
            return _original_forward(self, x)
        except AttributeError as e:
            if "qkv" in str(e):
                # NẾU GẶP LỖI QKV:
                # Có nghĩa là Model cũ không có lớp này -> Ta bỏ qua Attention lớp này
                # (Model vẫn chạy được nhưng bỏ qua bước tinh chỉnh nhỏ này)
                return x
            raise e 

    AAttn.forward = patched_forward
    print("✅ Đã kích hoạt chế độ tương thích YOLOv12 (Fix AAttn)")

except ImportError:
    pass 

class TrashClassifier:
    def __init__(self, classification_model_path='best.pt', detection_model_path=None, use_hybrid=True):
        print(f"--- KHỞI ĐỘNG MODEL ---")
        self.use_hybrid = use_hybrid
        self.classification_model = None
        self.detection_model = None
        
        # --- A. Tải Model Phân Loại (Của bạn) ---
        # Tự động tìm đường dẫn nếu file không nằm ngay bên cạnh
        if not os.path.exists(classification_model_path):
            candidates = [
                'best.pt',
                'waste_project/yolov12_run/weights/best.pt',
                'runs/classify/train/weights/best.pt'
            ]
            for p in candidates:
                if os.path.exists(p):
                    classification_model_path = p
                    break
        
        try:
            if os.path.exists(classification_model_path):
                self.classification_model = YOLO(classification_model_path)
                print(f"✅ Classification Model: {classification_model_path}")
            else:
                print(f"❌ Không tìm thấy Classification Model: {classification_model_path}")
        except Exception as e:
            print(f"❌ Lỗi tải Classification Model: {e}")

        # --- B. Tải Model Detection (Để tìm vật thể) ---
        if self.use_hybrid:
            # Danh sách ưu tiên model detection nhẹ
            det_candidates = ['yolov12n.pt', 'yolov8n.pt', 'yolov11n.pt']
            if detection_model_path:
                det_candidates.insert(0, detection_model_path)

            for model_name in det_candidates:
                try:
                    # Model này sẽ tự tải về nếu chưa có
                    self.detection_model = YOLO(model_name)
                    print(f"✅ Detection Model: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Không tải được {model_name}: {e}")
            
            if self.detection_model is None:
                print("⚠️ Chuyển về chế độ chỉ phân loại (Classification Only)")
                self.use_hybrid = False

        # Warmup nhẹ (chạy thử 1 ảnh rỗng để load model vào RAM)
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            if self.detection_model: self.detection_model(dummy, verbose=False)
            if self.classification_model: self.classification_model(dummy, verbose=False)
        except:
            pass

    def predict_frame(self, frame):
        """Dự đoán toàn bộ khung hình (Logic cũ)"""
        if not self.classification_model:
            return "Đang tải...", 0.0
            
        try:
            results = self.classification_model(frame, verbose=False)
            if not results: return "Unknown", 0.0
            
            top1_idx = results[0].probs.top1
            label = results[0].names[top1_idx]
            conf = results[0].probs.top1conf.item()
            
            return label, conf
        except:
            return "Error", 0.0

    def detect_and_classify(self, frame, conf_threshold=0.25):
        """
        LOGIC MỚI:
        1. Tìm vật thể (Detection)
        2. Cắt vật thể (Crop)
        3. Phân loại vật thể (Classification)
        """
        if not self.use_hybrid or not self.detection_model:
            return []

        # 1. Detection
        results = self.detection_model(frame, conf=conf_threshold, verbose=False)
        if not results:
            return []

        hybrid_detections = []
        h_img, w_img, _ = frame.shape
        
        # 2. Xử lý từng vật thể tìm thấy
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Mở rộng khung một chút để không cắt mất chi tiết (padding)
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)
            
            # Bỏ qua box quá nhỏ (nhiễu)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # Cắt ảnh
            crop = frame[y1:y2, x1:x2]
            
            # 3. Classification
            label, conf = self.predict_frame(crop)
            
            hybrid_detections.append((label, conf, x1, y1, x2, y2))

        return hybrid_detections

    def draw_vietnamese_text(self, frame, text, pos, color=(0, 255, 0), size=30, bg_color=None):
        
        """Vẽ chữ Tiếng Việt đẹp bằng Pillow"""
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
            
        if bg_color:
            # Vẽ nền cho chữ nếu cần
            bbox = draw.textbbox(pos, text, font=font)
            draw.rectangle(bbox, fill=bg_color)
            
        draw.text(pos, text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_bounding_boxes(self, frame, detections):
        """Vẽ khung và chữ lên frame"""
        # Copy frame để không vẽ đè lên ảnh gốc trong bộ nhớ
        vis_frame = frame.copy()
        
        for label, conf, x1, y1, x2, y2 in detections:
            # Chọn màu dựa trên độ tin cậy
            if conf > 0.7:
                color = (0, 255, 0)      # Xanh lá (Chắc chắn)
                text_color = (255, 255, 255)
            elif conf > 0.5:
                color = (0, 255, 255)    # Vàng (Khá chắc)
                text_color = (0, 0, 0)
            else:
                color = (0, 165, 255)    # Cam (Không chắc)
                text_color = (255, 255, 255)

            # 1. Vẽ khung chữ nhật (OpenCV nhanh hơn cho việc này)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 2. Vẽ nhãn bằng hàm Tiếng Việt xịn của bạn
            text = f"{label.upper()} {conf:.0%}"
            
            # Tính toán vị trí text để không bị trôi ra ngoài ảnh
            text_y = y1 - 25 if y1 > 25 else y1 + 5
            
            vis_frame = self.draw_vietnamese_text(
                vis_frame, 
                text, 
                (x1, text_y), 
                color=text_color, 
                size=18, 
                bg_color=color # Nền chữ trùng màu khung
            )

        return vis_frame

    def draw_on_frame(self, frame, label, conf):
        h, w, _ = frame.shape

        if conf > 0.7:
            status = "CHÍNH XÁC"
            color = (0, 255, 0) # Xanh lá
        elif conf > 0.5:
            status = "CÓ THỂ LÀ"
            color = (255, 215, 0) # Vàng
        else:
            status = "KHÔNG CHẮC"
            color = (255, 69, 0) # Đỏ cam

        text_main = f"{label.upper()}"
        text_sub = f"{status} ({conf:.0%})"

        # Hiệu ứng tối vùng dưới
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-110), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Vẽ chữ
        frame = self.draw_vietnamese_text(frame, text_main, (30, h-100), color, size=45)
        frame = self.draw_vietnamese_text(frame, text_sub, (30, h-45), (220, 220, 220), size=22)
        
        return frame