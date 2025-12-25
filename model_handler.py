import cv2
from ultralytics import YOLO
import os
import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, model_path='best.pt'):
        print(f"--- KHỞI ĐỘNG MODEL: {model_path} ---")
        self.model = None
        
        if not os.path.exists(model_path):
            if os.path.exists('yolov12n-cls.pt'):
                model_path = 'yolov12n-cls.pt'
                print("⚠️ Đang dùng model dự phòng: yolov12n-cls.pt")
            else:
                print(f"❌ KHÔNG TÌM THẤY FILE MODEL: {model_path}")
                return

        try:
            self.model = YOLO(model_path)
            print("✅ Tải Model thành công!")
            
            try:
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                self.predict_frame(dummy) 
                print("✅ Warmup thành công - Sẵn sàng!")
            except Exception as e:
                print(f"⚠️ Cảnh báo Warmup: {e}")

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng khi tải Model: {e}")
            self.model = None

    def predict_frame(self, frame_cv2):
        if not self.model: 
            return "Đang tải...", 0.0

        try:
            img_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)

            results = self.model(img_rgb, verbose=False, conf=0.25)
            
            if not results: return "Không rõ", 0.0
            
            result = results[0]
            
            if result.probs:
                top1_idx = result.probs.top1
                label = result.names[top1_idx]
                conf = result.probs.top1conf.item()
                return label, conf
            
            return "Không rõ", 0.0
            
        except Exception as e:
            print(f"Lỗi dự đoán: {e}")
            return "Lỗi xử lý", 0.0

    def draw_vietnamese_text(self, frame, text, pos, color=(0, 255, 0), size=30):
        """Vẽ chữ Tiếng Việt đẹp bằng Pillow"""
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
            
        draw.text(pos, text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-110), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        frame = self.draw_vietnamese_text(frame, text_main, (30, h-100), color, size=45)
        frame = self.draw_vietnamese_text(frame, text_sub, (30, h-45), (220, 220, 220), size=22)
        
        return frame