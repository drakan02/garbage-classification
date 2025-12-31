from ultralytics import YOLO
import cv2

# 1. Load model của bạn (Đảm bảo đường dẫn đúng)
model = YOLO("taco2-v12m.pt") 

# 2. Chạy dự đoán trên Camera
# source=0: Webcam mặc định
# show=True: Hiển thị cửa sổ kết quả
# conf=0.3: Chỉ hiện những vật có độ tin cậy > 30% (Do model chưa train xong nên để thấp thôi)
# imgsz=640: Kích thước ảnh đầu vào
print("Đang mở camera... Nhấn 'q' để thoát.")

# Thêm verbose=False
results = model.predict(source=0, show=True, conf=0.4, verbose=False, imgsz=640)