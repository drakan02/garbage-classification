import cv2
from ultralytics import YOLO
import os

def test_custom_model():
    model_path = 'garbage_project/train_run/weights/best.pt' 
    # ==========================================

    # Kiểm tra model
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy file model tại '{model_path}'")
        print("Vui lòng kiểm tra lại đường dẫn file .pt của bạn.")
        return

    print(f"Đang tải model từ: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return

    # Mở camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    print("Đang chạy model... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy dự đoán
        # conf=0.5: Chỉ hiện các vật thể có độ tin cậy > 50%
        results = model(frame, conf=0.5, verbose=False)

        # Vẽ kết quả lên hình
        annotated_frame = results[0].plot()

        # Hiển thị
        cv2.imshow('Custom Model Test', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_custom_model()