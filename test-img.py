import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import os

# --- CẤU HÌNH ---
MODEL_PATH = 'garbage_project/train_run/weights/best.pt'  # Đường dẫn đến file model đã train xong

def main():
    # 1. Kiểm tra model
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file model '{MODEL_PATH}'")
        return

    print("Đang tải mô hình...")
    model = YOLO(MODEL_PATH)
    print("Đã tải xong! Đang mở cửa sổ chọn ảnh...")

    # 2. Vòng lặp để test nhiều ảnh liên tục
    while True:
        # Tạo cửa sổ ẩn của tkinter 
        root = tk.Tk()
        root.withdraw() 

        # Mở hộp thoại chọn file
        print("\nHãy chọn file ảnh trong cửa sổ vừa hiện ra...")
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để test rác",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        # Nếu người dùng bấm Cancel hoặc tắt cửa sổ thì dừng chương trình
        if not file_path:
            print("Đã hủy chọn ảnh. Kết thúc chương trình.")
            break

        print(f"Đang nhận diện ảnh: {os.path.basename(file_path)}")

        # 3. Dự đoán
        results = model.predict(source=file_path, conf=0.4, save=False)

        # 4. Hiển thị kết quả bằng OpenCV
        for result in results:
            img_result = result.plot() # Vẽ box lên ảnh

            # Resize lại ảnh nếu quá to để vừa màn hình
            h, w = img_result.shape[:2]
            if h > 800: # Nếu cao quá 800px thì thu nhỏ
                scale = 800 / h
                img_result = cv2.resize(img_result, (int(w * scale), 800))

            cv2.imshow("Ket qua nhan dien (Nhan phim bat ky de chon anh tiep theo)", img_result)
            
            print("Nhấn [X] để chọn ảnh tiếp theo...")
            print("Nhấn phím 'q' để tắt.")
            
            # Đợi người dùng nhấn phím (0 nghĩa là đợi mãi mãi)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Nếu nhấn phím 'q' hoặc 'Esc' (27) thì thoát luôn
            if key == 27 or key == ord('q'):
                print("Kết thúc.")
                return

if __name__ == "__main__":
    main()