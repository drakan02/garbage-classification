from ultralytics import YOLO

def main():
    # 1. Load mô hình
    # Dùng bản 'n' (nano) để train nhanh nhất và nhẹ nhất
    # Nếu máy bạn có GPU mạnh (RTX 3060 trở lên), có thể thử 'yolov12s.pt' (small) để chính xác hơn
    print("Đang tải mô hình YOLOv12n...")
    model = YOLO('yolov12n.pt') 

    # 2. Bắt đầu Train
    # data: trỏ vào file yaml vừa tạo
    # epochs: số vòng lặp học (50-100 là ổn cho 1500 ảnh)
    # imgsz: kích thước ảnh (640 là chuẩn)
    # batch: số ảnh học cùng lúc (16 là an toàn cho RAM, nếu lỗi OutOfMemory thì giảm xuống 8 hoặc 4)
    print("Bắt đầu huấn luyện...")
    results = model.train(
        data='taco.yaml',      # đường dẫn file data.yaml
        imgsz=640,
        epochs=250,            # data nhỏ → train lâu hơn
        batch=16,              # T4 / 8GB OK
        device=0,              # GPU
        optimizer="AdamW",     # ổn định cho data nhỏ
        lr0=0.001,             # learning rate thấp tránh overfit
        freeze=10,             # FREEZE backbone (rất quan trọng)
        patience=50,           # early stopping
        cos_lr=True,           # cosine LR
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,       # tắt mosaic ở cuối
        workers=2,
        verbose=True,
        plots=True
    )

if __name__ == '__main__':
    main()