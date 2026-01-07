from ultralytics import YOLO
import torch

def main():
    print("Đang tải mô hình YOLO12n ...")
    model = YOLO("yolo12n.pt")

    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,

        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        patience=30,

        augment=True,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,

        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,

        workers=4,
        project="garbage_project",
        name="train_run"
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()