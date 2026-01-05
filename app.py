import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import torch
import time

# =====================
# CONFIG
# =====================
MODEL_PATH = "garbage_project/train_run/weights/best.pt"
IMG_SIZE = 640
INTERVAL = 0.5          # ~2 FPS (ổn định cho demo)
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =====================
# LOAD MODEL
# =====================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Không tìm thấy model")

model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"✅ Model loaded on {DEVICE}")

# =====================
# GLOBAL STATE
# =====================
last_time = 0

# =====================
# UTILS
# =====================
def resize(img):
    return img.resize((IMG_SIZE, IMG_SIZE))

def summarize(result):
    counts = {}
    names = result.names

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id]
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        return "⚠️ Không phát hiện rác."

    text = "### 📊 Kết quả:\n"
    for k, v in counts.items():
        text += f"- **{k.capitalize()}**: {v}\n"
    return text

# =====================
# REALTIME WEBCAM
# =====================
def predict_cam(frame, conf):
    global last_time

    if frame is None:
        return None, "⚠️ Chưa có dữ liệu webcam"

    now = time.time()
    if now - last_time < INTERVAL:
        return gr.update(), gr.update()

    last_time = now

    img = Image.fromarray(frame)
    img = resize(img)

    result = model.predict(
        img,
        imgsz=IMG_SIZE,
        conf=conf,
        device=DEVICE,
        verbose=False
    )[0]

    plotted = result.plot()[..., ::-1]   # BGR → RGB
    return plotted, summarize(result)

# =====================
# UI
# =====================
with gr.Blocks(title="Realtime Garbage Detection YOLO") as demo:
    gr.Markdown("# ♻️ PHÂN LOẠI RÁC THẢI – WEBCAM REALTIME")

    conf = gr.Slider(
        0.1, 0.9, 0.4,
        step=0.05,
        label="Độ tin cậy"
    )

    with gr.Row():
        cam_in = gr.Image(
            source="webcam",
            type="numpy",
            streaming=True,      # 🔥 QUAN TRỌNG
            label="Webcam"
        )
        cam_out = gr.Image(label="Nhận dạng (YOLO)")

    cam_stat = gr.Markdown()

    cam_in.change(
        fn=predict_cam,
        inputs=[cam_in, conf],
        outputs=[cam_out, cam_stat],
        show_progress=False
    )

# =====================
# RUN
# =====================
if __name__ == "__main__":
    demo.launch(share=True)
