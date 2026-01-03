import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =====================
# CONFIG
# =====================
MODEL_PATH = "taco1-v12s.pt"   # detection model
CONF_THRESHOLD = 0.5           # giống notebook
# =====================

# Load model
print("⏳ Loading model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

# =====================
# DETECTION FUNCTION (CHUẨN YOLO)
# =====================
def predict(image):
    if image is None:
        return None

    # Đảm bảo PIL Image (YOLO xử lý tốt nhất)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Inference (KHÔNG resize thủ công)
    results = model.predict(
        image,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    # Vẽ box bằng plot() của Ultralytics
    plotted = results[0].plot()      # BGR numpy
    plotted = plotted[..., ::-1]     # BGR → RGB

    return plotted

# =====================
# UI
# =====================
with gr.Blocks(
    title="♻️ YOLO Detection – Chuẩn kết quả",
    theme=gr.themes.Soft()
) as demo:

    camera_state = gr.State(False)

    gr.Markdown("# ♻️ HỆ THỐNG PHÂN LOẠI RÁC THẢI (YOLO Detection)")

    with gr.Tabs():

        # -------- TAB 1: IMAGE --------
        with gr.TabItem("🖼️ Ảnh tĩnh"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Ảnh đầu vào")
                img_output = gr.Image(label="Kết quả detection")

            btn_run = gr.Button("🔍 Phát hiện", variant="primary")
            btn_run.click(predict, img_input, img_output)

        # -------- TAB 2: WEBCAM --------
        with gr.TabItem("🎥 Webcam"):

            btn_toggle = gr.Button("📷 BẬT CAMERA", variant="primary")
            off_message = gr.Markdown(
                "### ⚠️ Camera đang tắt. Bấm nút để bật.",
                visible=True
            )

            with gr.Row():
                cam_input = gr.Image(
                    source="webcam",
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                    visible=False
                )

                cam_output = gr.Image(
                    label="Realtime Detection",
                    visible=False
                )

            # Toggle camera
            def toggle_camera(is_on):
                is_on = not is_on
                if is_on:
                    return (
                        is_on,
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        "🔴 TẮT CAMERA"
                    )
                else:
                    return (
                        is_on,
                        gr.update(visible=False, value=None),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        "📷 BẬT CAMERA"
                    )

            btn_toggle.click(
                toggle_camera,
                camera_state,
                [camera_state, cam_input, cam_output, off_message, btn_toggle]
            )

            # Stream detection
            cam_input.stream(
                fn=predict,
                inputs=cam_input,
                outputs=cam_output
            )

# =====================
# RUN
# =====================
if __name__ == "__main__":
    print("🚀 Launching app...")
    demo.launch(share=True)
