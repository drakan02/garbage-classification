import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# =====================
# 1. LOAD MODEL
# =====================
MODEL_PATH = "waste_project/yolov12_cls_run/weights/best.pt" 

try:
    model = YOLO(MODEL_PATH)
    print("✅ Đã nạp model thành công!")
except Exception as e:
    print(f"⚠️ Không thấy file model: {e}")
    model = YOLO("yolov8n-cls.pt")

# =====================
# 2. HÀM DỰ ĐOÁN
# =====================
def predict(image):
    if image is None:
        return None
    
    results = model(image, verbose=False)
    r = results[0]
    
    probs = {}
    if hasattr(r.probs, 'top5'):
        top_k = min(3, len(r.probs.top5))
        for i in range(top_k):
            idx = r.probs.top5[i]
            score = float(r.probs.top5conf[i])
            label = r.names[idx]
            probs[label] = score
    
    return probs

# Hàm để xử lý logic Bật/Tắt Camera (Ẩn/Hiện giao diện)
def toggle_camera_visibility(is_on):
    # Đảo ngược trạng thái: Đang bật -> Tắt và ngược lại
    is_on = not is_on
    
    if is_on:
        # Nếu bật: Hiện Camera, Hiện Output, Đổi nút thành "Tắt"
        return (
            is_on,                      # Cập nhật biến trạng thái
            gr.update(visible=True),    # Hiện Camera
            gr.update(visible=True),    # Hiện Kết quả
            gr.update(visible=False),   # Ẩn thông báo "Camera đang tắt"
            "🔴 TẮT CAMERA"             # Đổi tên nút
        )
    else:
        # Nếu tắt: Ẩn Camera, Ẩn Output, Hiện thông báo, Đổi nút thành "Bật"
        return (
            is_on,
            gr.update(visible=False, value=None), # Ẩn Camera và Xóa hình cũ
            gr.update(visible=False),             # Ẩn Kết quả
            gr.update(visible=True),              # Hiện thông báo
            "📷 BẬT CAMERA"                       # Đổi tên nút
        )

# =====================
# 3. GIAO DIỆN
# =====================
custom_css = ".gradio-container {background-color: #f0f2f6}"

with gr.Blocks(title="♻️ Phân loại rác thải AI", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Biến trạng thái để nhớ Camera đang bật hay tắt (Mặc định là False - Tắt)
    camera_state = gr.State(False)

    gr.Markdown("# ♻️ HỆ THỐNG PHÂN LOẠI RÁC THẢI")
    
    with gr.Tabs():
        
        # -------- TAB 1: ẢNH TĨNH --------
        with gr.TabItem("🖼️ Tải Ảnh Lên"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Chọn ảnh từ máy")
                img_output = gr.Label(num_top_classes=3, label="Kết quả")
            
            btn_run = gr.Button("🔍 Phân loại ngay", variant="primary")
            btn_run.click(fn=predict, inputs=img_input, outputs=img_output)

        # -------- TAB 2: CAMERA --------
        with gr.TabItem("🎥 Webcam"):
            
            # Nút BẬT/TẮT CAMERA TO
            btn_toggle = gr.Button("📷 BẬT CAMERA", variant="primary")
            
            # Thông báo khi tắt camera
            off_message = gr.Markdown("### ⚠️ Camera đang tắt. Hãy bấm nút phía trên để bắt đầu.", visible=True)

            with gr.Row():
                # Camera Input (Mặc định ẩn - visible=False)
                cam_input = gr.Image(
                    source="webcam", 
                    streaming=True, 
                    label="Webcam Stream",
                    type="numpy",
                    visible=False  # <--- Quan trọng: Ẩn ngay từ đầu
                )
                
                # Output (Mặc định ẩn)
                cam_output = gr.Label(
                    num_top_classes=3, 
                    label="Kết quả Realtime",
                    visible=False 
                )

            # --- SỰ KIỆN ---
            
            # 1. Bấm nút Bật/Tắt -> Gọi hàm ẩn hiện giao diện
            btn_toggle.click(
                fn=toggle_camera_visibility,
                inputs=[camera_state],
                outputs=[camera_state, cam_input, cam_output, off_message, btn_toggle]
            )

            # 2. Luồng xử lý AI (Chạy ngầm, nhưng chỉ hoạt động khi cam_input có dữ liệu)
            cam_input.stream(
                fn=predict, 
                inputs=cam_input, 
                outputs=cam_output
            )

# =====================
# 4. CHẠY APP
# =====================
if __name__ == "__main__":
    print("🚀 Đang khởi động Web App...")
    demo.launch(share=True, app_kwargs={"docs_url": None, "redoc_url": None})