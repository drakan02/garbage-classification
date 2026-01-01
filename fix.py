import onnx

# Tên file bị lỗi opset 22
input_model = "bestv22.onnx"
# Tên file mới sẽ được tạo ra
output_model = "bestv21.onnx"

try:
    print(f"Đang tải {input_model}...")
    model = onnx.load(input_model)

    # Kiểm tra version hiện tại
    current_version = model.opset_import[0].version
    print(f"Opset hiện tại: {current_version}")

    if current_version > 21:
        # Hạ version xuống 21 (hoặc 17, 11 tùy ý)
        print("Phát hiện Opset quá cao. Đang hạ xuống mức 21...")
        model.opset_import[0].version = 21
        
        # Lưu file mới
        onnx.save(model, output_model)
        print(f"XONG! Đã lưu file sửa lỗi tại: {output_model}")
        print("Hãy dùng file này để chạy code.")
    else:
        print("Opset đã thấp sẵn rồi, không cần sửa.")

except Exception as e:
    print(f"Có lỗi xảy ra: {e}")