import os
import matplotlib.pyplot as plt

# Đường dẫn đến thư mục dữ liệu 
ROOT_DIR = 'data_approved'

# Các định dạng ảnh chấp nhận
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def count_images():
    if not os.path.exists(ROOT_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{ROOT_DIR}'")
        return

    class_counts = {}
    
    # Lấy danh sách các folder con
    subfolders = sorted([f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))])
    
    print(f"{'CLASS NAME':<15} | {'SỐ LƯỢNG ẢNH'}")
    print("-" * 35)

    for class_name in subfolders:
        class_path = os.path.join(ROOT_DIR, class_name)
        # Đếm file có đuôi ảnh
        files = [f for f in os.listdir(class_path) if f.lower().endswith(IMG_EXTENSIONS)]
        count = len(files)
        class_counts[class_name] = count
        
        print(f"{class_name:<15} | {count}")

    print("-" * 35)
    total = sum(class_counts.values())
    print(f"{'TỔNG CỘNG':<15} | {total} ảnh")

    # --- Vẽ biểu đồ ---
    if class_counts:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_counts.keys(), class_counts.values(), color='#36a2eb')
        plt.xlabel('Tên lớp (Class)')
        plt.ylabel('Số lượng ảnh')
        plt.title('Thống kê số lượng ảnh gốc trong data_approved')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Hiển thị số trên đầu cột
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height}', ha='center', va='bottom')
        
        plt.show()

if __name__ == "__main__":
    count_images()