import os
import glob
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# --- CẤU HÌNH ---
# Hãy đảm bảo tên thư mục đúng với folder bạn vừa tạo ở bước trước
dataset_path = 'taco_yolo' 

# Danh sách class theo chiến thuật 5 nhóm mới
class_names = {
    0: 'Soft Plastic',    # Túi, Vỏ kẹo
    1: 'Hard Plastic',    # Chai, Hộp, Cốc
    2: 'Paper',           # Giấy, Bìa
    3: 'Metal & Glass',   # Lon, Chai thủy tinh
    4: 'Cigarette'        # Thuốc lá
}

# Màu sắc cho biểu đồ (tương ứng với từng loại rác cho dễ nhìn)
colors = ['#4287f5', '#004aad', '#f5e042', '#a0a0a0', '#eb7134']
#         Nhựa mềm   Nhựa cứng   Giấy       Kim loại    Thuốc lá

def count_classes():
    # 1. Kiểm tra thư mục
    if not os.path.exists(dataset_path):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{dataset_path}'.") 
        print("👉 Bạn hãy chạy file prepare_taco.py (logic mới) trước nhé!")
        return

    # 2. Quét file nhãn
    print(f"🔍 Đang quét dữ liệu trong: {dataset_path} ...")
    train_txt = glob.glob(os.path.join(dataset_path, 'labels', 'train', '*.txt'))
    val_txt = glob.glob(os.path.join(dataset_path, 'labels', 'val', '*.txt'))
    all_files = train_txt + val_txt

    if len(all_files) == 0:
        print("❌ Không tìm thấy file .txt nào trong thư mục labels.")
        return

    print(f"✅ Tìm thấy tổng cộng {len(all_files)} file ảnh đã gán nhãn.")
    
    cnt = Counter()

    # 3. Đếm số lượng object
    for file_path in all_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        cnt[class_id] += 1
                except (ValueError, IndexError):
                    continue

    # 4. Hiển thị bảng kết quả
    print("\n" + "="*60)
    print("{:<5} | {:<20} | {:<10} | {:<10}".format("ID", "Tên Class", "Số lượng", "Tỷ lệ %"))
    print("-" * 60)
    
    names = []
    counts = []
    total_objects = sum(cnt.values())
    
    if total_objects == 0:
        print("⚠️ Chưa có object nào được gán nhãn!")
        return

    # Duyệt từ 0 đến 4 để hiển thị đúng thứ tự
    for i in range(5):
        name = class_names.get(i, "Unknown")
        count = cnt[i]
        percent = (count / total_objects) * 100
        
        names.append(name)
        counts.append(count)
        
        print("{:<5} | {:<20} | {:<10} | {:.1f}%".format(i, name, count, percent))

    print("-" * 60)
    print(f"TỔNG CỘNG: {total_objects} vật thể rác")
    print("="*60)

    # 5. Vẽ biểu đồ
    plt.figure(figsize=(12, 7))
    bars = plt.bar(names, counts, color=colors, edgecolor='black', alpha=0.8)
    
    # Viết số lượng và phần trăm lên đầu mỗi cột
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percent = (count / total_objects) * 100
        label = f"{count}\n({percent:.1f}%)"
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, label, 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.title(f'Thống kê dataset TACO (Chiến lược 5 Class Cân bằng)', fontsize=14, fontweight='bold')
    plt.xlabel('Loại rác', fontsize=12)
    plt.ylabel('Số lượng mẫu (Objects)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, max(counts) * 1.15) # Tăng chiều cao biểu đồ để số không bị che
    
    plt.show()

if __name__ == "__main__":
    count_classes()