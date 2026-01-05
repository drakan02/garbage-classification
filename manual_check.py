import os
import cv2
import shutil

# --- CẤU HÌNH ---
INPUT_DIR = 'data'           # Thư mục CHỜ DUYỆT
APPROVED_DIR = 'data_approved'        # Thư mục ĐÃ DUYỆT
REJECT_DIR = '_REJECTED'    # Thư mục BỊ LOẠI 

# Tên cửa sổ
WINDOW_NAME = "Review Data (Left: OK | Right: Delete | Q: Quit)"

# Map ID sang tên
ID_TO_NAME = {
    0: 'battery', 1: 'biological', 2: 'cardboard', 3: 'clothes', 
    4: 'glass', 5: 'metal', 6: 'paper', 7: 'plastic', 8: 'shoe', 9: 'trash'
}

COLOR_BOX = (0, 255, 0)      # Xanh lá (cho khung)
COLOR_TEXT = (0, 0, 0)       # ĐEN (cho chữ)
COLOR_WARNING = (0, 0, 255)  # Đỏ (cho cảnh báo lỗi)

# Cấu hình Font chữ
FONT_SCALE_LABEL = 0.4       # Kích thước chữ của nhãn (trên object) - ĐÃ GIẢM
FONT_SCALE_INFO = 0.5        # Kích thước chữ thông tin (trên cùng) - ĐÃ GIẢM
FONT_THICKNESS = 1           # Độ dày nét chữ (mỏng lại cho dễ nhìn)

current_action = None

def on_mouse_click(event, x, y, flags, param):
    global current_action
    if event == cv2.EVENT_LBUTTONDOWN:
        current_action = 'approve'
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_action = 'reject'

def move_file_pair(img_path, dest_root_folder):
    parent_dir = os.path.dirname(img_path)
    sub_folder_name = os.path.basename(parent_dir) 
    
    dest_folder = os.path.join(dest_root_folder, sub_folder_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    filename = os.path.basename(img_path)
    dest_img_path = os.path.join(dest_folder, filename)
    shutil.move(img_path, dest_img_path)
    
    txt_path = os.path.splitext(img_path)[0] + ".txt"
    if os.path.exists(txt_path):
        dest_txt_path = os.path.join(dest_folder, os.path.basename(txt_path))
        shutil.move(txt_path, dest_txt_path)
        
    return dest_folder

def main():
    global current_action

    if not os.path.exists(APPROVED_DIR): os.makedirs(APPROVED_DIR)
    if not os.path.exists(REJECT_DIR): os.makedirs(REJECT_DIR)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

    print("="*50)
    print("CHẾ ĐỘ REVIEW BẰNG CHUỘT")
    print("Chuột TRÁI : OK (Duyệt)")
    print("Chuột PHẢI : XÓA (Reject)")
    print("Phím [Q]   : THOÁT")
    print("="*50)

    all_images = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_images.append(os.path.join(root, file))

    total = len(all_images)
    if total == 0:
        print(f"Thư mục '{INPUT_DIR}' trống trơn!")
        return

    idx = 0
    while idx < total:
        img_path = all_images[idx]
        
        if not os.path.exists(img_path):
            idx += 1
            continue

        txt_path = os.path.splitext(img_path)[0] + ".txt"
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Lỗi ảnh: {img_path}")
            move_file_pair(img_path, REJECT_DIR)
            idx += 1
            continue

        h_img, w_img, _ = img.shape
        
        num_objects = 0
        has_label = False
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                num_objects = len(lines)
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, x_n, y_n, w_n, h_n = map(float, parts)
                        cls_id = int(cls_id)
                        x_center, y_center = x_n * w_img, y_n * h_img
                        w_box, h_box = w_n * w_img, h_n * h_img
                        x1, y1 = int(x_center - w_box/2), int(y_center - h_box/2)
                        x2, y2 = int(x_center + w_box/2), int(y_center + h_box/2)

                        # Vẽ khung xanh
                        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BOX, 2)
                        
                        # Label Text
                        label_name = ID_TO_NAME.get(cls_id, str(cls_id))
                        (w_text, h_text), _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_LABEL, FONT_THICKNESS)
                        
                        # Vẽ nền trắng nhỏ hơn khớp với chữ bé
                        cv2.rectangle(img, (x1, y1 - h_text - 8), (x1 + w_text, y1), (255, 255, 255), -1)
                        
                        # Vẽ chữ bé
                        cv2.putText(img, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_LABEL, COLOR_TEXT, FONT_THICKNESS)
                        has_label = True
        
        if not has_label:
            cv2.putText(img, "NO LABEL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WARNING, 2)

        # Resize hiển thị
        display_h = 800
        if h_img > display_h:
            scale = display_h / h_img
            new_w = int(w_img * scale)
            img = cv2.resize(img, (new_w, display_h))

        # --- Hiển thị thông tin Header ---
        # Thanh trắng trên cùng (nhỏ lại, cao 30px thay vì 40px)
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (255, 255, 255), -1) 
        
        info = f"[{idx+1}/{total}] Objects: {num_objects} | {os.path.basename(img_path)}"
        # Chữ bé và mảnh hơn
        cv2.putText(img, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, COLOR_TEXT, FONT_THICKNESS)
        
        cv2.imshow(WINDOW_NAME, img)
        
        current_action = None
        while current_action is None:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                current_action = 'quit'
                break
        
        if current_action == 'quit':
            print("\nĐã thoát.")
            break
            
        elif current_action == 'approve':
            dest = move_file_pair(img_path, APPROVED_DIR)
            print(f"OK ({num_objects} objs): {os.path.basename(img_path)}")
            idx += 1
            
        elif current_action == 'reject':
            dest = move_file_pair(img_path, REJECT_DIR)
            print(f"Xóa: {os.path.basename(img_path)}")
            idx += 1

    cv2.destroyAllWindows()
    if idx >= total:
        print("\nĐÃ DUYỆT XONG TOÀN BỘ!")

if __name__ == "__main__":
    main()