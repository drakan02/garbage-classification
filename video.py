import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

MODEL_PATH = "garbage_project/train_run/weights/best.pt"
VIDEO_PATH = "input.mp4" 
OUTPUT_PATH = "output.mp4"
FONT_PATH = "arial.ttf"

SPEED_FACTOR = 0.75 
WARNING_TARGET_CLASS = "battery"

CONF_TRACK = 0.05         
CONF_OTHER_FILTER = 0.50  

MAX_BATTERY_AREA_RATIO = 0.15

X_MIN_SCAN_RATIO = 0.15 
X_MAX_SCAN_RATIO = 0.85

Y_MIN_VALID_RATIO = 0.30 
Y_MAX_VALID_RATIO = 0.80

COLOR_SATURATION_THRESHOLD = 60 
LOCK_CONFIDENCE_THRESHOLD = 0.70

def analyze_saturation(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])

def draw_text(img_cv2, text_lines, font_path, font_size=50, color=(0, 0, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype(font_path, font_size)
    except: font = ImageFont.load_default()

    W, H = img_pil.size
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in text_lines]
    total_h = sum([b[3]-b[1] for b in bboxes]) + 10 * (len(text_lines)-1)
    start_y = (H - total_h) // 2
    
    current_y = start_y
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (W - w) // 2
        draw.text((x-2, current_y-2), line, font=font, fill=(255,255,255))
        draw.text((x+2, current_y+2), line, font=font, fill=(255,255,255))
        draw.text((x, current_y), line, font=font, fill=color)
        current_y += h + 10
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    print(f"🔄 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_area = width * height

    # Tọa độ vùng an toàn
    y_min_valid = int(height * Y_MIN_VALID_RATIO)
    y_max_valid = int(height * Y_MAX_VALID_RATIO)
    
    # Tọa độ vùng quét ngang
    x_min_scan = int(width * X_MIN_SCAN_RATIO)
    x_max_scan = int(width * X_MAX_SCAN_RATIO)

    output_fps = fps * SPEED_FACTOR
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, output_fps, (width, height))

    warning_counter = 0
    warning_duration_frames = int(5 * output_fps) 
    
    locked_objects = {}
    confirmed_dangerous_ids = set()

    frame_idx = 0
    print("✅ Bắt đầu")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        results = model.track(frame, persist=True, conf=CONF_TRACK, verbose=False, 
                              tracker="bytetrack.yaml", agnostic_nms=True, iou=0.5)
        
        annotated_frame = frame.copy()
        current_frame_has_battery = False

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                area_ratio = (w*h) / total_area
                cy = (y1 + y2) / 2
                cx = (x1 + x2) / 2

                if cy < y_min_valid or cy > y_max_valid:
                    continue 

                is_entering_or_exiting = cx < x_min_scan or cx > x_max_scan
                
                if is_entering_or_exiting: continue 
                
                detected_class_name = model.names[int(cls)]
                corrected_class_name = detected_class_name
                if detected_class_name.lower() == "metal":
                    mean_sat = analyze_saturation(frame, box)
                    if mean_sat < COLOR_SATURATION_THRESHOLD:
                        corrected_class_name = "Paper"
                        if track_id in locked_objects and locked_objects[track_id]["class"].lower() == "metal":
                            del locked_objects[track_id]

                if area_ratio > MAX_BATTERY_AREA_RATIO:
                    if track_id in locked_objects and locked_objects[track_id]["class"].lower() == WARNING_TARGET_CLASS:
                        del locked_objects[track_id]
                        if track_id in confirmed_dangerous_ids:
                            confirmed_dangerous_ids.remove(track_id)

                if track_id not in locked_objects:
                    should_lock = False
                    if conf >= LOCK_CONFIDENCE_THRESHOLD: should_lock = True
                    if corrected_class_name.lower() == WARNING_TARGET_CLASS and area_ratio <= MAX_BATTERY_AREA_RATIO: 
                        should_lock = True
                    
                    if should_lock:
                        locked_objects[track_id] = {
                            "class": corrected_class_name,
                            "conf": conf
                        }
                else:
                    if corrected_class_name.lower() == WARNING_TARGET_CLASS and locked_objects[track_id]["class"].lower() != WARNING_TARGET_CLASS:
                         if area_ratio <= MAX_BATTERY_AREA_RATIO:
                            locked_objects[track_id] = {
                                "class": corrected_class_name,
                                "conf": conf
                            }

                if track_id in locked_objects:
                    final_class_name = locked_objects[track_id]["class"]
                    final_conf = locked_objects[track_id]["conf"]
                else:
                    final_class_name = corrected_class_name
                    final_conf = conf

                is_battery_class = final_class_name.lower() == WARNING_TARGET_CLASS
                
                if is_battery_class:
                    if area_ratio > MAX_BATTERY_AREA_RATIO:
                        is_dangerous = False
                        final_class_name = detected_class_name if detected_class_name.lower() != "battery" else "Object"
                    else:
                        confirmed_dangerous_ids.add(track_id)
                        is_dangerous = True
                else:
                    is_dangerous = False

                if is_dangerous:
                    color = (0, 0, 255)
                    label = f"WARNING: {final_class_name} {final_conf:.2f}"
                    current_frame_has_battery = True
                else:
                    if final_conf < CONF_OTHER_FILTER and track_id not in locked_objects: continue
                    color = (0, 255, 0) 
                    label = f"{final_class_name} {final_conf:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = y1 - 10 if y1 > 30 else y1 + 25
                cv2.rectangle(annotated_frame, (x1, label_y - text_h - 5), (x1 + text_w, label_y + 5), color, -1)
                cv2.putText(annotated_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if current_frame_has_battery:
            warning_counter = warning_duration_frames
        
        if warning_counter > 0:
            warning_counter -= 1
            if (frame_idx % 10) < 5:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                annotated_frame = draw_text(
                    annotated_frame, 
                    ["WARNING: BATTERY DETECTED! \n \n", "DO NOT THROW IN THE TRASH"], 
                    FONT_PATH, font_size=60
                )

        out.write(annotated_frame)
        if frame_idx % 30 == 0: print(f"Processing {frame_idx}...")

    cap.release()
    out.release()
    print(f"✅ Video output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()