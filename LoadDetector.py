import cv2
import time
import os
from ultralytics import YOLO

# --- Setup Output Folder ---
output_dir = "saved_frames"
os.makedirs(output_dir, exist_ok=True)

# --- Define Helper Function ---
def draw_bounding_box(frame, x1, y1, x2, y2, conf):
    top_left = (int(x1), int(y1))
    bottom_right = (int(x2), int(y2))
    
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    
    label = f"Bin: {conf:.2f}"
    label_y = top_left[1] - 10 if top_left[1] - 10 > 10 else top_left[1] + 20
    
    cv2.putText(frame, label, (top_left[0], label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    return frame

# --- Main Execution ---
model = YOLO("best.pt") 
video_path = "./input.mp4" 
cap = cv2.VideoCapture(video_path)

print(f"{'Frame':<8} | {'Bounding Box [x1, y1, x2, y2]':<32} | {'Conf':<6} | {'Speed(ms)':<8}")
print("-" * 70)
print(f"Saving individual frames to the '{output_dir}' folder...")

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    start_time = time.time()

    # Run Inference
    results = model.predict(frame, imgsz=320, conf=0.25, verbose=False)

    inference_time = (time.time() - start_time) * 1000

    # Extract, format, and draw the bounding boxes
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]

            print(f"{frame_idx:<8} | [{x1:^4}, {y1:^4}, {x2:^4}, {y2:^4}]{' ':<9} | {conf:.2f} | {inference_time:.1f}")

            # Draw the bounding box
            frame = draw_bounding_box(frame, x1, y1, x2, y2, conf)

    # Save the frame as an image instead of trying to open a window
    cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.jpg", frame)

    frame_idx += 1

# Clean up
cap.release()
print(f"Finished! You can now open the '{output_dir}' folder and click through the images.")