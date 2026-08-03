import cv2
import os

# Input video
video_path = "Drop.avi"

# Output folder
output_folder = "dataset"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    filename = os.path.join(output_folder, f"frame{frame_count:04d}.png")
    cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()

print(f"Done! Extracted {frame_count} frames.")