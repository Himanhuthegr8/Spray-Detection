from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/segment/spray_seg/weights/best.pt')

# Run prediction on the video
# stream=True is CRITICAL for long videos so it doesn't try to load all frames into memory at once!
print("Starting video prediction... This might take a while.")
for result in model.predict(
    source='Drop.avi', 
    save=True, 
    stream=True,
    half=True,          # Uses FP16 precision to cut GPU memory usage in half!
    show_labels=False,  # Hides the text labels (e.g., 'droplet')
    show_conf=False,    # Hides the confidence scores (e.g., '0.95')
    show_boxes=False,   # Hides the bounding box rectangles
):
    pass  # The frames are automatically processed and saved to the video output

print("Finished! Check the runs/segment/predict folder for the video.")
