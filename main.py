from ultralytics import YOLO

# 1. Load the AI Model
# 'yolov8n-pose.pt' is the specific "brain" file. 
# 'n' stands for 'nano' (the fastest/smallest version, perfect for laptops).
# 'pose' means it looks for skeletons (joints), not just boxes.
print("Loading AI model...")
model = YOLO('yolov8n-pose.pt') 

# 2. Run the Tracker
# source="input.mp4": The video file to analyze.
# show=True: Opens a window on your screen so you can watch the AI work.
# conf=0.3: The AI must be at least 30% sure it's a person to draw a box.
# save=True: Automatically saves the processed video to your disk.
print("Starting video analysis...")
results = model.track(source="input.mp4", show=True, save=True, conf=0.3)

print("Analysis Complete!")