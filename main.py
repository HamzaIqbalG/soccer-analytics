from ultralytics import YOLO
import cv2
import pandas as pd

# 1. Load the Model
model = YOLO('yolov8n-pose.pt')

# 2. Open the video
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Store player data here
player_data = []

print("Processing video frames...")

# 3. Process frame by frame
results = model.track(source=video_path, show=True, stream=True, conf=0.3)

for frame_idx, result in enumerate(results):
    # 'result.keypoints' contains the skeleton data
    if result.keypoints is not None and result.boxes.id is not None:
        
        # Get the ID (Who is this?) and Keypoints (Where are they?)
        track_ids = result.boxes.id.int().cpu().tolist()
        keypoints = result.keypoints.xyn.cpu().numpy()  # xyn = normalized coordinates (0-1)

        for i, track_id in enumerate(track_ids):
            # YOLO Keypoint 15 is Left Ankle, 16 is Right Ankle
            # We take the average of both feet to find the "center" of the player
            left_ankle = keypoints[i][15]
            right_ankle = keypoints[i][16]
            
            # Simple math: average the x and y coordinates
            center_x = (left_ankle[0] + right_ankle[0]) / 2
            center_y = (left_ankle[1] + right_ankle[1]) / 2

            # Save this moment in time
            player_data.append({
                "Frame": frame_idx,
                "PlayerID": track_id,
                "x": center_x,
                "y": center_y
            })

# 4. Save to Excel/CSV
print("Saving data...")
df = pd.DataFrame(player_data)
df.to_csv("player_movements.csv", index=False)
print("Done! Data saved to player_movements.csv")