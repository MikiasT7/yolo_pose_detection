from ultralytics import YOLO
import cv2

PERSON_POSE = {
    "Nose": 0,
    "Left_eye": 1,
    "Right_eye": 2,
    "Left_ear": 3,
    "Right_ear": 4,
    "Left_shoulder": 5,
    "Right_shoulder": 6,
    "Left_elbow": 7,
    "Right_elbow": 8,
    "Left_wrist": 9,
    "Right_wrist": 10,
    "Left_hip": 11,
    "Right_hip": 12,
    "Left_knee": 13,
    "Right_knee": 14,
    "Left_ankle": 15,
    "Right_ankle": 16
}

PERSON_POSE_PAIRS = [
    ["Nose", "Left_eye"], ["Left_eye", "Right_eye"], ["Nose", "Left_ear"], ["Left_ear", "Right_ear"],  # Nose to eyes and ears
    ["Left_shoulder", "Right_shoulder"],  # Shoulders
    ["Left_shoulder", "Left_elbow"], ["Left_elbow", "Left_wrist"],  # Left arm
    ["Right_shoulder", "Right_elbow"], ["Right_elbow", "Right_wrist"],  # Right arm
    ["Left_shoulder", "Left_hip"], ["Right_shoulder", "Right_hip"],  # Shoulders to hips
    ["Left_hip", "Right_hip"],  # Hips
    ["Left_hip", "Left_knee"], ["Left_knee", "Left_ankle"],  # Left leg
    ["Right_hip", "Right_knee"], ["Right_knee", "Right_ankle"]  # Right leg
]

# Load the model
model_path = "./yolo11n-pose.pt" # Model path
model = YOLO(model_path) 

# Load the image
image_path = "./img2.png" # Image path 
img = cv2.imread(image_path)

# Run the model to make predictions
results = model(image_path)[0]  # Assuming results contains multiple detections

# Iterate through each detection result
for result in results:
    keypoints = result.keypoints  # Get the keypoints for this result

    if keypoints:
        # Convert keypoints to list (you can access .xy or .xyn)
        keypoints_xy = keypoints.xy.tolist()[0]  # Get coordinates as a list (access the first list)

        print("Keypoints (xy):", keypoints_xy)
        
        # Draw circles for each keypoint
        keypoint_coords = {}
        for idx, keypoint in enumerate(keypoints_xy):
            if len(keypoint) == 2 and keypoint[0] != 0 and keypoint[1] != 0:  # [x, y]
                x, y = keypoint  # Unpack the [x, y] coordinates
                keypoint_coords[list(PERSON_POSE.keys())[idx]] = (int(x), int(y))
                # Draw a circle at each keypoint (x, y)
                cv2.circle(img, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)  

        # Draw lines for the specified part pairs
        for part1, part2 in PERSON_POSE_PAIRS:
            if part1 in keypoint_coords and part2 in keypoint_coords:
                # Get the coordinates of the two parts
                x1, y1 = keypoint_coords[part1]
                x2, y2 = keypoint_coords[part2]
                
                # Print out the coordinates to inspect the structure
                print(part1, part2)
                print(keypoint_coords[part1], keypoint_coords[part2])
                print(" ")
                
                # Draw a line between the two parts (keypoints)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=2)  

# Display the image with keypoints and lines
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
