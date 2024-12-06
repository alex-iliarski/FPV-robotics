import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load custom YOLOv8 model
model = YOLO("goal_yolov8n.pt")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables for selected object tracking
selected_box = None
selected_label = None


def detect_goal_objects(frame):
    """Detect goal objects using YOLO."""
    results = model(frame)
    bounding_boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        label = results[0].names[int(box.cls)]  # Get object label
        bounding_boxes.append((x1, y1, x2 - x1, y2 - y1, label))  # Return bounding boxes with labels
    return bounding_boxes


def find_closest_object(target_point, detections):
    """Find the bounding box closest to the pointing target."""
    closest_object = None
    min_distance = float('inf')

    for (bx, by, bw, bh, label) in detections:
        center_x = bx + bw // 2
        center_y = by + bh // 2
        distance = np.linalg.norm([target_point[0] - center_x, target_point[1] - center_y])
        if distance < min_distance:
            min_distance = distance
            closest_object = (bx, by, bw, bh, label)

    return closest_object


# Initialize video capture
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    if selected_box is None:
        # Detect goal objects in the frame
        detections = detect_goal_objects(frame)

        # Detect hand and find the pointing direction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        target_point = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Visualize hand joints
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get fingertip (index finger tip) coordinates
                fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                target_point = np.array([int(fingertip.x * w), int(fingertip.y * h)])

                # Draw fingertip point
                cv2.circle(frame, tuple(target_point), 10, (255, 0, 0), -1)

        # Select the closest object to the fingertip
        if target_point is not None:
            closest_object = find_closest_object(target_point, detections)
            if closest_object:
                bx, by, bw, bh, label = closest_object
                selected_box = (bx, by, bw, bh)
                selected_label = label
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"Selected: {label}", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw all detected objects
        for (x, y, w, h, label) in detections:
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        # Use YOLO to detect the selected object by label
        detections = detect_goal_objects(frame)
        found = False

        for x, y, w, h, label in detections:
            if label == selected_label:
                selected_box = (x, y, w, h)  # Update box
                found = True
                break

        # Draw the selected object
        if found:
            x, y, w, h = selected_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking: {selected_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost - Press R to reset", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Pointing-Based Object Selection and Tracking', frame)

    # Check for reset or quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Reset the tracker
        selected_box = None
        selected_label = None
    elif key == ord('q'):  # Quit the application
        break

cap.release()
cv2.destroyAllWindows()
