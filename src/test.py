import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
import mediapipe as mp


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables to handle selection and tracking
selected_box = None
dominant_color = None

def detect_objects_with_contours(frame, min_area=500, max_area=50000):
    """Detect objects using contours and edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def find_closest_object(target_point, detections):
    """Find the bounding box closest to the pointing target."""
    closest_object = None
    min_distance = float('inf')

    for (bx, by, bw, bh) in detections:
        center_x = bx + bw // 2
        center_y = by + bh // 2
        distance = np.linalg.norm([target_point[0] - center_x, target_point[1] - center_y])
        if distance < min_distance:
            min_distance = distance
            closest_object = (bx, by, bw, bh)

    return closest_object

# Initialize video capture
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Detect objects in the current frame
    detections = detect_objects_with_contours(frame)

    # Hand detection and pointing direction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    target_point = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]  # Wrist
            fingertip = hand_landmarks.landmark[8]  # Index finger tip

            wrist_coords = np.array([int(wrist.x * w), int(wrist.y * h)])
            fingertip_coords = np.array([int(fingertip.x * w), int(fingertip.y * h)])
            target_point = fingertip_coords

            # Draw pointing vector
            cv2.arrowedLine(frame, tuple(wrist_coords), tuple(fingertip_coords), (0, 255, 0), 2)
            cv2.circle(frame, tuple(target_point), 10, (255, 0, 0), -1)

    # Select closest object to pointing gesture
    if target_point is not None:
        closest_object = find_closest_object(target_point, detections)
        if closest_object:
            bx, by, bw, bh = closest_object
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, "Selected", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw all detected objects
    for (bx, by, bw, bh) in detections:
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
        cv2.putText(frame, "Object", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the frame
    cv2.imshow('Object Detection and Pointing Selection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
