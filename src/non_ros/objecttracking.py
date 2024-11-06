import cv2

# Initialize tracker (can use 'KCF', 'MIL', 'CSRT', etc.)
tracker = cv2.TrackerKCF_create()

# Start video capture
cap = cv2.VideoCapture(1)
ret, frame = cap.read()

# Select the object to track (bounding box)
bbox = cv2.selectROI("Tracking", frame, False)

# Initialize tracker with first frame and bounding box
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()

    # Update tracker
    success, bbox = tracker.update(frame)

    # Draw bounding box
    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Object Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
