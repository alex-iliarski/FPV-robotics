import cv2
import numpy as np

# Load or capture a frame
frame = cv2.imread('image_with_oriented_object.jpg')

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Use edge detection to find contours
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and detect orientation
for contour in contours:
    # Fit a minimum area rectangle around the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the rectangle
    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    # Get the angle of orientation
    angle = rect[-1]
    print(f"Orientation angle: {angle}")

# Display the frame with orientation
cv2.imshow('Orientation Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
