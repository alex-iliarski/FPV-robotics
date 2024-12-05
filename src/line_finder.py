#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray

class LineDetector:
    def __init__(self):
        # Initialize ROS Node
        rospy.init_node('line_detector', anonymous=True)

        # Subscribe to the camera image and bounding box topics
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.bbox_sub = rospy.Subscriber("/turtlebot_detector/bounding_box", Int32MultiArray, self.bbox_callback)

        self.line_search = False
        self.bbox = None
        
        # Publish the detected line
        self.line_pub = rospy.Publisher(f"/line_detector/line", Int32MultiArray, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

    def bbox_callback(self, msg):
        # check if the message is a valid bounding box
        if len(msg.data) == 4:
            self.line_search = True
            self.bbox = msg.data
        else:
            self.line_search = False

    def image_callback(self, msg):
        # Do not search for the guiding line if turtlebot is not detected
        if not self.line_search:
            self.line_pub.publish(Int32MultiArray(data=[]))
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # crop fram to fit in bounding box
        x1, y1, x2, y2 = self.bbox
        frame = frame[y1:y2, x1:x2]

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue color ranges in HSV space
        lower_blue = np.array([100, 100, 100])  # Lower bound for blue hue, saturation, and value
        upper_blue = np.array([140, 255, 255])  # Upper bound for blue hue, saturation, and value
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Clean up the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel for cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (we assume it's the rectangular shape)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit a minimum area rectangle to the contour
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)  # Get the 4 vertices of the rectangle
            box = np.int32(box)  # Convert the vertices to integers

            # Draw the bounding rectangle
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # Find the long side of the rectangle (by comparing side lengths)
            side_lengths = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
            longest_side_idx = np.argmax(side_lengths)

            # Get the coordinates of the long side
            p1 = box[longest_side_idx]
            p2 = box[(longest_side_idx + 1) % 4]

            if p2[0] - p1[0] != 0:  # Avoid division by zero
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                b = p1[1] - m * p1[0]

                # Set the extension length (30 pixels)
                extension_length = 30

                # Calculate direction of the line (normalize the direction vector)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx**2 + dy**2)
                dx_normalized = dx / length
                dy_normalized = dy / length

                # Extend the line by 30 pixels in both directions
                p1_extended = (int(p1[0] - dx_normalized * extension_length), int(p1[1] - dy_normalized * extension_length))
                p2_extended = (int(p2[0] + dx_normalized * extension_length), int(p2[1] + dy_normalized * extension_length))

                # put line in the original frame
                p1_original = (p1_extended[0] + x1, p1_extended[1] + y1)
                p2_original = (p2_extended[0] + x1, p2_extended[1] + y1)

                line = Int32MultiArray(data=[p1_original[0], p1_original[1], p2_original[0], p2_original[1]])
                self.line_pub.publish(line)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = LineDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass


