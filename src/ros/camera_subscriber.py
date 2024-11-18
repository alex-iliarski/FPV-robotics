#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TurtleBotDetector:
    def __init__(self):
        rospy.init_node('camera_subscriber', anonymous=True)
        self.image_sub = rospy.Subscriber('/eye_camera/image_raw', Image, self.image_callback)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CvBridge error: {0}".format(e))
            return

        self.detect_turtlebot(cv_image)

    def detect_turtlebot(self, image):
        # Get the image dimensions and calculate the center
        height, width = image.shape[:2]
        frame_center = (width // 2, height // 2)

        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([60, 60, 60])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        closest_contour = None
        closest_distance = float('inf')

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                # Get bounding box and center of the contour
                x, y, w, h = cv2.boundingRect(contour)
                contour_center = (x + w // 2, y + h // 2)

                # Calculate Euclidean distance to the frame center
                distance = ((contour_center[0] - frame_center[0]) ** 2 + (contour_center[1] - frame_center[1]) ** 2) ** 0.5

                if distance < closest_distance:
                    closest_distance = distance
                    closest_contour = (x, y, w, h)

        # If a closest contour is found, draw a bounding box
        if closest_contour:
            x, y, w, h = closest_contour
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rospy.loginfo(f"TurtleBot detected at position: x={x}, y={y}, distance={closest_distance}")

        # Display the processed image
        cv2.imshow("TurtleBot Detection", image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = TurtleBotDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass