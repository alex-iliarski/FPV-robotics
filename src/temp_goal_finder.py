#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class temp_goal_finder:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('temp_goal_finder', anonymous=True)

        # Publisher for the bounding box coordinates
        self.pub = rospy.Publisher('/goal_detector/bounding_box', Int32MultiArray, queue_size=10)

        # Subscriber to the image topic
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        # Bridge to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()

    def image_callback(self, data):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

            # Process the image to find the bounding box
            bbox = self.find_green_object(cv_image)

            if bbox:
                # Publish the bounding box coordinates
                bbox_msg = Int32MultiArray(data=bbox)
                self.pub.publish(bbox_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def find_green_object(self, image):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for green color
        lower_green = np.array([40, 40, 40])  # Adjust these values
        upper_green = np.array([80, 255, 255])  # Adjust these values

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            tr = 50
            if w >= tr or h >= tr:
                return [x, y, x + w, y + h]
        else:
            # No contours found
            return None

if __name__ == '__main__':
    try:
        detector = temp_goal_finder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
