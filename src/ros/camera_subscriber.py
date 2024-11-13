#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TurtleBotDetector:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('camera_subscriber', anonymous=True)
        
        # Create a subscriber to the camera image topic
        self.image_sub = rospy.Subscriber('/eye_camera/image_raw', Image, self.image_callback)
        
        # Initialize cv_bridge
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert the ROS Image message to a CV2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CvBridge error: {0}".format(e))
            return

        # Detect the TurtleBot using color (for example, detecting a green object)
        self.detect_turtlebot(cv_image)

    def detect_turtlebot(self, image):
        # Convert the image to HSV color space for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color range for TurtleBot detection (e.g., green)
        lower_color = np.array([40, 40, 40])
        upper_color = np.array([80, 255, 255])
        
        # Create a mask for green colors
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter small contours to reduce noise
            if cv2.contourArea(contour) > 500:
                # Draw bounding box around the detected TurtleBot
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rospy.loginfo("TurtleBot detected at position: x={}, y={}".format(x, y))

        # Display the processed image
        cv2.imshow("TurtleBot Detection", image)
        cv2.waitKey(1)  # Add a delay to allow image display updates

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = TurtleBotDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
