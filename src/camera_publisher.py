#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def publish_camera():
    rospy.init_node('camera_publisher', anonymous=True)
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    rate = rospy.Rate(30)  # Publish at 30 Hz
    bridge = CvBridge()
    cap = cv2.VideoCapture(0)  # Open the camera (device 2 for usb camera)

    if not cap.isOpened():
        rospy.logerr("Unable to open camera")
        return

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to capture frame")
            break

        # Convert the frame to a ROS Image message
        image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(image_msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        publish_camera()
    except rospy.ROSInterruptException:
        pass
