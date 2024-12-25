#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

def draw_bounding_boxes(image, box, color, label=None):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if label:
        cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_line(image, line, color):
    x1, y1, x2, y2 = line
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

class BoundingBoxVisualizer:
    def __init__(self):
        rospy.init_node('bounding_box_visualizer', anonymous=True)

        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        self.goal_sub = rospy.Subscriber('/fpv_controller/goal_bounding_box', Int32MultiArray, self.goal_callback)
        self.turtlebot_sub = rospy.Subscriber('/fpv_controller/turtlebot_bounding_box', Int32MultiArray, self.turtlebot_callback)
        self.line_sub = rospy.Subscriber('/fpv_controller/turtlebot_orientation_line', Int32MultiArray, self.line_callback)


        self.bridge = CvBridge()

        # Store bounding box data
        self.turtlebot = None
        self.goal = None

        # Store line data
        self.line = None

    def goal_callback(self, msg):
        self.goal = np.array(msg.data)

    def turtlebot_callback(self, msg):
        self.turtlebot = np.array(msg.data)

    def line_callback(self, msg):
        if len(msg.data) == 4:
            self.line = np.array(msg.data)


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        if self.turtlebot is not None:
            if len(self.turtlebot) == 4:
                draw_bounding_boxes(cv_image, self.turtlebot, (255, 0, 0), label="Turtlebot")
                self.turtlebot = None
            
        if self.goal is not None:
            if len(self.goal) == 4:
                draw_bounding_boxes(cv_image, self.goal, (0, 255, 0), label="Goal")
                self.goal = None

        if self.line is not None:
            draw_line(cv_image, self.line, (0, 0, 255))
            self.line = None
            

        cv2.imshow('Bounding Boxes', cv_image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        visualizer = BoundingBoxVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
