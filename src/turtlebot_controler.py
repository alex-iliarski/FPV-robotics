#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math

class TurtlebotControler:
    def __init__(self):
        rospy.init_node('turtlebot_controler', anonymous=True)

        # Subscribe to Camera feed
        self.img_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)

        # Subscribe to the line, goal, and turtlebot bounding box topics
        self.line_sub = rospy.Subscriber("/fpv_controller/turtlebot_orientation_line", Int32MultiArray, self.line_callback)
        self.goal_sub = rospy.Subscriber("/fpv_controller/goal_bounding_box", Int32MultiArray, self.goal_callback)
        self.turtlebot_sub = rospy.Subscriber("/fpv_controller/turtlebot_bounding_box", Int32MultiArray, self.turtlebot_callback)

        # Initialize variables for Camera feed
        self.img = None
        self.bridge = CvBridge()

        # Initialize variables for goal
        self.goal_found = False
        self.goal_center = None

        # Initialize variables for turtlebot
        self.turtlebot_center = None

        # Initialize variables for control
        self.forward_direction = None  # Calibrated forward direction

        # Initialize variables for calibration
        self.prev_distance = None
        self.is_calibrating = False

        # Publisher for control commands
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.cmd = Twist()

        # Parameters
        self.angular_threshold = rospy.get_param("~angular_threshold", 0.1)  # Default 0.05 radians
        self.distance_threshold = rospy.get_param("~distance_threshold", 10)  # Default 10 pixels

        # Rate
        self.rate = rospy.Rate(10)  # 10 Hz

    def goal_callback(self, msg):
        # Check if the message is a valid bounding box
        if len(msg.data) == 4:
            x1, y1, x2, y2 = msg.data
            self.goal_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.goal_found = True
        else:
            self.goal_found = False
            self.goal_center = None

    def turtlebot_callback(self, msg):
        # Check if the message is a valid bounding box
        if len(msg.data) == 4:
            x1, y1, x2, y2 = msg.data
            self.turtlebot_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")


    def calibrate_forward_direction(self):
        # Set flag to indicate calibration
        self.is_calibrating = True

        # Measure initial distance
        xg, yg = self.goal_center
        xt, yt = self.turtlebot_center
        initial_distance = math.sqrt((xg - xt) ** 2 + (yg - yt) ** 2)

        # Move forward slightly
        rospy.loginfo("Calibrating: Moving forward")
        self.cmd.linear.x = 0.5  # Move forward at 0.1 m/s
        self.cmd.angular.z = 0.0
        self.cmd_pub.publish(self.cmd)
        rospy.sleep(1.5) 
        self.cmd.linear.x = 0.0  # stop
        self.cmd.angular.z = 0.0
        self.cmd_pub.publish(self.cmd)
        rospy.sleep(1.0)  # Wait for 1 second

        # Measure new distance
        xg, yg = self.goal_center
        xt, yt = self.turtlebot_center
        new_distance = math.sqrt((xg - xt) ** 2 + (yg - yt) ** 2)

        # Determine direction
        if new_distance < initial_distance:
            self.forward_direction = "forward"
            rospy.loginfo("Calibration complete: Forward direction set as 'forward'")
        else:
            self.forward_direction = "backward"
            rospy.loginfo("Calibration complete: Forward direction set as 'backward'")

        # Stop the robot after calibration
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.cmd_pub.publish(self.cmd)

        self.is_calibrating = False

    def line_callback(self, msg):
        # Line detection callback (main control logic)
        # Check if the message is a valid line
        if len(msg.data) != 4 or not self.goal_found:
            return
        if self.is_calibrating:
            return

        # line coordinates
        x1, y1, x2, y2 = msg.data

        # goal and turtlebot center coordinates
        xg, yg = self.goal_center
        xt, yt = self.turtlebot_center

        # Calculate the slope of the detected line and the goal line
        m_line = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        m_goal = (yg - y1) / (xg - x1) if xg != x1 else float('inf')

        # Calculate the angle between the lines
        if m_line != float('inf') and m_goal != float('inf'):
            tan_theta = abs((m_goal - m_line) / (1 + m_goal * m_line))
            angle = math.atan(tan_theta)  # Angle in radians
        else:
            angle = math.pi / 2  # 90 degrees for vertical lines

        # Calculate the distance between the turtlebot and the goal
        distance = math.sqrt((xg - xt) ** 2 + (yg - yt) ** 2)

        # Control logic based on angle and distance
        if angle > self.angular_threshold:
            # Rotate toward the goal
            self.cmd.angular.z = 0.5 * angle if m_line > m_goal else -0.5 * angle
            self.cmd.linear.x = 0.0  # Stop linear motion while rotating
        elif distance > self.distance_threshold:
            # Calibrate forward direction if not already done
            if self.forward_direction is None:
                rospy.loginfo("Starting calibration sequence")
                self.calibrate_forward_direction()
                return
            
            # Move forward or backward based on calibrated direction
            self.cmd.linear.x = 0.2 if self.forward_direction == "forward" else -0.2
            self.cmd.angular.z = 0.0
        else:
            # Stop when close enough to the goal
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
            rospy.loginfo("Goal reached!")

        # Publish the control command
        self.cmd_pub.publish(self.cmd)

        # Visualization
        if self.img is not None:
            cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(self.img, (int(xg), int(yg)), 5, (0, 255, 0), -1)
            cv2.circle(self.img, (int(xt), int(yt)), 5, (255, 0, 0), -1)
            cv2.putText(self.img, f"Angle: {math.degrees(angle):.2f} deg, Distance: {distance:.2f}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Line and Goal", self.img)
            cv2.waitKey(1)

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == "__main__":
    try:
        controler = TurtlebotControler()
        controler.run()
    except rospy.ROSInterruptException:
        pass
