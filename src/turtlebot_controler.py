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
        # Initialize the ROS node
        rospy.init_node('turtlebot_controler', anonymous=True)

        # Subscribe to the line topic
        self.line_sub = rospy.Subscriber("/line_detector/line", Int32MultiArray, self.line_callback)
        self.goal_sub = rospy.Subscriber("/goal_detector/bounding_box", Int32MultiArray, self.goal_callback)
        self.img_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)

        self.img = None
        self.bridge = CvBridge()

        self.goad_found = False
        self.goal_center = None

        self.prev_distance = None
        self.prev_move = None


        # Publish the control commands
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Initialize the control commands
        self.cmd = Twist()

    def goal_callback(self, msg):
        if len(msg.data) == 4:
            x1, y1, x2, y2 = msg.data
            self.goal_center = (x1 + x2) / 2, (y1 + y2) / 2
            self.goal_found = True
        else:
            self.goal_found = False
            self.goal_center = None

    def image_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return


    def line_callback(self, msg):
        # Check if the message is a valid line
        if len(msg.data) != 4:
            return
        # Get the line coordinates
        x1, y1, x2, y2 = msg.data

        if self.goal_found:
            xg, yg = self.goal_center

            # Calculate the slope of the detected line
            if x2 != x1:
                m_line = (y2 - y1) / (x2 - x1)
            else:
                m_line = float('inf')  # Vertical line

            # Calculate the slope of the line from the goal to the line's first point
            if xg != x1:
                m_goal = (yg - y1) / (xg - x1)
            else:
                m_goal = float('inf')  # Vertical line to goal

            # Calculate the angle between the two lines
            if m_line != float('inf') and m_goal != float('inf'):
                tan_theta = abs((m_goal - m_line) / (1 + m_goal * m_line))
                angle = math.atan(tan_theta)  # Angle in radians
            else:
                angle = math.pi / 2  # 90 degrees for a vertical line

            # Calculate the distance to the goal point
            distance = math.sqrt((xg - x1) ** 2 + (yg - y1) ** 2)

            # Control logic based on angle and distance
            angular_threshold = 0.1  # Radians
            distance_threshold = 10  # Pixels

            if angle > angular_threshold:
                # Rotate toward the goal
                if m_line > m_goal:  # Goal is to the left of the line
                    self.cmd.angular.z = 0.5 * angle
                else:  # Goal is to the right of the line
                    self.cmd.angular.z = -0.5 * angle
                self.cmd.linear.x = 0.0  # Stop linear motion while rotating
            elif distance > distance_threshold:
                # Move toward the goal
                self.cmd.linear.x = 0.1  # Forward velocity
                self.cmd.angular.z = 0.0  # Stop rotation when aligned
            else:
                # Stop when close enough to the goal
                self.cmd.linear.x = 0.0
                self.cmd.angular.z = 0.0

            # Log information
            rospy.loginfo(f"Angle: {math.degrees(angle):.2f} degrees, Distance: {distance:.2f}")
            rospy.loginfo(f"Control Command: Linear X: {self.cmd.linear.x}, Angular Z: {self.cmd.angular.z}")

            # Publish the control command
            self.cmd_pub.publish(self.cmd)

            # Visualization (Optional)
            if self.img is not None:
                cv2.line(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.circle(self.img, (int(xg), int(yg)), 5, (0, 255, 0), -1)
                cv2.putText(self.img, f"Angle: {math.degrees(angle):.2f} deg, Dist: {distance:.2f}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Line", self.img)
                cv2.waitKey(1)


    # def line_callback(self, msg):
    #     # Check if the message is a valid line
    #     if len(msg.data) != 4:
    #         return
    #     # Get the line coordinates
    #     x1, y1, x2, y2 = msg.data

    #     # calculate the intersect of the line with the goal
    #     if self.goal_found:
    #         xg, yg = self.goal_center
    #         m = (y2 - y1) / (x2 - x1)
    #         b = y1 - m * x1
    #         if m == 0:
    #             xg_line = xg
    #         else:
    #             xg_line = (yg - b) / m
    #         # Draw the line
    #         cv2.line(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    #         cv2.circle(self.img, (int(xg_line), int(yg)), 5, (0, 0, 255), -1)
    #         cv2.imshow("Line", self.img)
    #         cv2.waitKey(1)

    #         # Calculate the error
    #         error = xg_line - xg
    #         ## calculate distance from x1, y1 to xg, yg
    #         distance = ((xg - x1) ** 2 + (yg - y1) ** 2) ** 0.5

    #         # Calculate the control command
    #         self.cmd.angular.z = -0.001 * error

    #         rospy.loginfo(f"Error: {error}, Distance: {distance}")
    #         rospy.loginfo(f"Control Command: {self.cmd.angular.z}")

    #         # if error is small enough, move forward
    #         if abs(error) == 0:
    #             # if the previous move was forward and the distance is smaller than the previous distance keep moving forward
    #             if self.prev_move == "forward" and distance < self.prev_distance:
    #                 rospy.loginfo("Moving forward")
    #                 self.cmd.linear.x = 0.1
    #                 self.prev_move = "forward"
    #                 self.prev_distance = distance
    #             # if the previous move was backward and the distance is smaller than the previous distance move backward
    #             elif self.prev_move == "backward" and distance < self.prev_distance:
    #                 rospy.loginfo("Moving backward")
    #                 self.cmd.linear.x = -0.1
    #                 self.prev_move = "backward"
    #                 self.prev_distance = distance
    #             # if the previous move was forward and the distance is bigger than the previous distance move backward
    #             elif self.prev_move == "forward" and distance >= self.prev_distance:
    #                 rospy.loginfo("Moving backward")
    #                 self.cmd.linear.x = -0.1
    #                 self.prev_move = "backward"
    #                 self.prev_distance = distance
    #             # if the previous move was backward and the distance is bigger than the previous distance move forward
    #             elif self.prev_move == "backward" and distance >= self.prev_distance:
    #                 rospy.loginfo("Moving forward")
    #                 self.cmd.linear.x = 0.1
    #                 self.prev_move = "forward"
    #                 self.prev_distance = distance
    #             # if the previous move was None, move forward
    #             else:
    #                 rospy.loginfo("Moving forward")
    #                 self.cmd.linear.x = 0.1
    #                 self.prev_move = "forward"
    #                 self.prev_distance = distance
                
    #         # Publish the control command
    #         self.cmd_pub.publish(self.cmd)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        controler = TurtlebotControler()
        controler.run()
    except rospy.ROSInterruptException:
        pass

