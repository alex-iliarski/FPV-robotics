#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from ultralytics import YOLO
import time
import mediapipe as mp
import rospkg
import os

class GoalFinder:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('goal_finder', anonymous=True)

        # Load goal detection YOLOv8 model
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('fpv_robotics')
        model_path = os.path.join(package_path, "src", "models", "goal_yolov8n.pt")
        self.model = YOLO(model_path)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

        # Variables for selected object tracking
        self.selected_label = None

        # Stability tracking variables
        self.current_label = None

        self.selection_start_time = None
        self.selection_duration = 2.0   # Duration (seconds) to confirm selection

        # Publisher for the bounding box coordinates
        self.pub = rospy.Publisher('/fpv_controller/goal_bounding_box', Int32MultiArray, queue_size=10)

        # Subscriber to the image topic
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        # Bridge to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        rospy.sleep(4)

    def image_callback(self, data):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            h, w, _ = cv_image.shape

            # Select the goal object to track
            if self.selected_label is None:
                detections = self.detect_goal_objects(cv_image)

                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_image)

                target_point = None
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        fingertip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        target_point = np.array([int(fingertip.x * w), int(fingertip.y * h)])

                # if we have a target point, find the closest object to it
                if target_point is not None:
                    closest_object = self.find_closest_object(target_point, detections)

                    # if a closest object is found
                    if closest_object is not None:
                        bx, by, bw, bh, label = closest_object

                        if self.current_label == label:
                            # start the selection timer
                            if self.selection_start_time is None:
                                self.selection_start_time = time.time()
                            # finalize the selection if the timer has elapsed
                            elif time.time() - self.selection_start_time >= self.selection_duration:
                                    self.selected_label = label
                                    self.selection_start_time = None

                        # reset the selection timer if the current label changes
                        else:
                            self.current_label = label
                            self.selection_start_time = None

            # goal is selected, publish the bounding box
            else:
                detections = self.detect_goal_objects(cv_image)
                found = False

                for (x, y, w, h, label) in detections:
                    if label == self.selected_label:
                        found = True
                        self.pub.publish(Int32MultiArray(data=[x, y, x + w, y + h]))
                        break

                if not found:
                    self.pub.publish(Int32MultiArray(data=[]))


        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def detect_goal_objects(self, image):
        excluded_labels = ["water bottle"]
        results = self.model.predict(source=image, save=False, save_txt=False, imgsz=640, verbose=False)
        bounding_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results[0].names[int(box.cls)]
            if label not in excluded_labels:
                bounding_boxes.append([x1, y1, x2 - x1, y2 - y1, label])
        return bounding_boxes
    
    def find_closest_object(self, target_point, detections):
        closest_object = None
        min_distance = float('inf')

        for (bx, by, bw, bh, label) in detections:
            center_x = bx + bw // 2
            center_y = by + bh // 2
            distance = np.linalg.norm([target_point[0] - center_x, target_point[1] - center_y])
            if distance < min_distance:
                min_distance = distance
                closest_object = [bx, by, bw, bh, label]

        return closest_object
    
    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = GoalFinder()
        detector.run()
    except rospy.ROSInterruptException:
        pass