#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

class TurtlebotDetector:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('turtlebot_detector', anonymous=True)

        # Set up image subscriber and CvBridge
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        # Publish annotated images and bounding boxes
        self.image_pub = rospy.Publisher('/turtlebot_detector/detections_image', Image, queue_size=10)
        self.bbox_pub = rospy.Publisher("/turtlebot_detector/bounding_box", Float32MultiArray, queue_size=10)

        # CvBridge for converting ROS Image messages to OpenCV format
        self.bridge = CvBridge()

        self.display_visualization = rospy.get_param("turtlebot_visualization", False)

        # Load YOLOv8 model
        rospy.loginfo("Loading YOLOv8 model")
        self.model = YOLO(r'/home/isaac/catkin_ws/src/fpv_robotics/src/ros/ turtlebot3_yolov8n.pt')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        self.detect_turtlebot(cv_image)

    def detect_turtlebot(self, image):
        # Run YOLO inference
        try:
            results = self.model.predict(source=image, save=False, save_txt=False, imgsz=640, verbose=False)
        except Exception as e:
            rospy.logerr(f"YOLO Error: {e}")
            return

        best_box = None
        best_score = 0.5 # Minimum confidence threshold
        best_class = None

    # Iterate over results to find the highest confidence detection
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, score, cls in zip(boxes, scores, classes):
                if score > best_score:  # Update if this detection is the highest confidence so far
                    best_score = score
                    best_box = box
                    best_class = cls

        # Draw the highest confidence detection, if found
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            class_name = self.model.names[int(best_class)]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} {best_score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # rospy.loginfo(f"Best detection: {class_name} with confidence {best_score:.2f}")

        # Publish the annotated image
        annotated_image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.image_pub.publish(annotated_image_msg)

        # Publish the bounding box coordinates and confidence score if a detection was found
        if best_box is not None:
            bbox = Float32MultiArray()
            bbox.data = [x1, y1, x2, y2]
            self.bbox_pub.publish(bbox)

        # Display the annotated image if enabled
        if self.display_visualization:
            cv2.imshow("YOLO Detection", image)
            cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = TurtlebotDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
