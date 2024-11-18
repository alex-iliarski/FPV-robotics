#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class YoloROSNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolo_detector', anonymous=True)

        # Set up image subscriber and CvBridge
        self.image_sub = rospy.Subscriber('/eye_camera/image_raw', Image, self.image_callback)

        # Optional: Publish annotated images
        self.image_pub = rospy.Publisher('/yolo/detections_image', Image, queue_size=1)
        self.bridge = CvBridge()

        # Load YOLOv8 model
        rospy.loginfo("Loading YOLOv8 model")
        self.model = YOLO(r'/home/isaac/catkin_ws/src/FPV-robotics/src/ros/turtlebot3_yolov8n.pt')

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
        rospy.loginfo("Received image from camera")
        results = self.model.predict(source=image, save=False, save_txt=False, imgsz=640)
        rospy.loginfo("YOLO inference completed")

        # Draw detections on the image
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[int(cls)]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                rospy.loginfo(f"Detected {class_name} with confidence {score:.2f}")

        # Publish the annotated image
        annotated_image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")

        rospy.loginfo("Publishing detections to /yolo/detections_image")
        self.image_pub.publish(annotated_image_msg)

        cv2.imshow("YOLO Detection", image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = YoloROSNode()
        detector.run()
    except rospy.ROSInterruptException:
        pass
