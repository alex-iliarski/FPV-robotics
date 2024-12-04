#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class TurtleBotDetector:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('camera_subscriber', anonymous=True)
        
        # Create a subscriber to the camera image topic
        self.image_sub = rospy.Subscriber('/eye_camera/image_raw', Image, self.image_callback)
        
        # Initialize cv_bridge
        self.bridge = CvBridge()
        
        # Load the YOLOv8 model
        #self.model = YOLO('yolov8_model.pt')  # Replace with the path to your YOLOv8 model file
        #self.model = YOLO(r'~/hw2_catkin_ws/src/FPV-robotics/src/ros/turtlebot3_yolov8n.pt')
        self.model = YOLO(r'/home/vboxuser/hw2_catkin_ws/src/FPV-robotics/src/ros/turtlebot3_yolov8n.pt')


    def image_callback(self, msg):
        # Convert the ROS Image message to a CV2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CvBridge error: {0}".format(e))
            return

        # Detect the TurtleBot using YOLOv8
        self.detect_turtlebot(cv_image)

    def detect_turtlebot(self, image):
        # Run the YOLOv8 model on the input image
        results = self.model.predict(image, verbose=False)
        
        # Parse the results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Extract confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Extract class IDs
            
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if confidence > 0.5:  # Filter detections with confidence > 0.5
                    # Draw the bounding box on the image
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID {int(class_id)}: {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    rospy.loginfo(f"Detected ID {int(class_id)} at x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={confidence:.2f}")

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
