#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from pynput import keyboard

class HeadController:
    def __init__(self):
        rospy.init_node("head_keyboard_controller", anonymous=True)
        self.pub_x = rospy.Publisher('/jC1Head_rotx_position_controller/command', Float64, queue_size=10)
        self.pub_y = rospy.Publisher('/jC1Head_roty_position_controller/command', Float64, queue_size=10)
        self.angle_x = 0.0
        self.angle_y = 0.0

    def on_press(self, key):
        try:
            if key == keyboard.Key.up: # move head up
                self.angle_y -= 0.1
            elif key == keyboard.Key.down: # move head down
                self.angle_y += 0.1
            elif key == keyboard.Key.left: # move head left
                self.angle_x += 0.1
            elif key == keyboard.Key.right: # move head right
                self.angle_x -= 0.1
            self.publish_angles()
        except Exception as e:
            rospy.logerr(e)

    def publish_angles(self):
        self.pub_x.publish(self.angle_x)
        self.pub_y.publish(self.angle_y)
        rospy.loginfo(f"Set angles: rotx={self.angle_x}, roty={self.angle_y}")

    def run(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

if __name__ == '__main__':
    try:
        controller = HeadController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
