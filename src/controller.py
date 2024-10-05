#!/usr/bin/env python
import numpy as np
import rospy
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import JointState
from tf.broadcaster import TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Float64MultiArray  
from pynput.keyboard import Key, Listener

# Constants for wheel speeds
FORWARD_SPEED = 1.0*2.5
TURN_SPEED = 0.5*2.5
STOP_SPEED = 0.0

def on_press(key):
    """ Handle the press of a key to modify wheel velocities based on the command """
    move = Float64MultiArray()
    try:
        if key.char == 'w':  # Move forward
            rospy.set_param('left_wheel', FORWARD_SPEED)
            rospy.set_param('right_wheel', FORWARD_SPEED)
        elif key.char == 'a':  # Turn left
            rospy.set_param('left_wheel', STOP_SPEED)
            rospy.set_param('right_wheel', TURN_SPEED)
        elif key.char == 'd':  # Turn right
            rospy.set_param('left_wheel', TURN_SPEED)
            rospy.set_param('right_wheel', STOP_SPEED)
        elif key.char == 's':  # Stop
            rospy.set_param('left_wheel', STOP_SPEED)
            rospy.set_param('right_wheel', STOP_SPEED)
    except AttributeError:
        pass

def vel_move():
    """ Continuously move the robot based on the wheel parameters """
    pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(100)  # 100 Hz

    while not rospy.is_shutdown():
        wl = rospy.get_param('left_wheel')  # Get left wheel speed
        wr = rospy.get_param('right_wheel')  # Get right wheel speed
        move = Float64MultiArray()
        move.data = [wl, wr]  # Set wheel speeds
        pub.publish(move)  # Publish wheel speeds
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node("velocity_command")
    rospy.set_param('left_wheel', STOP_SPEED)  # Initialize wheel parameters
    rospy.set_param('right_wheel', STOP_SPEED)
    
    # Start the keyboard listener
    listener = Listener(on_press=on_press)
    listener.start()
    
    # Start the robot movement function
    vel_move()
    listener.join()
    rospy.spin()


    




