#!/usr/bin/env python
from pynput.keyboard import Listener, Key  # Importing Listener and Key classes from pynput.keyboard for keyboard events
import rospy  # Import rospy for ROS Python utilities
from std_msgs.msg import Float64MultiArray  # ROS standard message type for publishing wheel velocities
from geometry_msgs.msg import Twist  # ROS message type for receiving velocity commands

# Set robot physical parameters
wheel_radius = 0.035  # Wheel radius in meters
wheel_base_distance = 0.235  # Distance between the wheels in meters

# Initialize wheel velocities to zero
left_wheel_vel = 0.0  # Initial left wheel velocity
right_wheel_vel = 0.0  # Initial right wheel velocity

# Initialize velocities from keyboard or ROS commands
linear_vel = 0.0  # Linear velocity
angular_vel = 0.0  # Angular velocity

# Dictionary to track the state of specific keys
keys_pressed = {'w': False, 's': False, 'a': False, 'd': False, 'x': False}  # Tracks which keys are pressed

def update_velocities_from_keyboard():
    """Adjusts linear and angular velocities based on keyboard inputs."""
    
    global linear_vel, angular_vel
    incremental_linear = 0.0  # Incremental change to linear velocity
    incremental_angular = 0.0  # Incremental change to angular velocity

    # Increase/decrease linear velocity based on 'w' and 's' keys
    if keys_pressed['w']:
        incremental_linear += 0.01
    if keys_pressed['s']:
        incremental_linear -= 0.0001

    # Increase/decrease angular velocity based on 'a' and 'd' keys
    if keys_pressed['a']:
        incremental_angular += 0.01
    if keys_pressed['d']:
        incremental_angular -= 0.01
    
    # Apply incremental changes to global velocities
    linear_vel += incremental_linear
    angular_vel += incremental_angular

    # Reset velocities if 'x' is pressed
    if keys_pressed['x']:
        linear_vel = 0.0
        angular_vel = 0.0

    # Update wheel velocities based on updated linear and angular velocities
    update_wheel_velocities()

def update_wheel_velocities():
    """Calculates wheel velocities using differential drive kinematics."""
    
    global left_wheel_vel, right_wheel_vel
    # Calculate left and right wheel velocities from linear and angular velocities
    left_wheel_vel = (linear_vel - angular_vel * wheel_base_distance / 2) / wheel_radius
    right_wheel_vel = (linear_vel + angular_vel * wheel_base_distance / 2) / wheel_radius

def on_press(key):
    """Callback for key press event to update velocity based on pressed key."""
    try:
        # Set the corresponding key's state to True if pressed
        if key.char in keys_pressed:
            keys_pressed[key.char] = True
            update_velocities_from_keyboard()
    except AttributeError:
        pass  # Exception handling for special keys that do not have a char attribute

def on_release(key):
    """Callback for key release event to update velocity based on released key."""
    try:
        # Set the corresponding key's state to False when released
        if key.char in keys_pressed:
            keys_pressed[key.char] = False
            update_velocities_from_keyboard()
    except AttributeError:
        pass  # Exception handling for special keys that do not have a char attribute
    if key == Key.esc:
        return False  # Stop listener if escape key is pressed

def cmdvelCallback(msg):
    """ROS callback that updates robot's linear and angular velocities from cmd_vel messages."""
    
    global linear_vel, angular_vel
    linear_vel = msg.linear.x  # Update global linear velocity from ROS message
    angular_vel = msg.angular.z  # Update global angular velocity from ROS message
    update_wheel_velocities()  # Recalculate wheel velocities

def wheelControl():
    """Main loop for publishing wheel velocities to a ROS topic."""
    
    wheel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=10)  # ROS publisher for wheel velocities
    rate = rospy.Rate(50)  # Set loop rate to 50 Hz

    while not rospy.is_shutdown():
        wheel_msg = Float64MultiArray()  # Create a Float64MultiArray message for the velocities
        wheel_msg.data = [left_wheel_vel, right_wheel_vel]  # Set message data to current wheel velocities
        wheel_pub.publish(wheel_msg)  # Publish the wheel velocities
        rate.sleep()  # Sleep to maintain loop rate

if __name__ == '__main__':
    
    rospy.init_node('wheel_control_node', anonymous=True)  # Initialize the ROS node
    rospy.Subscriber('/turtlebot/kobuki/cmd_vel', Twist, cmdvelCallback, queue_size=10)  # Subscribe to cmd_vel for velocity commands
    listener = Listener(on_press=on_press, on_release=on_release)  # Start listener for keyboard events
    listener.start()  # Start the listener
    wheelControl()  # Start the main control loop
    listener.join()  # Wait for the listener thread to finish
    rospy.spin()  # Keep the node running until shutdown
