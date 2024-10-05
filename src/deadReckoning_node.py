#!/usr/bin/env python
import math
import rospy
import tf
import numpy as np
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import *
import threading
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped,Point
# from plotstate import *
# Defines a ROS node for dead reckoning based on wheel encoders and IMU data.
class DeadReckoningNode:
    def __init__(self):
        rospy.init_node('dead_reckoning_node', anonymous=True)  # Initialize ROS node
        
        # Initialize wheel velocities and flags to check if velocities are received
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_velocity_received = False
        self.right_wheel_velocity_received = False
        self.lock = threading.Lock()
    
        # Robot parameters: wheel radius and distance between wheels
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.235
        #Robot Dimension
        self.xB_dim=3
        #Feature Dimension
        self.xF_dim=2
        ###############  Initial robot state and its covariance matrix ###############
        self.xk= np.zeros((self.xB_dim,1))
        self.Pk=np.diag([0.1,0.2,0.1]).reshape(3,3)
        
        #############  Odometry noise covariance matrix ########################
        self.Qk = np.diag(np.array([0.2**2, 0.2**2, 0.01**2])) 
        
        # Tracks the last time a message was received
        self.last_time = rospy.Time.now()  
        
        ############# Publishers for odometry messages ##########
        self.odom_pub = rospy.Publisher("turtlebot/kobuki/odom_predict", Odometry, queue_size=10)
        self.path_pub = rospy.Publisher("turtlebot/kobuki/path_update", Path, queue_size=10)
        self.path_marker_pub = rospy.Publisher("turtlebot/kobuki/path_marker_update", Marker, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "world_ned"
        self.marker_pub_odom = rospy.Publisher('visualization_marker/odom', Marker, queue_size=10)

      
        ########## Subscribe to joint states ##########
        self.js_sub=rospy.Subscriber("turtlebot/joint_states", JointState, self.joint_states_callback, queue_size=10)
        
        
        #TF broadcaster to publish transformations between coordinate frames
        self.odom_broadcaster = tf.TransformBroadcaster()
        #     # Instantiate the StateEstimator
        # self.state_estimator = StateEstimator(kSteps=100)   # Adjust kSteps as needed
        
        
    def wrap_angle(self,angle):
        
        "Normalizes an angle to be within the range of [-pi, pi]"
        #corrects an angle to be within the range of [-pi, pi]
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

    def joint_states_callback(self, msg):
        """
        Callback function that processes messages from the joint states topic.
        
        :param msg: The message received from the joint states topic, containing the names of joints and their velocities.
        """
       
        # Define the names for the left and right wheel joints for easier identification.
        self.left_wheel_name = 'turtlebot/kobuki/wheel_left_joint'
        self.right_wheel_name = 'turtlebot/kobuki/wheel_right_joint'
        
        # Check the first joint name in the message and assign the corresponding velocity.
        if msg.name[0] == self.left_wheel_name:
            self.left_wheel_velocity = msg.velocity[0]
            self.left_wheel_velocity_received = True  # Mark the left wheel velocity as received.
            
        elif msg.name[0] == self.right_wheel_name:
            self.right_wheel_velocity = msg.velocity[0]
            
        # Proceed only if both wheel velocities have been received.
        if self.left_wheel_velocity_received:  
            # Compute the linear velocity for each wheel by multiplying the angular velocity by the wheel radius.
            self.left_linear_velocity = self.left_wheel_velocity * self.wheel_radius
            self.right_linear_velocity = self.right_wheel_velocity * self.wheel_radius
            
            # Calculate the overall linear and angular velocities.
            self.linear_velocity = (self.left_linear_velocity + self.right_linear_velocity) / 2
            self.angular_velocity = (self.left_linear_velocity - self.right_linear_velocity) / self.wheel_base_distance
            
            # Compute the current time from the message stamp and calculate the time elapsed since the last update.
            self.current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            self.time = (self.current_time - self.last_time).to_sec()
            self.last_time = self.current_time
            
            # Update the robot's pose using the motion model.
            self.xk[0,0] = self.xk[0,0]+np.cos(self.xk[2,0]) * self.linear_velocity * self.time
            self.xk[1,0] =self.xk[1,0]+ np.sin(self.xk[2,0]) * self.linear_velocity * self.time
            
            # Normalize the robot's orientation angle.
            self.xk[2,0] = self.wrap_angle(self.xk[2,0] + self.angular_velocity * self.time)
            
            # operation modifying self.xk
            self.xk, self.Pk = self.prediction(self.xk, self.Pk, self.linear_velocity, self.angular_velocity, self.time)
            
            #Update the path.
            self.update_path() 
            
            # Reset the velocity reception flags for the next iteration.
            self.left_wheel_velocity_received = False
            
        
            
              
    #####################################################  Prediction Step  ########################################################
    def prediction(self, xk, Pk, v, w, t):
        """
        Predicts the next state of the robot using the motion model.

        :param xk: Current state vector of the robot.
        :param Pk: Current state covariance matrix.
        :param v: Linear velocity of the robot.
        :param w: Angular velocity of the robot.
        :param t: Time interval since the last update.
        :return: Updated state vector and covariance matrix after prediction.
        """
        
        #Acquire the lock to ensure thread safety during the prediction update.
     
        # Extract the base state from the state vector.
        xk_robot = self.xk[:self.xB_dim]
        # Calculate the Jacobian of the motion model with respect to the robot's state.
        """ Ak is the Jacobian matrix of the motion model with respect to the robot state"""
        Ak = np.array([
            [1.0, 0.0, -np.sin(self.xk[2,0]) * v * t],
            [0.0, 1.0,  np.cos(self.xk[2,0]) * v * t],
            [0.0, 0.0, 1.0]
        ])

        #Calculate the Jacobian of the motion model with respect to the process noise.
   
        """ Wk is essentially the Jacobian matrix of the motion model with respect to the wheel velocities """
        
        ### linear velocity,v=(left linear velocity+right linear velocity)/2 =wheelradius*(left wheel velocity+right wheel velocity)
        ## [[diff(x/vl) diff(x/vr) , 0.0],     #t*r*cos(theta) #t*r*cos(theta)
        # [diff(y/vl),  diff(y/vr),0.0],             #t*r*sin(theta) #t*r*sin(theta)
        # [diff(theta)/vl, diff(theta/vr),1.0  ]]    #t*r/wheel_base_distance  #-t*r/wheel_base_distance
        
        # # Calculate the process noise covariance matrix.
        Wk = np.array([[np.cos(self.xk[2,0])*t*0.5*self.wheel_radius, np.cos(self.xk[2,0])*t*0.5*self.wheel_radius,0.0],    
                            [np.sin(self.xk[2,0])*t*0.5*self.wheel_radius, np.sin(self.xk[2,0])*t*0.5*self.wheel_radius,0.0],   
                                
                            [(t*self.wheel_radius)/self.wheel_base_distance, -(t*self.wheel_radius)/self.wheel_base_distance,1.0]])     
     
        # Retrieve the process noise covariance matrix.
        Qk = self.Qk # Process noise covariance matrix accounting for uncertainties in the motion model.
        
       
        # Check if there are additional state components beyond the robot's base state.
        if len(self.xk) > self.xB_dim:
            # If additional states exist, extract and concatenate them to the robot's base state.
            xk_extend = self.xk[self.xB_dim:]
            self.xk = np.concatenate((xk_robot, xk_extend))
        
        else:
            # If no additional states, use the base state directly.
            self.xk = xk_robot
            
        
        # Prepare the extended Jacobians and covariance matrices to include additional states.
        Fk1 = np.eye(len(self.xk))
        Fk2 = np.zeros((len(self.xk), len(Qk)))

        # Assign the Jacobian blocks to the extended matrices.
        Fk1[:self.xB_dim, :self.xB_dim] = Ak
        Fk2[:self.xB_dim, :len(Qk)] = Wk

        # Update the state covariance matrix.
        self.Pk = Fk1 @ self.Pk @ Fk1.T + Fk2 @ Qk @ Fk2.T
        
        # Publishing the predicted odometry 
        self.publish_odometry(self.xk,self.Pk)
        
        
        return self.xk, self.Pk


    #################### Publish Odometry #####################
 
    def publish_odometry(self,xk,Pk):
        """
        Publishes odometry data to ROS.

        :param xk: Current state vector [x, y, theta].
        :param Pk: Covariance matrix of the state.
        :param linear_velocity: Linear velocity of the robot.
        :param angular_velocity: Angular velocity of the robot.
        :param current_time: Current ROS time, used as the timestamp for the odometry message.
        """
        
        # Convert the yaw angle to a quaternion for representing 3D orientations.
        self.q = quaternion_from_euler(0, 0, xk[2,0])

        # Initialize an odometry message.
        odom = Odometry()
        odom.header.stamp =self.current_time
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"
       
        # Set the position in the odometry message.
        odom.pose.pose.position.x = xk[0,0]
        odom.pose.pose.position.y = xk[1,0]
       
        # Set the orientation in the odometry message.
        odom.pose.pose.orientation.x = self.q[0]
        odom.pose.pose.orientation.y = self.q[1]
        odom.pose.pose.orientation.z = self.q[2]
        odom.pose.pose.orientation.w = self.q[3]

        # Set the velocities in the odometry message.
        odom.twist.twist.linear.x = self.linear_velocity
        odom.twist.twist.angular.z = self.angular_velocity

        
        # Convert covariance matrix from np.array to list
        P_list = Pk.tolist()  
        # Update the diagonal elements directly for variance
        odom.pose.covariance[0] = P_list[0][0]  # Variance in x
        odom.pose.covariance[7] = P_list[1][1]  # Variance in y
        odom.pose.covariance[35] = P_list[2][2]  # Variance in yaw

        # Update the off-diagonal elements for covariance between variables
        odom.pose.covariance[1] = P_list[0][1]  # Covariance between x and y
        odom.pose.covariance[6] = P_list[1][0]  # Covariance between y and x 

        odom.pose.covariance[5] = P_list[0][2]  # Covariance between x and yaw
        odom.pose.covariance[30] = P_list[2][0]  # Covariance between yaw and x

        odom.pose.covariance[11] = P_list[1][2]  # Covariance between y and yaw
        odom.pose.covariance[31] = P_list[2][1]  # Covariance between yaw and y
         
        ############## Marker visulaization for Covariance of robot state  #######3###
        uncertainity=self.Pk[0:2,0:2]
        # print('uncertainity',uncertainity)
        eigenvalues, eigenvectors=np.linalg.eigh(uncertainity)
        yaw_un=np.arctan2(eigenvectors[1,0],eigenvectors[0,0])  # Rotation about z axis
        q_e=quaternion_from_euler(0,0,yaw_un)
        
        odom_uncertainity=Marker()
        odom_uncertainity.header.frame_id="world_ned"
        odom_uncertainity.header.stamp=self.current_time
        odom_uncertainity.ns="odom_uncertainity"
        odom_uncertainity.type=Marker.CYLINDER
        odom_uncertainity.action=Marker.ADD
        odom_uncertainity.pose.position.x=xk[0,0]
        odom_uncertainity.pose.position.y=xk[1,0]
        odom_uncertainity.pose.orientation.x=q_e[0]
        odom_uncertainity.pose.orientation.y=q_e[1]
        odom_uncertainity.pose.orientation.z=q_e[2]
        odom_uncertainity.pose.orientation.w=q_e[3]
        # scale.x and scale.y are set to twice the square root of the eigenvalues, representing the standard deviation of the robot's position uncertainty in x and y. 
        odom_uncertainity.scale.x=2*math.sqrt(eigenvalues[0]) #Cylinder dimensions
        odom_uncertainity.scale.y=2*math.sqrt(eigenvalues[1])
        
        odom_uncertainity.scale.z=0.02
        odom_uncertainity.color=ColorRGBA(0.0,1.0,0.7,1.0)
        odom_uncertainity.lifetime=rospy.Duration(0.1)
        self.marker_pub_odom.publish(odom_uncertainity)
        self.odom_pub.publish(odom)
        
        # Publish the transform over tf (transformation frames in ROS).
        self.odom_broadcaster.sendTransform((xk[0,0], xk[1,0], 0.0), self.q, rospy.Time.now(), odom.child_frame_id, odom.header.frame_id)
        
    ################################3 Update Path #########################################
    def update_path(self):
        """ Update the path with the current position of the robot"""
        
        quaternion = quaternion_from_euler(0, 0, self.xk[2, 0])
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world_ned"
        pose.pose.position.x = self.xk[0, 0]
        pose.pose.position.y = self.xk[1, 0]
        pose.pose.orientation = Quaternion(*quaternion)
        self.path.poses.append(pose)
        self.path.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path)

        point = Point(pose.pose.position.x, pose.pose.position.y,0.0)
        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        marker.points.append(point)
        self.path_marker_pub.publish(marker)
        
if __name__ == '__main__':
    try:
        DeadReckoningNode()
        # Keep the program running until rospy is shut down
        rospy.spin()
    except rospy.ROSInterruptException:
        pass