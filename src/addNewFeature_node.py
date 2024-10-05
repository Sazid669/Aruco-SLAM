#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, ColorRGBA
from deadReckoning_node import DeadReckoningNode
from nav_msgs.msg import Odometry, Path
import threading
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from math import *
from ellipse import *
from marker import *
import matplotlib.pyplot as plt

# Defines a ROS node for adding new features and updating the robot state.
class AddNewFeatureNode(DeadReckoningNode):
    def __init__(self):
        # Initialize the base class (DeadReckoningNode) to set up dead reckoning functionalities.
        super(AddNewFeatureNode, self).__init__()
    
        # Subscribe to  ArUco marker position throttle data, triggering `aruco_callback` upon receiving data.
        """ Throttle the Aruco position, which observe each aruco after every 2 seconds and update it accordingly.. """
        self.aruco_sub = rospy.Subscriber("/turtlebot/kobuki/aruco_position_throttle", Float64MultiArray, self.aruco_callback, queue_size=10)
        #Subscribe to IMU data topics.
        self.imu_sub= rospy.Subscriber("turtlebot/kobuki/sensors/imu", Imu, self.imu_callback, queue_size=10)
        
        # Publishers for visualizing ArUco markers.
        self.markers_pub_aruco_marker = rospy.Publisher("aruco/visualization_aruco", MarkerArray, queue_size=10)
    
        # Lock for thread-safe operations.
        self.lock = threading.Lock()
        
        # Handler for managing markers.
        self.marker_handler = MarkerHandler()
       
    
    def aruco_callback(self, msg):
        """
        Callback function for ArUco marker detection messages.
        This function processes incoming data from a topic subscribed to ArUco marker positions.

        :param msg: The message received from the topic, expected to contain ArUco marker data.
        """
       
        self.aruco_detected=True
        # Extract the ArUco marker's x, y, z coordinates and ID from the received message.
        self.aruco_x, self.aruco_y, self.aruco_z, self.aruco_id = msg.data[0], msg.data[1], msg.data[2], msg.data[3]
    
        # Form the ArUco marker's position as a numpy array.
        aruco_position = np.array([[self.aruco_x], [self.aruco_y], [self.aruco_z]])
    
        # Transform the position from the camera frame to the feature frame.
        CxF = np.array([[self.aruco_z], [self.aruco_x], [self.aruco_y]]) 
        print('Camera_frame_to_feature_frame', CxF)
        
        # Transform the position from the robot frame to the feature frame.
        RxF = self.robot_frame_to_feature_frame(CxF)
        print('Robot_frame_to_feature_frame', RxF)
        
        # Extract only the (x, y) position.
        znp = RxF[:2]
        self.zk_feature = znp
    
        # Define the measurement noise covariance matrix.
        Rnp = np.diag([0.001, 5.0])
    
        # Update the state and covariance matrix with the new observation.
        self.xk, self.Pk = self.update(self.xk, self.Pk, znp, Rnp, self.aruco_id)
    
        rospy.logwarn(f"observed Aruco List, {self.marker_handler.observed_arucos}")
    
       
     
    def addNewFeature(self, xk, Pk, znp, Rnp, aruco_id):
        """
        Adds a new feature to the state vector and updates the covariance matrix accordingly.

        :param xk: Current state vector.
        :param Pk: Current covariance matrix.
        :param znp: Measurement vector of the new feature.
        :param Rnp: Measurement noise covariance matrix of the new feature.
        :param aruco_id: ID of the observed ArUco marker.
        """
        # Calculate the number of features from the measurement vector.
        nf = int(len(znp) // self.xF_dim)
        
        
        for i in range(nf):
            # Extract individual feature measurements from the measurement vector.
            start_idx = i * self.xF_dim
            end_idx = start_idx + self.xF_dim
            znpi = znp[start_idx:end_idx]

            # Extract the robot pose from the state vector.
            NxB = self.xk[0:self.xB_dim]
            
            # Compute the state of the new feature (World frame to feature frame) based on the robot's pose and the measurement.
            NxF = self.g(NxB, znpi)
            
            # Add the new marker to the marker handler.
            self.marker_handler.add_marker(self.aruco_id, NxF)
           
            # Compute the Jacobians of the motion model with respect to the state and noise.
            Gx = self.Jgx(NxB, znpi)
            Gv = self.Jgv(NxB, znpi)

            # Extract the relevant sub-matrix of the noise covariance matrix for the current feature.
            Rfpi = Rnp[start_idx:end_idx, start_idx:end_idx]

            # Initialize an extended covariance matrix to accommodate the new feature.
            new_dim = self.xk.shape[0] + NxF.shape[0]
            extended_Pk = np.zeros((new_dim, new_dim))
            
            # Copy the existing covariance matrix into the extended covariance matrix.
            extended_Pk[:self.Pk.shape[0], :self.Pk.shape[0]] = self.Pk
            
            # Compute the covariance block for the new feature.
            End_block = Gx @ self.Pk[:self.xB_dim, :self.xB_dim] @ Gx.T + Gv @ Rfpi @ Gv.T
            
            # Insert the new covariance block into the extended covariance matrix.
            offset = self.Pk.shape[0]
            extended_Pk[offset:offset + self.xF_dim, offset:offset + self.xF_dim] = End_block
            
            # Compute cross-covariance blocks and update the extended covariance matrix.
            side_block = Gx @ self.Pk[:self.xB_dim, :]
            extended_Pk[:self.Pk.shape[0], offset:offset + self.xF_dim] = side_block.T
            extended_Pk[offset:offset + self.xF_dim, :self.Pk.shape[0]] = side_block
            
            # Update the covariance matrix with the extended matrix.
            self.Pk = extended_Pk
            
            # Extend the state vector with the new feature state.
            self.xk = np.vstack((xk, NxF))
        

        
        return self.xk, self.Pk

    def update(self, xk, Pk, znp, Rnp, aruco_id):
        """
        Updates the state vector and covariance matrix based on the new observation.

        :param xk: Current state vector.
        :param Pk: Current covariance matrix.
        :param znp: Measurement vector of the new feature.
        :param Rnp: Measurement noise covariance matrix of the new feature.
        :param aruco_id: ID of the observed ArUco marker.
        :return: Updated state vector and covariance matrix.
        """
        self.xk = xk
        self.Pk = Pk
        
        # If the ArUco marker has not been observed before, add it as a new feature.
        if aruco_id not in self.marker_handler.observed_arucos:
            self.xk, self.Pk = self.addNewFeature(xk, Pk, znp, Rnp, aruco_id)
              
        else:
        
            # Get the index of the observed ArUco marker.
            index = self.marker_handler.get_index(aruco_id)
           
            # Calculate the start and end indices for the feature in the state vector.
            start_idx = 3 + index * self.xF_dim
            end_idx = start_idx + self.xF_dim
        
            self.zk_feature = znp

            # Create the actual observation (Robot frame to Feature frame)
            zk = znp.reshape(-1, 1)
            
            # Extract the feature state from the state vector.
            NxF = self.xk[start_idx:end_idx].reshape(-1, 1)
            
            # Extract the robot state from the state vector.
            NxR = self.xk[:self.xB_dim].reshape(-1, 1)
            NxR_x, NxR_y, NxR_yaw = NxR[0, 0], NxR[1, 0], NxR[2, 0]
           
           #Using ominus to convert to Robot frame to world frame. 
            RxN_x = -NxR_x * np.cos(NxR_yaw) - NxR_y * np.sin(NxR_yaw) 
            RxN_y = NxR_x * np.sin(NxR_yaw) - NxR_y * np.cos(NxR_yaw)
            RxN_yaw = -NxR_yaw
            
            # Extract the Feature state from the World frame to Feature frame.
            NxF_x, NxF_y = NxF[0, 0], NxF[1, 0]

            # Expected Observation (Robot frame to Feature frame)
            h = np.array([[RxN_x + NxF_x * np.cos(RxN_yaw) - NxF_y * np.sin(RxN_yaw)],
                          [RxN_y + NxF_x * np.sin(RxN_yaw) + NxF_y * np.cos(RxN_yaw)]])

            # Calculate the innovation (difference between expected and actual observations).
            innovation = zk - h
            print('innovation', innovation)

            # Initialize the Observation matrix.
            
            Hk = np.zeros((2, len(xk)))
            Hk[:, :3] = np.array([
                [-np.cos(NxR_yaw), -np.sin(NxR_yaw),
                 NxR_x * np.sin(NxR_yaw) - NxR_y * np.cos(NxR_yaw) - NxF_x * np.sin(NxR_yaw) + NxF_y * np.cos(NxR_yaw)],
                [np.sin(NxR_yaw), -np.cos(NxR_yaw),
                 NxR_x * np.cos(NxR_yaw) + NxR_y * np.sin(NxR_yaw) - NxF_x * np.cos(NxR_yaw) - NxF_y * np.sin(NxR_yaw)]
            ])
            Hk[:, start_idx:end_idx] = np.array([
                [np.cos(NxR_yaw), np.sin(NxR_yaw)],
                [-np.sin(NxR_yaw), np.cos(NxR_yaw)]
            ])

            # Define the Observation noise matrix.
            Vk = np.eye(2)
            
            # Calculate the innovation covariance matrix.
            S = Hk @ self.Pk @ Hk.T + Vk @ Rnp @ Vk.T
            
            # Calculate the Kalman gain.
            Kk = self.Pk @ Hk.T @ np.linalg.inv(S)
            
            I = np.eye(len(self.xk))
            
            #Acquire lock 
            self.lock.acquire()
            # Update the state vector.
            self.xk = self.xk + Kk @ innovation
            
            # Update the covariance matrix.
            self.Pk = (I - Kk @ Hk) @ self.Pk @ (I - Kk @ Hk).T 
             
            # Update the robot's state based on the received yaw measurement. The `update` function is assumed to handle this update by incorporating the new yaw measurement into the robot's state estimation process.
            self.yaw_update_orientation(self.yaw_measurement)
            
            # Publish the updated odometry.
            self.publish_odometry(self.xk, self.Pk)
            
            # Release the lock after the operations.
            self.lock.release()
            
            if aruco_id in self.marker_handler.observed_arucos:
                # Get the index of the observed ArUco marker.
                self.idx = self.marker_handler.get_index(aruco_id)
                
                # Calculate the start and end indices for the feature in the state vector.
                start_idx = 3 + self.marker_handler.get_index(aruco_id) * self.xF_dim
                end_idx = start_idx + self.xF_dim
                
                # Extract the feature state and its uncertainty from the state vector and covariance matrix.
                self.aruco_marker_data = self.xk[start_idx:end_idx].reshape(1, 2)
                self.aruco_marker_uncertainty = self.Pk[start_idx:end_idx, start_idx:end_idx]
                # Visualize the ArUco marker.
                self.aruco_visualization(self.idx, self.aruco_marker_data, self.aruco_marker_uncertainty)
            
            # Update the path.
            self.update_path()
    
        return self.xk, self.Pk
    
    ################################################### Imu Callback #################################################################
    def imu_callback(self, msg):
            """
            Callback function to handle IMU sensor data. It processes the orientation data provided in quaternion format,
            converts it to Euler angles, and updates the robot state using the yaw (orientation around the vertical axis).

            :param msg: The message received from the IMU topic, containing orientation data in quaternion format.
            """
        
            # Extract the quaternion tuple from the IMU message, which includes x, y, z, and w components.
            quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

            # Convert the quaternion to Euler angles. Since only yaw is needed for 2D motion estimation, 
            # the roll and pitch values are discarded (denoted by underscores).
            _, _, yaw_measurement = euler_from_quaternion(quaternion)
            self.yaw_measurement=yaw_measurement-np.pi/2    
            
    ################################################ Update Orientation #########################################################
    def yaw_update_orientation(self, yaw_measurement):
        """
        Performs a Kalman filter update step using the yaw measurement from the IMU. This method integrates the new yaw
        measurement into the state estimate, updating both the state vector and its covariance.

        :param yaw_measurement: The yaw angle measured by the IMU.
        
        """
        # Actual measurement received from the IMU callback.
        self.zk = np.array([yaw_measurement]).reshape(-1, 1)
    
        # Jacobian of the observation model with respect to the noise vector.
        self.Vk = np.diag([0.01])  # It's an identity matrix because the observation noise directly affects the measured yaw.

        # Covariance matrix of the observation noise.
        self.Rk = np.array([[0.0001]])

        # Expected observation from the current state estimate.
        self.h = np.array([self.xk[2,0]]).reshape(-1, 1)# The expected yaw from the state.
    
        # Innovation: difference between the actual measurement and the expected observation.
        innovation = self.wrap_angle(self.zk - self.h)
        

        # Compute the Jacobian of the observation model with respect to the state vector.
        # Jacobian of the observation model (Hk) with respect to the state
        self.Hk = np.zeros((1, len(self.xk)))
        self.Hk[0, 2] = 1  # Only the yaw component, which affects the measurement

            
        # Compute the Kalman gain.
        S = self.Hk @ self.Pk @ self.Hk.T + self.Vk @ self.Rk @ self.Vk.T
        Kk = self.Pk @ self.Hk.T @ np.linalg.inv(S)
        

        # Update the state estimate using the Kalman gain and the innovation.
        self.xk = self.xk + Kk @ innovation

        # Update the covariance of the estimate.
        I = np.eye(len(self.xk))  # Identity matrix of the same dimension as the state vector.
        self.Pk= (I - Kk @ self.Hk) @ self.Pk 
        

        return self.xk, self.Pk


    def euler_to_quaternion(self, R):
        """
        Convert a 2D rotation matrix R to a quaternion.
        :param R: The 2D rotation matrix.
        :return: The quaternion corresponding to the rotation matrix.
        """
        R_3d = np.eye(4)
        R_3d[0:2, 0:2] = R
        quaternion = tf.transformations.quaternion_from_matrix(R_3d)
        # Normalize the quaternion
        norm = np.linalg.norm(quaternion)
        normalized_quaternion = quaternion / norm
        return normalized_quaternion
    
    
    #################################### Marker Visualization #########################################
    def aruco_visualization(self, idx, aruco_position, aruco_marker_uncertainty):
        """
        Visualize the ArUco marker position and its uncertainty.

        :param idx: Index of the ArUco marker.
        :param aruco_position: Position of the ArUco marker.
        :param aruco_marker_uncertainty: Uncertainty of the ArUco marker position.
        """
        marker_array = MarkerArray()
        
        aruco_id = list(self.marker_handler.observed_arucos.keys())[idx]
        
        # Create a sphere marker for the ArUco marker.
        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = int(aruco_id)  # Use the actual ArUco ID as the marker ID
        marker.header.stamp = rospy.Time.now()
        marker.pose.position.x = aruco_position[0, 0]
        marker.pose.position.y = aruco_position[0, 1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0) 
        marker_array.markers.append(marker)

        # Calculate the Rotation matrix from the uncertainty matrix.
        R, _, _ = np.linalg.svd(aruco_marker_uncertainty[:2, :2], full_matrices=False)
        
        # Create an uncertainty marker.
        uncertainty_marker = Marker()
        uncertainty_marker.header.frame_id = 'world_ned'
        uncertainty_marker.header.stamp = rospy.Time.now()
        uncertainty_marker.ns = "feature_ellipse"
        uncertainty_marker.type = Marker.SPHERE_LIST
        uncertainty_marker.action = Marker.ADD
        uncertainty_marker.id = int(aruco_id) + 1000  # Make the uncertainty marker ID unique
        quaternion = self.euler_to_quaternion(R)
        uncertainty_marker.pose.orientation.x = quaternion[0]
        uncertainty_marker.pose.orientation.y = quaternion[1]
        uncertainty_marker.pose.orientation.z = quaternion[2]
        uncertainty_marker.pose.orientation.w = quaternion[3]
        uncertainty_marker.scale.x = 0.02
        uncertainty_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)

        # Create points for the Uncertainty ellipse.
        for point in np.transpose(GetEllipse(aruco_position, aruco_marker_uncertainty, sigma=1)):
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            uncertainty_marker.points.append(p)

        marker_array.markers.append(uncertainty_marker)

        # Create a text label for the ArUco marker.
        text_marker = Marker()
        text_marker.header.frame_id = "world_ned"
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.id = int(aruco_id) + 2000  # Make the text marker ID unique
        text_marker.header.stamp = rospy.Time.now()
        text_marker.pose.position.x = aruco_position[0, 0]
        text_marker.pose.position.y = aruco_position[0, 1]
        text_marker.pose.position.z = 0.2
        text_marker.scale.z = 0.15
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"ArUco {int(aruco_id)}"

        marker_array.markers.append(text_marker)
        rospy.loginfo(f"Publishing marker with ID {marker.id} at position ({marker.pose.position.x}, {marker.pose.position.y})")

        # Publish the marker array.
        self.markers_pub_aruco_marker.publish(marker_array)
        
    def robot_frame_to_feature_frame(self, CxF):
        """
        Transform the position from the robot frame to the feature frame.

        :param CxF: Position in the camera frame.
        :return: Position in the feature frame.
        """
        ########### Camera Frame to Feature Frame #######
        CxF_x = CxF[0, 0]
        CxF_y = CxF[1, 0]
        CxF_z = CxF[2, 0]
        ##########  Robot frame to Camera Frame ##########
        RxC_x = 0.122 
        RxC_y = -0.033 
        RxC_z = 0.082  
        RxC_yaw = 0.0
        
        # Calculate the feature position in the robot frame.
        RxF_x = RxC_x + CxF_x * np.cos(RxC_yaw) - CxF_y * np.sin(RxC_yaw)
        RxF_y = RxC_y + CxF_y * np.sin(RxC_yaw) + CxF_y * np.cos(RxC_yaw)
        RxF_z = RxC_z + CxF_z
        return np.array([[RxF_x], [RxF_y], [RxF_z]])
        
    def boxplus(self, NxB, BxF):
        """
        Compounds the  Robot state and a feature state to compute the feature's position in the world frame.

        :param NxB: The robot's state vector (x, y, theta).
        :param BxF: The feature's position respect to the robot frame.
        :return: The feature's position in the world frame.
        """
        x, y, theta = NxB[0, 0], NxB[1, 0], NxB[2, 0]
        BxF_x, BxF_y = BxF[0, 0], BxF[1, 0]
        NxF_x = x + np.cos(theta) * BxF_x - np.sin(theta) * BxF_y
        NxF_y = y + np.sin(theta) * BxF_x + np.cos(theta) * BxF_y
        return np.array([[NxF_x], [NxF_y]])

    def g(self, NxB, znp):
        """
        Applies the inverse observation model to compute the feature's state in the world frame.

        :param NxB: The robot's state in the world frame.
        :param znp: The feature observation.
        :return: The feature's state respect to the world frame.
        """
        return self.boxplus(NxB, znp)

    def J1_boxplus(self, NxB, BxF):
        """
        Computes the Jacobian of the boxplus operation with respect to the robot's state.

        :param NxB: The robot's state vector.
        :param BxF: The feature's relative position in the body frame.
        :return: Jacobian matrix with respect to the robot's state.
        """
        theta = NxB[2, 0]
        BxF_x, BxF_y = BxF[0, 0], BxF[1, 0]
        return np.array([
            [1.0, 0.0, -BxF_x * np.sin(theta) - BxF_y * np.cos(theta)],
            [0.0, 1.0, BxF_x * np.cos(theta) - BxF_y * np.sin(theta)]
        ])

    def Jgx(self, NxB, BxF):
        """
        Computes the Jacobian of the g function  with respect to the state.

        :param NxB: The robot's state.
        :param BxF: The feature's observed state in the body frame.
        :return: Jacobian matrix of the transformation.
        """
        return self.J1_boxplus(NxB, BxF)

    def J_2boxplus(self, NxB):
        """
        Computes the Jacobian of the boxplus operation with respect to the feature's position.

        :param NxB: The robot's state vector.
        :return: Jacobian matrix with respect to the feature's position.
        """
        theta = NxB[2, 0]
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    def Jgv(self, NxB, BxF):
        """
        Computes the Jacobian of the g function with respect to the noise.

        :param NxB: The robot's navigation state.
        :param BxF: The observed feature in the body frame.
        :return: Jacobian matrix with respect to the noise.
        """
        J2 = self.J_2boxplus(NxB)
        return J2
    
 

if __name__ == '__main__':
    try:
        AddNewFeatureNode()
        # Keep the ROS node running until interrupted
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
