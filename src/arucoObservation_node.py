#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ArucoObservationNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("image_sub" , anonymous=True)
        """Initialize the ArucoDetector node."""
        self.bridge = CvBridge()  # Bridge to convert ROS Image messages to OpenCV images
        
        ############################## Subscribe to camera image topic ################################
        self.image_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/realsense/color/image_color", Image, self.image_callback)
        ######################### Publisher for ArUco marker positions ################################
        self.pose_pub = rospy.Publisher("turtlebot/kobuki/aruco_position", Float64MultiArray, queue_size=10)
    def calculate_noise(self, gray_image):
        # Calculate the standard deviation of the pixel intensities
        noise = np.std(gray_image)
        return noise
    def image_callback(self,msg):
        """Callback function for image data received from the camera."""
        try:
            # Convert the ROS image to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        # Marker identification settings
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        frame = cv_image
        
        # Detect markers in the image
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
        
        """ Hardcoded camera matrix extractions from "rostopic echo /turtlebot/kobuki/sensors/realsense/color/camera_info" """
        
        # Pre-defined camera calibration
        camera_matrix = np.array([[1396.8086675255468, 0.0, 960.0],
                                  [0.0, 1396.8086675255468, 540.0],   
                                  [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Process each detected marker
        if marker_ids is not None:
            for i in range(len(marker_ids)):
                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i], 0.15, camera_matrix, dist_coeffs)
                
                # Draw marker boundaries and axes on the image
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, (0, 255, 0))
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.05)

                # Extract the translation vectors (x, y, z)
                x, y, z = tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2]

                # Display the X, Y, and Z coordinates on the image
                cv2.putText(frame, f"X: {x:.2f}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Y: {y:.2f}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {z:.2f}", (10, 85), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

                # Prepare a message to publish the pose
                point_msg = Float64MultiArray()
                point_msg.data = [x, y, z, float(marker_ids[i])]
                rospy.loginfo(point_msg.data)
                self.pose_pub.publish(point_msg)

        # Show the image with markers
        # cv2.imshow("Camera", frame)
        # cv2.waitKey(3)  # A small delay to update the window

if __name__ == '__main__':
    try:
        ArucoObservationNode()
        # Keep the node running until interrupted
        rospy.spin()  
    except rospy.ROSInterruptException:
        pass