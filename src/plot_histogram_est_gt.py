import rosbag
import pandas as pd
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import math

# Load the bag file
bag_file_path = '/home/syma/catkin_ws/src/hands_on_localization/src/main_odom.bag'
bag = rosbag.Bag(bag_file_path)

# Initialize lists to hold the data
estimated_poses = []
ground_truth_poses = []

# Extract estimated poses
for topic, msg, t in bag.read_messages(topics=['/turtlebot/kobuki/odom_predict']):
    # print(t.to_sec())
    # Extracting position and orientation
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    _, _, yaw = euler_from_quaternion(orientation_list)
    estimated_poses.append([t.to_sec(), x, y, yaw])

# Extract ground truth poses
for topic, msg, t in bag.read_messages(topics=['/turtlebot/kobuki/ground_truth']):
    # Extracting position and orientation
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    _, _, yaw = euler_from_quaternion(orientation_list)
    ground_truth_poses.append([t.to_sec(), x-3.0, y+0.78, yaw-math.pi/2])
# 3.0 -0.78 -0.2
bag.close()

# Create dataframes
estimated_df = pd.DataFrame(estimated_poses, columns=['time', 'x', 'y', 'yaw'])
ground_truth_df = pd.DataFrame(ground_truth_poses, columns=['time', 'x', 'y', 'yaw'])

# Merge dataframes on time
merged_df = pd.merge_asof(estimated_df, ground_truth_df, on='time', suffixes=('_est', '_gt'))

# Calculate errors
merged_df['error_x'] = merged_df['x_est'] - merged_df['x_gt']
merged_df['error_y'] = merged_df['y_est'] - merged_df['y_gt']
merged_df['error_yaw'] = merged_df['yaw_est'] - merged_df['yaw_gt']

# Calculate mean of the errors
mean_error_x = merged_df['error_x'].mean()
mean_error_y = merged_df['error_y'].mean()
mean_error_yaw = merged_df['error_yaw'].mean()

# Subtract the mean to create zero-mean errors
zero_mean_error_x = merged_df['error_x'] - mean_error_x
zero_mean_error_y = merged_df['error_y'] - mean_error_y
zero_mean_error_yaw = merged_df['error_yaw'] - mean_error_yaw

# Plot zero-mean X error histogram
plt.figure(figsize=(6, 6))
plt.hist(zero_mean_error_x, bins=20, density=True, facecolor='g', alpha=0.75)
min_ylim, max_ylim = plt.ylim()
plt.xlabel('Zero-Mean X Error')
plt.title('error histogram', fontstyle='italic')
plt.ylim(0, 2)
plt.savefig('zero_mean_x_error_histogram.png')
plt.show()

# Plot zero-mean Y error histogram
plt.figure(figsize=(6, 6))
plt.hist(zero_mean_error_y, bins=20, density=True, facecolor='r', alpha=0.75)
min_ylim, max_ylim = plt.ylim()
plt.xlabel('Zero-Mean Y Error')
plt.ylabel('Number of Errors')
plt.title('error histogram', fontstyle='italic')
plt.ylim(0, 2)
plt.savefig('zero_mean_y_error_histogram.png')
plt.show()

# Plot zero-mean Yaw error histogram
plt.figure(figsize=(6, 6))
plt.hist(zero_mean_error_yaw, bins=20, density=True, facecolor='y', alpha=0.75)
min_ylim, max_ylim = plt.ylim()
plt.xlabel('Zero-Mean Yaw Error')
plt.title('error histogram', fontstyle='italic')
plt.ylim(0, 2)
plt.savefig('zero_mean_yaw_error_histogram.png')
plt.show()




# Plot X values over time
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(merged_df['time'], merged_df['x_gt'], 'g', label='Ground Truth X')
plt.plot(merged_df['time'], merged_df['x_est'], 'b', label='Estimated X')
plt.xlabel('Time')
plt.ylabel('X Position')
plt.title('X Position Over Time', fontstyle='italic')
plt.legend()

# Plot Y values over time
plt.subplot(3, 1, 2)
plt.plot(merged_df['time'], merged_df['y_gt'], 'g', label='Ground Truth Y')
plt.plot(merged_df['time'], merged_df['y_est'], 'b', label='Estimated Y')
plt.xlabel('Time')
plt.ylabel('Y Position')
plt.title('Y Position Over Time', fontstyle='italic')
plt.legend()

# Plot Yaw values over time
plt.subplot(3, 1, 3)
plt.plot(merged_df['time'], merged_df['yaw_gt'], 'g', label='Ground Truth Yaw')
plt.plot(merged_df['time'], merged_df['yaw_est'], 'b', label='Estimated Yaw')
plt.xlabel('Time')
plt.ylabel('Yaw')
plt.title('Yaw Over Time', fontstyle='italic')
plt.legend()
plt.tight_layout()
plt.savefig('pose_comparison.png')
plt.show()



# Calculate errors
merged_df['error_x'] = merged_df['x_est'] - merged_df['x_gt']
merged_df['error_y'] = merged_df['y_est'] - merged_df['y_gt']
merged_df['error_yaw'] = merged_df['yaw_est'] - merged_df['yaw_gt']

# Plot errors over time
plt.figure(figsize=(10, 12))

# Plot X error over time
plt.subplot(3, 1, 1)
plt.plot(merged_df['time'], merged_df['error_x'], 'r')
plt.xlabel('Time')
plt.ylabel('X Error')
plt.title('X Error Over Time', fontstyle='italic')

# Plot Y error over time
plt.subplot(3, 1, 2)
plt.plot(merged_df['time'], merged_df['error_y'], 'r')
plt.xlabel('Time')
plt.ylabel('Y Error')
plt.title('Y Error Over Time', fontstyle='italic')

# Plot Yaw error over time
plt.subplot(3, 1, 3)
plt.plot(merged_df['time'], merged_df['error_yaw'], 'r')
plt.xlabel('Time')
plt.ylabel('Yaw Error')
plt.title('Yaw Error Over Time', fontstyle='italic')

plt.tight_layout()
plt.savefig('error_over_time.png')
plt.show()