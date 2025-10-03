import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import distance_transform_edt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from deployment.src.topic_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC
from deployment.src.utils import clip_angle
from config_controller import *
from config_navigation import *
from deployment.saferpath.utils_mpsves import *
from trajectory_optimizer import trajectory_optimizer
from utils_mpc import mpc_optimize
from plotting import PlottingHandler

from unitree_api.msg import Request
import json

class TrajectoryOptimizer(Node):
    def __init__(self):
        super().__init__('trajectory_optimizer')

        # Subscribers
        self.sub_score_map = self.create_subscription(
            OccupancyGrid, SCORE_MAP_TOPIC, self.score_map_callback, 1)
        self.sub_waypoint = self.create_subscription(
            Float32MultiArray, WAYPOINT_TOPIC, self.waypoint_callback, 1)
        self.sub_reached_goal = self.create_subscription(
            Bool, REACHED_GOAL_TOPIC, self.reached_goal_callback, 1)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 1)
        
        # Publisher for control inputs
        # self.pub_control = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_control = self.create_publisher(Request, VEL_TOPIC, 1)

        # Timer
        self.timer = self.create_timer(dt, self.process_trajectory)
        
        self.traj_queue = deque(maxlen=5)
        self.current_traj = None
        self.current_score_map = None
        self.current_state = np.zeros(3)
        self.current_pose = None
        self.current_yaw = 0.0
        self.reached_goal = False
        self.pending_score_map = None
        self.step_count = 0
        self.mpc_trajectory = []  # Store MPC actual trajectory
        self.mpc_predicted_traj = None

        # Rotation-related variables
        self.is_rotating = False
        self.rotation_start_time = None
        self.rotation_target_angle = 0.0
        self.rotation_timer = None

        # Initialize plotting
        # self.plot_handler = PlottingHandler()

        self.get_logger().info("Trajectory Optimizer initialized.")

    def score_map_callback(self, msg):
        try:
            map_data = np.array(msg.data, dtype=float) / 100.0
            map_width = msg.info.width
            map_height = msg.info.height
            map_data_2d = map_data.reshape(map_height, map_width)
            grad_x = np.abs(np.diff(map_data_2d, axis=1, prepend=map_data_2d[:, :1]))
            grad_y = np.abs(np.diff(map_data_2d, axis=0, prepend=map_data_2d[:1, :]))
            boundaries = np.logical_or(grad_x > 0.5, grad_y > 0.5).astype(float)
            obstacle_mask = map_data_2d > 0.5
            passable_mask = map_data_2d < 0.5
            dist_obstacle = distance_transform_edt(obstacle_mask)
            dist_passable = distance_transform_edt(passable_mask)
            boundary_value = 0.0
            distance_scale = 0.5
            smooth_cost_map = np.zeros_like(map_data_2d, dtype=float)
            smooth_cost_map[boundaries > 0.5] = boundary_value
            smooth_cost_map[obstacle_mask] = boundary_value + dist_obstacle[obstacle_mask] * distance_scale
            smooth_cost_map[passable_mask] = boundary_value - dist_passable[passable_mask] * distance_scale
            max_cost = np.max(np.abs(smooth_cost_map))
            if max_cost > 0:
                smooth_cost_map = smooth_cost_map / max_cost
            smooth_cost_map = np.clip(smooth_cost_map, -1, 1)
            self.pending_score_map = {
                'data': smooth_cost_map.flatten(),
                'width': map_width,
                'height': map_height
            }
            plt.imsave('cost_map.png', smooth_cost_map, cmap='RdBu')
        except Exception as e:
            self.get_logger().error(f"score_map_callback error: {str(e)}")

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.current_pose = position
        self.current_yaw = 2 * math.atan2(orientation.z, orientation.w)
        self.current_state[0] = position.x
        self.current_state[1] = position.y
        self.current_state[2] = self.current_yaw
        self.mpc_trajectory.append(self.current_state.copy())

    def waypoint_callback(self, msg):
        try:
            data = np.array(msg.data).reshape(-1, 2)
            traj = np.zeros((3, data.shape[0]))
            traj[:2, :] = data.T
            dx = np.diff(traj[0, :])
            dy = np.diff(traj[1, :])
            traj[2, :-1] = np.arctan2(dy, dx)
            traj[2, -1] = traj[2, -2] if TRACKING_STEPS > 1 else 0.0

            if self.pending_score_map is None:
                self.get_logger().warn("No score map available, skipping trajectory processing.")
                score_map_info = {
                    'data': np.zeros((1, 1)),
                    'width': 1,
                    'height': 1
                }
            else:
                score_map_info = self.pending_score_map
                self.pending_score_map = None
            
            self.traj_queue.append((traj, score_map_info))
        except Exception as e:
            self.get_logger().error(f"Trajectory parsing failed: {str(e)}")

    def reached_goal_callback(self, msg):
        self.reached_goal = msg.data
        if self.reached_goal:
            self.get_logger().info("Goal reached!")
            # self.plot_handler.close()

    def pad_trajectory(self, traj):
        if traj.shape[1] >= N + 1:
            return traj[:, :N + 1]
        dx = traj[0, -1] - traj[0, -2] if traj.shape[1] > 1 else 0
        dy = traj[1, -1] - traj[1, -2] if traj.shape[1] > 1 else 0
        padded_traj = np.zeros((3, N + 1))
        padded_traj[:, :traj.shape[1]] = traj
        for k in range(traj.shape[1], N + 1):
            padded_traj[0, k] = padded_traj[0, k-1] + dx
            padded_traj[1, k] = padded_traj[1, k-1] + dy
            padded_traj[2, k] = padded_traj[2, k-1]
        return padded_traj

    def convert_to_relative_trajectory(self, x_ref, current_state):
        """Convert absolute trajectory to relative trajectory"""
        rel_traj = np.zeros_like(x_ref)  # [x, y, theta]
        cos_theta = np.cos(current_state[2])
        sin_theta = np.sin(current_state[2])
        
        for t in range(x_ref.shape[1]):
            dx = x_ref[0, t] - current_state[0]
            dy = x_ref[1, t] - current_state[1]
            rel_traj[0, t] = dx * cos_theta + dy * sin_theta
            rel_traj[1, t] = -dx * sin_theta + dy * cos_theta
            rel_traj[2, t] = clip_angle(x_ref[2, t] - current_state[2])
        
        return rel_traj

    def check_rotation_completion(self):
        """Check if rotation is complete"""
        if self.is_rotating:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.rotation_start_time).nanoseconds / 1e9
            angle_diff = clip_angle(self.current_state[2] - self.rotation_target_angle)
            
            if elapsed_time >= ROTATION_DURATION or abs(angle_diff) < 0.05:
                self.get_logger().info("Rotation completed.")
                self.is_rotating = False
                self.rotation_timer.destroy()
                self.rotation_timer = None
                
                control_msg = Request()
                control_msg.header.identity.api_id = 1002
                control_msg.parameter = json.dumps({"x": 0.0, "y": 0.0, "z": 0.0})
                # control_msg = Twist()
                # control_msg.linear.x = 0.0
                # control_msg.angular.z = 0.0
                self.pub_control.publish(control_msg)
                if self.current_traj is not None and self.current_score_map is not None:
                    self.traj_queue.append((self.current_traj, self.current_score_map))

    def process_trajectory(self):
        self.step_count += 1
        if self.reached_goal:
            return
        
        if self.is_rotating:
            return
        
        if len(self.traj_queue) > 0:
            self.current_traj, self.current_score_map = self.traj_queue[-1]
            self.traj_queue.clear()

            try:
                current_x = self.current_state[0]
                current_y = self.current_state[1]
                dists = np.sqrt((self.current_traj[0, :] - current_x)**2 + (self.current_traj[1, :] - current_y)**2)
                i_start = np.argmin(dists)
                ref_window = self.current_traj[:, i_start : i_start + N + 1]
                if ref_window.shape[1] < N + 1:
                    print("Padding trajectory")
                    ref_window = self.pad_trajectory(ref_window)

                ref_start_pose = self.current_traj[:, 0]

                map_data = self.current_score_map['data']
                map_width = self.current_score_map['width']
                map_height = self.current_score_map['height']
                
                fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
                fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
                cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
                cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
                dist_coeffs = DIST_COEFFS
                camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
                camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]

                result = trajectory_optimizer(
                    current_state=self.current_state,
                    x_ref=ref_window,
                    ref_start_pose=ref_start_pose,
                    map_width=map_width,
                    map_height=map_height,
                    map_data=map_data,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    dist_coeffs=dist_coeffs,
                    camera_height=camera_height,
                    camera_x_offset=camera_x_offset
                )
                X_opt = result['X']
                
                if not result['valid']:
                    self.get_logger().error("Trajectory optimization failed: No valid trajectories found.")
                    cost_map_2d = self.current_score_map['data'].reshape(self.current_score_map['height'], self.current_score_map['width'])
                    bottom_rows = cost_map_2d[-20:, :] if cost_map_2d.shape[0] >= 20 else cost_map_2d
                    mid_col = cost_map_2d.shape[1] // 2
                    left_passable = np.sum(bottom_rows[:, :mid_col] < 0)
                    right_passable = np.sum(bottom_rows[:, mid_col:] < 0)
                    self.get_logger().info(f"Left passable pixels: {left_passable}, Right passable pixels: {right_passable}")

                    rotation_speed = ROTATION_ANGLE / ROTATION_DURATION
                    
                    # control_msg = Twist()
                    # if left_passable > right_passable:
                    #     control_msg.linear.x = 0.0
                    #     control_msg.angular.z = float(rotation_speed)
                    #     self.rotation_target_angle = self.current_state[2] + ROTATION_ANGLE
                    # else:
                    #     control_msg.linear.x = 0.0
                    #     control_msg.angular.z = float(-rotation_speed)
                    #     self.rotation_target_angle = self.current_state[2] - ROTATION_ANGLE

                    control_msg = Request()
                    control_msg.header.identity.api_id = 1008
                    if left_passable > right_passable:
                        control_msg.parameter = json.dumps({"x": 0.0, "y": 0.0, "z": float(rotation_speed)})
                        self.rotation_target_angle = self.current_state[2] + ROTATION_ANGLE
                    else:
                        control_msg.parameter = json.dumps({"x": 0.0, "y": 0.0, "z": float(-rotation_speed)})
                        self.rotation_target_angle = self.current_state[2] - ROTATION_ANGLE
                    
                    self.pub_control.publish(control_msg)
                    self.is_rotating = True
                    self.rotation_start_time = self.get_clock().now()
                    self.rotation_timer = self.create_timer(0.1, self.check_rotation_completion)
                    return

                # self.get_logger().info(f"Optimized trajectory: {X_opt}")

                x_ref = self.convert_to_relative_trajectory(X_opt, self.current_state)
                x0_rel = self.current_state - X_opt[:, 0]

                u_opt, predicted_traj = mpc_optimize(
                    x0=x0_rel,
                    x_ref=x_ref
                )
                self.mpc_predicted_traj = predicted_traj
                self.get_logger().info(f"MPC control input: v={u_opt[0]:.3f}, w={u_opt[1]:.3f}")

                # control_msg = Twist()
                # control_msg.linear.x = float(u_opt[0])
                # control_msg.angular.z = float(u_opt[1])
                
                control_msg = Request()
                control_msg.header.identity.api_id = 1008
                # if u_opt[0] > V_MAX:
                #     u_opt[0] = V_MAX
                #     u_opt[1] = u_opt[1] * (V_MAX / u_opt[0])
                u_opt[0] = np.clip(u_opt[0], V_MIN, V_MAX)
                u_opt[1] = np.clip(u_opt[1], -W_MAX, W_MAX)

                self.pub_control.publish(control_msg)

                # u_list, v_list = project_trajectory(X_opt, ref_start_pose, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height)
                # self.plot_handler.plot_trajectory_on_cost_map(map_data, u_list, v_list, ref_start_pose, X_opt, self.current_traj, self.mpc_predicted_traj)
            except Exception as e:
                self.get_logger().error(f"Trajectory optimization or MPC failed: {str(e)}")

    def destroy_node(self):
        # self.plot_handler.close()
        super().destroy_node()
