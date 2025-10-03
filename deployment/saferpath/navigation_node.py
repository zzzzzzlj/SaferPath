import sys
import os
import time
import yaml
import torch
import numpy as np
import cv2
import math
from typing import List
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import OccupancyGrid, Odometry
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from deployment.saferpath.utils_tsm import compute_batch_cost_maps, project_points, transform_to_absolute_trajectory, get_camera_params
from deployment.src.utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from deployment.saferpath.config_navigation import *
from visualization import VisualizationHandler
import argparse

# Depth model configurations
depth_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class ExplorationNode(Node):
    def __init__(self, args):
        super().__init__('exploration_node')
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Instance variables
        self.context_queue: List[PILImage.Image] = []
        self.context_size = None
        self.subgoal = []
        self.current_pose = None
        self.current_yaw = 0.0
        self.closest_node = 0
        self.reached_goal = False
        self.topomap = []
        self.model_params = None
        self.num_diffusion_iters = None
        self.noise_scheduler = None
        self.depth_model = None
        self.model = None
        self.depth_save_dir = "depth_outputs"
        if not os.path.exists(self.depth_save_dir):
            os.makedirs(self.depth_save_dir)

        self._load_topomap()
        self._load_model_and_configs()
        self._init_depth_model()
        self._setup_ros()
        self._init_visualization()
        self.timer = self.create_timer(1.0 / RATE, self.navigation_loop)
        print("Navigation started. Waiting for image observations...")

    def _load_model_and_configs(self):
        """Load model parameters and weights"""
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[self.args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]

        ckpt_path = model_paths[self.args.model]["ckpt_path"]
        if os.path.exists(ckpt_path):
            print(f"Loading model from {ckpt_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
        self.model = load_model(ckpt_path, self.model_params, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        
        assert -1 <= self.args.goal_node < len(self.topomap), "Invalid goal node index"
        print("Total topomap nodes:", len(self.topomap))
        if self.args.goal_node == -1:
            self.goal_node = len(self.topomap) - 1
        else:
            self.goal_node = self.args.goal_node

    def _load_topomap(self):
        """Load topomap images"""
        topomap_filenames = sorted(os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, self.args.dir)), key=lambda x: int(x.split(".")[0]))
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{self.args.dir}"
        num_nodes = len(os.listdir(topomap_dir))
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            self.topomap.append(PILImage.open(image_path))

    def _init_depth_model(self):
        """Initialize Depth Anything V2 model"""
        self.depth_model = DepthAnythingV2(**depth_model_configs[self.args.depth_encoder])
        self.depth_model.load_state_dict(torch.load(
            f'../../Depth_Anything_V2/checkpoints/depth_anything_v2_{self.args.depth_encoder}.pth', 
            map_location='cuda'
        ))
        self.depth_model = self.depth_model.to(self.device).eval()

    def _setup_ros(self):
        """Setup ROS2 publishers and subscribers"""
        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 1
        )
        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.callback_odom, 10)
        
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 1)
        self.cost_map_pub = self.create_publisher(OccupancyGrid, "/score_map", 1)

    def _init_visualization(self):
        """Initialize visualization (currently disabled)"""
        pass

    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def plot_depth_map(self, obs_img):
        """Compute and display cost map for the current observation."""
        target_height, target_width = 96, 96
        cropped_img = obs_img.resize((target_width, target_height), PILImage.LANCZOS)
        
        obs_img_np = np.array(cropped_img)
        obs_img_np = cv2.cvtColor(obs_img_np, cv2.COLOR_RGB2BGR)
        
        with torch.no_grad():
            depth = self.depth_model.infer_image(obs_img_np)
        
        depth_np = depth
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        depth_tensor = torch.from_numpy(depth_np).to(self.device).unsqueeze(0)
        cost_map = compute_batch_cost_maps(depth_tensor, threshold_factor=1.0).squeeze(0)
        cost_map_np = cost_map.cpu().numpy()
        
        cost_map_msg = OccupancyGrid()
        cost_map_msg.header.stamp = self.get_clock().now().to_msg()
        cost_map_msg.header.frame_id = "map"
        cost_map_msg.info.width = cost_map_np.shape[1]
        cost_map_msg.info.height = cost_map_np.shape[0]
        cost_map_msg.info.resolution = 0.1
        cost_map_msg.data = (cost_map_np.flatten() * 100).astype(np.int8).tolist()
        self.cost_map_pub.publish(cost_map_msg)

        return cost_map_np

    def callback_odom(self, msg):
        """Callback function to receive odometry data"""
        try:
            self.current_pose = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
            cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
            self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
            self.get_logger().debug(f"Updated pose: x={self.current_pose.x:.2f}, y={self.current_pose.y:.2f}, yaw={self.current_yaw:.2f}")
        except Exception as e:
            self.get_logger().error(f"Error in callback_odom: {str(e)}")

    def select_and_visualize_trajectory(self, trajectories, cost_map_np):
        """Project trajectories to cost map, compute costs, and select the best trajectory."""
        camera_matrix, dist_coeffs, camera_height, camera_x_offset = get_camera_params()
        image_width = self.model_params["image_size"][1]
        SCALE = 1

        map_height, map_width = cost_map_np.shape
        cost_map_flat = cost_map_np.flatten()

        trajectories_scaled = trajectories * SCALE
        uv = project_points(trajectories_scaled, camera_height, camera_x_offset, camera_matrix, dist_coeffs)
        uv = uv.reshape(trajectories.shape[0], trajectories.shape[1], 2)

        costs = []
        u_lists = []
        v_lists = []
        for i in range(trajectories.shape[0]):
            u_list = uv[i, :, 0]
            v_list = uv[i, :, 1]
            u_list = image_width - u_list
            u_list = np.clip(u_list, 0, map_width - 1)
            v_list = np.clip(v_list, 0, map_height - 1)
            traj_cost = 0
            for u, v in zip(u_list, v_list):
                u_idx = int(np.floor(u))
                v_idx = int(np.floor(v))
                idx = u_idx + v_idx * map_width
                cost = cost_map_flat[idx]
                traj_cost += cost
            costs.append(traj_cost)
            u_lists.append(u_list)
            v_lists.append(v_list)

        min_cost_idx = np.argmin(costs)
        chosen_trajectory = trajectories[min_cost_idx]
        chosen_cost = costs[min_cost_idx]
        self.get_logger().info(f"Chosen trajectory index: {min_cost_idx}, cost: {chosen_cost}")

        return chosen_trajectory, min_cost_idx, chosen_cost

    def navigation_loop(self):
        chosen_waypoint = np.zeros(4)
        if len(self.context_queue) > self.model_params["context_size"]:
            current_obs_img = self.context_queue[-1]
            cost_map_np = self.plot_depth_map(current_obs_img)

            if self.model_params["model_type"] == "nomad":
                obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1)
                obs_images = obs_images.to(self.device)
                mask = torch.zeros(1).long().to(self.device)

                start = max(self.closest_node - self.args.radius, 0)
                end = min(self.closest_node + self.args.radius + 1, self.goal_node)
                goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(self.device) for g_img in self.topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                self.closest_node = min_idx + start
                print("closest node:", self.closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                with torch.no_grad():
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
                    
                    noisy_action = torch.randn(
                        (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
                    naction = noisy_action

                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    start_time = time.time()
                    for k in self.noise_scheduler.timesteps:
                        noise_pred = self.model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                naction = to_numpy(get_action(naction))
                naction = naction * 0.5
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate(([0.0], naction.flatten())).astype(float).tolist()
                self.sampled_actions_pub.publish(sampled_actions_msg)
                
                chosen_trajectory = naction[0]
                initial_point = np.zeros((1, chosen_trajectory.shape[1]))
                chosen_trajectory = np.vstack([initial_point, chosen_trajectory])
                len_traj_pred = chosen_trajectory.shape[0]
                N = 5  
                if len_traj_pred >= N + 1:
                    chosen_trajectory = chosen_trajectory[:N+1]
                else:
                    padding = np.repeat(chosen_trajectory[-1:], N + 1 - len_traj_pred, axis=0)
                    chosen_trajectory = np.vstack([chosen_trajectory, padding])
                
                absolute_trajectory = transform_to_absolute_trajectory(self, chosen_trajectory)
                
                waypoint_msg = Float32MultiArray()
                waypoint_msg.data = absolute_trajectory.flatten().astype(float).tolist()
                self.waypoint_pub.publish(waypoint_msg)

                cost_map_msg = OccupancyGrid()
                cost_map_msg.header.stamp = self.get_clock().now().to_msg()
                cost_map_msg.header.frame_id = "map"
                cost_map_msg.info.width = cost_map_np.shape[1]
                cost_map_msg.info.height = cost_map_np.shape[0]
                cost_map_msg.info.resolution = 0.1
                cost_map_msg.data = (cost_map_np.flatten() * 100).astype(np.int8).tolist()
                self.cost_map_pub.publish(cost_map_msg)
                
                chosen_waypoint = chosen_trajectory[self.args.waypoint]
            else:
                start = max(self.closest_node - self.args.radius, 0)
                end = min(self.closest_node + self.args.radius + 1, self.goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(self.topomap[start:end + 1]):
                    transf_obs_img = transform_images(self.context_queue, self.model_params["image_size"])
                    goal_data = transform_images(sg_img, self.model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(self.device)

                distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                waypoints = waypoints * 0.5
                waypoints = waypoints[..., :2]
                
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] > self.args.close_threshold:
                    chosen_trajectory = waypoints[min_dist_idx]
                    self.closest_node = start + min_dist_idx
                else:
                    chosen_trajectory = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)]
                    self.closest_node = min(start + min_dist_idx + 1, self.goal_node)
                
                initial_point = np.zeros((1, chosen_trajectory.shape[1]))
                chosen_trajectory = np.vstack([initial_point, chosen_trajectory])
                
                len_traj_pred = chosen_trajectory.shape[0]
                N = 5
                if len_traj_pred >= N + 1:
                    chosen_trajectory = chosen_trajectory[:N+1]
                else:
                    padding = np.repeat(chosen_trajectory[-1:], N + 1 - len_traj_pred, axis=0)
                    chosen_trajectory = np.vstack([chosen_trajectory, padding])
                
                absolute_trajectory = transform_to_absolute_trajectory(self, chosen_trajectory)
                
                waypoint_msg = Float32MultiArray()
                waypoint_msg.data = absolute_trajectory.flatten().astype(float).tolist()
                self.waypoint_pub.publish(waypoint_msg)
                
                cost_map_msg = OccupancyGrid()
                cost_map_msg.header.stamp = self.get_clock().now().to_msg()
                cost_map_msg.header.frame_id = "map"
                cost_map_msg.info.width = cost_map_np.shape[1]
                cost_map_msg.info.height = cost_map_np.shape[0]
                cost_map_msg.info.resolution = 0.1
                cost_map_msg.data = (cost_map_np.flatten() * 100).astype(np.int8).tolist()
                self.cost_map_pub.publish(cost_map_msg)
                
        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)
        chosen_waypoint = [float(x) for x in chosen_waypoint]
        self.reached_goal = self.closest_node == self.goal_node
        goal_msg = Bool()
        goal_msg.data = bool(self.reached_goal)
        self.goal_pub.publish(goal_msg)
        if self.reached_goal:
            print("Goal reached! Stopping...")
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

    def destroy_node(self):
        super().destroy_node()
