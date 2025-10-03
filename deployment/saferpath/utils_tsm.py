import cv2
import numpy as np
import torch
import math
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
from deployment.saferpath.config_navigation import DATA_CONFIG_PATH

def compute_batch_cost_maps(
    depth_imgs: torch.Tensor,
    threshold_factor: float = 1.0
) -> torch.Tensor:
    """
    Generate binary cost maps based on depth image gradients using OpenCV.

    Args:
        depth_imgs (torch.Tensor): Batch of depth images, shape (B, H, W)
        threshold_factor (float): Threshold factor to adjust sensitivity

    Returns:
        torch.Tensor: Binary cost maps, shape (B, H, W), 0 for traversable, 1 for obstacles
    """
    B, H, W = depth_imgs.shape
    device = depth_imgs.device
    cost_maps = []

    for i in range(B):
        depth_img_np = depth_imgs[i].cpu().numpy().astype(np.float32)

        min_val = np.min(depth_img_np)
        max_val = np.max(depth_img_np)
        if max_val - min_val > 1e-6:
            normalized_depth = (depth_img_np - min_val) / (max_val - min_val)
        else:
            normalized_depth = np.zeros_like(depth_img_np)

        smoothed = cv2.GaussianBlur(normalized_depth, (5, 5), 0)
        grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.abs(grad_y)

        position_weight = np.zeros_like(grad_magnitude)
        for j in range(H):
            weight = 0.1 + 0.9 * (j / H)
            position_weight[j, :] = weight

        weighted_grad = position_weight * grad_magnitude
        threshold = np.mean(weighted_grad) * threshold_factor

        cost_map_np = np.ones_like(grad_magnitude, dtype=np.uint8)
        cost_map_np[weighted_grad >= threshold] = 0
        cost_map_np[-1:, :] = 0

        cost_map = torch.from_numpy(cost_map_np).to(device)
        cost_maps.append(cost_map)

    return torch.stack(cost_maps, dim=0)

def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) or (batch_size, horizon, num_points, 2)
            representing (x, y) coordinates. The latter is for expanded trajectories.
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        uv: array of shape (batch_size, horizon, 2) or (batch_size, horizon, num_points, 2)
            representing (u, v) coordinates on the 2D image plane
    """
    if xy.ndim == 3:
        batch_size, horizon, _ = xy.shape
        num_points = 1
    elif xy.ndim == 4:
        batch_size, horizon, num_points, _ = xy.shape
    else:
        raise ValueError("xy must have shape (batch_size, horizon, 2) or (batch_size, horizon, num_points, 2)")

    xy_reshaped = xy.reshape(batch_size * horizon * num_points, 2)
    xyz = np.concatenate(
        [xy_reshaped, -camera_height * np.ones((batch_size * horizon * num_points, 1))], axis=-1
    )
    xyz[:, 0] += camera_x_offset
    xyz_cv = np.stack([xyz[:, 1], -xyz[:, 2], xyz[:, 0]], axis=-1)
    rvec = (0, 0, 0)
    tvec = (0, 0, 0)
    uv, _ = cv2.projectPoints(
        xyz_cv, rvec, tvec, camera_matrix, dist_coeffs
    )
    if num_points == 1:
        uv = uv.reshape(batch_size, horizon, 2)
    else:
        uv = uv.reshape(batch_size, horizon, num_points, 2)
    return uv

def transform_to_absolute_trajectory(self, relative_trajectory):
    """Transform relative trajectory to absolute coordinates"""
    if self.current_pose is None:
        self.get_logger().warn("No odometry data available, returning relative trajectory")
        return relative_trajectory
    
    absolute_trajectory = np.zeros_like(relative_trajectory)
    current_x = self.current_pose.x
    current_y = self.current_pose.y
    current_yaw = self.current_yaw
    
    for i in range(relative_trajectory.shape[0]):
        rel_x = relative_trajectory[i, 0]
        rel_y = relative_trajectory[i, 1]
        
        abs_x = current_x + rel_x * math.cos(current_yaw) - rel_y * math.sin(current_yaw)
        abs_y = current_y + rel_x * math.sin(current_yaw) + rel_y * math.cos(current_yaw)
        
        absolute_trajectory[i, 0] = abs_x
        absolute_trajectory[i, 1] = abs_y
        
        if relative_trajectory.shape[1] > 2:
            absolute_trajectory[i, 2:] = relative_trajectory[i, 2:]
    
    return absolute_trajectory

def get_camera_params(dataset_name="turtlebot3"):
    """Extract camera parameters from config"""
    with open(DATA_CONFIG_PATH, "r") as f:
        data_config = yaml.safe_load(f)
    fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
    fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
    cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
    cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([
        data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"],
        data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"],
        data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"],
        data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"],
        data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
    ])
    camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
    camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]
    return camera_matrix, dist_coeffs, camera_height, camera_x_offset
