import os
import yaml

# Constants
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
DATA_CONFIG_PATH = "/train/vint_train/data/data_config.yaml"

# Load robot config
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = 2 

# Topic names 
WAYPOINT_TOPIC = "/waypoint"
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"
SCORE_MAP_TOPIC = "/score_map"

# You need to modify the following three topics according to your robot platform
ODOM_TOPIC = "/odom"
VEL_TOPIC = "/api/sport/request"
# VEL_TOPIC = "/cmd_vel"
IMAGE_TOPIC = "/camera/color/image_raw" 

