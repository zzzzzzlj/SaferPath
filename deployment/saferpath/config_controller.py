import yaml
import numpy as np

# Path to the system configuration file
DATA_CONFIG_PATH = "../../train/vint_train/data/data_config.yaml"

# Load configuration data
with open(DATA_CONFIG_PATH, "r") as f:
    data_config = yaml.safe_load(f)

# Dataset and camera parameters
dataset_name = "turtlebot3"
camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]
IMAGE_SIZE = [96, 96]

# Camera distortion coefficients
k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
DIST_COEFFS = np.array([k1, k2, p1, p2, k3])

# Timing and control parameters
RATE = 2
dt = 1 / RATE
WAYPOINT_TIMEOUT = 1.0
TRACKING_STEPS = 5
DT = 1 / RATE

# Trajectory optimizer parameters
N = 5
Q_BASE = np.diag([100, 100, 50])
COST_THRESHOLD = -0.08
CONSTRAINT_PENALTY = 100.0
SCALE = 1
NUM_INTERP_POINTS = 3
K = 100
RHO = 5
SIGMA_XY = 0.2
SIGMA_THETA = 0.2
LAMBDA = 2.0
ETA = 0.7
GAMMA = 0.2
KERNEL_H = 0.02
NUM_ITER = 10
SMOOTHNESS_WEIGHT = 5
MAP_WEIGHT = 50
MAP_STARTPOINT = 4
ROBOT_WIDTH = 0.3

# Model Predictive Control (MPC) parameters
Q_MPC = np.diag([20.0, 40.0, 20.0])
Q_T = np.diag([20.0, 40.0, 20.0])
R_MPC = np.diag([2.0, 2.0])
V_MAX = 0.4
V_MIN = 0.2 # V_MIN is set to ensure some forward movement, because unitree go2 won't move if v is too small. In simulation, it can be 0.
W_MAX = 1.0
HORIZON = N

# Rotation parameters
ROTATION_ANGLE = np.deg2rad(45)
ROTATION_DURATION = 1.0
