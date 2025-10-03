import rclpy
import argparse
from navigation_node import ExplorationNode

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: vint)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=4,
        type=int,
        help="index of waypoint for navigation (0 to number of waypoints predicted, default: 2)",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images (default: topomap)",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index in topomap (-1 for last node, default: -1)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="temporal distance to localize to next node (default: 3)",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="number of local nodes for localization (default: 4)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="number of actions sampled from exploration model (default: 8)",
    )
    parser.add_argument(
        "--depth_encoder",
        default="vits",
        type=str,
        help="depth encoder type (default: vitl, options: vits, vitb, vitl, vitg)",
    )
    parsed_args = parser.parse_args()
    print(f"Using {parsed_args.model} model with depth encoder {parsed_args.depth_encoder}")
    
    rclpy.init(args=[])
    exploration_node = ExplorationNode(parsed_args)
    rclpy.spin(exploration_node)
    exploration_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()