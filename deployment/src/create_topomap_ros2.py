import os
import time
import shutil
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

IMAGE_TOPIC = "/camera/color/image_raw"
TOPOMAP_IMAGES_DIR = "../topomaps/images"


class TopomapGenerator(Node):
    def __init__(self, args):
        super().__init__("CREATE_TOPOMAP")
        self.args = args
        self.obs_img = None
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 10)

        self.subgoals_pub = self.create_publisher(Image, "/subgoals", 10)

        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            self.get_logger().info(f"{self.topomap_name_dir} already exists, clearing old images...")
            self.remove_files_in_dir(self.topomap_name_dir)

        self.start_time = float("inf")
        self.i = 0
        self.timer = self.create_timer(args.dt, self.timer_callback)

    def remove_files_in_dir(self, dir_path: str):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                self.get_logger().error(f"Failed to delete {file_path}, reason: {e}")

    def callback_obs(self, msg: Image):
        self.obs_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def timer_callback(self):
        if self.obs_img is not None:
            image_path = os.path.join(self.topomap_name_dir, f"{self.i}.png")
            cv2.imwrite(image_path, self.obs_img)
            self.get_logger().info(f"Published image {self.i}")
            self.i += 1
            self.obs_img = None

        if time.time() - self.start_time > 2 * self.args.dt:
            self.get_logger().warn(f"Topic {IMAGE_TOPIC} stopped publishing images, shutting down node...")
            rclpy.shutdown()

    def main(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Generate topomap images from /usb_cam/image_raw topic")
    parser.add_argument(
        "--dir", "-d", default="topomap", type=str,
        help="Path to store topomap images (default: topomap)"
    )
    parser.add_argument(
        "--dt", "-t", default=1.0, type=float,
        help="Time interval to capture images from /usb_cam/image_raw topic (default: 1.0 seconds)"
    )
    args = parser.parse_args()

    topomap_gen = TopomapGenerator(args)
    topomap_gen.main()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
