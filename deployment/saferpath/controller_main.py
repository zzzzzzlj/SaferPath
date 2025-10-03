import rclpy
from controller_node import TrajectoryOptimizer

def main():
    rclpy.init()
    optimizer = TrajectoryOptimizer()
    try:
        rclpy.spin(optimizer)
    except KeyboardInterrupt:
        pass
    finally:
        optimizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()