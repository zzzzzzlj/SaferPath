import matplotlib.pyplot as plt
import numpy as np
from deployment.saferpath.config_controller import *
from deployment.saferpath.utils_mpsves import project_trajectory

class PlottingHandler:
    def __init__(self):
        self.traj_artists = []
        self.contour = None
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.axs[0].set_title('Optimized and MPC Trajectories vs Reference on Smooth Cost Map')
        self.axs[0].set_xlabel('u (pixels)')
        self.axs[0].set_ylabel('v (pixels)')
        self.axs[0].invert_yaxis()
        self.axs[1].set_title('Absolute Trajectories in World Coordinates')
        self.axs[1].set_xlabel('x (meters)')
        self.axs[1].set_ylabel('y (meters)')
        self.cost_image = None
        plt.ion()
        self.fig.show()

    def plot_trajectory_on_cost_map(self, cost_map, u_list, v_list, ref_start_pose, X_opt, current_traj, mpc_predicted_traj):
        # Clear both subplots
        self.axs[0].clear()
        self.axs[1].clear()
        
        # Set subplot titles and labels
        self.axs[0].set_title('Optimized and MPC Trajectories vs Reference on Smooth Cost Map')
        self.axs[0].set_xlabel('u (pixels)')
        self.axs[0].set_ylabel('v (pixels)')
        self.axs[0].invert_yaxis()
        self.axs[1].set_title('Absolute Trajectories in World Coordinates')
        self.axs[1].set_xlabel('x (meters)')
        self.axs[1].set_ylabel('y (meters)')

        # Plot cost map (left subplot)
        cost_map_2d = cost_map.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0])
        self.cost_image = self.axs[0].imshow(
            cost_map_2d, 
            cmap='RdBu',
            origin='upper', 
            extent=[0, IMAGE_SIZE[0], IMAGE_SIZE[1], 0],
            vmin=cost_map_2d.min(),
            vmax=cost_map_2d.max()
        )

        self.contour = self.axs[0].contour(
            np.linspace(0, IMAGE_SIZE[0], IMAGE_SIZE[0]),
            np.linspace(0, IMAGE_SIZE[1], IMAGE_SIZE[1]),
            cost_map_2d,
            levels=[0.0],
            colors='black',
            linewidths=2
        )

        self.axs[0].set_xlim(0, IMAGE_SIZE[0])
        self.axs[0].set_ylim(IMAGE_SIZE[1], 0)

        # Clear previous trajectory artists
        for artist in self.traj_artists:
            artist.remove()
        self.traj_artists = []

        # Plot optimized trajectory (left subplot, projected coordinates)
        line, = self.axs[0].plot(u_list, v_list, 'b-', linewidth=2, label='Optimized Trajectory')
        self.traj_artists.append(line)
        points, = self.axs[0].plot(u_list, v_list, 'ro', markersize=5, label='Optimized Points')
        self.traj_artists.append(points)

        # Plot robot width boundaries for optimized trajectory (left subplot)
        fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
        fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
        cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
        cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
        dist_coeffs = DIST_COEFFS
        camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
        camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]
        map_width = IMAGE_SIZE[0]
        map_height = IMAGE_SIZE[1]
        half_width = ROBOT_WIDTH / 2.0
        u_left, v_left = project_trajectory(X_opt, ref_start_pose, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height, offset=-half_width)
        u_right, v_right = project_trajectory(X_opt, ref_start_pose, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height, offset=half_width)
        left_line, = self.axs[0].plot(u_left, v_left, 'b--', linewidth=1, label='Optimized Left Boundary')
        self.traj_artists.append(left_line)
        right_line, = self.axs[0].plot(u_right, v_right, 'b--', linewidth=1, label='Optimized Right Boundary')
        self.traj_artists.append(right_line)

        # Plot reference trajectory (left subplot, projected coordinates)
        if current_traj is not None:
            ref_traj = current_traj[:, :min(current_traj.shape[1], N+1)]
            u_ref, v_ref = project_trajectory(ref_traj, ref_start_pose, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height)
            ref_line, = self.axs[0].plot(u_ref, v_ref, 'g--', linewidth=2, label='Reference Trajectory')
            self.traj_artists.append(ref_line)
            ref_points, = self.axs[0].plot(u_ref, v_ref, 'go', markersize=5, label='Reference Points')
            self.traj_artists.append(ref_points)

        # Plot absolute trajectories (right subplot, world coordinates)
        # Optimized trajectory
        opt_line, = self.axs[1].plot(X_opt[0, :], X_opt[1, :], 'b-', linewidth=2, label='Optimized Trajectory')
        self.traj_artists.append(opt_line)
        opt_points, = self.axs[1].plot(X_opt[0, :], X_opt[1, :], 'ro', markersize=5, label='Optimized Points')
        self.traj_artists.append(opt_points)

        # Reference trajectory
        if current_traj is not None:
            ref_traj = current_traj[:, :min(current_traj.shape[1], N+1)]
            ref_line, = self.axs[1].plot(ref_traj[0, :], ref_traj[1, :], 'g--', linewidth=2, label='Reference Trajectory')
            self.traj_artists.append(ref_line)
            ref_points, = self.axs[1].plot(ref_traj[0, :], ref_traj[1, :], 'go', markersize=5, label='Reference Points')
            self.traj_artists.append(ref_points)

        # Plot MPC predicted trajectory (right subplot, world coordinates)
        if mpc_predicted_traj is not None:
            pred_x = mpc_predicted_traj[:, 0]
            pred_y = mpc_predicted_traj[:, 1]
            
            pred_line, = self.axs[1].plot(
                pred_x,
                pred_y,
                color='darkorange', 
                linewidth=3, 
                linestyle='-',
                alpha=0.8,
                label='MPC Predicted Trajectory'
            )
            self.traj_artists.append(pred_line)
            
            pred_points, = self.axs[1].plot(
                pred_x,
                pred_y,
                color='orange',
                marker='D',
                markersize=6,
                linestyle='',
                alpha=0.9,
                label='MPC Predicted Points'
            )
            self.traj_artists.append(pred_points)
            
            start_point, = self.axs[1].plot(
                pred_x[0],
                pred_y[0],
                color='red',
                marker='s',
                markersize=10,
                label='MPC Prediction Start'
            )
            self.traj_artists.append(start_point)

        # Add legends
        self.axs[0].legend()
        self.axs[1].legend()

        # Add colorbar (left subplot only)
        if not hasattr(self.fig, 'colorbar'):
            self.fig.colorbar(self.cost_image, ax=self.axs[0], label='Cost (0 at boundary, >0 obstacles, <0 passable)')

        # Adjust layout and save
        self.fig.tight_layout()
        self.fig.savefig('optimized_trajectory.png')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
