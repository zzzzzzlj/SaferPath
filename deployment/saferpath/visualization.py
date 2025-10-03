import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os

class VisualizationHandler:
    def __init__(self, interactive=False):
        self.interactive = interactive
        self.waypoint_artists = []
        self.trajectory_artists = []
        self.fig, self.ax = self._init_waypoint_fig()
        self.cost_fig, self.cost_ax = self._init_cost_fig()
        if self.interactive:
            self.fig.show()
            self.cost_fig.show()

    def _init_waypoint_fig(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('X Displacement (dx)')
        ax.set_ylabel('Y Displacement (dy)')
        ax.set_title('Waypoints Visualization')
        ax.grid(True)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal', adjustable='box')
        return fig, ax

    def _init_cost_fig(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title('Real-time Cost Map')
        ax.axis('off')
        return fig, ax

    def plot_waypoints(self, waypoints, chosen_waypoint, model_type='vint'):
        """Update waypoints dynamically, connecting trajectories for nomad."""
        # Remove previous artists
        for artist in self.waypoint_artists:
            artist.remove()
        self.waypoint_artists = []

        # Plot waypoints
        if waypoints is not None:
            if model_type == 'nomad':
                # Connect waypoints in each trajectory
                for sample in waypoints:  # sample: (len_traj_pred, 2) or (len_traj_pred, 4)
                    dx = sample[:, 0]
                    dy = sample[:, 1]
                    # Plot trajectory line
                    line, = self.ax.plot(dx, dy, 'b-', linewidth=1, alpha=0.5)
                    self.waypoint_artists.append(line)

                    # Plot points and arrows
                    for wp in sample:
                        if len(wp) == 4:
                            dx, dy, hx, hy = wp
                            point, = self.ax.plot(dx, dy, 'bo', markersize=5)
                            self.waypoint_artists.append(point)
                            arrow_length = 0.2
                            arrow = self.ax.arrow(dx, dy, hx * arrow_length, hy * arrow_length,
                                                  head_width=0.05, head_length=0.1, fc='r', ec='r')
                            self.waypoint_artists.append(arrow)
                        elif len(wp) == 2:
                            dx, dy = wp
                            point, = self.ax.plot(dx, dy, 'bo', markersize=5)
                            self.waypoint_artists.append(point)
            else:
                # Vint: plot discrete points
                for sub_goal in waypoints:
                    for wp in sub_goal:
                        if len(wp) == 4:
                            dx, dy, hx, hy = wp
                            point, = self.ax.plot(dx, dy, 'bo', markersize=5)
                            self.waypoint_artists.append(point)
                            arrow_length = 0.2
                            arrow = self.ax.arrow(dx, dy, hx * arrow_length, hy * arrow_length,
                                                  head_width=0.05, head_length=0.1, fc='r', ec='r')
                            self.waypoint_artists.append(arrow)
                        elif len(wp) == 2:
                            dx, dy = wp
                            point, = self.ax.plot(dx, dy, 'bo', markersize=5)
                            self.waypoint_artists.append(point)

        # Highlight chosen waypoint
        if chosen_waypoint is not None:
            if len(chosen_waypoint) == 4:
                dx, dy, hx, hy = chosen_waypoint
                point, = self.ax.plot(dx, dy, 'go', markersize=8, label='Chosen Waypoint')
                self.waypoint_artists.append(point)
                arrow_length = 0.2
                arrow = self.ax.arrow(dx, dy, hx * arrow_length, hy * arrow_length,
                                      head_width=0.08, head_length=0.15, fc='g', ec='g')
                self.waypoint_artists.append(arrow)
            elif len(chosen_waypoint) == 2:
                dx, dy = chosen_waypoint
                point, = self.ax.plot(dx, dy, 'go', markersize=8, label='Chosen Waypoint')
                self.waypoint_artists.append(point)
            if self.ax.get_legend() is not None:
                self.ax.get_legend().remove()
            self.ax.legend()

        # Save and update display
        self.fig.savefig('waypoints_plot.png')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_depth_map(self, cost_map_np):
        """Display cost map for the current observation."""
        # self.cost_ax.clear()
        # self.cost_ax.set_title('Real-time Cost Map')
        # self.cost_ax.axis('off')
        # self.cost_image = self.cost_ax.imshow(cost_map_np, cmap='binary')  # Use binary colormap for cost map (0: passable, 1: obstacle)
        
        # # MODIFIED: Save cost map figure
        # self.cost_fig.savefig('cost_map.png')  # Changed to save only cost map
        # self.cost_fig.canvas.draw()
        # self.cost_fig.canvas.flush_events()

        pass  # 原代码中绘图部分已注释，保持原样

    def close(self):
        plt.close(self.fig)
        plt.close(self.cost_fig)