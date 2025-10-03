import numpy as np
from deployment.saferpath.config_controller import *
from deployment.saferpath.utils_mpsves import mpes, project_trajectory, cost_map_interpolation_numpy

def trajectory_optimizer(current_state, x_ref, ref_start_pose, map_width, map_height, map_data, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset):
    mean = np.ones(2)
    C = np.array([[SIGMA_XY**2, 0], [0, SIGMA_XY**2]])
    min_costs_history = []
    final_wx_wy = None
    final_costs = None
    final_constraint_violated = None

    dx_ref = x_ref[0, :] - ref_start_pose[0]
    dy_ref = x_ref[1, :] - ref_start_pose[1]

    for iter_idx in range(NUM_ITER):
        delta_W = np.random.multivariate_normal([0, 0], C, K)
        wx = mean[0] + delta_W[:, 0]
        wy = mean[1] + delta_W[:, 1]

        Xs = np.zeros((K, N + 1, 3))
        for k in range(K):
            Xs[k, :, 0] = wx[k] * dx_ref + ref_start_pose[0]
            Xs[k, :, 1] = wy[k] * dy_ref + ref_start_pose[1]
            Xs[k, :, 2] = x_ref[2, :]

        costs = np.zeros(K)
        x_ref_expanded = x_ref.T[np.newaxis, :, :]
        err = Xs - x_ref_expanded
        state_costs = np.einsum('kti,ij,ktj->k', err, Q_BASE, err)

        dx = Xs[:, :, 0] - ref_start_pose[0]
        dy = Xs[:, :, 1] - ref_start_pose[1]
        theta = ref_start_pose[2]
        rel_x = dx * np.cos(theta) + dy * np.sin(theta)
        rel_x = np.maximum(rel_x, 0)
        rel_y = -dx * np.sin(theta) + dy * np.cos(theta)

        xyz = np.zeros((K, N + 1, 3))
        xyz[:, :, 0] = SCALE * rel_x + camera_x_offset
        xyz[:, :, 1] = SCALE * rel_y
        xyz[:, :, 2] = -camera_height
        xyz_cv = np.stack([xyz[:, :, 1], -xyz[:, :, 2], xyz[:, :, 0]], axis=-1)
        x_prime = xyz_cv[:, :, 0] / (xyz_cv[:, :, 2] + 1e-6)
        y_prime = xyz_cv[:, :, 1] / (xyz_cv[:, :, 2] + 1e-6)
        r2 = x_prime**2 + y_prime**2
        k1, k2, p1, p2, k3 = dist_coeffs
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x_dist = x_prime * radial + 2 * p1 * x_prime * y_prime + p2 * (r2 + 2 * x_prime**2)
        y_dist = y_prime * radial + p1 * (r2 + 2 * y_prime**2) + 2 * p2 * x_prime * y_prime
        u = fx * x_dist + cx
        v = fy * y_dist + cy
        u = map_width - u
        u = np.clip(u, 0, map_width - 1)
        v = np.clip(v, 0, map_height - 1)

        u_idx = np.floor(u).astype(int)
        v_idx = np.floor(v).astype(int)
        idx = u_idx + v_idx * map_width
        idx = np.clip(idx, 0, len(map_data) - 1)
        cost_map_costs = map_data[idx]
        cost_map_costs = MAP_WEIGHT * cost_map_costs
        constraint_violated = cost_map_costs > COST_THRESHOLD

        half_width = ROBOT_WIDTH / 2.0
        for side in [-1, 1]:
            offset_x = side * half_width * -np.sin(Xs[:, :, 2])
            offset_y = side * half_width * np.cos(Xs[:, :, 2])
            rel_x_offset = (dx + offset_x) * np.cos(theta) + (dy + offset_y) * np.sin(theta)
            rel_y_offset = -(dx + offset_x) * np.sin(theta) + (dy + offset_y) * np.cos(theta)
            rel_x_offset = np.maximum(rel_x_offset, 0)

            xyz_offset = np.zeros((K, N + 1, 3))
            xyz_offset[:, :, 0] = SCALE * rel_x_offset + camera_x_offset
            xyz_offset[:, :, 1] = SCALE * rel_y_offset
            xyz_offset[:, :, 2] = -camera_height
            xyz_cv_offset = np.stack([xyz_offset[:, :, 1], -xyz_offset[:, :, 2], xyz_offset[:, :, 0]], axis=-1)
            x_prime_offset = xyz_cv_offset[:, :, 0] / (xyz_cv_offset[:, :, 2] + 1e-6)
            y_prime_offset = xyz_cv_offset[:, :, 1] / (xyz_cv_offset[:, :, 2] + 1e-6)
            r2_offset = x_prime_offset**2 + y_prime_offset**2
            radial_offset = 1 + k1 * r2_offset + k2 * r2_offset**2 + k3 * r2_offset**3
            x_dist_offset = x_prime_offset * radial_offset + 2 * p1 * x_prime_offset * y_prime_offset + p2 * (r2_offset + 2 * x_prime_offset**2)
            y_dist_offset = y_prime_offset * radial_offset + p1 * (r2_offset + 2 * y_prime_offset**2) + 2 * p2 * x_prime_offset * y_prime_offset
            u_offset = fx * x_dist_offset + cx
            v_offset = fy * y_dist_offset + cy
            u_offset = map_width - u_offset
            u_offset = np.clip(u_offset, 0, map_width - 1)
            v_offset = np.clip(v_offset, 0, map_height - 1)

            u_idx_offset = np.floor(u_offset).astype(int)
            v_idx_offset = np.floor(v_offset).astype(int)
            idx_offset = u_idx_offset + v_idx_offset * map_width
            idx_offset = np.clip(idx_offset, 0, len(map_data) - 1)
            cost_offset = map_data[idx_offset]
            cost_map_costs[:, MAP_STARTPOINT:] += MAP_WEIGHT * cost_offset[:, MAP_STARTPOINT:]
            constraint_violated[:, MAP_STARTPOINT:] |= cost_offset[:, MAP_STARTPOINT:] > COST_THRESHOLD

        for m in range(1, NUM_INTERP_POINTS + 1):
            alpha = m / (NUM_INTERP_POINTS + 1.0)
            x_interp = (1 - alpha) * Xs[:, :-1, 0] + alpha * Xs[:, 1:, 0]
            y_interp = (1 - alpha) * Xs[:, :-1, 1] + alpha * Xs[:, 1:, 1]
            theta_interp = (1 - alpha) * Xs[:, :-1, 2] + alpha * Xs[:, 1:, 2]
            dx_interp = x_interp - ref_start_pose[0]
            dy_interp = y_interp - ref_start_pose[1]
            rel_x_interp = dx_interp * np.cos(theta) + dy_interp * np.sin(theta)
            rel_y_interp = -dx_interp * np.sin(theta) + dy_interp * np.cos(theta)

            xyz_interp = np.zeros((K, N, 3))
            xyz_interp[:, :, 0] = SCALE * rel_x_interp + camera_x_offset
            xyz_interp[:, :, 1] = SCALE * rel_y_interp
            xyz_interp[:, :, 2] = -camera_height
            xyz_cv_interp = np.stack([xyz_interp[:, :, 1], -xyz_interp[:, :, 2], xyz_interp[:, :, 0]], axis=-1)
            x_prime_interp = xyz_cv_interp[:, :, 0] / (xyz_cv_interp[:, :, 2] + 1e-6)
            y_prime_interp = xyz_cv_interp[:, :, 1] / (xyz_cv_interp[:, :, 2] + 1e-6)
            r2_interp = x_prime_interp**2 + y_prime_interp**2
            radial_interp = 1 + k1 * r2_interp + k2 * r2_interp**2 + k3 * r2_interp**3
            x_dist_interp = x_prime_interp * radial_interp + 2 * p1 * x_prime_interp * y_prime_interp + p2 * (r2_interp + 2 * x_prime_interp**2)
            y_dist_interp = y_prime_interp * radial_interp + p1 * (r2_interp + 2 * y_prime_interp**2) + 2 * p2 * x_prime_interp * y_prime_interp
            u_interp = fx * x_dist_interp + cx
            v_interp = fy * y_dist_interp + cy
            u_interp = map_width - u_interp
            u_interp = np.clip(u_interp, 0, map_width - 1)
            v_interp = np.clip(v_interp, 0, map_height - 1)
            u_idx_interp = np.floor(u_interp).astype(int)
            v_idx_interp = np.floor(v_interp).astype(int)
            idx_interp = u_idx_interp + v_idx_interp * map_width
            idx_interp = np.clip(idx_interp, 0, len(map_data) - 1)
            cost_interp = map_data[idx_interp]
            cost_map_costs[:, :-1] += MAP_WEIGHT * cost_interp
            constraint_violated[:, :-1] |= cost_interp > COST_THRESHOLD

            for side in [-1, 1]:
                offset_x_interp = side * half_width * -np.sin(theta_interp)
                offset_y_interp = side * half_width * np.cos(theta_interp)
                rel_x_offset_interp = (dx_interp + offset_x_interp) * np.cos(theta) + (dy_interp + offset_y_interp) * np.sin(theta)
                rel_y_offset_interp = -(dx_interp + offset_x_interp) * np.sin(theta) + (dy_interp + offset_y_interp) * np.cos(theta)
                rel_x_offset_interp = np.maximum(rel_x_offset_interp, 0)
                xyz_offset_interp = np.zeros((K, N, 3))
                xyz_offset_interp[:, :, 0] = SCALE * rel_x_offset_interp + camera_x_offset
                xyz_offset_interp[:, :, 1] = SCALE * rel_y_offset_interp
                xyz_offset_interp[:, :, 2] = -camera_height
                xyz_cv_offset_interp = np.stack([xyz_offset_interp[:, :, 1], -xyz_offset_interp[:, :, 2], xyz_offset_interp[:, :, 0]], axis=-1)
                x_prime_offset_interp = xyz_cv_offset_interp[:, :, 0] / (xyz_cv_offset_interp[:, :, 2] + 1e-6)
                y_prime_offset_interp = xyz_cv_offset_interp[:, :, 1] / (xyz_cv_offset_interp[:, :, 2] + 1e-6)
                r2_offset_interp = x_prime_offset_interp**2 + y_prime_offset_interp**2
                radial_offset_interp = 1 + k1 * r2_offset_interp + k2 * r2_offset_interp**2 + k3 * r2_offset_interp**3
                x_dist_offset_interp = x_prime_offset_interp * radial_offset_interp + 2 * p1 * x_prime_offset_interp * y_prime_offset_interp + p2 * (r2_offset_interp + 2 * x_prime_offset_interp**2)
                y_dist_offset_interp = y_prime_offset_interp * radial_offset_interp + p1 * (r2_offset_interp + 2 * y_prime_offset_interp**2) + 2 * p2 * x_prime_offset_interp * y_prime_offset_interp
                u_offset_interp = fx * x_dist_offset_interp + cx
                v_offset_interp = fy * y_dist_offset_interp + cy
                u_offset_interp = map_width - u_offset_interp
                u_offset_interp = np.clip(u_offset_interp, 0, map_width - 1)
                v_offset_interp = np.clip(v_offset_interp, 0, map_height - 1)
                u_idx_offset_interp = np.floor(u_offset_interp).astype(int)
                v_idx_offset_interp = np.floor(v_offset_interp).astype(int)
                idx_offset_interp = u_idx_offset_interp + v_idx_offset_interp * map_width
                idx_offset_interp = np.clip(idx_offset_interp, 0, len(map_data) - 1)
                cost_offset_interp = map_data[idx_offset_interp]
                cost_map_costs[:, MAP_STARTPOINT-1:-1] += MAP_WEIGHT * cost_offset_interp[:, MAP_STARTPOINT-1:]
                constraint_violated[:, MAP_STARTPOINT-1:-1] |= cost_offset_interp[:, MAP_STARTPOINT-1:] > COST_THRESHOLD

        smoothness_cost = np.sum((Xs[:, 1:, :] - Xs[:, :-1, :])**2, axis=(1, 2)) * SMOOTHNESS_WEIGHT
        constraint_penalty = np.sum(np.maximum(cost_map_costs - COST_THRESHOLD, 0), axis=1) * CONSTRAINT_PENALTY
        costs = state_costs + np.sum(cost_map_costs, axis=1) + smoothness_cost + constraint_penalty

        valid_indices = ~np.any(constraint_violated, axis=1)
        noise_list = [delta_W[k] for k in range(K)]

        # mpes update
        mean, C = mpes(
            mean, C, noise_list, costs,
            eta=ETA, lamb=LAMBDA, n_rep=len(noise_list)
        )

        min_cost = np.min(costs) if len(costs) > 0 else np.inf
        min_costs_history.append(min_cost if min_cost != np.inf else None)

        if iter_idx == NUM_ITER - 1:
            final_wx_wy = np.stack([wx, wy], axis=1)
            final_costs = costs
            final_constraint_violated = constraint_violated

    best_cost = np.inf
    best_wx_wy = np.array([1.0, 1.0])
    found_valid = False
    valid_indices = ~np.any(final_constraint_violated, axis=1)
    valid_costs = final_costs[valid_indices]
    valid_wx_wy = final_wx_wy[valid_indices]
    if len(valid_costs) > 0:
        min_idx = np.argmin(valid_costs)
        best_cost = valid_costs[min_idx]
        best_wx_wy = valid_wx_wy[min_idx]
        found_valid = True

    if not found_valid:
        print("Warning: No trajectories satisfy constraints in final iteration!")
        return {'X': x_ref, 'min_costs_history': min_costs_history, 'valid': False}

    X_opt = np.zeros_like(x_ref)
    X_opt[0, :] = best_wx_wy[0] * dx_ref + ref_start_pose[0]
    X_opt[1, :] = best_wx_wy[1] * dy_ref + ref_start_pose[1]
    X_opt[2, :] = x_ref[2, :]

    u_list, v_list = project_trajectory(X_opt, ref_start_pose, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height)
    for i, (u, v, x, y) in enumerate(zip(u_list, v_list, X_opt[0, :], X_opt[1, :])):
        cost = cost_map_interpolation_numpy(u, v, map_data, map_width, map_height)

    return {'X': X_opt, 'min_costs_history': min_costs_history, 'valid': True}