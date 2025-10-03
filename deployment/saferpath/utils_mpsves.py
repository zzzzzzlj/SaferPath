import numpy as np
import math

def isPD_np(B):
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD_np(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD_np(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A3))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD_np(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def mpes(mean, C, noise_list, cost_list, eta, lamb, n_rep):
    S = np.array(cost_list)
    epsilon = np.stack(noise_list)
    maxS = np.max(S, axis=0)
    minS = np.min(S, axis=0)
    h = 10
    lamb = max(0.02, (maxS - minS)/h)

    expS = np.exp(-1/lamb*S)
    P = expS / np.sum(expS)

    d_mean = np.average(epsilon, axis=0, weights=P)
    mean = mean + eta * d_mean
    Cmat = np.matmul(np.expand_dims(epsilon, axis=2), np.expand_dims(epsilon, axis=1))
    d_C = np.sum(np.array([Cmat[i]*(P[i]-1/n_rep) for i in range(P.shape[0])]), axis=0)
    C = C + eta * d_C
    if not isPD_np(C):
        C = nearestPD_np(C)
    return mean, C

def project_point_numpy(x, y, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height):
    xyz = np.array([x + camera_x_offset, y, -camera_height])
    xyz_cv = np.array([xyz[1], -xyz[2], xyz[0]])
    x_prime = xyz_cv[0] / (xyz_cv[2] + 1e-6)
    y_prime = xyz_cv[1] / (xyz_cv[2] + 1e-6)
    k1, k2, p1, p2, k3 = dist_coeffs
    r2 = x_prime**2 + y_prime**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_dist = x_prime * radial + 2 * p1 * x_prime * y_prime + p2 * (r2 + 2 * x_prime**2)
    y_dist = y_prime * radial + p1 * (r2 + 2 * y_prime**2) + 2 * p2 * x_prime * y_prime
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    u = map_width - u
    u = np.clip(u, 0, map_width - 1)
    v = np.clip(v, 0, map_height - 1)
    return u, v

def cost_map_interpolation_numpy(u, v, map_data, map_width, map_height):
    u_idx = np.floor(u).astype(int)
    v_idx = np.floor(v).astype(int)
    u_idx = np.clip(u_idx, 0, map_width - 1)
    v_idx = np.clip(v_idx, 0, map_height - 1)
    idx = u_idx + v_idx * map_width
    idx = np.clip(idx, 0, len(map_data) - 1)
    cost = map_data[idx]
    return cost

def project_trajectory(X, ref_start_pose, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height, offset=0.0):
    u_list, v_list = [], []
    theta = ref_start_pose[2]
    half_width = offset
    for k in range(X.shape[1]):
        abs_x, abs_y, abs_theta = X[0, k], X[1, k], X[2, k]
        dx = abs_x - ref_start_pose[0]
        dy = abs_y - ref_start_pose[1]
        offset_x = half_width * -math.sin(abs_theta)
        offset_y = half_width * math.cos(abs_theta)
        dx_offset = dx + offset_x
        dy_offset = dy + offset_y
        rel_x = dx_offset * math.cos(theta) + dy_offset * math.sin(theta)
        rel_y = -dx_offset * math.sin(theta) + dy_offset * math.cos(theta)
        u, v = project_point_numpy(rel_x, rel_y, fx, fy, cx, cy, dist_coeffs, camera_height, camera_x_offset, map_width, map_height)
        u_list.append(u)
        v_list.append(v)
    return np.array(u_list), np.array(v_list)