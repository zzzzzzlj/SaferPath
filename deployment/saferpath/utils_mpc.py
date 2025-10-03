import numpy as np
from scipy.optimize import minimize
from deployment.saferpath.config_controller import dt, V_MAX, W_MAX, Q_MPC, Q_T, R_MPC, HORIZON
from deployment.src.utils import clip_angle

def predict_trajectory(x0, u):
    """Predict MPC trajectory based on differential drive model."""
    x = np.zeros((HORIZON + 1, 3))
    x[0] = x0
    for t in range(HORIZON):
        v, w = u[t]
        x[t + 1, 0] = x[t, 0] + v * np.cos(x[t, 2]) * dt
        x[t + 1, 1] = x[t, 1] + v * np.sin(x[t, 2]) * dt
        x[t + 1, 2] = x[t, 2] + w * dt
        x[t + 1, 2] = clip_angle(x[t + 1, 2])
    return x

def mpc_cost(u_flat, x0, x_ref):
    """MPC cost function including terminal cost."""
    u = u_flat.reshape(HORIZON, 2)
    x = predict_trajectory(x0, u)

    # Stage cost
    state_error = x - x_ref.T
    state_cost = np.sum(np.einsum('ti,ij,tj->t', state_error, Q_MPC, state_error))
    
    # Terminal cost
    terminal_error = x[-1] - x_ref[:, -1]
    terminal_cost = terminal_error.T @ Q_T @ terminal_error
    
    # Control cost
    control_cost = np.sum(np.einsum('ti,ij,tj->t', u, R_MPC, u))
    if HORIZON > 1:
        control_diff = u[1:] - u[:-1]
        control_cost += np.sum(np.einsum('ti,ij,tj->t', control_diff, R_MPC, control_diff))
    
    total_cost = state_cost + control_cost + terminal_cost
    return total_cost

def mpc_optimize(x0, x_ref):
    """Perform MPC optimization."""
    u0 = np.zeros((HORIZON, 2))  # Initial control input
    bounds = [(-V_MAX, V_MAX), (-W_MAX, W_MAX)] * HORIZON

    result = minimize(
        fun=mpc_cost,
        x0=u0.flatten(),
        args=(x0, x_ref),
        method='SLSQP',
        bounds=bounds,
    )
    u_opt = result.x.reshape(HORIZON, 2)
    predicted_states = predict_trajectory(x0, u_opt)
    return u_opt[0], predicted_states  # Return the first control input
