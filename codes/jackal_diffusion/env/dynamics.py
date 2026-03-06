import numpy as np

# Physical constants
WHEEL_SEPARATION = 0.262   # m (track width)
WHEEL_RADIUS = 0.098       # m
MAX_LINEAR_VEL = 2.0       # m/s
MAX_ANGULAR_VEL = 4.0      # rad/s
DT = 0.05                  # s (20 Hz control loop)
ROBOT_RADIUS = 0.27        # m (safety footprint)


def step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """
    First-order unicycle kinematics with a simple velocity filter
    to approximate Jackal's actual response lag.

    Args:
        state  : [x, y, theta, v, omega]
        action : [v_cmd, omega_cmd] both normalized to [-1, 1]

    Returns:
        next_state : [x, y, theta, v, omega]
    """
    x, y, theta, v, omega = state
    v_cmd = float(action[0]) * MAX_LINEAR_VEL
    omega_cmd = float(action[1]) * MAX_ANGULAR_VEL

    x_new = x + v_cmd * np.cos(theta) * DT
    y_new = y + v_cmd * np.sin(theta) * DT
    theta_new = np.arctan2(
        np.sin(theta + omega_cmd * DT),
        np.cos(theta + omega_cmd * DT),
    )

    # First-order lag filter (mimics Jackal's velocity controller)
    v_new = 0.85 * v + 0.15 * v_cmd
    omega_new = 0.85 * omega + 0.15 * omega_cmd

    return np.array([x_new, y_new, theta_new, v_new, omega_new], dtype=np.float32)
