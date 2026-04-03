from __future__ import annotations

import numpy as np

WHEEL_SEPARATION = 0.262
WHEEL_RADIUS = 0.098
MAX_LINEAR_VEL = 2.0
MAX_ANGULAR_VEL = 4.0
DT = 0.05
ROBOT_RADIUS = 0.27


def step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    x, y, theta, v, omega = state
    v_cmd = float(action[0]) * MAX_LINEAR_VEL
    omega_cmd = float(action[1]) * MAX_ANGULAR_VEL

    x_new = x + v_cmd * np.cos(theta) * DT
    y_new = y + v_cmd * np.sin(theta) * DT
    theta_new = np.arctan2(
        np.sin(theta + omega_cmd * DT),
        np.cos(theta + omega_cmd * DT),
    )

    v_new = 0.85 * v + 0.15 * v_cmd
    omega_new = 0.85 * omega + 0.15 * omega_cmd
    return np.array([x_new, y_new, theta_new, v_new, omega_new], dtype=np.float32)
