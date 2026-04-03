from __future__ import annotations

from typing import Dict, List

import numpy as np

from jackal_diffusion.env import arena, dynamics

LEFT_WAYPOINTS = [(4.0, 8.7), (6.0, 8.7), (8.0, 8.7)]
RIGHT_WAYPOINTS = [(4.0, 3.3), (6.0, 3.3), (8.0, 3.3)]
MAX_STEPS = 200
WAYPOINT_TOL = 0.6
CRUISE_V_CMD = 0.6


def pure_pursuit(state: np.ndarray, waypoint: tuple[float, float]) -> np.ndarray:
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    dx = waypoint[0] - x
    dy = waypoint[1] - y
    target_heading = np.arctan2(dy, dx)
    heading_error = np.arctan2(
        np.sin(target_heading - theta),
        np.cos(target_heading - theta),
    )

    omega = 2.2 * heading_error
    omega_cmd = np.clip(omega / dynamics.MAX_ANGULAR_VEL, -1.0, 1.0)
    return np.array([CRUISE_V_CMD, omega_cmd], dtype=np.float32)


def collect_demo(side: str, seed: int) -> Dict:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    rng = np.random.default_rng(seed)
    x0 = float(arena.START[0] + rng.uniform(-0.3, 0.3))
    y0 = float(arena.START[1] + rng.uniform(-0.3, 0.3))
    theta0 = float(arena.START[2] + rng.uniform(-0.2, 0.2))
    state = np.array([x0, y0, theta0, 0.0, 0.0], dtype=np.float32)

    route = LEFT_WAYPOINTS if side == "left" else RIGHT_WAYPOINTS
    waypoints = list(route) + [tuple(arena.GOAL.tolist())]
    waypoint_idx = 0

    obs_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []

    for _ in range(MAX_STEPS):
        obs_list.append(arena.get_observation(state, arena.GOAL))

        waypoint = waypoints[waypoint_idx]
        if (
            np.hypot(state[0] - waypoint[0], state[1] - waypoint[1]) <= WAYPOINT_TOL
            and waypoint_idx < len(waypoints) - 1
        ):
            waypoint_idx += 1
            waypoint = waypoints[waypoint_idx]

        action = pure_pursuit(state, waypoint)
        action_list.append(action)

        next_state = dynamics.step(state, action)
        if arena.check_collision(next_state[0], next_state[1]):
            return {
                "obs": np.asarray(obs_list, dtype=np.float32),
                "action": np.asarray(action_list, dtype=np.float32),
                "side": side,
                "success": False,
            }

        state = next_state
        if arena.check_goal(state[0], state[1], arena.GOAL):
            return {
                "obs": np.asarray(obs_list, dtype=np.float32),
                "action": np.asarray(action_list, dtype=np.float32),
                "side": side,
                "success": True,
            }

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "action": np.asarray(action_list, dtype=np.float32),
        "side": side,
        "success": False,
    }


def collect_all(n_per_side: int = 25) -> list[Dict]:
    demos: list[Dict] = []
    for side, seed_base in (("left", 1000), ("right", 2000)):
        collected = 0
        attempts = 0
        while collected < n_per_side:
            demo = collect_demo(side=side, seed=seed_base + attempts)
            attempts += 1
            if demo["success"] and len(demo["obs"]) >= 12:
                demos.append(
                    {
                        "obs": demo["obs"],
                        "action": demo["action"],
                        "side": demo["side"],
                    }
                )
                collected += 1
            if attempts > 5000:
                raise RuntimeError(f"Failed to collect enough successful {side} demos.")
    return demos
