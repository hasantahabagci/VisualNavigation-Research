from __future__ import annotations

import numpy as np

from jackal_diffusion.env import dynamics

ARENA_SIZE = 12.0
START = np.array([1.0, 6.0, 0.0], dtype=np.float32)
GOAL = np.array([11.0, 6.0], dtype=np.float32)

OBSTACLE = dict(cx=6.0, cy=6.0, w=1.5, h=4.0)
GOAL_RADIUS = 0.5

N_RAYS = 8
MAX_RANGE = ARENA_SIZE
_RAY_ANGLES = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)


def _ray_segment_intersection(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    ex, ey = bx - ax, by - ay
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return np.inf
    t = ((ax - ox) * ey - (ay - oy) * ex) / denom
    s = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0 and 0 <= s <= 1:
        return t
    return np.inf


def get_range_readings(x: float, y: float, theta: float) -> np.ndarray:
    hw = OBSTACLE["w"] / 2.0
    hh = OBSTACLE["h"] / 2.0
    cx, cy = OBSTACLE["cx"], OBSTACLE["cy"]
    obs_segs = [
        (cx - hw, cy - hh, cx + hw, cy - hh),
        (cx + hw, cy - hh, cx + hw, cy + hh),
        (cx + hw, cy + hh, cx - hw, cy + hh),
        (cx - hw, cy + hh, cx - hw, cy - hh),
    ]
    size = ARENA_SIZE
    wall_segs = [
        (0, 0, size, 0),
        (size, 0, size, size),
        (size, size, 0, size),
        (0, size, 0, 0),
    ]
    all_segs = obs_segs + wall_segs

    readings = np.full(N_RAYS, MAX_RANGE, dtype=np.float32)
    for i, rel_angle in enumerate(_RAY_ANGLES):
        angle = theta + rel_angle
        dx = np.cos(angle)
        dy = np.sin(angle)
        best = MAX_RANGE
        for ax, ay, bx, by in all_segs:
            t = _ray_segment_intersection(x, y, dx, dy, ax, ay, bx, by)
            if t < best:
                best = t
        readings[i] = min(best, MAX_RANGE)
    return readings


OBS_DIM = 8 + N_RAYS


def check_collision(x: float, y: float) -> bool:
    radius = dynamics.ROBOT_RADIUS
    if x < radius or x > ARENA_SIZE - radius or y < radius or y > ARENA_SIZE - radius:
        return True
    left = OBSTACLE["cx"] - OBSTACLE["w"] / 2.0 - radius
    right = OBSTACLE["cx"] + OBSTACLE["w"] / 2.0 + radius
    bottom = OBSTACLE["cy"] - OBSTACLE["h"] / 2.0 - radius
    top = OBSTACLE["cy"] + OBSTACLE["h"] / 2.0 + radius
    return left <= x <= right and bottom <= y <= top


def check_goal(x: float, y: float, goal: np.ndarray) -> bool:
    return np.hypot(x - float(goal[0]), y - float(goal[1])) <= GOAL_RADIUS


def get_observation(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    x, y, theta, v, omega = state
    goal_dx = float(goal[0]) - x
    goal_dy = float(goal[1]) - y
    goal_dist = np.hypot(goal_dx, goal_dy)
    ranges = get_range_readings(float(x), float(y), float(theta))
    return np.concatenate(
        [
            np.array(
                [x, y, theta, v, omega, goal_dx, goal_dy, goal_dist],
                dtype=np.float32,
            ),
            ranges,
        ]
    )
