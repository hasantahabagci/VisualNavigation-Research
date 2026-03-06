import numpy as np

from . import dynamics

ARENA_SIZE = 12.0
START = np.array([1.0, 6.0, 0.0], dtype=np.float32)
GOAL = np.array([11.0, 6.0], dtype=np.float32)

OBSTACLE = dict(cx=6.0, cy=6.0, w=1.5, h=4.0)
GOAL_RADIUS = 0.5

# ---------- range sensor configuration ----------
N_RAYS = 8                       # rays equally spaced around the robot
MAX_RANGE = ARENA_SIZE           # max reading (metres)
_RAY_ANGLES = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)  # relative to heading


def _ray_segment_intersection(
    ox: float, oy: float, dx: float, dy: float,
    ax: float, ay: float, bx: float, by: float,
) -> float:
    """
    Intersect ray (ox,oy)+t*(dx,dy)  with line segment A-B.
    Returns t >= 0 if hit, else inf.
    """
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
    """
    Cast N_RAYS rays from (x, y) at angles evenly spaced around the robot
    heading.  Each ray returns the distance to the nearest surface (obstacle
    edge or arena wall), capped at MAX_RANGE.

    Returns: np.ndarray of shape (N_RAYS,)
    """
    # Pre-compute obstacle edges (4 segments, CCW)
    hw = OBSTACLE["w"] / 2.0
    hh = OBSTACLE["h"] / 2.0
    cx, cy = OBSTACLE["cx"], OBSTACLE["cy"]
    obs_segs = [
        (cx - hw, cy - hh, cx + hw, cy - hh),  # bottom
        (cx + hw, cy - hh, cx + hw, cy + hh),  # right
        (cx + hw, cy + hh, cx - hw, cy + hh),  # top
        (cx - hw, cy + hh, cx - hw, cy - hh),  # left
    ]
    # Arena wall segments
    S = ARENA_SIZE
    wall_segs = [
        (0, 0, S, 0),   # bottom wall
        (S, 0, S, S),   # right wall
        (S, S, 0, S),   # top wall
        (0, S, 0, 0),   # left wall
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


OBS_DIM = 8 + N_RAYS  # 16 with default N_RAYS=8


def check_collision(x: float, y: float) -> bool:
    """AABB check expanded by robot radius and arena bounds."""
    r = dynamics.ROBOT_RADIUS

    if x < r or x > ARENA_SIZE - r or y < r or y > ARENA_SIZE - r:
        return True

    left = OBSTACLE["cx"] - OBSTACLE["w"] / 2.0 - r
    right = OBSTACLE["cx"] + OBSTACLE["w"] / 2.0 + r
    bottom = OBSTACLE["cy"] - OBSTACLE["h"] / 2.0 - r
    top = OBSTACLE["cy"] + OBSTACLE["h"] / 2.0 + r
    return left <= x <= right and bottom <= y <= top


def check_goal(x: float, y: float, goal: np.ndarray) -> bool:
    return np.hypot(x - float(goal[0]), y - float(goal[1])) <= GOAL_RADIUS


def get_observation(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    Returns (8 + N_RAYS)-dim observation vector:
    [x, y, theta, v, omega, goal_dx, goal_dy, goal_dist, range_0, ..., range_{N-1}]
    """
    x, y, theta, v, omega = state
    goal_dx = float(goal[0]) - x
    goal_dy = float(goal[1]) - y
    goal_dist = np.hypot(goal_dx, goal_dy)
    ranges = get_range_readings(float(x), float(y), float(theta))
    return np.concatenate([
        np.array([x, y, theta, v, omega, goal_dx, goal_dy, goal_dist], dtype=np.float32),
        ranges,
    ])
