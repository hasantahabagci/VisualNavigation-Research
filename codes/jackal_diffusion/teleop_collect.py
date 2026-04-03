#!/usr/bin/env python3
"""
Jackal Diffusion — Teleop Dataset Collector
============================================
Generates a random (but sensible) start / goal / obstacle for each episode.
The user drives the robot with the keyboard; all episodes are saved to disk.

Controls
--------
  W / ↑      Forward
  S / ↓      Backward
  A / ←      Turn left
  D / →      Turn right
  Enter      Save episode + new scene
  R          Discard episode + new scene
  Q / Esc    Quit and save

Output: data/teleop_demos.npz
Loading example:
    data = np.load("data/teleop_demos.npz", allow_pickle=True)
    n    = int(data["n_demos"])
    demos = [
        {
            "obs":      data[f"obs_{i}"],       # (T, OBS_DIM) float32
            "action":   data[f"action_{i}"],     # (T, 2)       float32
            "goal":     data[f"goal_{i}"],        # (2,)         float32
            "start":    data[f"start_{i}"],       # (3,)         float32  [x,y,θ]
            "obstacle": data[f"obstacle_{i}"],    # (4,)         float32  [cx,cy,w,h]
            "success":  bool(data[f"success_{i}"]),
        }
        for i in range(n)
    ]
Running
-------
    cd codes/jackal_diffusion
    python teleop_collect.py [--seed 42] [--save data/teleop_demos.npz] [--fps 20]
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

import env.arena as _arena
from env import dynamics

# ── tunables ─────────────────────────────────────────────────────────────────
ARENA_S      = _arena.ARENA_SIZE
ROBOT_R      = dynamics.ROBOT_RADIUS
DEFAULT_FPS  = 20
DEFAULT_SAVE = "data/teleop_demos.npz"

# Normalised command magnitudes (all values in [-1, 1])
V_FWD        = 0.60   # forward command
V_BWD        = 0.30   # backward command (absolute value)
OMEGA_TURN   = 0.75   # turning command magnitude

# Clearance around the obstacle for safe point sampling (robot radius + buffer)
SAMPLE_MARGIN = ROBOT_R + 0.50


# ── sensible scene generation ───────────────────────────────────────────────

def _in_obstacle(cx: float, cy: float, w: float, h: float,
                  px: float, py: float, margin: float = 0.0) -> bool:
    """Return True if the point (px, py) is inside the obstacle AABB (with optional margin)."""
    return (abs(px - cx) <= w / 2.0 + margin and
            abs(py - cy) <= h / 2.0 + margin)


def _sample_free_point(
    rng,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    cx: float, cy: float, w: float, h: float,
    margin: float,
) -> tuple[float, float]:
    """Sample a random point outside the obstacle AABB."""
    for _ in range(400):
        x = float(rng.uniform(*x_range))
        y = float(rng.uniform(*y_range))
        if not _in_obstacle(cx, cy, w, h, x, y, margin):
            return x, y
    raise RuntimeError("Could not find an obstacle-free point (400 attempts)")


def random_scenario(rng) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Constraints:
      • Start   : left  region  x ∈ [0.8, 4.5],  y ∈ [0.8, 11.2]
      • Goal    : right region  x ∈ [7.5, 11.2], y ∈ [0.8, 11.2]
      • Obstacle: middle region cx ∈ [4.0, 8.0],  cy ∈ [2.5, 9.5]
      • Obstacle must leave ≥ 1.2 m passage on the top OR bottom
      • ‖start − goal‖₂ ≥ 5 m
      • Initial heading (θ) close to goal direction (±0.5 rad)
    """
    for _ in range(600):
        cx = float(rng.uniform(4.0, 8.0))
        cy = float(rng.uniform(2.5, 9.5))
        w  = float(rng.uniform(1.0, 2.5))
        h  = float(rng.uniform(2.0, 5.0))

        # Obstacle must not block the entire arena: at least 1.2 m gap top or bottom
        top_gap = ARENA_S - (cy + h / 2.0)
        bot_gap = cy - h / 2.0
        if top_gap < 1.2 and bot_gap < 1.2:
            continue

        try:
            sx, sy = _sample_free_point(rng, (0.8, 4.5),  (0.8, 11.2), cx, cy, w, h, SAMPLE_MARGIN)
            gx, gy = _sample_free_point(rng, (7.5, 11.2), (0.8, 11.2), cx, cy, w, h, SAMPLE_MARGIN)
        except RuntimeError:
            continue

        if np.hypot(gx - sx, gy - sy) < 5.0:
            continue

        # Goal direction + small noise
        base_theta = float(np.arctan2(gy - sy, gx - sx))
        theta = float(np.arctan2(
            np.sin(base_theta + rng.uniform(-0.5, 0.5)),
            np.cos(base_theta + rng.uniform(-0.5, 0.5)),
        ))

        start    = np.array([sx, sy, theta], dtype=np.float32)
        goal     = np.array([gx, gy],        dtype=np.float32)
        obstacle = dict(cx=cx, cy=cy, w=w, h=h)
        return start, goal, obstacle

    # Fallback: default configuration
    return _arena.START.copy(), _arena.GOAL.copy(), dict(**_arena.OBSTACLE)


def _apply_obstacle(obs_dict: dict) -> None:
    """Patch the arena module's global OBSTACLE dict in-place."""
    _arena.OBSTACLE["cx"] = obs_dict["cx"]
    _arena.OBSTACLE["cy"] = obs_dict["cy"]
    _arena.OBSTACLE["w"]  = obs_dict["w"]
    _arena.OBSTACLE["h"]  = obs_dict["h"]


# ── keyboard → action ───────────────────────────────────────────────────────

def action_from_keys(pressed: set) -> np.ndarray:
    v, omega = 0.0, 0.0
    if "w" in pressed or "up" in pressed:
        v += V_FWD
    if "s" in pressed or "down" in pressed:
        v -= V_BWD
    if "a" in pressed or "left" in pressed:
        omega += OMEGA_TURN
    if "d" in pressed or "right" in pressed:
        omega -= OMEGA_TURN
    return np.array(
        [np.clip(v, -1.0, 1.0), np.clip(omega, -1.0, 1.0)],
        dtype=np.float32,
    )


# ── main class ───────────────────────────────────────────────────────────────

class TeleopCollector:
    def __init__(self, seed: int, save_path: str, fps: int) -> None:
        self.save_path = save_path
        self.fps       = fps
        self.rng       = np.random.default_rng(seed)
        self.pressed: set = set()

        # persistent storage — load existing file so new episodes are appended
        self.all_demos: List[Dict] = self._load_existing(save_path)

        # episode state
        self._ep_obs: List[np.ndarray] = []
        self._ep_act: List[np.ndarray] = []
        self.state:    np.ndarray      = np.zeros(5, dtype=np.float32)
        self.goal:     np.ndarray      = np.zeros(2, dtype=np.float32)
        self.obstacle: dict            = {}
        self.start:    np.ndarray      = np.zeros(3, dtype=np.float32)
        self.traj:     List[tuple]     = []
        self.ep_done   = False
        self.ep_status = "running"   # "running" | "success" | "collision"  (English keys)

        self._build_fig()
        self._reset()

    # ── episode management ──────────────────────────────────────────────────

    @staticmethod
    def _load_existing(save_path: str) -> "List[Dict]":
        """Return demos already stored in *save_path*, or an empty list."""
        if not os.path.isfile(save_path):
            return []
        try:
            data = np.load(save_path, allow_pickle=True)
            n = int(data["n_demos"])
            demos = [
                {
                    "obs":      data[f"obs_{i}"],
                    "action":   data[f"action_{i}"],
                    "goal":     data[f"goal_{i}"],
                    "start":    data[f"start_{i}"],
                    "obstacle": data[f"obstacle_{i}"],
                    "success":  bool(data[f"success_{i}"]),
                }
                for i in range(n)
            ]
            print(f"[Loaded]  {n} existing episodes from {save_path}")
            return demos
        except Exception as exc:
            print(f"[Warning] Could not load existing file ({exc}); starting fresh.")
            return []

    def _reset(self) -> None:
        start, goal, obs_dict = random_scenario(self.rng)
        _apply_obstacle(obs_dict)
        self.start    = start
        self.goal     = goal
        self.obstacle = obs_dict
        self.state    = np.array(
            [start[0], start[1], start[2], 0.0, 0.0], dtype=np.float32
        )
        self.traj      = [(float(start[0]), float(start[1]))]
        self._ep_obs   = []
        self._ep_act   = []
        self.ep_done   = False
        self.ep_status = "running"
        self._refresh_static()

    def _save_ep(self, success: bool) -> None:
        if len(self._ep_obs) < 5:
            print("[Skip]  Episode too short (< 5 steps), discarded.")
            return
        self.all_demos.append({
            "obs":      np.asarray(self._ep_obs, dtype=np.float32),
            "action":   np.asarray(self._ep_act, dtype=np.float32),
            "goal":     self.goal.copy(),
            "start":    self.start.copy(),
            "obstacle": np.array(
                [self.obstacle["cx"], self.obstacle["cy"],
                 self.obstacle["w"],  self.obstacle["h"]],
                dtype=np.float32,
            ),
            "success": success,
        })
        print(
            f"[Ep {len(self.all_demos):>3}]  "
            f"T={len(self._ep_obs):>4} steps | success={success}"
        )

    def _persist(self) -> None:
        if not self.all_demos:
            print("[Info]  No episodes to save.")
            return
        save_dir = os.path.dirname(os.path.abspath(self.save_path))
        os.makedirs(save_dir, exist_ok=True)

        npz_data: dict = {"n_demos": np.array(len(self.all_demos))}
        for i, d in enumerate(self.all_demos):
            npz_data[f"obs_{i}"]      = d["obs"]
            npz_data[f"action_{i}"]   = d["action"]
            npz_data[f"goal_{i}"]     = d["goal"]
            npz_data[f"start_{i}"]    = d["start"]
            npz_data[f"obstacle_{i}"] = d["obstacle"]
            npz_data[f"success_{i}"]  = np.array(d["success"])

        np.savez(self.save_path, **npz_data)
        print(f"[Saved]  {len(self.all_demos)} episodes → {self.save_path}")

    # ── keyboard events ────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        key = (event.key or "").lower()
        self.pressed.add(key)

        if key in ("enter", "return"):
            if not self.ep_done:
                self._save_ep(success=False)
            self._reset()

        elif key == "r":
            self._reset()

        elif key in ("q", "escape"):
            if not self.ep_done and self._ep_obs:
                self._save_ep(success=False)
            self._persist()
            plt.close(self.fig)

    def _on_release(self, event) -> None:
        self.pressed.discard((event.key or "").lower())

    # ── matplotlib setup ───────────────────────────────────────────────────

    def _build_fig(self) -> None:
        self.fig, (self.ax, self.ax_hud) = plt.subplots(
            1, 2, figsize=(13, 7),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        self.fig.patch.set_facecolor("#1e1e2e")
        self.fig.canvas.mpl_connect("key_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_release)

        # ── Arena axis
        ax = self.ax
        ax.set_facecolor("#13131f")
        ax.set_xlim(-0.4, ARENA_S + 0.4)
        ax.set_ylim(-0.4, ARENA_S + 0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", color="white")
        ax.set_ylabel("y [m]", color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        # Arena boundary
        boundary = Rectangle(
            (0, 0), ARENA_S, ARENA_S,
            fill=False, ec="#666", lw=1.5, ls="--",
        )
        ax.add_patch(boundary)

        # Dynamic artists
        self._patch_obs  = Rectangle((0, 0), 1, 1, color="#e06c75", alpha=0.85, zorder=2)
        ax.add_patch(self._patch_obs)

        (self._ln_traj,) = ax.plot([], [], "-",  color="#61afef", lw=1.2, alpha=0.7, zorder=3)
        (self._pt_start,) = ax.plot([], [], "o", color="#98c379", ms=9,              zorder=4)
        (self._pt_goal,)  = ax.plot([], [], "*", color="#e5c07b", ms=14,             zorder=4)
        self._circ_goal = Circle(
            (0, 0), _arena.GOAL_RADIUS,
            color="#e5c07b", fill=False, ls="--", lw=1.2, alpha=0.6, zorder=4,
        )
        ax.add_patch(self._circ_goal)

        (self._pt_robot,) = ax.plot([], [], "o", color="#c678dd", ms=11, zorder=6)
        (self._ln_head,)  = ax.plot([], [], "-", color="#c678dd", lw=2.5, zorder=7)

        self._title = ax.set_title(
            "Jackal Teleop | saved: 0", color="white", fontsize=11,
        )

        # ── HUD axis
        hud = self.ax_hud
        hud.set_facecolor("#13131f")
        hud.axis("off")

        hud.text(
            0.05, 0.99,
            "Controls\n"
            "────────────────\n"
            "W / ↑    Forward\n"
            "S / ↓    Backward\n"
            "A / ←    Turn left\n"
            "D / →    Turn right\n"
            "Enter    Save + New\n"
            "R        Discard + New\n"
            "Q/Esc    Quit & Save",
            transform=hud.transAxes,
            va="top", color="white", fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#282c36", ec="#555"),
        )

        self._txt_info = hud.text(
            0.05, 0.40, "",
            transform=hud.transAxes,
            va="top", color="#abb2bf", fontsize=10, fontfamily="monospace",
        )
        plt.tight_layout()

    def _refresh_static(self) -> None:
        """Update static artists when a new scene is set up."""
        obs = self.obstacle
        self._patch_obs.set_xy((obs["cx"] - obs["w"] / 2, obs["cy"] - obs["h"] / 2))
        self._patch_obs.set_width(obs["w"])
        self._patch_obs.set_height(obs["h"])

        self._pt_start.set_data([self.start[0]], [self.start[1]])
        self._pt_goal.set_data([self.goal[0]], [self.goal[1]])
        self._circ_goal.set_center((float(self.goal[0]), float(self.goal[1])))
        self._ln_traj.set_data([], [])
        self._title.set_text(
            f"Jackal Teleop | saved: {len(self.all_demos)}"
        )
        self.fig.canvas.draw_idle()

    def _update_artists(self) -> None:
        """Update dynamic artists on every animation frame."""
        xs, ys = zip(*self.traj)
        self._ln_traj.set_data(xs, ys)

        x, y, theta = float(self.state[0]), float(self.state[1]), float(self.state[2])
        self._pt_robot.set_data([x], [y])
        self._ln_head.set_data(
            [x, x + 0.55 * np.cos(theta)],
            [y, y + 0.55 * np.sin(theta)],
        )

        action = action_from_keys(self.pressed)
        v_ms   = float(action[0]) * dynamics.MAX_LINEAR_VEL
        w_rads = float(action[1]) * dynamics.MAX_ANGULAR_VEL

        status_color = {
            "running":   "#98c379",
            "success":   "#e5c07b",
            "collision": "#e06c75",
        }.get(self.ep_status, "#abb2bf")

        info = (
            f"Step   : {len(self._ep_obs):>4}\n"
            f"x      : {x:.2f} m\n"
            f"y      : {y:.2f} m\n"
            f"θ      : {np.degrees(theta):.1f}°\n"
            f"v_cmd  : {v_ms:+.2f} m/s\n"
            f"ω_cmd  : {w_rads:+.2f} r/s\n"
            f"\nStatus : {self.ep_status}\n"
            f"Total  : {len(self.all_demos)} eps"
        )
        self._txt_info.set_text(info)
        self._txt_info.set_color(status_color)

    # ── simulation step ───────────────────────────────────────────────────

    def _sim_step(self) -> None:
        if self.ep_done:
            return

        action = action_from_keys(self.pressed)
        self._ep_obs.append(_arena.get_observation(self.state, self.goal))
        self._ep_act.append(action.copy())

        nxt  = dynamics.step(self.state, action)
        nx, ny = float(nxt[0]), float(nxt[1])

        if _arena.check_collision(nx, ny):
            self.ep_status = "collision"
            self.ep_done   = True
            self._save_ep(success=False)
            # Collision: user can press R or Enter to reset
            return

        self.state = nxt
        self.traj.append((nx, ny))

        if _arena.check_goal(nx, ny, self.goal):
            self.ep_status = "success"
            self.ep_done   = True
            self._save_ep(success=True)
            self._reset()

    # ── animation ────────────────────────────────────────────────────────

    def _frame(self, _frame_idx: int) -> None:
        self._sim_step()
        self._update_artists()

    def run(self) -> None:
        self._anim = FuncAnimation(
            self.fig,
            self._frame,
            interval=int(1000 / self.fps),
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jackal Diffusion — Teleop Dataset Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save", type=str, default=DEFAULT_SAVE,
                        help=f"Output .npz file path (default: {DEFAULT_SAVE})")
    parser.add_argument("--fps",  type=int, default=DEFAULT_FPS,
                        help=f"Animation speed in Hz (default: {DEFAULT_FPS})")
    args = parser.parse_args()

    print("=" * 56)
    print("  Jackal Diffusion — Teleop Dataset Collector")
    print("=" * 56)
    print(f"  Seed      : {args.seed}")
    print(f"  Save      : {args.save}")
    print(f"  FPS       : {args.fps}")
    print("-" * 56)
    print("  W/↑ Fwd  S/↓ Back  A/← Left  D/→ Right")
    print("  Enter: Save+New  R: Discard+New  Q: Quit")
    print("=" * 56)

    collector = TeleopCollector(seed=args.seed, save_path=args.save, fps=args.fps)
    collector.run()


if __name__ == "__main__":
    main()
