from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from jackal_diffusion.env import arena
from jackal_diffusion.env_runner.base_lowdim_runner import BaseLowdimRunner
from jackal_diffusion.eval.visualize import _draw_scene
from jackal_diffusion.policy.base_lowdim_policy import BaseLowdimPolicy


def random_start(
    seed: int,
    xy_jitter: float = 0.3,
    theta_jitter: float = 0.2,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = float(arena.START[0] + rng.uniform(-xy_jitter, xy_jitter))
    y = float(arena.START[1] + rng.uniform(-xy_jitter, xy_jitter))
    theta = float(arena.START[2] + rng.uniform(-theta_jitter, theta_jitter))
    return x, y, theta


def random_starts(
    n_rollouts: int,
    start_seed: int = 999,
    xy_jitter: float = 0.3,
    theta_jitter: float = 0.2,
) -> list[tuple[float, float, float]]:
    return [
        random_start(start_seed + idx, xy_jitter=xy_jitter, theta_jitter=theta_jitter)
        for idx in range(n_rollouts)
    ]


class JackalLowdimRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir: str,
        n_rollouts: int = 50,
        start_seed: int = 999,
        max_steps: int = 300,
        exec_horizon: int = 1,
        xy_jitter: float = 0.3,
        theta_jitter: float = 0.2,
        save_plot: bool = True,
        plot_name: str = "latest_rollouts.png",
    ) -> None:
        super().__init__(output_dir)
        self.n_rollouts = int(n_rollouts)
        self.start_seed = int(start_seed)
        self.max_steps = int(max_steps)
        self.exec_horizon = int(exec_horizon)
        self.xy_jitter = float(xy_jitter)
        self.theta_jitter = float(theta_jitter)
        self.save_plot = bool(save_plot)
        self.plot_name = plot_name

    def run(self, policy: BaseLowdimPolicy) -> dict:
        starts = random_starts(
            n_rollouts=self.n_rollouts,
            start_seed=self.start_seed,
            xy_jitter=self.xy_jitter,
            theta_jitter=self.theta_jitter,
        )
        rollouts = [
            policy.run_episode(
                start=start,
                goal=arena.GOAL,
                max_steps=self.max_steps,
                exec_horizon=self.exec_horizon,
            )
            for start in starts
        ]
        if self.save_plot:
            self._save_plot(rollouts)

        successes = np.array([r["success"] for r in rollouts], dtype=np.float32)
        collisions = np.array([r["collision"] for r in rollouts], dtype=np.float32)
        steps = np.array([r["steps"] for r in rollouts], dtype=np.float32)
        left = np.array([r["side"] == "left" for r in rollouts], dtype=np.float32)
        right = np.array([r["side"] == "right" for r in rollouts], dtype=np.float32)
        success_steps = steps[successes > 0.5]

        return {
            "test_mean_score": float(successes.mean()) if len(successes) else 0.0,
            "test_collision_rate": float(collisions.mean()) if len(collisions) else 0.0,
            "test_mean_steps": float(steps.mean()) if len(steps) else 0.0,
            "test_success_mean_steps": float(success_steps.mean()) if len(success_steps) else 0.0,
            "test_left_fraction": float(left.mean()) if len(left) else 0.0,
            "test_right_fraction": float(right.mean()) if len(right) else 0.0,
        }

    def _save_plot(self, rollouts: list[dict]) -> None:
        media_dir = os.path.join(self.output_dir, "media")
        os.makedirs(media_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_scene(ax, arena.OBSTACLE)
        ax.set_title("Jackal Rollouts")
        for rollout in rollouts:
            traj = np.asarray(rollout["trajectory"], dtype=np.float32)
            color = "tab:blue" if rollout["side"] == "left" else "tab:red"
            alpha = 0.8 if rollout["success"] else 0.35
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, lw=1.2)
        fig.tight_layout()
        fig.savefig(os.path.join(media_dir, self.plot_name), dpi=150)
        plt.close(fig)
