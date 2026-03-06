from __future__ import annotations

import numpy as np
import torch

from data.dataset import DemoDataset
from data.expert import collect_all
from env import arena
from eval.visualize import (
    plot_multimodal_trajectories,
    plot_training_loss,
    print_summary,
)
from policy.diffusion import DiffusionPolicy
from policy.mlp_bc import MLPBC


TO = 2
TA = 8


def _random_starts(n: int, seed: int = 123) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(n):
        x = float(arena.START[0] + rng.uniform(-0.3, 0.3))
        y = float(arena.START[1] + rng.uniform(-0.3, 0.3))
        theta = float(arena.START[2] + rng.uniform(-0.2, 0.2))
        starts.append((x, y, theta))
    return starts


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    demos = collect_all(n_per_side=25)
    n_left = sum(d["side"] == "left" for d in demos)
    n_right = sum(d["side"] == "right" for d in demos)
    print(f"[Data]     {len(demos)} demos collected | Left: {n_left}  Right: {n_right}")

    dataset = DemoDataset(demos, to=TO, ta=TA)
    train_idx, val_idx = dataset.get_split_indices(val_ratio=0.1, seed=0)
    dataset.fit_normalizers(train_idx)

    obs_shape, action_shape = dataset.sample_shapes()
    print(
        f"[Data]     Dataset size: {len(dataset)} samples | "
        f"obs {obs_shape} | action {action_shape}"
    )
    print(
        "[Data]     Norm stats: "
        f"obs_mean[0]={dataset.obs_mean[0]:.3f}, obs_std[0]={dataset.obs_std[0]:.3f}, "
        f"act_min={dataset.action_min.tolist()}, act_max={dataset.action_max.tolist()}"
    )

    dp = DiffusionPolicy(to=TO, ta=TA, obs_dim=arena.OBS_DIM, device="cpu")
    dp_losses = dp.train(
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=500,
        batch_size=64,
    )

    mlp = MLPBC(to=TO, ta=TA, obs_dim=arena.OBS_DIM, device="cpu")
    mlp_losses = mlp.train_model(
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=500,
        batch_size=64,
    )

    print("[Eval]     Running 50 rollouts per method...")
    starts = _random_starts(n=50, seed=999)

    dp_rollouts = [dp.run_episode(start=s, goal=arena.GOAL, max_steps=300) for s in starts]
    mlp_rollouts = [
        mlp.run_episode(start=s, goal=arena.GOAL, max_steps=300) for s in starts
    ]

    plot_training_loss(dp_losses, mlp_losses)
    plot_multimodal_trajectories(dp_rollouts, mlp_rollouts, arena.OBSTACLE)
    print_summary(dp_rollouts, mlp_rollouts)


if __name__ == "__main__":
    main()
