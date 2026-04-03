from __future__ import annotations

import argparse
import os

import hydra

from jackal_diffusion.config_utils import load_config
from jackal_diffusion.env import arena
from jackal_diffusion.env_runner import random_starts
from jackal_diffusion.eval import (
    plot_multimodal_trajectories,
    plot_training_loss,
    print_summary,
)


def _train_workspace(config_name: str, epochs: int, output_dir: str):
    cfg = load_config(config_name)
    cfg.training.resume = False
    cfg.training.device = "cpu"
    cfg.training.num_epochs = int(epochs)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.run()
    workspace.model.eval()
    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both policies and compare them.")
    parser.add_argument("--dp-epochs", type=int, default=500)
    parser.add_argument("--mlp-epochs", type=int, default=500)
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=999)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--work-dir", type=str, default="data/compare_runs")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    dp_workspace = _train_workspace(
        config_name="train_diffusion_lowdim_workspace",
        epochs=args.dp_epochs,
        output_dir=os.path.join(args.work_dir, "diffusion"),
    )
    mlp_workspace = _train_workspace(
        config_name="train_mlp_lowdim_workspace",
        epochs=args.mlp_epochs,
        output_dir=os.path.join(args.work_dir, "mlp"),
    )

    starts = random_starts(args.n_rollouts, start_seed=args.start_seed)
    dp_rollouts = [
        dp_workspace.model.run_episode(start=start, goal=arena.GOAL, max_steps=300)
        for start in starts
    ]
    mlp_rollouts = [
        mlp_workspace.model.run_episode(start=start, goal=arena.GOAL, max_steps=300)
        for start in starts
    ]

    loss_path = plot_training_loss(
        dp_workspace.train_losses,
        mlp_workspace.train_losses,
        out_dir=args.results_dir,
    )
    traj_path = plot_multimodal_trajectories(
        dp_rollouts,
        mlp_rollouts,
        obstacle=arena.OBSTACLE,
        out_dir=args.results_dir,
    )
    print(f"[Saved]    {loss_path}")
    print(f"[Saved]    {traj_path}")
    print_summary(dp_rollouts, mlp_rollouts)


if __name__ == "__main__":
    main()
