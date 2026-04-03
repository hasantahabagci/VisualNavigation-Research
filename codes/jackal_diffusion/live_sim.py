from __future__ import annotations

import argparse
import os
import time

import hydra
import numpy as np

from jackal_diffusion.config_utils import load_config
from jackal_diffusion.env import arena
from jackal_diffusion.env_runner import random_start
from jackal_diffusion.eval import animate_rollout
from jackal_diffusion.policy import (
    JackalDiffusionLowdimPolicy,
    JackalMLPLowdimPolicy,
)
from jackal_diffusion.workspace import load_policy_from_checkpoint


DEFAULT_DIFFUSION_LEGACY = "results/dp_model.pt"
DEFAULT_MLP_LEGACY = "results/mlp_model.pt"


def _policy_to_config_name(policy_name: str) -> str:
    return (
        "train_diffusion_lowdim_workspace"
        if policy_name == "diffusion"
        else "train_mlp_lowdim_workspace"
    )


def _resolve_start(args) -> tuple[float, float, float]:
    if args.start_x is not None or args.start_y is not None or args.start_theta is not None:
        sx = args.start_x if args.start_x is not None else float(arena.START[0])
        sy = args.start_y if args.start_y is not None else float(arena.START[1])
        st = args.start_theta if args.start_theta is not None else float(arena.START[2])
        return sx, sy, st
    return random_start(args.seed)


def _resolve_goal(args) -> np.ndarray:
    if args.goal_x is not None or args.goal_y is not None:
        gx = args.goal_x if args.goal_x is not None else float(arena.GOAL[0])
        gy = args.goal_y if args.goal_y is not None else float(arena.GOAL[1])
        return np.array([gx, gy], dtype=np.float32)
    return arena.GOAL.copy()


def _train_from_scratch(policy_name: str, epochs: int, output_dir: str | None):
    cfg = load_config(_policy_to_config_name(policy_name))
    cfg.training.resume = False
    cfg.training.device = "cpu"
    cfg.training.num_epochs = int(epochs)
    cfg.training.rollout_every = max(1, int(epochs))
    cfg.training.checkpoint_every = max(1, int(epochs))
    cfg.task.env_runner.n_rollouts = min(int(cfg.task.env_runner.n_rollouts), 5)

    if output_dir is None:
        output_dir = os.path.join(
            "data",
            "live_sim",
            time.strftime("%Y%m%d-%H%M%S"),
            policy_name,
        )
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.run()
    policy = workspace.model
    policy.eval()
    return workspace, policy


def _load_legacy_policy(policy_name: str, model_path: str, device: str = "cpu"):
    if policy_name == "diffusion":
        return JackalDiffusionLowdimPolicy.load_legacy(model_path, device=device)
    return JackalMLPLowdimPolicy.load_legacy(model_path, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 2D Jackal rollout viewer")
    parser.add_argument(
        "--policy",
        choices=["diffusion", "mlp", "dp"],
        default="diffusion",
        help="Policy family to use when training or loading legacy files.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Workspace checkpoint path.")
    parser.add_argument(
        "--legacy-model",
        type=str,
        default=None,
        help="Legacy .pt model path for one-way compatibility loading.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Epoch count when training from scratch inside the wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory used when training from scratch.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for start pose.")
    parser.add_argument("--max-steps", type=int, default=300, help="Episode step limit.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--no-animate", action="store_true", help="Skip matplotlib animation.")
    parser.add_argument(
        "--save-legacy",
        type=str,
        default=None,
        help="Optional path to export the loaded/trained policy in legacy .pt format.",
    )
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--start-theta", type=float, default=None)
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    args = parser.parse_args()

    policy_name = "diffusion" if args.policy == "dp" else args.policy
    start = _resolve_start(args)
    goal = _resolve_goal(args)

    if args.checkpoint is not None:
        workspace, policy = load_policy_from_checkpoint(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device="cpu",
        )
        title = f"Live Rollout - {policy.__class__.__name__} (checkpoint)"
    elif args.legacy_model is not None:
        policy = _load_legacy_policy(policy_name, args.legacy_model, device="cpu")
        workspace = None
        title = f"Live Rollout - {policy.__class__.__name__} (legacy)"
    else:
        default_legacy = DEFAULT_DIFFUSION_LEGACY if policy_name == "diffusion" else DEFAULT_MLP_LEGACY
        if os.path.exists(default_legacy):
            policy = _load_legacy_policy(policy_name, default_legacy, device="cpu")
            workspace = None
            title = f"Live Rollout - {policy.__class__.__name__} (legacy default)"
        else:
            workspace, policy = _train_from_scratch(
                policy_name=policy_name,
                epochs=args.epochs,
                output_dir=args.output_dir,
            )
            title = f"Live Rollout - {policy.__class__.__name__}"

    if args.save_legacy is not None:
        policy.save_legacy(args.save_legacy)
        print(f"[Saved]    {args.save_legacy}")

    rollout = policy.run_episode(start=start, goal=goal, max_steps=args.max_steps)
    print(
        f"[Eval]     success={rollout['success']} | "
        f"collision={rollout.get('collision', False)} | steps={rollout['steps']}"
    )

    if not args.no_animate:
        animate_rollout(
            rollout=rollout,
            obstacle=arena.OBSTACLE,
            goal=goal,
            start=start,
            title=title,
            speed=max(args.speed, 1e-3),
            n_obs_steps=policy.n_obs_steps,
            n_action_steps=policy.n_action_steps,
        )


if __name__ == "__main__":
    main()
