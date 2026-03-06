from __future__ import annotations

import argparse
import os

import numpy as np

from data.dataset import DemoDataset
from data.expert import collect_all
from env import arena
from eval.visualize import animate_rollout
from policy.diffusion import DiffusionPolicy
from policy.mlp_bc import MLPBC

TO = 8
TA = 16

DEFAULT_DP_PATH = "results/dp_model.pt"
DEFAULT_MLP_PATH = "results/mlp_model.pt"


def random_start(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = float(arena.START[0] + rng.uniform(-0.3, 0.3))
    y = float(arena.START[1] + rng.uniform(-0.3, 0.3))
    theta = float(arena.START[2] + rng.uniform(-0.2, 0.2))
    return x, y, theta


def build_dataset() -> tuple[DemoDataset, np.ndarray, np.ndarray]:
    demos = collect_all(n_per_side=25)
    print(
        f"[Data]     {len(demos)} demos collected | "
        f"Left: {sum(d['side'] == 'left' for d in demos)}  "
        f"Right: {sum(d['side'] == 'right' for d in demos)}"
    )

    dataset = DemoDataset(demos, to=TO, ta=TA)
    train_idx, val_idx = dataset.get_split_indices(val_ratio=0.1, seed=0)
    dataset.fit_normalizers(train_idx)
    print(f"[Data]     Dataset size: {len(dataset)}")
    return dataset, train_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 2D rollout viewer")
    parser.add_argument(
        "--policy",
        choices=["dp", "mlp"],
        default="dp",
        help="Which policy to train and animate.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Rollout start random seed.")
    parser.add_argument("--max-steps", type=int, default=300, help="Episode step limit.")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = realtime).",
    )
    parser.add_argument(
        "--dp-epochs",
        type=int,
        default=250,
        help="Training epochs for Diffusion Policy.",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=200,
        help="Training epochs for MLP-BC.",
    )

    # -- Save / Load ----------------------------------------------------------
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the trained model after training.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load a previously saved model (skip training).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path for saving/loading the model file.",
    )

    # -- Manual start / goal --------------------------------------------------
    parser.add_argument("--start-x", type=float, default=None, help="Override start X.")
    parser.add_argument("--start-y", type=float, default=None, help="Override start Y.")
    parser.add_argument("--start-theta", type=float, default=None, help="Override start heading (rad).")
    parser.add_argument("--goal-x", type=float, default=None, help="Override goal X.")
    parser.add_argument("--goal-y", type=float, default=None, help="Override goal Y.")

    args = parser.parse_args()

    import torch

    np.random.seed(0)
    torch.manual_seed(0)

    # Determine model path
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = DEFAULT_DP_PATH if args.policy == "dp" else DEFAULT_MLP_PATH

    # Resolve start pose
    if args.start_x is not None or args.start_y is not None or args.start_theta is not None:
        sx = args.start_x if args.start_x is not None else float(arena.START[0])
        sy = args.start_y if args.start_y is not None else float(arena.START[1])
        st = args.start_theta if args.start_theta is not None else float(arena.START[2])
        start = (sx, sy, st)
        print(f"[Start]    Manual start: x={sx:.2f}, y={sy:.2f}, θ={st:.2f}")
    else:
        start = random_start(args.seed)
        print(f"[Start]    Random start (seed={args.seed}): x={start[0]:.2f}, y={start[1]:.2f}, θ={start[2]:.2f}")

    # Resolve goal
    if args.goal_x is not None or args.goal_y is not None:
        gx = args.goal_x if args.goal_x is not None else float(arena.GOAL[0])
        gy = args.goal_y if args.goal_y is not None else float(arena.GOAL[1])
        goal = np.array([gx, gy], dtype=np.float32)
        print(f"[Goal]     Manual goal: x={gx:.2f}, y={gy:.2f}")
    else:
        goal = arena.GOAL
        print(f"[Goal]     Default goal: x={goal[0]:.2f}, y={goal[1]:.2f}")

    # Build or load model
    if args.load and os.path.exists(model_path):
        if args.policy == "dp":
            model = DiffusionPolicy.load(model_path, device="cpu")
            title = "Live Rollout – Diffusion Policy (loaded)"
        else:
            model = MLPBC.load(model_path, device="cpu")
            title = "Live Rollout – MLP-BC (loaded)"
    else:
        if args.load:
            print(f"[Warn]     {model_path} not found — training from scratch.")
        dataset, train_idx, val_idx = build_dataset()
        if args.policy == "dp":
            model = DiffusionPolicy(to=TO, ta=TA, obs_dim=arena.OBS_DIM, device="cpu")
            model.train(
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                epochs=args.dp_epochs,
                batch_size=64,
            )
            title = "Live Rollout – Diffusion Policy"
        else:
            model = MLPBC(to=TO, ta=TA, obs_dim=arena.OBS_DIM, device="cpu")
            model.train_model(
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                epochs=args.mlp_epochs,
                batch_size=64,
            )
            title = "Live Rollout – MLP-BC"

        if args.save:
            model.save(model_path)

    rollout = model.run_episode(start=start, goal=goal, max_steps=args.max_steps)

    print(
        f"[Eval]     success={rollout['success']} | "
        f"collision={rollout.get('collision', False)} | steps={rollout['steps']}"
    )
    animate_rollout(
        rollout=rollout,
        obstacle=arena.OBSTACLE,
        goal=goal,
        start=start,
        title=title,
        speed=max(args.speed, 1e-3),
        to=TO,
        ta=TA,
    )


if __name__ == "__main__":
    main()
