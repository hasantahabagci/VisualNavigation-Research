from __future__ import annotations

import argparse
import json
import os

import hydra

from jackal_diffusion.workspace import load_workspace_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a workspace checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to workspace checkpoint.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where eval metrics and media will be written.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for policy evaluation.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    workspace = load_workspace_from_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    policy = workspace.model
    policy.to(args.device)
    policy.eval()

    env_runner = hydra.utils.instantiate(
        workspace.cfg.task.env_runner,
        output_dir=args.output_dir,
    )
    runner_log = env_runner.run(policy)

    out_path = os.path.join(args.output_dir, "eval_log.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(runner_log, f, indent=2, sort_keys=True)

    for key, value in sorted(runner_log.items()):
        print(f"{key}: {value}")
    print(f"[Saved]    {out_path}")


if __name__ == "__main__":
    main()
