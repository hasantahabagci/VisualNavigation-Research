from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from jackal_diffusion.env import arena


def _ensure_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_training_loss(dp_losses, mlp_losses, out_dir: str = "results") -> str:
    out_dir = _ensure_dir(out_dir)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(dp_losses) + 1), dp_losses, label="Diffusion Policy")
    plt.plot(np.arange(1, len(mlp_losses) + 1), mlp_losses, label="MLP-BC")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log)")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "training_loss.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _draw_scene(ax, obstacle, goal=None, start=None) -> None:
    ax.set_xlim(0, arena.ARENA_SIZE)
    ax.set_ylim(0, arena.ARENA_SIZE)
    ax.set_aspect("equal")

    rect = Rectangle(
        (obstacle["cx"] - obstacle["w"] / 2.0, obstacle["cy"] - obstacle["h"] / 2.0),
        obstacle["w"],
        obstacle["h"],
        color="lightgray",
        ec="k",
        alpha=0.8,
    )
    ax.add_patch(rect)

    sx, sy = (start[0], start[1]) if start is not None else (arena.START[0], arena.START[1])
    gx, gy = (goal[0], goal[1]) if goal is not None else (arena.GOAL[0], arena.GOAL[1])
    ax.plot(sx, sy, "o", color="green", ms=7, label="Start")
    ax.plot(gx, gy, marker="*", color="gold", ms=12, label="Goal")

    goal_circle = plt.Circle(
        (gx, gy),
        arena.GOAL_RADIUS,
        color="gold",
        fill=False,
        ls="--",
        lw=1.0,
        alpha=0.6,
        label="Goal radius",
    )
    ax.add_patch(goal_circle)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_multimodal_trajectories(
    dp_rollouts,
    mlp_rollouts,
    obstacle,
    out_dir: str = "results",
) -> str:
    out_dir = _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    ax_dp, ax_mlp = axes
    _draw_scene(ax_dp, obstacle)
    _draw_scene(ax_mlp, obstacle)

    ax_dp.set_title("Diffusion Policy")
    for rollout in dp_rollouts:
        traj = np.asarray(rollout["trajectory"], dtype=np.float32)
        color = "tab:blue" if rollout.get("side") == "left" else "tab:red"
        ax_dp.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.65, lw=1.3)

    ax_mlp.set_title("MLP-BC")
    for rollout in mlp_rollouts:
        traj = np.asarray(rollout["trajectory"], dtype=np.float32)
        ax_mlp.plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.65, lw=1.2)

    handles, labels = ax_dp.get_legend_handles_labels()
    if handles:
        ax_dp.legend(loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "multimodal_trajectories.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def animate_rollout(
    rollout: dict,
    obstacle: dict,
    title: str = "Live Rollout",
    speed: float = 1.0,
    goal=None,
    start=None,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
) -> None:
    from jackal_diffusion.env import dynamics

    traj = np.asarray(rollout["trajectory"], dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("rollout['trajectory'] must contain [x, y, ...] states.")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _draw_scene(ax, obstacle, goal=goal, start=start)
    ax.set_title(title)

    path_line, = ax.plot([], [], color="tab:blue", lw=2.0, alpha=0.9, label="Trajectory")
    obs_line, = ax.plot(
        [],
        [],
        color="cyan",
        lw=2.5,
        alpha=0.85,
        marker="o",
        ms=5,
        label=f"$T_o$ = {n_obs_steps}",
    )
    act_line, = ax.plot(
        [],
        [],
        color="magenta",
        lw=2.0,
        ls="--",
        alpha=0.80,
        marker="s",
        ms=4,
        label=f"$T_a$ = {n_action_steps}",
    )

    robot = plt.Circle(
        (traj[0, 0], traj[0, 1]),
        radius=dynamics.ROBOT_RADIUS,
        color="tab:orange",
        alpha=0.85,
    )
    ax.add_patch(robot)
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )
    ax.legend(loc="upper right", fontsize=8)

    plt.ion()
    plt.show(block=False)
    pause_s = max(1e-3, dynamics.DT / max(speed, 1e-3))
    n_steps = traj.shape[0]

    for i in range(n_steps):
        xy = traj[: i + 1, :2]
        path_line.set_data(xy[:, 0], xy[:, 1])
        robot.center = (float(traj[i, 0]), float(traj[i, 1]))

        obs_start = max(0, i - n_obs_steps + 1)
        obs_seg = traj[obs_start : i + 1, :2]
        obs_line.set_data(obs_seg[:, 0], obs_seg[:, 1])

        act_end = min(n_steps, i + n_action_steps + 1)
        act_seg = traj[i:act_end, :2]
        act_line.set_data(act_seg[:, 0], act_seg[:, 1])

        status_text.set_text(
            f"step: {i}/{n_steps - 1}\n"
            f"x: {traj[i, 0]:.2f}  y: {traj[i, 1]:.2f}\n"
            f"$T_o$: [{obs_start}-{i}]  $T_a$: [{i}-{min(n_steps - 1, i + n_action_steps)}]\n"
            f"success: {rollout.get('success', False)}"
        )
        fig.canvas.draw_idle()
        plt.pause(pause_s)

    plt.ioff()
    plt.show()


def _rate(results) -> float:
    return 100.0 * float(sum(r["success"] for r in results)) / max(len(results), 1)


def _avg_steps(results):
    success_steps = [r["steps"] for r in results if r["success"]]
    if not success_steps:
        return "-"
    return f"{int(np.mean(success_steps))}"


def print_summary(dp_results, mlp_results) -> None:
    dp_success = _rate(dp_results)
    mlp_success = _rate(mlp_results)

    dp_left = 100.0 * sum(r.get("side") == "left" for r in dp_results) / max(len(dp_results), 1)
    dp_right = 100.0 * sum(r.get("side") == "right" for r in dp_results) / max(len(dp_results), 1)

    print("\n========== RESULTS ==========")
    print("Method            | Diffusion | MLP-BC")
    print(f"Success rate      |   {dp_success:>3.0f}%     |  {mlp_success:>3.0f}%")
    print(f"Left  path chosen |   {dp_left:>3.0f}%     |   -")
    print(f"Right path chosen |   {dp_right:>3.0f}%     |   -")
    print(f"Avg steps         |   {_avg_steps(dp_results):>3}     |  {_avg_steps(mlp_results):>3}")
    print("=============================")
