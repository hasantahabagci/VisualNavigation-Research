from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import DemoDataset
from env import arena, dynamics


class MLPBC(nn.Module):
    """
    Naive behavior cloning baseline.
    Maps flattened obs_seq -> action_seq directly.
    """

    def __init__(
        self,
        to: int = 2,
        ta: int = 8,
        obs_dim: int = 8,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_horizon = int(to)
        self.action_horizon = int(ta)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)

        self.net = nn.Sequential(
            nn.Linear(self.obs_horizon * self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_horizon * 2),
        ).to(self.device)

        self.dataset: DemoDataset | None = None
        self.losses: list[float] = []

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        out = self.net(obs_flat)
        return out.reshape(-1, self.action_horizon, 2)

    def train_model(
        self,
        dataset: DemoDataset,
        train_idx: np.ndarray | None = None,
        val_idx: np.ndarray | None = None,
        epochs: int = 500,
        batch_size: int = 64,
    ) -> list[float]:
        self.dataset = dataset

        if train_idx is None or val_idx is None:
            train_idx, val_idx = dataset.get_split_indices(val_ratio=0.1, seed=42)

        if not dataset.is_fitted:
            dataset.fit_normalizers(train_idx)

        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-6)
        criterion = nn.MSELoss()

        self.losses = []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            self.net.train()
            running = 0.0
            count = 0
            for obs_seq, action_seq in train_loader:
                obs_seq = obs_seq.to(self.device)
                action_seq = action_seq.to(self.device)
                bsz = obs_seq.shape[0]

                obs_flat = obs_seq.reshape(bsz, -1)
                pred = self.forward(obs_flat)
                loss = criterion(pred, action_seq)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running += float(loss.item()) * bsz
                count += bsz

            train_loss = running / max(count, 1)
            self.losses.append(train_loss)

            self.net.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for obs_seq, action_seq in val_loader:
                    obs_seq = obs_seq.to(self.device)
                    action_seq = action_seq.to(self.device)
                    bsz = obs_seq.shape[0]
                    pred = self.forward(obs_seq.reshape(bsz, -1))
                    val_loss = criterion(pred, action_seq)
                    val_running += float(val_loss.item()) * bsz
                    val_count += bsz

            val_loss_epoch = val_running / max(val_count, 1)

            if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                print(f"[Train-ML] Epoch {epoch}/{epochs} | Loss: {train_loss:.4f}")

            if val_loss_epoch < 0.001:
                elapsed = time.time() - t0
                print(
                    f"[Train-ML] Early stop at epoch {epoch}/{epochs} | "
                    f"Loss: {train_loss:.4f} ({elapsed:.1f} s)"
                )
                break

        if len(self.losses) == epochs:
            elapsed = time.time() - t0
            print(
                f"[Train-ML] Epoch {epochs}/{epochs} | Loss: {self.losses[-1]:.4f} "
                f"({elapsed:.1f} s)"
            )

        return self.losses

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        if self.dataset is None or not self.dataset.is_fitted:
            raise RuntimeError("Baseline is not fitted to a normalized dataset.")

        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.shape != (self.obs_horizon, self.obs_dim):
            raise ValueError(
                f"obs_seq must have shape ({self.obs_horizon}, {self.obs_dim}), got {obs_seq.shape}"
            )

        obs_norm = self.dataset.normalize_obs(obs_seq)[None, ...]
        obs_tensor = torch.tensor(obs_norm.tolist(), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_norm = self.forward(obs_tensor.reshape(1, -1))[0]

        action_norm_np = np.asarray(action_norm.detach().cpu().tolist(), dtype=np.float32)
        action = self.dataset.denormalize_action(action_norm_np)
        v_center = 0.5 * (self.dataset.action_min[0] + self.dataset.action_max[0])
        action[:, 0] = v_center
        action[:, 1] = np.clip(
            action[:, 1],
            float(self.dataset.action_min[1]),
            float(self.dataset.action_max[1]),
        )
        return np.clip(action, -1.0, 1.0)

    def run_episode(
        self,
        start: tuple[float, float, float],
        goal: np.ndarray,
        max_steps: int = 300,
        exec_horizon: int = 1,
    ) -> dict:
        state = np.array([start[0], start[1], start[2], 0.0, 0.0], dtype=np.float32)
        trajectory = [state.copy()]

        obs_history = [arena.get_observation(state, goal) for _ in range(self.obs_horizon)]
        success = False
        collision = False
        steps = 0

        while steps < max_steps and not success and not collision:
            obs_seq = np.asarray(obs_history[-self.obs_horizon :], dtype=np.float32)
            action_seq = self.predict(obs_seq)

            for action in action_seq[:exec_horizon]:
                action = np.clip(action, -1.0, 1.0)
                state = dynamics.step(state, action)
                trajectory.append(state.copy())
                obs_history.append(arena.get_observation(state, goal))
                steps += 1

                if arena.check_collision(state[0], state[1]):
                    collision = True
                    break

                if arena.check_goal(state[0], state[1], goal):
                    success = True
                    break

                if steps >= max_steps:
                    break

        return {
            "trajectory": trajectory,
            "success": bool(success),
            "collision": bool(collision),
            "steps": int(steps),
        }
