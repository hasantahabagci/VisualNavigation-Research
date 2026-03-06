from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import DemoDataset
from env import arena, dynamics
from model.noise_net import NoiseNet
from model.scheduler import CosineScheduler


class DiffusionPolicy:
    """
    Wraps NoiseNet + CosineScheduler for training and closed-loop inference.
    """

    def __init__(
        self,
        to: int = 2,
        ta: int = 8,
        obs_dim: int = 8,
        device: str = "cpu",
    ) -> None:
        self.to = int(to)
        self.ta = int(ta)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)

        self.net = NoiseNet(to=self.to, obs_dim=self.obs_dim).to(self.device)
        self.scheduler = CosineScheduler(k_train=100, k_infer=10)

        self.dataset: DemoDataset | None = None
        self.losses: list[float] = []
        self.val_losses: list[float] = []

    def train(
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

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=1e-3, weight_decay=1e-6
        )
        criterion = nn.MSELoss()

        self.losses = []
        self.val_losses = []
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
                k = torch.randint(
                    low=1,
                    high=self.scheduler.k_train + 1,
                    size=(bsz,),
                    device=self.device,
                )

                eps = torch.randn_like(action_seq)
                noisy_actions = self.scheduler.add_noise(action_seq, eps, k)
                eps_pred = self.net(noisy_actions, obs_flat, k)
                loss = criterion(eps_pred, eps)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running += float(loss.item()) * bsz
                count += bsz

            train_loss = running / max(count, 1)

            self.net.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for obs_seq, action_seq in val_loader:
                    obs_seq = obs_seq.to(self.device)
                    action_seq = action_seq.to(self.device)
                    bsz = obs_seq.shape[0]

                    obs_flat = obs_seq.reshape(bsz, -1)
                    k = torch.randint(
                        low=1,
                        high=self.scheduler.k_train + 1,
                        size=(bsz,),
                        device=self.device,
                    )
                    eps = torch.randn_like(action_seq)
                    noisy_actions = self.scheduler.add_noise(action_seq, eps, k)
                    eps_pred = self.net(noisy_actions, obs_flat, k)
                    val_loss = criterion(eps_pred, eps)

                    val_running += float(val_loss.item()) * bsz
                    val_count += bsz

            val_loss_epoch = val_running / max(val_count, 1)
            self.losses.append(train_loss)
            self.val_losses.append(val_loss_epoch)

            if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                print(f"[Train-DP] Epoch {epoch}/{epochs} | Loss: {train_loss:.4f}")

            if val_loss_epoch < 0.005:
                elapsed = time.time() - t0
                print(
                    f"[Train-DP] Early stop at epoch {epoch}/{epochs} | "
                    f"Loss: {train_loss:.4f} ({elapsed:.1f} s)"
                )
                break

        if len(self.losses) == epochs:
            elapsed = time.time() - t0
            print(
                f"[Train-DP] Epoch {epochs}/{epochs} | Loss: {self.losses[-1]:.4f} "
                f"({elapsed:.1f} s)"
            )

        return self.losses

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        """Returns action sequence of shape (Ta, 2)."""
        if self.dataset is None or not self.dataset.is_fitted:
            raise RuntimeError("Policy is not fitted to a normalized dataset.")

        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.shape != (self.to, self.obs_dim):
            raise ValueError(
                f"obs_seq must have shape ({self.to}, {self.obs_dim}), got {obs_seq.shape}"
            )

        obs_norm = self.dataset.normalize_obs(obs_seq)[None, ...]
        obs_tensor = torch.tensor(obs_norm.tolist(), dtype=torch.float32, device=self.device)
        obs_flat = obs_tensor.reshape(1, -1)

        with torch.no_grad():
            action_norm = self.scheduler.sample(self.net, obs_flat, self.ta)[0]

        action_norm_np = np.asarray(action_norm.detach().cpu().tolist(), dtype=np.float32)
        action = self.dataset.denormalize_action(action_norm_np)

        # Keep commands close to expert support for stable rollouts.
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
        """
        Returns:
            {'trajectory': list[state], 'success': bool, 'side': str}
        """
        state = np.array([start[0], start[1], start[2], 0.0, 0.0], dtype=np.float32)
        trajectory = [state.copy()]

        obs_history = [arena.get_observation(state, goal) for _ in range(self.to)]
        success = False
        collision = False
        steps = 0

        while steps < max_steps and not success and not collision:
            obs_seq = np.asarray(obs_history[-self.to :], dtype=np.float32)
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

        ys = np.array([s[1] for s in trajectory], dtype=np.float32)
        up = float(ys.max() - arena.START[1])
        down = float(arena.START[1] - ys.min())
        side = "left" if up >= down else "right"

        return {
            "trajectory": trajectory,
            "success": bool(success),
            "collision": bool(collision),
            "side": side,
            "steps": int(steps),
        }

    # ---- persistence ---------------------------------------------------------
    def save(self, path: str = "results/dp_model.pt") -> None:
        """Save trained model weights and dataset normalizers."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "net_state": self.net.state_dict(),
            "to": self.to,
            "ta": self.ta,
            "obs_dim": self.obs_dim,
        }
        if self.dataset is not None and self.dataset.is_fitted:
            payload["obs_mean"] = self.dataset.obs_mean
            payload["obs_std"] = self.dataset.obs_std
            payload["action_min"] = self.dataset.action_min
            payload["action_max"] = self.dataset.action_max
        torch.save(payload, path)
        print(f"[Save]     Model saved to {path}")

    @classmethod
    def load(cls, path: str = "results/dp_model.pt", device: str = "cpu") -> "DiffusionPolicy":
        """Load a previously saved model (no dataset/training needed)."""
        payload = torch.load(path, map_location=device, weights_only=False)
        obj = cls(
            to=payload["to"],
            ta=payload["ta"],
            obs_dim=payload["obs_dim"],
            device=device,
        )
        obj.net.load_state_dict(payload["net_state"])
        obj.net.eval()

        # Reconstruct a minimal dataset stub for normalizers
        if "obs_mean" in payload:
            stub = DemoDataset.__new__(DemoDataset)
            stub.obs_mean = payload["obs_mean"]
            stub.obs_std = payload["obs_std"]
            stub.action_min = payload["action_min"]
            stub.action_max = payload["action_max"]
            stub.obs_tensor = torch.empty(0)
            stub.action_tensor = torch.empty(0)
            stub.to = payload["to"]
            stub.ta = payload["ta"]
            obj.dataset = stub

        print(f"[Load]     Model loaded from {path}")
        return obj
