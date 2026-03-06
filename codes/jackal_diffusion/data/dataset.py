from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    """
    Sliding-window dataset with receding horizon formulation.

    To = 2   observation horizon
    Ta = 8   action horizon

    Each sample:
        obs_seq    : Tensor (To, 8)
        action_seq : Tensor (Ta, 2)

    Normalization:
        actions -> min-max scaled to [-1, 1] per dimension
        obs     -> zero-mean, unit-variance
    """

    def __init__(
        self,
        demos: Sequence[dict],
        to: int = 2,
        ta: int = 8,
        stride: int = 2,
    ) -> None:
        self.to = int(to)
        self.ta = int(ta)
        self.stride = int(stride)

        obs_windows = []
        action_windows = []
        side_windows = []

        for demo in demos:
            obs = np.asarray(demo["obs"], dtype=np.float32)
            action = np.asarray(demo["action"], dtype=np.float32)
            side = str(demo.get("side", "unknown"))

            T = min(len(obs), len(action))
            if T < self.to + self.ta:
                continue

            obs = obs[:T]
            action = action[:T]

            for t in range(self.to - 1, T - self.ta + 1, self.stride):
                obs_seq = obs[t - self.to + 1 : t + 1]
                action_seq = action[t : t + self.ta]
                obs_windows.append(obs_seq)
                action_windows.append(action_seq)
                side_windows.append(side)

        if not obs_windows:
            raise ValueError("No valid sliding-window samples could be built.")

        self.obs_windows = np.asarray(obs_windows, dtype=np.float32)
        self.action_windows = np.asarray(action_windows, dtype=np.float32)
        self.side_windows = np.asarray(side_windows)

        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.action_min: np.ndarray | None = None
        self.action_max: np.ndarray | None = None
        self.obs_tensor: torch.Tensor | None = None
        self.action_tensor: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.obs_windows.shape[0])

    def get_split_indices(
        self, val_ratio: float = 0.1, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return train_idx, val_idx

    def fit_normalizers(self, train_indices: np.ndarray) -> None:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        obs_train = self.obs_windows[train_indices].reshape(-1, self.obs_windows.shape[-1])
        action_train = self.action_windows[train_indices].reshape(
            -1, self.action_windows.shape[-1]
        )

        self.obs_mean = obs_train.mean(axis=0)
        self.obs_std = obs_train.std(axis=0) + 1e-6

        action_min = action_train.min(axis=0)
        action_max = action_train.max(axis=0)

        # Handle near-constant dimensions (e.g., fixed speed command).
        range_ = action_max - action_min
        tiny = range_ < 1e-6
        if np.any(tiny):
            center = 0.5 * (action_min + action_max)
            action_min[tiny] = center[tiny] - 1.0
            action_max[tiny] = center[tiny] + 1.0

        self.action_min = action_min
        self.action_max = action_max

        # Cache normalized tensors once to avoid expensive per-item conversion.
        obs_norm = self.normalize_obs(self.obs_windows)
        action_norm = self.normalize_action(self.action_windows)
        self.obs_tensor = torch.tensor(obs_norm.tolist(), dtype=torch.float32)
        self.action_tensor = torch.tensor(action_norm.tolist(), dtype=torch.float32)

    @property
    def is_fitted(self) -> bool:
        return (
            self.obs_mean is not None
            and self.obs_std is not None
            and self.action_min is not None
            and self.action_max is not None
            and self.obs_tensor is not None
            and self.action_tensor is not None
        )

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is None or self.obs_std is None:
            raise RuntimeError("Call fit_normalizers() before normalization.")
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_min is None or self.action_max is None:
            raise RuntimeError("Call fit_normalizers() before normalization.")
        return 2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        if self.action_min is None or self.action_max is None:
            raise RuntimeError("Call fit_normalizers() before denormalization.")
        return 0.5 * (action_norm + 1.0) * (self.action_max - self.action_min) + self.action_min

    def sample_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.to, self.obs_windows.shape[-1]), (self.ta, self.action_windows.shape[-1])

    def __getitem__(self, index: int):
        if not self.is_fitted:
            raise RuntimeError("Dataset normalizers are not fitted.")

        return self.obs_tensor[index], self.action_tensor[index]
