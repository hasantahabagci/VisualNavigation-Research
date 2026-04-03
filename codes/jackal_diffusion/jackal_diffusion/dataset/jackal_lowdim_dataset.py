from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch

from jackal_diffusion.dataset.base_dataset import BaseLowdimDataset
from jackal_diffusion.dataset.expert import collect_all
from jackal_diffusion.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _float_tensor(x) -> torch.Tensor:
    return torch.tensor(np.asarray(x).tolist(), dtype=torch.float32)


class JackalLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        n_demos_per_side: int = 25,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        stride: int = 2,
        val_ratio: float = 0.1,
        seed: int = 42,
        demos: Sequence[dict] | None = None,
    ) -> None:
        super().__init__()
        self.n_demos_per_side = int(n_demos_per_side)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.stride = int(stride)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        self.demos = list(collect_all(n_per_side=self.n_demos_per_side) if demos is None else demos)
        self.obs_windows, self.action_windows, self.side_windows = self._build_windows(self.demos)
        self.train_indices, self.val_indices = self._build_split_indices()
        self.indices = self.train_indices
        self.split_name = "train"
        self._normalizer: LinearNormalizer | None = None

    def _build_windows(
        self, demos: Sequence[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_windows = []
        action_windows = []
        side_windows = []
        for demo in demos:
            obs = np.asarray(demo["obs"], dtype=np.float32)
            action = np.asarray(demo["action"], dtype=np.float32)
            side = str(demo.get("side", "unknown"))
            length = min(len(obs), len(action))
            if length < self.n_obs_steps + self.n_action_steps:
                continue
            obs = obs[:length]
            action = action[:length]
            for t in range(self.n_obs_steps - 1, length - self.n_action_steps + 1, self.stride):
                obs_seq = obs[t - self.n_obs_steps + 1 : t + 1]
                action_seq = action[t : t + self.n_action_steps]
                obs_windows.append(obs_seq)
                action_windows.append(action_seq)
                side_windows.append(side)
        if not obs_windows:
            raise ValueError("No valid sliding-window samples could be built.")
        return (
            np.asarray(obs_windows, dtype=np.float32),
            np.asarray(action_windows, dtype=np.float32),
            np.asarray(side_windows),
        )

    def _build_split_indices(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        indices = np.arange(self.obs_windows.shape[0])
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * self.val_ratio))
        val_idx = np.sort(indices[:n_val])
        train_idx = np.sort(indices[n_val:])
        return train_idx, val_idx

    @property
    def obs_dim(self) -> int:
        return int(self.obs_windows.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.action_windows.shape[-1])

    def get_validation_dataset(self) -> "JackalLowdimDataset":
        val_set = copy.copy(self)
        val_set.indices = self.val_indices
        val_set.split_name = "val"
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self._normalizer is None:
            train_obs = self.obs_windows[self.train_indices]
            train_action = self.action_windows[self.train_indices]
            normalizer = LinearNormalizer()
            normalizer["obs"] = SingleFieldLinearNormalizer.create_fit(
                train_obs, last_n_dims=1, mode="gaussian", **kwargs
            )
            normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
                train_action, last_n_dims=1, mode="limits", **kwargs
            )
            self._normalizer = normalizer
        return self._normalizer

    def get_all_actions(self) -> torch.Tensor:
        return _float_tensor(self.action_windows[self.indices])

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        window_idx = int(self.indices[idx])
        return {
            "obs": _float_tensor(self.obs_windows[window_idx]),
            "action": _float_tensor(self.action_windows[window_idx]),
        }

    def describe(self) -> dict:
        left_demos = sum(d["side"] == "left" for d in self.demos)
        right_demos = sum(d["side"] == "right" for d in self.demos)
        return {
            "n_demos": len(self.demos),
            "n_left_demos": left_demos,
            "n_right_demos": right_demos,
            "n_total_windows": int(self.obs_windows.shape[0]),
            "n_train_windows": int(len(self.train_indices)),
            "n_val_windows": int(len(self.val_indices)),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
