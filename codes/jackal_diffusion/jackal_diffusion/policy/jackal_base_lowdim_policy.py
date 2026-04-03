from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from jackal_diffusion.env import arena, dynamics
from jackal_diffusion.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from jackal_diffusion.policy.base_lowdim_policy import BaseLowdimPolicy


class JackalBaseLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        n_obs_steps: int,
        n_action_steps: int,
        obs_dim: int,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "obs" not in obs_dict:
            raise KeyError("obs_dict must contain 'obs'.")
        obs = obs_dict["obs"]
        if obs.shape[-2:] != (self.n_obs_steps, self.obs_dim):
            raise ValueError(
                f"Expected obs shape (*, {self.n_obs_steps}, {self.obs_dim}), got {tuple(obs.shape)}"
            )
        nobs = self.normalizer["obs"].normalize(obs)
        return nobs.to(device=self.device, dtype=self.dtype)

    def _denormalize_action(self, naction: torch.Tensor) -> torch.Tensor:
        action = self.normalizer["action"].unnormalize(naction)
        return self._postprocess_action(action)

    def _postprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clone()
        stats = self.normalizer["action"].get_input_stats()
        action_min = stats["min"].to(device=action.device, dtype=action.dtype)
        action_max = stats["max"].to(device=action.device, dtype=action.dtype)
        v_center = 0.5 * (action_min[0] + action_max[0])
        action[..., 0] = v_center
        action[..., 1] = torch.clamp(action[..., 1], action_min[1], action_max[1])
        return torch.clamp(action, -1.0, 1.0)

    def run_episode(
        self,
        start: tuple[float, float, float],
        goal: np.ndarray,
        max_steps: int = 300,
        exec_horizon: int = 1,
    ) -> dict:
        state = np.array([start[0], start[1], start[2], 0.0, 0.0], dtype=np.float32)
        trajectory = [state.copy()]
        obs_history = [arena.get_observation(state, goal) for _ in range(self.n_obs_steps)]
        success = False
        collision = False
        steps = 0

        self.eval()
        while steps < max_steps and not success and not collision:
            obs_seq = np.asarray(obs_history[-self.n_obs_steps :], dtype=np.float32)
            obs_tensor = torch.tensor(obs_seq.tolist(), dtype=self.dtype, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_seq = self.predict_action({"obs": obs_tensor})["action"][0]
            actions = action_seq.detach().cpu().numpy()

            for action in actions[:exec_horizon]:
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

    def save_legacy(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._build_legacy_payload(), path)

    def _build_legacy_payload(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def _normalizer_from_legacy_stats(
        cls,
        obs_mean,
        obs_std,
        action_min,
        action_max,
    ) -> LinearNormalizer:
        obs_mean = np.asarray(obs_mean, dtype=np.float32)
        obs_std = np.asarray(obs_std, dtype=np.float32)
        action_min = np.asarray(action_min, dtype=np.float32)
        action_max = np.asarray(action_max, dtype=np.float32)

        obs_scale = 1.0 / np.maximum(obs_std, 1e-6)
        obs_offset = -obs_mean * obs_scale
        obs_stats = {
            "min": obs_mean - obs_std,
            "max": obs_mean + obs_std,
            "mean": obs_mean,
            "std": np.maximum(obs_std, 1e-6),
        }

        action_range = np.maximum(action_max - action_min, 1e-6)
        action_scale = 2.0 / action_range
        action_offset = -1.0 - action_scale * action_min
        action_stats = {
            "min": action_min,
            "max": action_max,
            "mean": 0.5 * (action_min + action_max),
            "std": np.maximum(0.5 * action_range, 1e-6),
        }

        normalizer = LinearNormalizer()
        normalizer["obs"] = SingleFieldLinearNormalizer.create_manual(
            scale=obs_scale,
            offset=obs_offset,
            input_stats_dict=obs_stats,
        )
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=action_scale,
            offset=action_offset,
            input_stats_dict=action_stats,
        )
        return normalizer
