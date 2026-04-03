from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jackal_diffusion.policy.jackal_base_lowdim_policy import JackalBaseLowdimPolicy


class JackalMLPLowdimPolicy(JackalBaseLowdimPolicy):
    def __init__(
        self,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        obs_dim: int = 16,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.n_obs_steps * self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_action_steps * self.action_dim),
        )

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return self.net(obs_flat).reshape(-1, self.n_action_steps, self.action_dim)

    def compute_loss(self, batch) -> torch.Tensor:
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"].to(device=self.device, dtype=self.dtype)
        action = nbatch["action"].to(device=self.device, dtype=self.dtype)
        pred = self.forward(obs.reshape(obs.shape[0], -1))
        return F.mse_loss(pred, action)

    def predict_action(self, obs_dict):
        nobs = self._normalize_obs(obs_dict)
        naction_pred = self.forward(nobs.reshape(nobs.shape[0], -1))
        action_pred = self._denormalize_action(naction_pred)
        return {
            "action": action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }

    def _build_legacy_payload(self) -> dict:
        return {
            "kind": "mlp",
            "model_state": self.net.state_dict(),
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "normalizer_state": self.normalizer.state_dict(),
        }

    @classmethod
    def load_legacy(cls, path: str, device: str = "cpu") -> "JackalMLPLowdimPolicy":
        payload = torch.load(path, map_location=device, weights_only=False)
        policy = cls(
            n_obs_steps=payload["n_obs_steps"],
            n_action_steps=payload["n_action_steps"],
            obs_dim=payload["obs_dim"],
            action_dim=payload.get("action_dim", 2),
            hidden_dim=payload.get("hidden_dim", 256),
        )
        policy.net.load_state_dict(payload["model_state"])
        if "normalizer_state" in payload:
            policy.normalizer.load_state_dict(payload["normalizer_state"])
        policy.to(device)
        policy.eval()
        return policy
