from __future__ import annotations

import torch
import torch.nn.functional as F

from jackal_diffusion.model.noise_net import NoiseNet
from jackal_diffusion.model.scheduler import CosineScheduler
from jackal_diffusion.policy.jackal_base_lowdim_policy import JackalBaseLowdimPolicy


class JackalDiffusionLowdimPolicy(JackalBaseLowdimPolicy):
    def __init__(
        self,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        obs_dim: int = 16,
        action_dim: int = 2,
        hidden_dim: int = 64,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
    ) -> None:
        super().__init__(
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        self.hidden_dim = int(hidden_dim)
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.net = NoiseNet(
            n_obs_steps=self.n_obs_steps,
            obs_dim=self.obs_dim,
            hidden=self.hidden_dim,
        )
        self.scheduler = CosineScheduler(
            k_train=self.num_train_timesteps,
            k_infer=self.num_inference_steps,
        )

    def compute_loss(self, batch) -> torch.Tensor:
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"].to(device=self.device, dtype=self.dtype)
        action = nbatch["action"].to(device=self.device, dtype=self.dtype)
        bsz = obs.shape[0]
        obs_flat = obs.reshape(bsz, -1)
        k = torch.randint(
            low=1,
            high=self.scheduler.k_train + 1,
            size=(bsz,),
            device=self.device,
        )
        eps = torch.randn_like(action)
        noisy_actions = self.scheduler.add_noise(action, eps, k)
        eps_pred = self.net(noisy_actions, obs_flat, k)
        return F.mse_loss(eps_pred, eps)

    def predict_action(self, obs_dict):
        nobs = self._normalize_obs(obs_dict)
        bsz = nobs.shape[0]
        obs_flat = nobs.reshape(bsz, -1)
        with torch.no_grad():
            naction_pred = self.scheduler.sample(
                self.net,
                obs_flat,
                self.n_action_steps,
            )
        action_pred = self._denormalize_action(naction_pred)
        return {
            "action": action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }

    def _build_legacy_payload(self) -> dict:
        payload = {
            "kind": "diffusion",
            "model_state": self.net.state_dict(),
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "normalizer_state": self.normalizer.state_dict(),
        }
        return payload

    @classmethod
    def load_legacy(cls, path: str, device: str = "cpu") -> "JackalDiffusionLowdimPolicy":
        payload = torch.load(path, map_location=device, weights_only=False)
        if "model_state" in payload:
            policy = cls(
                n_obs_steps=payload["n_obs_steps"],
                n_action_steps=payload["n_action_steps"],
                obs_dim=payload["obs_dim"],
                action_dim=payload.get("action_dim", 2),
                hidden_dim=payload.get("hidden_dim", 64),
                num_train_timesteps=payload.get("num_train_timesteps", 100),
                num_inference_steps=payload.get("num_inference_steps", 10),
            )
            policy.net.load_state_dict(payload["model_state"])
            if "normalizer_state" in payload:
                policy.normalizer.load_state_dict(payload["normalizer_state"])
        else:
            policy = cls(
                n_obs_steps=payload["to"],
                n_action_steps=payload["ta"],
                obs_dim=payload["obs_dim"],
            )
            policy.net.load_state_dict(payload["net_state"])
            if "obs_mean" in payload:
                normalizer = cls._normalizer_from_legacy_stats(
                    obs_mean=payload["obs_mean"],
                    obs_std=payload["obs_std"],
                    action_min=payload["action_min"],
                    action_max=payload["action_max"],
                )
                policy.set_normalizer(normalizer)
        policy.to(device)
        policy.eval()
        return policy
