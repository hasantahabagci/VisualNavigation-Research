from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(cond_dim, channels * 2)
        with torch.no_grad():
            self.affine.bias[:channels].fill_(1.0)
            self.affine.bias[channels:].zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.affine(cond)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class NoiseNet(nn.Module):
    def __init__(self, n_obs_steps: int = 2, obs_dim: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.obs_dim = obs_dim

        self.time_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.obs_proj = nn.Sequential(nn.Linear(self.n_obs_steps * obs_dim, hidden), nn.SiLU())
        self.conv1 = nn.Conv1d(2, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(hidden, 2, kernel_size=1)
        self.film1 = FiLM(hidden, hidden)
        self.film2 = FiLM(hidden, hidden)
        self.film3 = FiLM(hidden, hidden)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        obs: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x = noisy_actions.transpose(1, 2)
        t_emb = sinusoidal_embedding(timesteps, dim=64)
        cond = self.time_proj(t_emb) + self.obs_proj(obs)
        x = F.relu(self.film1(self.conv1(x), cond))
        x = F.relu(self.film2(self.conv2(x), cond))
        x = F.relu(self.film3(self.conv3(x), cond))
        return self.conv_out(x).transpose(1, 2)
