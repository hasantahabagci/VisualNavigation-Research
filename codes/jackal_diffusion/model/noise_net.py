from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Sinusoidal embedding for diffusion timestep k."""
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
    """
    Feature-wise Linear Modulation.
    Conditioning input -> MLP -> (gamma, beta)
    output = gamma * x + beta
    Applied channel-wise after each Conv1d layer.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.affine = nn.Linear(cond_dim, channels * 2)
        with torch.no_grad():
            self.affine.bias[:channels].fill_(1.0)
            self.affine.bias[channels:].zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.affine(cond)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * x + beta


class NoiseNet(nn.Module):
    """
    Predicts noise epsilon_theta(O_t, A^k_t, k).

    Inputs:
        noisy_actions : Tensor (B, Ta, 2)
        obs           : Tensor (B, To * 8)
        k             : Tensor (B,)
    """

    def __init__(self, to: int = 2, obs_dim: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.obs_horizon = to
        self.obs_dim = obs_dim

        self.time_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.obs_proj = nn.Sequential(nn.Linear(self.obs_horizon * obs_dim, hidden), nn.SiLU())

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
        k: torch.Tensor,
    ) -> torch.Tensor:
        # (B, Ta, 2) -> (B, 2, Ta)
        x = noisy_actions.transpose(1, 2)

        k_emb = sinusoidal_embedding(k, dim=64)
        cond_emb = self.time_proj(k_emb)
        obs_emb = self.obs_proj(obs)
        film_cond = cond_emb + obs_emb

        x = F.relu(self.film1(self.conv1(x), film_cond))
        x = F.relu(self.film2(self.conv2(x), film_cond))
        x = F.relu(self.film3(self.conv3(x), film_cond))
        x = self.conv_out(x)

        # (B, 2, Ta) -> (B, Ta, 2)
        return x.transpose(1, 2)
