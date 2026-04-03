from __future__ import annotations

import math

import numpy as np
import torch


class CosineScheduler:
    def __init__(self, k_train: int = 100, k_infer: int = 10, s: float = 0.008) -> None:
        self.k_train = int(k_train)
        self.k_infer = int(k_infer)
        self.s = float(s)

        steps = torch.arange(0, self.k_train + 1, dtype=torch.float32)
        t = steps / float(self.k_train)
        alpha_bar = torch.cos(((t + self.s) / (1.0 + self.s)) * math.pi / 2.0) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        self.alpha_bars = torch.clamp(alpha_bar, min=1e-6, max=1.0)
        self.infer_steps = self._build_infer_steps()

    def _build_infer_steps(self) -> list[int]:
        steps = np.round(np.linspace(self.k_train, 0, self.k_infer)).astype(int).tolist()
        clean_steps = [steps[0]]
        for step in steps[1:]:
            if step < clean_steps[-1]:
                clean_steps.append(int(step))
        if clean_steps[-1] != 0:
            clean_steps.append(0)
        return clean_steps

    def alpha_bar(self, k: torch.Tensor | int) -> torch.Tensor:
        if not torch.is_tensor(k):
            k = torch.tensor([k], dtype=torch.long)
        k = torch.clamp(k.long(), 0, self.k_train)
        return self.alpha_bars.to(k.device)[k]

    def add_noise(self, x0: torch.Tensor, eps: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bar(k).view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * eps

    def ddim_step(
        self,
        xt: torch.Tensor,
        eps_pred: torch.Tensor,
        k: torch.Tensor,
        k_prev: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar_k = self.alpha_bar(k).view(-1, 1, 1)
        alpha_bar_prev = self.alpha_bar(k_prev).view(-1, 1, 1)
        sqrt_alpha_bar_k = torch.sqrt(alpha_bar_k)
        sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_bar_k, min=1e-6))
        x0_pred = (xt - sqrt_one_minus * eps_pred) / torch.clamp(sqrt_alpha_bar_k, min=1e-6)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        return torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(
            torch.clamp(1.0 - alpha_bar_prev, min=1e-6)
        ) * eps_pred

    @torch.no_grad()
    def sample(self, net, obs: torch.Tensor, n_action_steps: int) -> torch.Tensor:
        batch_size = obs.shape[0]
        device = obs.device
        xt = torch.randn(batch_size, n_action_steps, 2, device=device)
        for i in range(len(self.infer_steps) - 1):
            k = self.infer_steps[i]
            k_prev = self.infer_steps[i + 1]
            k_tensor = torch.full((batch_size,), k, dtype=torch.long, device=device)
            k_prev_tensor = torch.full((batch_size,), k_prev, dtype=torch.long, device=device)
            eps_pred = net(xt, obs, k_tensor)
            xt = self.ddim_step(xt, eps_pred, k_tensor, k_prev_tensor)
        return torch.clamp(xt, -1.0, 1.0)
