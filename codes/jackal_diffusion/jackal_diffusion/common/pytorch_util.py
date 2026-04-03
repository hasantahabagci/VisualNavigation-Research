from __future__ import annotations

from typing import Callable, Dict

import torch


def dict_apply(
    x: Dict[str, torch.Tensor],
    func: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> torch.optim.Optimizer:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)
    return optimizer
