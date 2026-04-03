from __future__ import annotations

from typing import Dict

from jackal_diffusion.policy.base_lowdim_policy import BaseLowdimPolicy


class BaseLowdimRunner:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        raise NotImplementedError()
