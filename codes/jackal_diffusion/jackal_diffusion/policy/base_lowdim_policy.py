from __future__ import annotations

from typing import Dict

import torch

from jackal_diffusion.model.common.module_attr_mixin import ModuleAttrMixin
from jackal_diffusion.model.common.normalizer import LinearNormalizer


class BaseLowdimPolicy(ModuleAttrMixin):
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self) -> None:
        pass

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        raise NotImplementedError()
