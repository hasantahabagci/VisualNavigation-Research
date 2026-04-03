from __future__ import annotations

import hydra
import pathlib
from omegaconf import OmegaConf

from jackal_diffusion.workspace.train_base_lowdim_workspace import TrainBaseLowdimWorkspace


class TrainDiffusionLowdimWorkspace(TrainBaseLowdimWorkspace):
    pass


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg: OmegaConf):
    workspace = TrainDiffusionLowdimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
