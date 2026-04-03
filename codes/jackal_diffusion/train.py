from __future__ import annotations

import pathlib

import hydra
from omegaconf import OmegaConf


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("jackal_diffusion", "config")),
    config_name="train_diffusion_lowdim_workspace",
)
def main(cfg: OmegaConf) -> None:
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
