from __future__ import annotations

import pathlib

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = pathlib.Path(__file__).resolve().parent / "config"


def get_config_path(config_name: str) -> pathlib.Path:
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    path = CONFIG_DIR / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def load_config(config_name: str):
    config_name = pathlib.Path(config_name).stem
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg
