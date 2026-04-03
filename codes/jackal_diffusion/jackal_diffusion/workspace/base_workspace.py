from __future__ import annotations

import copy
import pathlib
import threading
from typing import Optional

import dill
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self) -> str:
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        raise NotImplementedError()

    def save_checkpoint(
        self,
        path=None,
        tag: str = "latest",
        exclude_keys=None,
        include_keys=None,
        use_thread: bool = False,
    ) -> str:
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: _torch_save_payload(payload, path)
            )
            self._saving_thread.start()
        else:
            _torch_save_payload(payload, path)
        return str(path.absolute())

    def get_checkpoint_path(self, tag: str = "latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self,
        path=None,
        tag: str = "latest",
        exclude_keys=None,
        include_keys=None,
        **kwargs,
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        with path.open("rb") as f:
            payload = torch.load(f, pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(
        cls,
        path,
        exclude_keys=None,
        include_keys=None,
        **kwargs,
    ):
        with open(path, "rb") as f:
            payload = torch.load(f, pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag: str = "latest") -> str:
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            torch.save(self, f, pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        with open(path, "rb") as f:
            return torch.load(f, pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    if isinstance(x, dict):
        return {k: _copy_to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_copy_to_cpu(v) for v in x]
    return copy.deepcopy(x)


def _torch_save_payload(payload, path: pathlib.Path) -> None:
    with path.open("wb") as f:
        torch.save(payload, f, pickle_module=dill)
