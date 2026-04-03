from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn

from jackal_diffusion.common.pytorch_util import dict_apply
from jackal_diffusion.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    available_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray],
        last_n_dims: int = 1,
        dtype=torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer") -> None:
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward: bool = True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result

        if "_default" not in self.params_dict:
            raise RuntimeError("Normalizer is not initialized.")
        params = self.params_dict["_default"]
        return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Normalizer is not initialized.")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self) -> Dict:
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    available_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray],
        last_n_dims: int = 1,
        dtype=torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        def to_tensor(x):
            return _to_tensor(x).flatten()

        scale = to_tensor(scale)
        offset = to_tensor(offset)
        params_dict = nn.ParameterDict(
            {
                "scale": scale,
                "offset": offset,
                "input_stats": nn.ParameterDict(dict_apply(input_stats_dict, to_tensor)),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, np.ndarray],
    last_n_dims: int = 1,
    dtype=torch.float32,
    mode: str = "limits",
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
    fit_offset: bool = True,
):
    if mode not in {"limits", "gaussian"}:
        raise ValueError("mode must be 'limits' or 'gaussian'")
    if last_n_dims < 0:
        raise ValueError("last_n_dims must be non-negative")
    if output_max <= output_min:
        raise ValueError("output_max must be greater than output_min")

    if not isinstance(data, torch.Tensor):
        data = _to_tensor(data)
    if dtype is not None:
        data = data.type(dtype)

    dim = 1
    if last_n_dims > 0:
        dim = int(np.prod(data.shape[-last_n_dims:]))
    data = data.reshape(-1, dim)

    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

    params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    for parameter in params.parameters():
        parameter.requires_grad_(False)
    return params


def _normalize(x, params, forward: bool = True):
    if not isinstance(x, torch.Tensor):
        x = _to_tensor(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    return x.reshape(src_shape)


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    array = np.asarray(x)
    torch_dtype = _numpy_dtype_to_torch(array.dtype)
    if array.ndim == 0:
        return torch.tensor(array.item(), dtype=torch_dtype)
    return torch.tensor(array.tolist(), dtype=torch_dtype)


def _numpy_dtype_to_torch(dtype) -> torch.dtype:
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return torch.bool
    if dtype.kind == "f":
        if dtype.itemsize <= 2:
            return torch.float16
        if dtype.itemsize <= 4:
            return torch.float32
        return torch.float64
    if dtype.kind == "c":
        if dtype.itemsize <= 8:
            return torch.complex64
        return torch.complex128
    if dtype.kind in {"i", "u"}:
        if dtype.itemsize <= 1:
            return torch.int8 if dtype.kind == "i" else torch.uint8
        if dtype.itemsize <= 2:
            return torch.int16
        if dtype.itemsize <= 4:
            return torch.int32
        return torch.int64
    return torch.float32
