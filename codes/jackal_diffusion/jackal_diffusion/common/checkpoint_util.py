from __future__ import annotations

import os
from typing import Dict, Optional


class TopKCheckpointManager:
    def __init__(
        self,
        save_dir: str,
        monitor_key: str,
        mode: str = "min",
        k: int = 1,
        format_str: str = "epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if k < 0:
            raise ValueError("k must be non-negative")
        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map: dict[str, float] = dict()

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == "max":
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None

        del self.path_value_map[delete_path]
        self.path_value_map[ckpt_path] = value

        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.exists(delete_path):
            os.remove(delete_path)
        return ckpt_path
