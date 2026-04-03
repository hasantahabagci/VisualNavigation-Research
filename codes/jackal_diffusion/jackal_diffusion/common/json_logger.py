from __future__ import annotations

import copy
import json
import numbers
import os
from typing import Any, Callable, Optional


class JsonLogger:
    def __init__(
        self,
        path: str,
        filter_fn: Optional[Callable[[str, Any], bool]] = None,
    ) -> None:
        if filter_fn is None:
            filter_fn = lambda k, v: isinstance(v, numbers.Number)
        self.path = path
        self.filter_fn = filter_fn
        self.file = None
        self.last_log = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            self.file = file = open(self.path, "r+", buffering=1)
        except FileNotFoundError:
            self.file = file = open(self.path, "w+", buffering=1)

        pos = file.seek(0, os.SEEK_END)
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        last_line_end = file.tell()

        pos = max(0, pos - 1)
        file.seek(pos, os.SEEK_SET)
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        last_line_start = file.tell()

        if last_line_start < last_line_end:
            last_line = file.readline()
            self.last_log = json.loads(last_line)

        file.seek(last_line_end)
        file.truncate()

    def stop(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self) -> "JsonLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def log(self, data: dict) -> None:
        filtered_data = dict(filter(lambda x: self.filter_fn(*x), data.items()))
        self.last_log = filtered_data
        for key, value in filtered_data.items():
            if isinstance(value, numbers.Integral):
                filtered_data[key] = int(value)
            elif isinstance(value, numbers.Number):
                filtered_data[key] = float(value)
        buf = json.dumps(filtered_data).replace("\n", "") + "\n"
        assert self.file is not None
        self.file.write(buf)

    def get_last_log(self) -> Optional[dict]:
        return copy.deepcopy(self.last_log)
