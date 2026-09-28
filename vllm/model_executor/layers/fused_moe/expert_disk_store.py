# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aligned on-disk records for paged MoE experts.

The store is deliberately independent of a particular quantization method.
One record contains every runtime tensor needed by one expert. Records are
fixed-stride and 4096-byte aligned so the serving path can use O_DIRECT.

create_for_streaming() is the important construction path for models that do
not fit in host memory: the loader knows tensor shapes before payloads arrive,
sizes the store, and writes one completed expert at a time. No full
[num_experts, ...] backing tensor is required.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ALIGN = 4096

_DTYPE_NAMES = {
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e8m0fnu: "float8_e8m0fnu",
}


@dataclass(frozen=True)
class ExpertStoreField:
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class DiskExpertStore:
    """One fixed-size record per expert for one MoE layer."""

    def __init__(
        self,
        path: str | Path,
        num_experts: int,
        fields: list[ExpertStoreField],
        *,
        direct_io: bool = True,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not fields:
            raise ValueError("at least one expert field is required")
        self.path = Path(path)
        self.num_experts = num_experts
        self.fields = {field.name: field for field in fields}
        if len(self.fields) != len(fields):
            raise ValueError("duplicate expert store field name")
        raw = max(field.offset + field.nbytes for field in fields)
        self.record_stride = _align_up(raw, ALIGN)
        self.direct_io = direct_io
        self.is_complete = False
        self._fd: int | None = None
        self._open_lock = threading.Lock()
        self._using_direct_io = False
        self._wfd: int | None = None
        self._lock_file: TextIO | None = None
        self._written: set[int] = set()
        self._identity: dict[str, object] = {}

    @staticmethod
    def make_fields(
        specs: list[tuple[str, tuple[int, ...], torch.dtype]],
    ) -> list[ExpertStoreField]:
        fields: list[ExpertStoreField] = []
        offset = 0
        for name, shape, dtype in specs:
            if dtype not in _DTYPE_NAMES:
                raise ValueError(f"unsupported expert store dtype: {dtype}")
            if any(dim < 0 for dim in shape):
                raise ValueError(f"negative dimension in {name}: {shape}")
            numel = 1
            for dim in shape:
                numel *= dim
            element_size = torch.empty((), dtype=dtype).element_size()
            nbytes = numel * element_size
            fields.append(
                ExpertStoreField(
                    name=name,
                    offset=offset,
                    nbytes=nbytes,
                    shape=shape,
                    dtype=dtype,
                )
            )
            offset += nbytes
        return fields

    @classmethod
    def create_for_streaming(
        cls,
        path: str | Path,
        num_experts: int,
        specs: list[tuple[str, tuple[int, ...], torch.dtype]],
        *,
        identity: dict[str, object] | None = None,
        direct_io: bool = True,
    ) -> "DiskExpertStore":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = cls.make_fields(specs)
        store = cls(path, num_experts, fields, direct_io=direct_io)
        store._identity = dict(identity or {})

        lock_file = open(str(path) + ".lock", "w")  # noqa: SIM115
        store._lock_file = lock_file
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        expected = store._fingerprint()
        sidecar = Path(str(path) + ".json")
        if sidecar.is_file() and path.is_file():
            try:
                with sidecar.open(encoding="utf-8") as f:
                    existing = json.load(f)
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing == expected and path.stat().st_size == (
                num_experts * store.record_stride
            ):
                store.is_complete = True
                store._release_lock()
                logger.info("Reusing expert disk store %s", path)
                return store

        tmp = Path(str(path) + ".tmp")
        store._wfd = os.open(
            tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644
        )
        os.ftruncate(store._wfd, num_experts * store.record_stride)
        return store

    def _fingerprint(self) -> dict[str, object]:
        return {
            "version": 1,
            "identity": self._identity,
            "num_experts": self.num_experts,
            "record_stride": self.record_stride,
            "fields": [
                {
                    "name": field.name,
                    "offset": field.offset,
                    "nbytes": field.nbytes,
                    "shape": list(field.shape),
                    "dtype": _DTYPE_NAMES[field.dtype],
                }
                for field in self.fields.values()
            ],
        }

    def field_view(self, record: torch.Tensor, name: str) -> torch.Tensor:
        """Return a typed view into one uint8 record buffer."""
        field = self.fields[name]
        if record.dtype != torch.uint8 or record.numel() != self.record_stride:
            raise ValueError(
                f"record must be uint8[{self.record_stride}], got "
                f"{record.dtype}[{record.numel()}]"
            )
        data = record[field.offset : field.offset + field.nbytes]
        return data.view(field.dtype).reshape(field.shape)

    def write_record(self, expert_id: int, record: torch.Tensor) -> None:
        """Write a completed expert during checkpoint streaming."""
        self._check_expert_id(expert_id)
        if self.is_complete or self._wfd is None:
            raise RuntimeError("expert store is not open for streaming writes")
        if record.device.type != "cpu":
            raise ValueError("streaming record must be a CPU tensor")
        if record.dtype != torch.uint8 or record.numel() != self.record_stride:
            raise ValueError(
                f"record must be uint8[{self.record_stride}]"
            )
        payload = memoryview(record.contiguous().numpy())
        offset = expert_id * self.record_stride
        written = 0
        while written < self.record_stride:
            nbytes = os.pwrite(self._wfd, payload[written:], offset + written)
            if nbytes <= 0:
                raise OSError(
                    f"short expert store write: expert={expert_id} "
                    f"{written}/{self.record_stride}"
                )
            written += nbytes
        self._written.add(expert_id)
        if len(self._written) % 16 == 0:
            os.fdatasync(self._wfd)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(self._wfd, 0, 0, os.POSIX_FADV_DONTNEED)

    def finalize(self) -> None:
        """Atomically publish a streaming store after every expert was written."""
        if self.is_complete:
            return
        if self._wfd is None:
            raise RuntimeError("expert store has no active streaming writer")
        if len(self._written) != self.num_experts:
            missing = self.num_experts - len(self._written)
            raise RuntimeError(
                f"cannot finalize expert store with {missing} unwritten experts"
            )

        os.fsync(self._wfd)
        os.close(self._wfd)
        self._wfd = None
        tmp = Path(str(self.path) + ".tmp")
        os.replace(tmp, self.path)
        sidecar = Path(str(self.path) + ".json")
        sidecar_tmp = Path(str(sidecar) + ".tmp")
        with sidecar_tmp.open("w", encoding="utf-8") as f:
            json.dump(self._fingerprint(), f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(sidecar_tmp, sidecar)
        self.is_complete = True
        self._release_lock()
        logger.info(
            "Published expert disk store %s (%d experts x %.2f MiB)",
            self.path,
            self.num_experts,
            self.record_stride / (1 << 20),
        )

    def read_record(self, expert_id: int, dst: torch.Tensor) -> int:
        """Read one complete record, preferring O_DIRECT when requested."""
        self._check_expert_id(expert_id)
        if not self.is_complete:
            raise RuntimeError("cannot read an incomplete expert store")
        if dst.device.type != "cpu":
            raise ValueError("expert read destination must be a CPU tensor")
        if dst.dtype != torch.uint8 or dst.numel() != self.record_stride:
            raise ValueError(
                f"destination must be uint8[{self.record_stride}]"
            )

        fd = self._open_reader()
        view = memoryview(dst.numpy())
        offset = expert_id * self.record_stride
        got = 0
        while got < self.record_stride:
            try:
                nbytes = os.preadv(fd, [view[got:]], offset + got)
            except OSError as exc:
                if self._using_direct_io:
                    raise OSError(
                        exc.errno,
                        "O_DIRECT expert read failed; destination must be "
                        "page-aligned and record size/alignment must satisfy "
                        f"the filesystem (expert={expert_id}, "
                        f"ptr_mod={dst.data_ptr() % ALIGN}, "
                        f"offset_mod={(offset + got) % ALIGN})",
                    ) from exc
                raise
            if nbytes <= 0:
                raise OSError(
                    f"short expert store read: expert={expert_id} "
                    f"{got}/{self.record_stride}"
                )
            got += nbytes
        return got

    def _open_reader(self) -> int:
        with self._open_lock:
            if self._fd is not None:
                return self._fd
            flags = os.O_RDONLY
            direct_flag = getattr(os, "O_DIRECT", 0) if self.direct_io else 0
            if direct_flag:
                try:
                    self._fd = os.open(self.path, flags | direct_flag)
                    self._using_direct_io = True
                    return self._fd
                except OSError as exc:
                    logger.warning_once(
                        "O_DIRECT unavailable for expert store %s (%s); "
                        "falling back to buffered reads.",
                        self.path,
                        exc,
                    )
            self._fd = os.open(self.path, flags)
            self._using_direct_io = False
            return self._fd

    def _check_expert_id(self, expert_id: int) -> None:
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(
                f"expert_id {expert_id} outside [0, {self.num_experts})"
            )

    def _release_lock(self) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._wfd is not None:
            os.close(self._wfd)
            self._wfd = None
        self._release_lock()


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
