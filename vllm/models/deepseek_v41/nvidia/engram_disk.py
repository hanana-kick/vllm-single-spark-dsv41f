# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed DeepSeek V4.1 Engram row reader for single-node serving."""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

_HEADER_LEN = struct.Struct("<Q")


def _tensor_location(
    model_dir: Path, weight_map: dict[str, str], tensor_name: str
) -> tuple[int, int, tuple[int, ...], int]:
    """Return fd, payload offset, shape and element bytes for one tensor."""
    try:
        shard_name = weight_map[tensor_name]
    except KeyError as exc:
        raise KeyError(f"Engram tensor missing from checkpoint index: {tensor_name}") from exc
    path = model_dir / shard_name
    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        raw = os.pread(fd, _HEADER_LEN.size, 0)
        if len(raw) != _HEADER_LEN.size:
            raise ValueError(f"truncated safetensors header: {path}")
        (header_len,) = _HEADER_LEN.unpack(raw)
        header_raw = os.pread(fd, header_len, _HEADER_LEN.size)
        if len(header_raw) != header_len:
            raise ValueError(f"truncated safetensors JSON header: {path}")
        header = json.loads(header_raw)
        meta = header[tensor_name]
        start, end = meta["data_offsets"]
        shape = tuple(int(v) for v in meta["shape"])
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = int(end) - int(start)
        if numel <= 0 or nbytes % numel:
            raise ValueError(
                f"invalid Engram tensor byte size for {tensor_name}: "
                f"shape={shape} bytes={nbytes}"
            )
        element_bytes = nbytes // numel
        return fd, _HEADER_LEN.size + header_len + int(start), shape, element_bytes
    except Exception:
        os.close(fd)
        raise


def _find_engram_name(
    weight_map: dict[str, str], layer_id: int, suffix: str
) -> str:
    """Accept both raw HF and loader-normalized checkpoint prefixes."""
    candidates = (
        f"layers.{layer_id}.engram.embed.{suffix}",
        f"model.layers.{layer_id}.engram.embed.{suffix}",
        f"layers.{layer_id}.engram.embed_tokens.{suffix}",
        f"model.layers.{layer_id}.engram.embed_tokens.{suffix}",
    )
    for name in candidates:
        if name in weight_map:
            return name
    expected = ", ".join(candidates)
    raise KeyError(f"Engram layer {layer_id} {suffix} not found; tried {expected}")


class DiskEngramTable:
    """Read only selected FP8 Engram rows directly from safetensors."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        layer_id: int,
        dim: int,
        block_size: int,
        row_start: int,
        num_rows: int,
        threads: int = 32,
    ) -> None:
        root = Path(model_dir)
        index_path = root / "model.safetensors.index.json"
        with index_path.open(encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        w_name = _find_engram_name(weight_map, layer_id, "weight")
        # Upstream checkpoints have used both "scale" and
        # "weight_scale_inv" naming across conversion paths.
        try:
            s_name = _find_engram_name(weight_map, layer_id, "scale")
        except KeyError:
            s_name = _find_engram_name(weight_map, layer_id, "weight_scale_inv")

        self.w_fd, w_base, w_shape, w_element = _tensor_location(
            root, weight_map, w_name
        )
        try:
            self.s_fd, s_base, s_shape, s_element = _tensor_location(
                root, weight_map, s_name
            )
        except Exception:
            os.close(self.w_fd)
            raise

        scale_cols = dim // block_size
        if w_shape[-1] != dim or s_shape[-1] != scale_cols:
            self.close()
            raise ValueError(
                f"Engram shape mismatch: weight={w_shape}, scale={s_shape}, "
                f"expected (*,{dim}) and (*,{scale_cols})"
            )
        if w_shape[0] != s_shape[0] or w_element != 1 or s_element != 1:
            self.close()
            raise ValueError(
                "Engram disk reader expects one-byte FP8 weights/scales with "
                f"matching rows, got {w_shape}/{s_shape} and "
                f"{w_element}/{s_element} bytes"
            )
        if not 0 <= row_start <= row_start + num_rows <= w_shape[0]:
            self.close()
            raise ValueError(
                f"Engram row shard [{row_start}, {row_start + num_rows}) "
                f"outside checkpoint rows {w_shape[0]}"
            )

        self.dim = dim
        self.scale_cols = scale_cols
        self.row_start = row_start
        self.num_rows = num_rows
        self.w_base = w_base + row_start * dim
        self.s_base = s_base + row_start * scale_cols
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, threads), thread_name_prefix="vllm-engram-disk"
        )

    @staticmethod
    def _pread_row(fd: int, offset: int, size: int) -> bytes:
        data = os.pread(fd, size, offset)
        if len(data) != size:
            raise OSError(f"short Engram read at {offset}: {len(data)}/{size}")
        return data

    def read_rows(
        self, local_rows: torch.Tensor, owned: torch.Tensor
    ) -> torch.Tensor:
        """Return BF16 [R, dim], zeroing rows not owned by this shard."""
        local_rows = local_rows.to(device="cpu", dtype=torch.int64).reshape(-1)
        owned = owned.to(device="cpu", dtype=torch.bool).reshape(-1)
        if local_rows.numel() != owned.numel():
            raise ValueError("local_rows and owned must have equal length")
        count = local_rows.numel()
        if count == 0:
            return torch.empty((0, self.dim), dtype=torch.bfloat16)

        valid_rows = local_rows[owned]
        if valid_rows.numel() and (
            int(valid_rows.min()) < 0 or int(valid_rows.max()) >= self.num_rows
        ):
            raise IndexError("Engram local row outside this rank's disk shard")

        unique, inverse = torch.unique(valid_rows, sorted=False, return_inverse=True)
        unique_list = [int(v) for v in unique.tolist()]

        w_futures = [
            self._pool.submit(
                self._pread_row, self.w_fd, self.w_base + row * self.dim, self.dim
            )
            for row in unique_list
        ]
        s_futures = [
            self._pool.submit(
                self._pread_row,
                self.s_fd,
                self.s_base + row * self.scale_cols,
                self.scale_cols,
            )
            for row in unique_list
        ]

        if unique_list:
            w_bytes = b"".join(f.result() for f in w_futures)
            s_bytes = b"".join(f.result() for f in s_futures)
            w = torch.frombuffer(bytearray(w_bytes), dtype=torch.uint8).reshape(
                len(unique_list), self.dim
            )
            s = torch.frombuffer(bytearray(s_bytes), dtype=torch.uint8).reshape(
                len(unique_list), self.scale_cols
            )
            vals = w.view(torch.float8_e4m3fn).to(torch.float32).reshape(
                len(unique_list), self.scale_cols, self.dim // self.scale_cols
            )
            scale = (s.to(torch.int32) << 23).view(torch.float32)
            dequant = (vals * scale[:, :, None]).reshape(
                len(unique_list), self.dim
            )
        else:
            dequant = torch.empty((0, self.dim), dtype=torch.float32)

        out = torch.zeros((count, self.dim), dtype=torch.bfloat16)
        if valid_rows.numel():
            out[owned] = dequant[inverse].to(torch.bfloat16)
        return out

    def close(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
            self._pool = None
        for name in ("w_fd", "s_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
