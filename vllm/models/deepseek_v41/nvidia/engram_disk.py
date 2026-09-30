# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed DeepSeek V4.1 Engram row reader for single-node serving."""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

_HEADER_LEN = struct.Struct("<Q")
_DROP_ENGRAM_PAGE_CACHE = (
    os.environ.get("VLLM_DSV41_ENGRAM_DROP_PAGE_CACHE", "1") != "0"
)
_ENGRAM_STAGE_POOL = ThreadPoolExecutor(
    max_workers=max(
        1, int(os.environ.get("VLLM_DSV41_ENGRAM_STAGE_WORKERS", "4"))
    ),
    thread_name_prefix="vllm-engram-stage",
)
# All Engram tables share one NVMe queue. The model submits every layer before
# decoder execution, so per-table pools would multiply thread count (e.g. two
# 32-worker pools) without increasing the device's useful queue depth.
_ENGRAM_IO_POOL = ThreadPoolExecutor(
    max_workers=max(
        1, int(os.environ.get("VLLM_DSV41_ENGRAM_IO_WORKERS", "32"))
    ),
    thread_name_prefix="vllm-engram-io",
)


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
        # Kept for API/backward compatibility; the process-wide I/O
        # pool controls actual queue depth across all Engram layers.
        self.threads = max(1, threads)
        self._pool = _ENGRAM_IO_POOL

    @staticmethod
    def _pread_exact(fd: int, offset: int, size: int) -> bytes:
        data = os.pread(fd, size, offset)
        if len(data) != size:
            raise OSError(f"short Engram read at {offset}: {len(data)}/{size}")
        if (
            _DROP_ENGRAM_PAGE_CACHE
            and hasattr(os, "posix_fadvise")
            and hasattr(os, "POSIX_FADV_DONTNEED")
        ):
            try:
                os.posix_fadvise(
                    fd, offset, size, os.POSIX_FADV_DONTNEED
                )
            except OSError:
                pass
        return data

    @staticmethod
    def _contiguous_runs(rows: list[int]) -> list[tuple[int, int]]:
        """Convert sorted unique row ids to inclusive-start/exclusive-end runs."""
        if not rows:
            return []
        runs: list[tuple[int, int]] = []
        start = prev = rows[0]
        for row in rows[1:]:
            if row != prev + 1:
                runs.append((start, prev + 1))
                start = row
            prev = row
        runs.append((start, prev + 1))
        return runs

    def _submit_runs(
        self,
        fd: int,
        base: int,
        runs: list[tuple[int, int]],
        row_bytes: int,
    ) -> list[tuple[int, int, Future[bytes]]]:
        """Submit one positional read per contiguous run.

        read_rows itself runs on the separate Engram staging executor, so these
        tasks can safely occupy the row-I/O pool without recursive pool waits.
        Random hash rows keep high NVMe queue depth, while adjacent rows still
        collapse into one syscall.
        """
        return [
            (
                start,
                end,
                self._pool.submit(
                    self._pread_exact,
                    fd,
                    base + start * row_bytes,
                    (end - start) * row_bytes,
                ),
            )
            for start, end in runs
        ]

    @staticmethod
    def _collect_runs(
        futures: list[tuple[int, int, Future[bytes]]],
        row_bytes: int,
    ) -> dict[int, bytes]:
        result: dict[int, bytes] = {}
        for start, end, future in futures:
            block = future.result()
            for i, row in enumerate(range(start, end)):
                lo = i * row_bytes
                result[row] = block[lo : lo + row_bytes]
        return result

    def submit_rows(
        self, local_rows: torch.Tensor, owned: torch.Tensor
    ) -> Future[torch.Tensor]:
        """Submit a row gather so decoder compute can overlap the disk I/O."""
        return _ENGRAM_STAGE_POOL.submit(self.read_rows, local_rows, owned)

    @staticmethod
    def _pread_into(fd: int, offset: int, view: memoryview) -> int:
        total = len(view)
        got = 0
        while got < total:
            nbytes = os.preadv(fd, [view[got:]], offset + got)
            if nbytes <= 0:
                raise OSError(
                    f"short Engram read at {offset}: {got}/{total}"
                )
            got += nbytes
        if (
            _DROP_ENGRAM_PAGE_CACHE
            and hasattr(os, "posix_fadvise")
            and hasattr(os, "POSIX_FADV_DONTNEED")
        ):
            try:
                os.posix_fadvise(
                    fd, offset, total, os.POSIX_FADV_DONTNEED
                )
            except OSError:
                pass
        return got

    def _submit_runs_into(
        self,
        fd: int,
        base: int,
        runs: list[tuple[int, int]],
        row_bytes: int,
        dst: torch.Tensor,
    ) -> list[Future[int]]:
        """Read sorted contiguous runs directly into a compact pinned tensor."""
        raw = memoryview(dst.numpy()).cast("B")
        futures: list[Future[int]] = []
        cursor = 0
        for start, end in runs:
            rows = end - start
            nbytes = rows * row_bytes
            lo = cursor * row_bytes
            futures.append(
                self._pool.submit(
                    self._pread_into,
                    fd,
                    base + start * row_bytes,
                    raw[lo : lo + nbytes],
                )
            )
            cursor += rows
        return futures

    def submit_raw_rows_compact(
        self, local_rows: torch.Tensor, owned: torch.Tensor
    ) -> Future[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Submit compact raw rows plus output-row -> unique-row mapping."""
        return _ENGRAM_STAGE_POOL.submit(
            self.read_raw_rows_compact, local_rows, owned
        )

    def read_raw_rows_compact(
        self, local_rows: torch.Tensor, owned: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unique raw rows and a pinned gather map.

        The gather map has one int32 entry per requested output row. -1 means
        the hash row is not owned by this table; otherwise it indexes the
        compact unique weight/scale buffers.
        """
        local_rows = local_rows.to(device="cpu", dtype=torch.int64).reshape(-1)
        owned = owned.to(device="cpu", dtype=torch.bool).reshape(-1)
        if local_rows.numel() != owned.numel():
            raise ValueError("local_rows and owned must have equal length")

        count = local_rows.numel()
        gather = torch.full(
            (count,), -1, dtype=torch.int32, pin_memory=True
        )
        valid_rows = local_rows[owned]
        if valid_rows.numel() and (
            int(valid_rows.min()) < 0 or int(valid_rows.max()) >= self.num_rows
        ):
            raise IndexError("Engram local row outside this rank's disk shard")
        if not valid_rows.numel():
            return (
                torch.empty(
                    (0, self.dim), dtype=torch.uint8, pin_memory=True
                ),
                torch.empty(
                    (0, self.scale_cols), dtype=torch.uint8, pin_memory=True
                ),
                gather,
            )

        unique, inverse = torch.unique(
            valid_rows, sorted=True, return_inverse=True
        )
        unique_list = [int(v) for v in unique.tolist()]
        runs = self._contiguous_runs(unique_list)

        weights = torch.empty(
            (len(unique_list), self.dim),
            dtype=torch.uint8,
            pin_memory=True,
        )
        scales = torch.empty(
            (len(unique_list), self.scale_cols),
            dtype=torch.uint8,
            pin_memory=True,
        )
        futures = self._submit_runs_into(
            self.w_fd, self.w_base, runs, self.dim, weights
        )
        futures += self._submit_runs_into(
            self.s_fd, self.s_base, runs, self.scale_cols, scales
        )
        for future in futures:
            future.result()

        gather[owned] = inverse.to(dtype=torch.int32)
        return weights, scales, gather

    def submit_raw_rows(
        self, local_rows: torch.Tensor, owned: torch.Tensor
    ) -> Future[tuple[torch.Tensor, torch.Tensor]]:
        """Submit raw FP8/E8M0 rows for GPU-side dequantization."""
        return _ENGRAM_STAGE_POOL.submit(
            self.read_raw_rows, local_rows, owned
        )

    def read_raw_rows(
        self, local_rows: torch.Tensor, owned: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility wrapper expanding compact rows on the CPU."""
        weights, scales, gather = self.read_raw_rows_compact(
            local_rows, owned
        )
        count = gather.numel()
        w_out = torch.zeros(
            (count, self.dim), dtype=torch.uint8, pin_memory=True
        )
        s_out = torch.zeros(
            (count, self.scale_cols), dtype=torch.uint8, pin_memory=True
        )
        valid = gather >= 0
        if valid.any():
            indices = gather[valid].to(dtype=torch.long)
            w_out[valid] = weights[indices]
            s_out[valid] = scales[indices]
        return w_out, s_out

    def read_rows(
        self, local_rows: torch.Tensor, owned: torch.Tensor
    ) -> torch.Tensor:
        """Return BF16 [R, dim], zeroing rows not owned by this shard.

        Unique row ids are sorted and adjacent ids are coalesced into one
        positional read. Long prefills therefore issue roughly one syscall per
        contiguous run instead of two syscalls per hash row.
        """
        local_rows = local_rows.to(device="cpu", dtype=torch.int64).reshape(-1)
        owned = owned.to(device="cpu", dtype=torch.bool).reshape(-1)
        if local_rows.numel() != owned.numel():
            raise ValueError("local_rows and owned must have equal length")
        count = local_rows.numel()
        if count == 0:
            return torch.empty(
                (0, self.dim), dtype=torch.bfloat16, pin_memory=True
            )

        valid_rows = local_rows[owned]
        if valid_rows.numel() and (
            int(valid_rows.min()) < 0 or int(valid_rows.max()) >= self.num_rows
        ):
            raise IndexError("Engram local row outside this rank's disk shard")

        if valid_rows.numel():
            unique, inverse = torch.unique(
                valid_rows, sorted=True, return_inverse=True
            )
            unique_list = [int(v) for v in unique.tolist()]
            runs = self._contiguous_runs(unique_list)

            w_futures = self._submit_runs(
                self.w_fd, self.w_base, runs, self.dim
            )
            s_futures = self._submit_runs(
                self.s_fd, self.s_base, runs, self.scale_cols
            )
            w_rows = self._collect_runs(w_futures, self.dim)
            s_rows = self._collect_runs(s_futures, self.scale_cols)

            w_bytes = b"".join(w_rows[row] for row in unique_list)
            s_bytes = b"".join(s_rows[row] for row in unique_list)
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
            inverse = torch.empty((0,), dtype=torch.int64)
            dequant = torch.empty((0, self.dim), dtype=torch.float32)

        out = torch.zeros(
            (count, self.dim), dtype=torch.bfloat16, pin_memory=True
        )
        if valid_rows.numel():
            out[owned] = dequant[inverse].to(torch.bfloat16)
        return out

    def close(self) -> None:
        # The process-wide pools outlive individual tables. Only this table's
        # file descriptors are owned here.
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
