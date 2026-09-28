# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental NVMe expert paging primitives.

This module deliberately contains no CUDA or model-loader integration yet.  It
provides the two pieces that need to be correct before the DeepSeek V4.1 data
path is changed:

* a zero-copy *index* of expert tensors inside safetensors shards (headers only;
  tensor payloads are never materialized), and
* a transactional LRU slot planner for a fixed-size resident expert arena.

The intended cache key is one routed expert in one transformer layer.  All
checkpoint tensors belonging to that expert (w1/w2/w3, scales, etc.) are grouped
into one :class:`ExpertPage`.
"""

from __future__ import annotations

import json
import re
import struct
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HEADER_LEN = struct.Struct("<Q")
# DeepSeek V4/V4.1 checkpoints use names such as:
#   model.layers.3.mlp.experts.17.w1.weight
# Keep the middle component flexible so this indexer stays useful if the model
# module is renamed without changing the experts.<id> checkpoint contract.
_EXPERT_TENSOR_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\..*?experts\."
    r"(?P<expert>\d+)\.(?P<tensor>.+)$"
)


@dataclass(frozen=True, order=True)
class ExpertPageKey:
    """Logical identity of one routed expert."""

    layer_id: int
    expert_id: int


@dataclass(frozen=True)
class SafeTensorSlice:
    """Byte range for one tensor payload in a safetensors shard."""

    path: Path
    tensor_name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class ExpertPage:
    """All safetensors slices that make up one routed expert."""

    key: ExpertPageKey
    tensors: Mapping[str, SafeTensorSlice]

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())


class SafetensorExpertIndex:
    """Header-only index of routed expert tensors.

    Safetensors stores data offsets relative to the start of the payload
    section.  Reading the 8-byte header length plus the JSON header is enough to
    locate every tensor without mapping or allocating its data.
    """

    def __init__(self, pages: Mapping[ExpertPageKey, ExpertPage]):
        self._pages = dict(pages)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "SafetensorExpertIndex":
        root = Path(model_dir)
        if not root.is_dir():
            raise ValueError(f"model_dir is not a directory: {root}")

        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"no safetensors shards found in {root}")

        grouped: dict[ExpertPageKey, dict[str, SafeTensorSlice]] = {}
        for shard in shards:
            for tensor_name, tensor_slice in _read_safetensors_header(shard).items():
                match = _EXPERT_TENSOR_RE.search(tensor_name)
                if match is None:
                    continue
                key = ExpertPageKey(
                    layer_id=int(match.group("layer")),
                    expert_id=int(match.group("expert")),
                )
                logical_name = match.group("tensor")
                tensors = grouped.setdefault(key, {})
                if logical_name in tensors:
                    prev = tensors[logical_name]
                    raise ValueError(
                        "duplicate expert tensor "
                        f"{key} {logical_name!r}: {prev.path} and {shard}"
                    )
                tensors[logical_name] = tensor_slice

        pages = {
            key: ExpertPage(key=key, tensors=dict(tensors))
            for key, tensors in grouped.items()
        }
        return cls(pages)

    def __len__(self) -> int:
        return len(self._pages)

    def __contains__(self, key: ExpertPageKey) -> bool:
        return key in self._pages

    def __getitem__(self, key: ExpertPageKey) -> ExpertPage:
        return self._pages[key]

    @property
    def total_bytes(self) -> int:
        return sum(page.nbytes for page in self._pages.values())

    def keys(self) -> tuple[ExpertPageKey, ...]:
        return tuple(sorted(self._pages))

    def pages_for_layer(self, layer_id: int) -> tuple[ExpertPage, ...]:
        return tuple(
            self._pages[key]
            for key in sorted(self._pages)
            if key.layer_id == layer_id
        )


def _read_safetensors_header(path: Path) -> dict[str, SafeTensorSlice]:
    size = path.stat().st_size
    with path.open("rb", buffering=0) as f:
        raw_len = f.read(_HEADER_LEN.size)
        if len(raw_len) != _HEADER_LEN.size:
            raise ValueError(f"truncated safetensors header length: {path}")
        (header_len,) = _HEADER_LEN.unpack(raw_len)
        if header_len == 0 or header_len > size - _HEADER_LEN.size:
            raise ValueError(
                f"invalid safetensors header length {header_len} for {path}"
            )
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"truncated safetensors JSON header: {path}")

    try:
        header: dict[str, Any] = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exc

    data_base = _HEADER_LEN.size + header_len
    result: dict[str, SafeTensorSlice] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise ValueError(f"invalid tensor metadata for {name!r} in {path}")

        offsets = meta.get("data_offsets")
        shape = meta.get("shape")
        dtype = meta.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) for v in offsets)
            or not isinstance(shape, list)
            or not all(isinstance(v, int) and v >= 0 for v in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"invalid tensor entry {name!r} in {path}")

        start, end = offsets
        if start < 0 or end < start or data_base + end > size:
            raise ValueError(f"out-of-range tensor {name!r} in {path}")

        result[name] = SafeTensorSlice(
            path=path,
            tensor_name=name,
            dtype=dtype,
            shape=tuple(shape),
            offset=data_base + start,
            nbytes=end - start,
        )
    return result


@dataclass(frozen=True)
class ExpertLoad:
    key: ExpertPageKey
    slot: int


@dataclass(frozen=True)
class ExpertEviction:
    key: ExpertPageKey
    slot: int


@dataclass(frozen=True)
class ExpertSlotPlan:
    """Transactional result of planning one batch's resident experts."""

    epoch: int
    required_slots: Mapping[ExpertPageKey, int]
    loads: tuple[ExpertLoad, ...]
    evictions: tuple[ExpertEviction, ...]
    _final_mapping: Mapping[ExpertPageKey, int]
    _final_lru: tuple[ExpertPageKey, ...]


class LRUExpertSlotCache:
    """Plan residency for a fixed number of expert slots.

    Planning is intentionally separated from commit.  NVMe reads can fail; a
    failed read must not silently mutate residency metadata.  Callers should:

      plan = cache.plan(required)
      perform_all_io(plan.loads)
      cache.commit(plan)

    A plan is rejected if another thread committed a newer plan first.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._mapping: dict[ExpertPageKey, int] = {}
        self._lru: OrderedDict[ExpertPageKey, None] = OrderedDict()
        self._epoch = 0
        self._lock = threading.RLock()

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def snapshot(self) -> dict[ExpertPageKey, int]:
        with self._lock:
            return dict(self._mapping)

    def reset(self) -> None:
        """Forget all residency after a failed physical slot update."""
        with self._lock:
            self._mapping.clear()
            self._lru.clear()
            self._epoch += 1

    def plan(self, required: Iterable[ExpertPageKey]) -> ExpertSlotPlan:
        required_order = tuple(dict.fromkeys(required))
        if len(required_order) > self.capacity:
            raise ValueError(
                f"batch needs {len(required_order)} unique experts but cache "
                f"has only {self.capacity} slots"
            )

        with self._lock:
            mapping = dict(self._mapping)
            lru: OrderedDict[ExpertPageKey, None] = OrderedDict(self._lru)
            epoch = self._epoch

        protected = set(required_order)

        # Hits become most-recent before choosing victims.
        for key in required_order:
            if key in mapping:
                lru.move_to_end(key)

        free_slots = [
            slot for slot in range(self.capacity) if slot not in mapping.values()
        ]
        loads: list[ExpertLoad] = []
        evictions: list[ExpertEviction] = []

        for key in required_order:
            if key in mapping:
                continue

            if free_slots:
                slot = free_slots.pop(0)
            else:
                victim = next((candidate for candidate in lru if candidate not in protected), None)
                if victim is None:
                    # len(required_order) <= capacity guarantees this should be
                    # unreachable unless the residency metadata is inconsistent.
                    raise RuntimeError("no evictable expert slot available")
                slot = mapping.pop(victim)
                lru.pop(victim)
                evictions.append(ExpertEviction(victim, slot))

            mapping[key] = slot
            lru[key] = None
            loads.append(ExpertLoad(key, slot))

        return ExpertSlotPlan(
            epoch=epoch,
            required_slots={key: mapping[key] for key in required_order},
            loads=tuple(loads),
            evictions=tuple(evictions),
            _final_mapping=mapping,
            _final_lru=tuple(lru),
        )

    def commit(self, plan: ExpertSlotPlan) -> None:
        with self._lock:
            if plan.epoch != self._epoch:
                raise RuntimeError(
                    "stale expert-slot plan: "
                    f"planned at epoch {plan.epoch}, current epoch {self._epoch}"
                )
            self._mapping = dict(plan._final_mapping)
            self._lru = OrderedDict((key, None) for key in plan._final_lru)
            self._epoch += 1
