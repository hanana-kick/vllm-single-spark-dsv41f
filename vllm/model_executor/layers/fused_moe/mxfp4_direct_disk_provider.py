# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct MXFP4 NVMe pager for FlashInfer CUTLASS on SM12x."""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.model_executor.layers.fused_moe.expert_disk_store import DiskExpertStore
from vllm.model_executor.layers.fused_moe.expert_pager import (
    ExpertPageKey,
    LRUExpertSlotCache,
)
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    ExpertWeightResult,
)


class FlashInferMxfp4DiskExpertProvider:
    """Page pre-converted FlashInfer MXFP4 expert records into fixed slots.

    Disk records are already in the exact runtime layout consumed by the
    FlashInfer CUTLASS MXFP8xMXFP4 kernel, so a cache miss is only:
      NVMe read -> slot copy -> expert-map update.
    No resident-set repacking is performed.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        global_num_experts: int,
        store: DiskExpertStore,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        cpu_backing: dict[str, torch.Tensor] | None = None,
        read_batch: int = 8,
    ) -> None:
        capacity = int(w13.shape[0])
        if not (
            capacity
            == int(w2.shape[0])
            == int(w13_scale.shape[0])
            == int(w2_scale.shape[0])
        ):
            raise ValueError("all paged expert tensors must share the slot dimension")
        if capacity <= 0 or capacity > global_num_experts:
            raise ValueError(
                f"invalid expert cache capacity {capacity} for "
                f"{global_num_experts} experts"
            )
        if not store.is_complete or store.num_experts != global_num_experts:
            raise ValueError("expert disk store is incomplete or has wrong expert count")

        self.layer_id = layer_id
        self.global_num_experts = global_num_experts
        self.store = store
        self.w13 = w13
        self.w2 = w2
        self.w13_scale = w13_scale
        self.w2_scale = w2_scale
        self.cpu_backing = cpu_backing
        self.uses_uva = cpu_backing is not None
        if self.uses_uva:
            assert cpu_backing is not None
            required = {"w13", "w2", "w13_scale", "w2_scale"}
            if set(cpu_backing) != required:
                raise ValueError(
                    f"UVA slot backing must contain {sorted(required)}"
                )
            for name, tensor in cpu_backing.items():
                if tensor.device.type != "cpu" or not tensor.is_pinned():
                    raise ValueError(
                        f"UVA backing {name} must be pinned CPU memory"
                    )
                if int(tensor.shape[0]) != capacity:
                    raise ValueError(
                        f"UVA backing {name} has wrong slot count "
                        f"{tensor.shape[0]} != {capacity}"
                    )
        self.cache = LRUExpertSlotCache(capacity)
        self.expert_map = torch.full(
            (global_num_experts,),
            -1,
            dtype=torch.int32,
            device=w13.device,
        )
        self.read_batch = max(1, int(read_batch))
        self._staging_owners: list[torch.Tensor] = []
        self._staging: list[torch.Tensor] = []
        if not self.uses_uva:
            # O_DIRECT needs page-aligned userspace buffers. Torch pinned
            # allocations are DMA-friendly but not guaranteed to be 4 KiB
            # aligned, so over-allocate and retain aligned views.
            for _ in range(self.read_batch):
                owner = torch.empty(
                    store.record_stride + 4096,
                    dtype=torch.uint8,
                    pin_memory=True,
                )
                shift = (-owner.data_ptr()) % 4096
                view = owner[shift : shift + store.record_stride]
                assert view.data_ptr() % 4096 == 0
                self._staging_owners.append(owner)
                self._staging.append(view)
        self._copy_done: torch.cuda.Event | None = None
        self._compute_done: torch.cuda.Event | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=self.read_batch,
            thread_name_prefix="vllm-expert-nvme",
        )
        self._lock = threading.RLock()
        self.disk_reads = 0
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def capacity(self) -> int:
        return self.cache.capacity

    def _required(self, topk_ids: torch.Tensor) -> tuple[ExpertPageKey, ...]:
        ids = topk_ids.detach().reshape(-1).to("cpu").tolist()
        result: list[ExpertPageKey] = []
        seen: set[int] = set()
        for raw in ids:
            expert_id = int(raw)
            if expert_id < 0:
                continue
            if expert_id >= self.global_num_experts:
                raise ValueError(f"routed expert id {expert_id} is out of range")
            if expert_id not in seen:
                seen.add(expert_id)
                result.append(ExpertPageKey(self.layer_id, expert_id))
        return tuple(result)

    def partition(
        self, topk_ids: torch.Tensor
    ) -> tuple[tuple[ExpertPageKey, ...], ...]:
        """Partition a wide prefill while leaving its hottest experts resident.

        Decode and small batches take the cheaper ordered-dedup path. Only a
        batch that could exceed resident capacity pays for frequency counting.

        Cold/rare experts run first. The final pass is exactly one cache worth
        of the most frequently routed experts (when the union exceeds
        capacity), so decode starts with a prompt-specific hot set instead of
        whichever expert IDs happened to occur last in token order.
        """
        if topk_ids.numel() <= self.capacity:
            return (self._required(topk_ids),)

        flat = [
            int(value)
            for value in topk_ids.detach().reshape(-1).to("cpu").tolist()
            if int(value) >= 0
        ]
        if not flat:
            return ((),)

        counts = Counter(flat)
        ordered_ids = sorted(counts, key=lambda expert: (counts[expert], expert))
        required = tuple(
            ExpertPageKey(self.layer_id, expert_id) for expert_id in ordered_ids
        )
        if len(required) <= self.capacity:
            return (required,)

        hot_start = len(required) - self.capacity
        cold = required[:hot_start]
        hot = required[hot_start:]
        groups = [
            tuple(cold[i : i + self.capacity])
            for i in range(0, len(cold), self.capacity)
        ]
        groups.append(tuple(hot))
        return tuple(groups)

    def _copy_record_to_slot(
        self, record: torch.Tensor, slot: int
    ) -> None:
        self.w13[slot].view(torch.uint8).copy_(
            self.store.field_view(record, "w13"), non_blocking=True
        )
        self.w2[slot].view(torch.uint8).copy_(
            self.store.field_view(record, "w2"), non_blocking=True
        )
        self.w13_scale[slot].view(torch.uint8).copy_(
            self.store.field_view(record, "w13_scale"), non_blocking=True
        )
        self.w2_scale[slot].view(torch.uint8).copy_(
            self.store.field_view(record, "w2_scale"), non_blocking=True
        )

    def _wait_compute_before_uva_write(self) -> None:
        if self.uses_uva and self._compute_done is not None:
            # CPU preadv is not ordered by a CUDA stream. Never overwrite a
            # mapped slot until the previous kernel using it has completed.
            self._compute_done.synchronize()
            self._compute_done = None

    def mark_compute_submitted(self) -> None:
        """Fence slot reuse after a kernel that consumed the current mapping."""
        if not self.uses_uva:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._compute_done = event

    def _load_records_uva(self, loads) -> None:
        assert self.cpu_backing is not None
        self._wait_compute_before_uva_write()

        def read_one(load) -> int:
            slot = load.slot
            return self.store.read_fields(
                load.key.expert_id,
                {
                    "w13": self.cpu_backing["w13"][slot],
                    "w2": self.cpu_backing["w2"][slot],
                    "w13_scale": self.cpu_backing["w13_scale"][slot],
                    "w2_scale": self.cpu_backing["w2_scale"][slot],
                },
            )

        for start in range(0, len(loads), self.read_batch):
            batch = loads[start : start + self.read_batch]
            futures = [self._pool.submit(read_one, load) for load in batch]
            for future in futures:
                future.result()
            self.disk_reads += len(batch)

    def _wait_staging_reuse(self) -> None:
        if self._copy_done is not None:
            # Only wait when the host is about to overwrite the same pinned
            # buffers. MoE compute itself stays ordered behind the async H2D on
            # the current CUDA stream without a host-side synchronization.
            self._copy_done.synchronize()
            self._copy_done = None

    def _load_records(self, loads) -> None:
        """Parallel NVMe reads into UVA slots or staged CUDA slots."""
        if not loads:
            return
        # Record-sized reads are large; stable expert-id order improves
        # locality/readahead for buffered I/O without changing slot placement.
        loads = sorted(loads, key=lambda load: load.key.expert_id)
        if self.uses_uva:
            self._load_records_uva(loads)
            return

        for start in range(0, len(loads), self.read_batch):
            self._wait_staging_reuse()
            batch = loads[start : start + self.read_batch]
            futures = [
                self._pool.submit(
                    self.store.read_record,
                    load.key.expert_id,
                    self._staging[i],
                )
                for i, load in enumerate(batch)
            ]
            for future in futures:
                future.result()

            for i, load in enumerate(batch):
                self._copy_record_to_slot(self._staging[i], load.slot)

            # All copies above are queued on the current stream. Record a fence
            # used only before these pinned buffers are recycled.
            self._copy_done = torch.cuda.Event()
            self._copy_done.record(torch.cuda.current_stream())
            self.disk_reads += len(batch)

    def prepare_keys(
        self, required: tuple[ExpertPageKey, ...]
    ) -> ExpertWeightResult:
        if len(required) > self.capacity:
            raise RuntimeError(
                f"expert group has {len(required)} experts but cache capacity "
                f"is {self.capacity}"
            )

        with self._lock:
            before = self.cache.snapshot()
            plan = self.cache.plan(required)
            self.cache_hits += sum(1 for key in required if key in before)
            self.cache_misses += len(plan.loads)

            for eviction in plan.evictions:
                self.expert_map[eviction.key.expert_id] = -1

            try:
                self._load_records(list(plan.loads))
                self.cache.commit(plan)
            except Exception:
                self.cache.reset()
                self.expert_map.fill_(-1)
                raise

            for key, slot in self.cache.snapshot().items():
                self.expert_map[key.expert_id] = slot

            return ExpertWeightResult(
                w1=self.w13,
                w2=self.w2,
                expert_map=self.expert_map,
            )

    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        required = self._required(topk_ids)
        if len(required) > self.capacity:
            raise RuntimeError(
                f"batch needs {len(required)} unique experts but resident "
                f"capacity is {self.capacity}; use partitioned prefill"
            )
        return self.prepare_keys(required)

    def close(self) -> None:
        self._wait_staging_reuse()
        self._wait_compute_before_uva_write()
        self._pool.shutdown(wait=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
