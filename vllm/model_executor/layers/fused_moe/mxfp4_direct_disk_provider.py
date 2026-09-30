# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct MXFP4 NVMe pager for FlashInfer CUTLASS on SM12x."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_disk_store import DiskExpertStore
from vllm.model_executor.layers.fused_moe.expert_pager import (
    ExpertLoad,
    ExpertPageKey,
    ExpertSlotPlan,
    LRUExpertSlotCache,
)
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    ExpertWeightResult,
)

logger = init_logger(__name__)

_NVME_IO_WORKERS = max(
    1, int(os.environ.get("VLLM_DSV41_NVME_IO_WORKERS", "32"))
)
_NVME_IO_POOL = ThreadPoolExecutor(
    max_workers=_NVME_IO_WORKERS,
    thread_name_prefix="vllm-dsv41-nvme",
)


@dataclass
class _PrefetchedExpertGroup:
    required: tuple[ExpertPageKey, ...]
    plan: ExpertSlotPlan
    futures: list[tuple[ExpertLoad, Future[tuple[bytearray, float]]]]
    deferred_loads: tuple[ExpertLoad, ...]


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
        # 384 int32 entries are only 1.5 KiB. Keep residency metadata on the
        # host and upload it in one copy only when the cache mapping changes.
        # Cache-hit decode therefore performs no tiny per-expert CUDA writes.
        self._expert_map_cpu = torch.full(
            (global_num_experts,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
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
        self._pool = _NVME_IO_POOL
        self._lock = threading.RLock()
        self.disk_reads = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.read_seconds = 0.0
        self.prepare_calls = 0
        self.stats_every = int(
            os.environ.get("VLLM_DSV41_NVME_STATS_EVERY", "0")
        )
        self.record_payload_bytes = sum(
            field.nbytes for field in store.fields.values()
        )

    @property
    def capacity(self) -> int:
        return self.cache.capacity

    def _required(self, topk_ids: torch.Tensor) -> tuple[ExpertPageKey, ...]:
        flat = topk_ids.detach().reshape(-1)
        # Decode-sized routing matrices are cheaper to copy directly. Wide
        # prefills must not synchronize tens of thousands of route ids to the
        # CPU just to discover at most global_num_experts unique values.
        if flat.numel() <= 256:
            ids = flat.to("cpu").tolist()
            seen: set[int] = set()
            unique_ids: list[int] = []
            for raw in ids:
                expert_id = int(raw)
                if expert_id < 0:
                    continue
                if expert_id >= self.global_num_experts:
                    raise ValueError(
                        f"routed expert id {expert_id} is out of range"
                    )
                if expert_id not in seen:
                    seen.add(expert_id)
                    unique_ids.append(expert_id)
        else:
            valid = flat[flat >= 0]
            if valid.numel() == 0:
                return ()
            unique = torch.unique(valid, sorted=False)
            if int(unique.max()) >= self.global_num_experts:
                raise ValueError("routed expert id is out of range")
            unique_ids = [int(value) for value in unique.to("cpu").tolist()]

        return tuple(
            ExpertPageKey(self.layer_id, expert_id) for expert_id in unique_ids
        )

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

        flat = topk_ids.detach().reshape(-1)
        valid = flat[flat >= 0]
        if valid.numel() == 0:
            return ((),)

        # Count on the accelerator, then transfer only <=384 expert IDs and
        # counts to the host. This avoids a large GPU->CPU synchronization on
        # every prefill layer.
        unique, counts_tensor = torch.unique(
            valid, sorted=False, return_counts=True
        )
        if int(unique.max()) >= self.global_num_experts:
            raise ValueError("routed expert id is out of range")
        ids_cpu = [int(value) for value in unique.to("cpu").tolist()]
        counts_cpu = [int(value) for value in counts_tensor.to("cpu").tolist()]
        counts = dict(zip(ids_cpu, counts_cpu, strict=True))
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
        # The first partition is the only one that also computes shared
        # experts, so Mxfp4MoEMethod must run it over the full token matrix.
        # Execute the hottest cold partition first to maximize useful routed
        # work in that unavoidable full-token pass. Keep the actual hot set
        # last so it remains resident when prefill hands off to decode.
        groups.reverse()
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
            started = time.perf_counter()
            futures = [self._pool.submit(read_one, load) for load in batch]
            for future in futures:
                future.result()
            self.read_seconds += time.perf_counter() - started
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
            started = time.perf_counter()
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
            self.read_seconds += time.perf_counter() - started

            for i, load in enumerate(batch):
                self._copy_record_to_slot(self._staging[i], load.slot)

            # All copies above are queued on the current stream. Record a fence
            # used only before these pinned buffers are recycled.
            self._copy_done = torch.cuda.Event()
            self._copy_done.record(torch.cuda.current_stream())
            self.disk_reads += len(batch)

    def _read_record_buffer_timed(
        self, expert_id: int
    ) -> tuple[bytearray, float]:
        started = time.perf_counter()
        buffer = self.store.read_record_buffer(expert_id)
        return buffer, time.perf_counter() - started

    def _copy_buffer_to_slot(
        self, buffer: bytearray, slot: int
    ) -> None:
        record = torch.frombuffer(buffer, dtype=torch.uint8)
        if self.uses_uva:
            assert self.cpu_backing is not None
            for name, backing in self.cpu_backing.items():
                backing[slot].reshape(-1).view(torch.uint8).copy_(
                    self.store.field_view(record, name).reshape(-1).view(torch.uint8)
                )
            return
        self._copy_record_to_slot(record, slot)

    def _publish_expert_map(self) -> None:
        self._expert_map_cpu.fill_(-1)
        for key, slot in self.cache.snapshot().items():
            self._expert_map_cpu[key.expert_id] = slot
        self.expert_map.copy_(self._expert_map_cpu, non_blocking=True)

    def prefetch_keys(
        self, required: tuple[ExpertPageKey, ...]
    ) -> _PrefetchedExpertGroup:
        """Start buffered NVMe reads without touching resident slots.

        The caller launches this before computing the current expert group and
        activates it only after that compute has been submitted. This preserves
        slot correctness while overlapping the next group's disk latency.
        """
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
            loads = sorted(plan.loads, key=lambda load: load.key.expert_id)
            # Bound look-ahead memory. A V4.1 MXFP4 expert is large; retaining
            # one bytearray per miss for a full 64-slot group can exceed a GiB.
            # Only the first read_batch records overlap current compute.
            ahead = loads[: self.read_batch]
            deferred = tuple(loads[self.read_batch :])
            futures = [
                (
                    load,
                    self._pool.submit(
                        self._read_record_buffer_timed, load.key.expert_id
                    ),
                )
                for load in ahead
            ]
            return _PrefetchedExpertGroup(
                required, plan, futures, deferred
            )

    def activate_prefetch(
        self, prefetched: _PrefetchedExpertGroup
    ) -> ExpertWeightResult:
        """Publish a prefetched group after the previous MoE launch."""
        with self._lock:
            if prefetched.plan.epoch != self.cache.epoch:
                raise RuntimeError(
                    "stale NVMe expert prefetch; cache mapping changed before "
                    "the prefetched group was activated"
                )

            completed = [
                (load, *future.result())
                for load, future in prefetched.futures
            ]
            # Reads in one prefetched group overlap in the shared executor.
            # The slowest individual read is a closer approximation of the
            # group's storage-service time than either the sum or host wait.
            if completed:
                self.read_seconds += max(seconds for _, _, seconds in completed)
            records = [(load, buffer) for load, buffer, _ in completed]

            try:
                if self.uses_uva and (records or prefetched.deferred_loads):
                    self._wait_compute_before_uva_write()
                for load, buffer in records:
                    self._copy_buffer_to_slot(buffer, load.slot)
                self.disk_reads += len(records)

                # Remaining misses use the normal bounded loader after current
                # compute has finished. This preserves the overlap benefit of
                # the first batch without unbounded temporary record storage.
                if prefetched.deferred_loads:
                    self._load_records(list(prefetched.deferred_loads))

                if records and not self.uses_uva:
                    self._copy_done = torch.cuda.Event()
                    self._copy_done.record(torch.cuda.current_stream())
                self.cache.commit(prefetched.plan)
            except Exception:
                self.cache.reset()
                self._expert_map_cpu.fill_(-1)
                self.expert_map.fill_(-1)
                raise

            if prefetched.plan.loads or prefetched.plan.evictions:
                self._publish_expert_map()

            self._maybe_log_stats()
            return ExpertWeightResult(
                w1=self.w13,
                w2=self.w2,
                expert_map=self.expert_map,
            )

    def _maybe_log_stats(self) -> None:
        self.prepare_calls += 1
        if self.stats_every <= 0 or self.prepare_calls % self.stats_every:
            return
        total = self.cache_hits + self.cache_misses
        hit_rate = 100.0 * self.cache_hits / total if total else 0.0
        read_gib = (
            self.disk_reads * self.record_payload_bytes / float(1 << 30)
        )
        read_gbps = (
            self.disk_reads * self.record_payload_bytes
            / self.read_seconds
            / 1e9
            if self.read_seconds > 0
            else 0.0
        )
        logger.info(
            "DSV4.1 NVMe layer=%d calls=%d hit=%.1f%% "
            "reads=%d (%.2f GiB) read=%.2f GB/s slots=%d UVA=%s",
            self.layer_id,
            self.prepare_calls,
            hit_rate,
            self.disk_reads,
            read_gib,
            read_gbps,
            self.capacity,
            self.uses_uva,
        )

    def split_resident(
        self, required: tuple[ExpertPageKey, ...]
    ) -> tuple[tuple[ExpertPageKey, ...], tuple[ExpertPageKey, ...]]:
        """Split required experts without changing LRU state."""
        resident_map = self.cache.snapshot()
        resident = tuple(key for key in required if key in resident_map)
        missing = tuple(key for key in required if key not in resident_map)
        return resident, missing

    def current_weights(self) -> ExpertWeightResult:
        """Return the currently published slots without changing residency."""
        return ExpertWeightResult(
            w1=self.w13,
            w2=self.w2,
            expert_map=self.expert_map,
        )

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

            try:
                self._load_records(list(plan.loads))
                self.cache.commit(plan)
            except Exception:
                self.cache.reset()
                self._expert_map_cpu.fill_(-1)
                self.expert_map.fill_(-1)
                raise

            if plan.loads or plan.evictions:
                self._publish_expert_map()

            self._maybe_log_stats()
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
        # Shared process-wide I/O pool remains alive for other MoE layers.

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
