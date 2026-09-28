# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous runtime provider for TRTLLM-layout MXFP4 expert records."""

from __future__ import annotations

import threading

import torch

from vllm.model_executor.layers.fused_moe.expert_disk_store import DiskExpertStore
from vllm.model_executor.layers.fused_moe.expert_pager import (
    ExpertPageKey,
    LRUExpertSlotCache,
)
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    ExpertWeightResult,
)


class Mxfp4DiskExpertProvider:
    """Fill a compact GPU expert array from an aligned per-layer disk store.

    The disk records are already in the exact TRTLLM runtime layout. The first
    implementation is intentionally synchronous: one page-aligned pinned row
    is reused for every miss and each CPU->GPU copy completes before the
    residency transaction commits. This is the correctness baseline before
    pipelined reads and GB10 zero-copy are introduced.
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
    ) -> None:
        capacity = int(w13.shape[0])
        if not (capacity == w2.shape[0] == w13_scale.shape[0] == w2_scale.shape[0]):
            raise ValueError("all paged expert tensors must have the same slot count")
        if capacity <= 0 or capacity > global_num_experts:
            raise ValueError(
                f"invalid expert cache capacity {capacity} for {global_num_experts} experts"
            )
        if store.num_experts != global_num_experts or not store.is_complete:
            raise ValueError("expert disk store is incomplete or has the wrong expert count")

        self.layer_id = layer_id
        self.global_num_experts = global_num_experts
        self.store = store
        self.w13 = w13
        self.w2 = w2
        self.w13_scale = w13_scale
        self.w2_scale = w2_scale
        self.cache = LRUExpertSlotCache(capacity)
        self._staging = torch.empty(
            store.record_stride, dtype=torch.uint8, pin_memory=True
        )
        self.expert_map = torch.full(
            (global_num_experts,),
            -1,
            dtype=torch.int32,
            device=w13.device,
        )
        self._lock = threading.RLock()
        self.disk_reads = 0

    @property
    def capacity(self) -> int:
        return self.cache.capacity

    def _required(self, topk_ids: torch.Tensor) -> tuple[ExpertPageKey, ...]:
        # A host sync is deliberate in the baseline. At decode this is a tiny
        # top-k matrix; prefill must be chunked so its union fits capacity.
        ids = topk_ids.detach().reshape(-1).to("cpu").tolist()
        ordered = []
        seen = set()
        for value in ids:
            expert_id = int(value)
            if expert_id < 0:
                continue
            if expert_id >= self.global_num_experts:
                raise ValueError(f"routed expert id {expert_id} is out of range")
            if expert_id not in seen:
                seen.add(expert_id)
                ordered.append(ExpertPageKey(self.layer_id, expert_id))
        return tuple(ordered)

    def _copy_record_to_slot(self, expert_id: int, slot: int) -> None:
        self.store.read_record(expert_id, self._staging)
        self.w13[slot].copy_(
            self.store.field_view(self._staging, "w13"), non_blocking=False
        )
        self.w2[slot].copy_(
            self.store.field_view(self._staging, "w2"), non_blocking=False
        )
        self.w13_scale[slot].view(torch.uint8).copy_(
            self.store.field_view(self._staging, "w13_scale"),
            non_blocking=False,
        )
        self.w2_scale[slot].view(torch.uint8).copy_(
            self.store.field_view(self._staging, "w2_scale"),
            non_blocking=False,
        )
        self.disk_reads += 1

    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        required = self._required(topk_ids)
        with self._lock:
            try:
                plan = self.cache.plan(required)
            except ValueError as exc:
                raise RuntimeError(
                    f"expert union does not fit {self.capacity} resident slots; "
                    "reduce max-num-batched-tokens for the synchronous paging "
                    "baseline or increase VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS"
                ) from exc

            # Invalidate victims before their physical bytes are overwritten.
            for eviction in plan.evictions:
                self.expert_map[eviction.key.expert_id] = -1

            try:
                for load in plan.loads:
                    self._copy_record_to_slot(load.key.expert_id, load.slot)
                self.cache.commit(plan)
            except Exception:
                # Physical slots may now contain a mixture of old and new
                # experts. Forget every mapping rather than risk silent reuse.
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
