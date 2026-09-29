# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first B12X expert pager for DeepSeek V4.1 on one GB10."""

from __future__ import annotations

import threading
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.expert_disk_store import DiskExpertStore
from vllm.model_executor.layers.fused_moe.expert_pager import (
    ExpertPageKey,
    LRUExpertSlotCache,
)
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    ExpertWeightResult,
)


class B12xDiskExpertProvider:
    """Page raw MXFP4 experts, then repack the compact B12X slot array.

    This baseline deliberately repacks all resident slots whenever any miss is
    admitted. It is not the performance design. It is the smallest path that
    preserves B12X's existing packing/scale contract and lets us validate
    routing, disk records, slot eviction, and model output before implementing
    incremental packed-slot updates.
    """

    def __init__(
        self,
        *,
        layer: Any,
        layer_id: int,
        global_num_experts: int,
        store: DiskExpertStore,
        b12x_experts: Any,
    ) -> None:
        self.layer = layer
        self.layer_id = layer_id
        self.global_num_experts = global_num_experts
        self.store = store
        self.b12x_experts = b12x_experts
        self.capacity = int(layer.w13_weight.shape[0])
        self.cache = LRUExpertSlotCache(self.capacity)
        self.expert_map = torch.full(
            (global_num_experts,),
            -1,
            dtype=torch.int32,
            device=layer.w13_weight.device,
        )
        self._staging = torch.empty(
            store.record_stride, dtype=torch.uint8, pin_memory=True
        )
        self._lock = threading.RLock()
        self.disk_reads = 0
        self.repacks = 0

    def _required(self, topk_ids: torch.Tensor) -> tuple[ExpertPageKey, ...]:
        ids = topk_ids.detach().reshape(-1).to("cpu").tolist()
        seen: set[int] = set()
        result: list[ExpertPageKey] = []
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

    def _load_slot(self, expert_id: int, slot: int) -> None:
        self.store.read_record(expert_id, self._staging)
        self.layer.w13_weight.data[slot].view(torch.uint8).copy_(
            self.store.field_view(self._staging, "w13"), non_blocking=False
        )
        self.layer.w2_weight.data[slot].view(torch.uint8).copy_(
            self.store.field_view(self._staging, "w2"), non_blocking=False
        )
        self.layer.w13_weight_scale.data[slot].view(torch.uint8).copy_(
            self.store.field_view(self._staging, "w13_scale"), non_blocking=False
        )
        self.layer.w2_weight_scale.data[slot].view(torch.uint8).copy_(
            self.store.field_view(self._staging, "w2_scale"), non_blocking=False
        )
        self.disk_reads += 1

    def _repack(self) -> None:
        experts = self.b12x_experts
        experts._refresh_quant_config(self.layer)
        prepared = experts._prepare_experts(
            w1=self.layer.w13_weight,
            w2=self.layer.w2_weight,
            activation=self.layer.activation,
            params_dtype=experts.moe_config.in_dtype,
        )
        experts._reuse_prepared_storage(self.layer, prepared)
        self.repacks += 1

    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        required = self._required(topk_ids)
        with self._lock:
            try:
                plan = self.cache.plan(required)
            except ValueError as exc:
                raise RuntimeError(
                    f"batch needs more than {self.capacity} unique routed "
                    "experts in one layer; reduce max-num-batched-tokens or "
                    "increase VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS"
                ) from exc

            for eviction in plan.evictions:
                self.expert_map[eviction.key.expert_id] = -1

            try:
                for load in plan.loads:
                    self._load_slot(load.key.expert_id, load.slot)
                if plan.loads:
                    self._repack()
                self.cache.commit(plan)
            except Exception:
                self.cache.reset()
                self.expert_map.fill_(-1)
                raise

            for key, slot in self.cache.snapshot().items():
                self.expert_map[key.expert_id] = slot

            return ExpertWeightResult(
                w1=self.layer.w13_weight,
                w2=self.layer.w2_weight,
                expert_map=self.expert_map,
            )
