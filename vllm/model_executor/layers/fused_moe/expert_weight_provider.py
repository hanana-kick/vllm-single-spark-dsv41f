# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime expert-weight provider interface.

The provider is intentionally a narrow seam between routing and the existing
MoE kernels. It does not define an alternate forward path: routing still
produces global expert ids and the regular modular kernel still performs the
expert computation.

A provider may make weights resident dynamically (for example from CPU or
NVMe), but it must return fixed-layout tensors plus an expert_map using the
same global-expert -> physical-slot convention as expert parallelism. Keeping
topk_ids global is required for quantized methods whose scales and other
metadata are indexed by logical expert id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ExpertWeightResult:
    """Expert tensors and slot mapping ready for one MoE invocation."""

    w1: torch.Tensor
    w2: torch.Tensor
    expert_map: torch.Tensor


class ExpertWeightProvider(Protocol):
    """Prepare experts selected by topk_ids for execution.

    Implementations may synchronize, perform I/O, or update residency state.
    For CUDA-graph support the returned tensors must have stable storage
    addresses and expert_map should be updated in-place between replays.
    """

    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        ...
