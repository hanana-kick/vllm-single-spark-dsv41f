# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import struct
from pathlib import Path

import pytest

from vllm.model_executor.layers.fused_moe.expert_pager import (
    ExpertPageKey,
    LRUExpertSlotCache,
    SafetensorExpertIndex,
)


def _write_fake_safetensors(path: Path, tensors: list[tuple[str, bytes]]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, data in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)

    raw_header = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def test_safetensor_expert_index_groups_expert_tensors(tmp_path: Path):
    _write_fake_safetensors(
        tmp_path / "model-00001-of-00002.safetensors",
        [
            ("model.layers.2.mlp.experts.7.w1.weight", b"aaaa"),
            ("model.layers.2.mlp.experts.7.w1.weight_scale_inv", b"bb"),
            ("model.layers.2.mlp.experts.7.w2.weight", b"cccccc"),
            ("model.layers.2.mlp.shared_experts.w1.weight", b"dense"),
        ],
    )
    _write_fake_safetensors(
        tmp_path / "model-00002-of-00002.safetensors",
        [
            ("model.layers.3.mlp.experts.9.w3.weight", b"xyz"),
        ],
    )

    index = SafetensorExpertIndex.from_model_dir(tmp_path)

    assert len(index) == 2
    key = ExpertPageKey(2, 7)
    assert key in index
    assert set(index[key].tensors) == {
        "w1.weight",
        "w1.weight_scale_inv",
        "w2.weight",
    }
    assert index[key].nbytes == 12
    assert index.total_bytes == 15
    assert [p.key for p in index.pages_for_layer(3)] == [ExpertPageKey(3, 9)]

    w1 = index[key].tensors["w1.weight"]
    with w1.path.open("rb") as f:
        f.seek(w1.offset)
        assert f.read(w1.nbytes) == b"aaaa"


def test_expert_slot_cache_lru_and_transactional_commit():
    a = ExpertPageKey(0, 1)
    b = ExpertPageKey(0, 2)
    c = ExpertPageKey(1, 3)
    cache = LRUExpertSlotCache(capacity=2)

    first = cache.plan([a, b])
    assert [(load.key, load.slot) for load in first.loads] == [(a, 0), (b, 1)]
    cache.commit(first)

    # Touch A so B becomes the least-recently-used resident.
    hit = cache.plan([a])
    assert hit.loads == ()
    cache.commit(hit)

    third = cache.plan([c])
    assert [(ev.key, ev.slot) for ev in third.evictions] == [(b, 1)]
    assert [(load.key, load.slot) for load in third.loads] == [(c, 1)]
    cache.commit(third)
    assert cache.snapshot() == {a: 0, c: 1}


def test_expert_slot_plan_rejects_stale_commit():
    a = ExpertPageKey(0, 1)
    b = ExpertPageKey(0, 2)
    cache = LRUExpertSlotCache(capacity=2)

    stale = cache.plan([a])
    fresh = cache.plan([b])
    cache.commit(fresh)

    with pytest.raises(RuntimeError, match="stale expert-slot plan"):
        cache.commit(stale)


def test_expert_slot_cache_rejects_batch_larger_than_capacity():
    cache = LRUExpertSlotCache(capacity=1)
    with pytest.raises(ValueError, match="2 unique experts"):
        cache.plan([ExpertPageKey(0, 1), ExpertPageKey(0, 2)])
