# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_disk_store import (
    DiskExpertStore,
)


def _record(store: DiskExpertStore, a: int, b: int) -> torch.Tensor:
    row = torch.zeros(store.record_stride, dtype=torch.uint8)
    store.field_view(row, "w13").fill_(a)
    store.field_view(row, "w2").fill_(b)
    return row


def test_streaming_store_round_trip_and_reuse(tmp_path: Path):
    path = tmp_path / "layer.experts"
    specs = [
        ("w13", (4,), torch.uint8),
        ("w2", (3,), torch.uint8),
    ]
    store = DiskExpertStore.create_for_streaming(
        path, 2, specs, identity={"model": "test", "layer": 3}, direct_io=False
    )
    assert not store.is_complete
    assert store.record_stride == 4096
    store.write_record(0, _record(store, 11, 21))
    store.write_record(1, _record(store, 12, 22))
    store.finalize()

    dst = torch.empty(store.record_stride, dtype=torch.uint8)
    assert store.read_record(1, dst) == store.record_stride
    assert store.field_view(dst, "w13").tolist() == [12] * 4
    assert store.field_view(dst, "w2").tolist() == [22] * 3
    store.close()

    reused = DiskExpertStore.create_for_streaming(
        path, 2, specs, identity={"model": "test", "layer": 3}, direct_io=False
    )
    assert reused.is_complete
    reused.close()


def test_store_identity_mismatch_rebuilds(tmp_path: Path):
    path = tmp_path / "layer.experts"
    specs = [("w13", (1,), torch.uint8)]
    first = DiskExpertStore.create_for_streaming(
        path, 1, specs, identity={"revision": "a"}, direct_io=False
    )
    first.write_record(0, _record_single(first, 1))
    first.finalize()
    first.close()

    second = DiskExpertStore.create_for_streaming(
        path, 1, specs, identity={"revision": "b"}, direct_io=False
    )
    assert not second.is_complete
    second.write_record(0, _record_single(second, 2))
    second.finalize()
    second.close()


def _record_single(store: DiskExpertStore, value: int) -> torch.Tensor:
    row = torch.zeros(store.record_stride, dtype=torch.uint8)
    store.field_view(row, "w13").fill_(value)
    return row


def test_finalize_rejects_missing_experts(tmp_path: Path):
    store = DiskExpertStore.create_for_streaming(
        tmp_path / "layer.experts",
        2,
        [("w13", (1,), torch.uint8)],
        direct_io=False,
    )
    store.write_record(0, _record_single(store, 1))
    with pytest.raises(RuntimeError, match="1 unwritten experts"):
        store.finalize()
    store.close()


def test_field_view_validates_record_shape(tmp_path: Path):
    store = DiskExpertStore.create_for_streaming(
        tmp_path / "layer.experts",
        1,
        [("w13", (2,), torch.uint8)],
        direct_io=False,
    )
    with pytest.raises(ValueError, match="record must be"):
        store.field_view(torch.empty(2, dtype=torch.uint8), "w13")
    store.close()

def test_partial_field_writes_do_not_require_record_staging(tmp_path: Path):
    path = tmp_path / "layer.experts"
    store = DiskExpertStore.create_for_streaming(
        path,
        1,
        [
            ("w13", (8,), torch.uint8),
            ("w2", (2,), torch.uint8),
        ],
        direct_io=False,
    )
    store.write_field(0, "w13", torch.tensor([1, 2, 3, 4], dtype=torch.uint8))
    store.write_field(
        0,
        "w13",
        torch.tensor([5, 6, 7, 8], dtype=torch.uint8),
        byte_offset=4,
    )
    store.write_field(0, "w2", torch.tensor([9, 10], dtype=torch.uint8))
    store.mark_expert_complete(0)
    store.finalize()

    dst = torch.empty(store.record_stride, dtype=torch.uint8)
    store.read_record(0, dst)
    assert store.field_view(dst, "w13").tolist() == list(range(1, 9))
    assert store.field_view(dst, "w2").tolist() == [9, 10]
    store.close()
