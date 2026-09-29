# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import struct
from pathlib import Path

import torch

from vllm.models.deepseek_v41.nvidia.engram_disk import DiskEngramTable


def _write_fake_safetensors(path: Path) -> None:
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
        dtype=torch.float8_e4m3fn,
    ).view(torch.uint8)
    scale = torch.full((3, 2), 127, dtype=torch.uint8)
    w = weight.numpy().tobytes()
    s = scale.numpy().tobytes()
    header = {
        "layers.1.engram.embed.weight": {
            "dtype": "F8_E4M3",
            "shape": [3, 4],
            "data_offsets": [0, len(w)],
        },
        "layers.1.engram.embed.scale": {
            "dtype": "U8",
            "shape": [3, 2],
            "data_offsets": [len(w), len(w) + len(s)],
        },
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + w + s)


def test_disk_engram_reads_selected_rows_and_zeroes_unowned(tmp_path: Path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_fake_safetensors(shard)
    index = {
        "weight_map": {
            "layers.1.engram.embed.weight": shard.name,
            "layers.1.engram.embed.scale": shard.name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    table = DiskEngramTable(
        tmp_path,
        layer_id=1,
        dim=4,
        block_size=2,
        row_start=0,
        num_rows=3,
        threads=2,
    )
    rows = table.read_rows(
        torch.tensor([2, 1, 0], dtype=torch.int64),
        torch.tensor([True, False, True]),
    )

    expected = torch.tensor(
        [[3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    table.close()


def test_disk_engram_accepts_model_prefix_and_weight_scale_inv(tmp_path: Path):
    weight = bytes([0] * 8)
    scale = bytes([127] * 4)
    header = {
        "model.layers.4.engram.embed.weight": {
            "dtype": "F8_E4M3",
            "shape": [2, 4],
            "data_offsets": [0, len(weight)],
        },
        "model.layers.4.engram.embed.weight_scale_inv": {
            "dtype": "U8",
            "shape": [2, 2],
            "data_offsets": [len(weight), len(weight) + len(scale)],
        },
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(raw)) + raw + weight + scale)
    index = {
        "weight_map": {
            "model.layers.4.engram.embed.weight": shard.name,
            "model.layers.4.engram.embed.weight_scale_inv": shard.name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    table = DiskEngramTable(
        tmp_path,
        layer_id=4,
        dim=4,
        block_size=2,
        row_start=0,
        num_rows=2,
        threads=1,
    )
    assert table.read_rows(
        torch.tensor([0]), torch.tensor([True])
    ).shape == (1, 4)
    table.close()
