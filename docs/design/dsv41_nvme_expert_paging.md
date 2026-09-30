# DeepSeek V4.1 single-Spark NVMe expert paging

This branch explores serving the upstream DeepSeek V4.1 implementation on one
DGX Spark without pruning routed experts.

## Goal

Keep the current vLLM DeepSeek V4.1 stack (multimodal, tools, reasoning,
continuous batching, prefix cache and eventually DSpark) while moving capacity
that cannot fit in the GB10's unified memory to local NVMe.

The target storage split is:

- dense / attention / shared weights: resident;
- Engram tables: disk-backed row lookup;
- routed experts: fixed-size resident arena with NVMe fallback;
- KV cache: ordinary vLLM shared pool.

No expert is removed from the routing space.  A cache miss is a latency event,
not a quality-changing substitution.

## Development order

1. Header-only safetensors expert index and transactional resident-slot planner.
2. Synchronous `pread` loader into a fixed resident expert arena, single
   sequence, eager execution.
3. Exactness checks against a fully resident reference deployment.
4. Batched miss coalescing and asynchronous I/O.
5. Disk-backed Engram on single Spark.
6. Multi-request scheduling, then vision.
7. DSpark and CUDA-graph recovery only after the eager path is exact.

The first milestone intentionally does not alter model loading or inference.  It
adds the metadata and residency primitives required to do that without making a
half-wired offload mode visible to users.

## Invariants

- Cache identity is `(layer_id, expert_id)`; all tensors/scales for that expert
  move together.
- A batch cannot evict an expert that the same batch requires.
- Residency metadata changes only after all planned I/O succeeds.
- Safetensors payloads are not mmap'ed merely to build the index: only the
  8-byte header length and JSON header are read.
- The initial implementation stays backend-specific to DeepSeek V4.1/MXFP4
  rather than pretending to be a generic vLLM offloader.

## Current experimental switch

The active DGX Spark path is opt-in and restricted to DeepSeek V4.1, TP1/EP1
and the FlashInfer CUTLASS MXFP8×MXFP4 backend:

```bash
export VLLM_DSV41_NVME_EXPERT_STORE_DIR=/fast-nvme/dsv41-experts
export VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS=64
export VLLM_DSV41_NVME_EXPERT_READ_BATCH=8
export VLLM_DSV41_NVME_UVA_SLOTS=1
export VLLM_DSV41_ENGRAM_DISK=1
export VLLM_DSV41_ENGRAM_DISK_DIR=/path/to/model/snapshot

vllm serve /path/to/model/snapshot \
  --moe-backend flashinfer_cutlass \
  --tensor-parallel-size 1 \
  --enforce-eager
```

The official safetensors are the source of truth. Routed experts are converted
once, expert-by-expert, to the exact FlashInfer runtime layout and persisted in
a fixed-stride NVMe store. No GGUF/Q2 weights and no expert pruning are used.

On GB10, the default slot mode uses pinned CPU backing exposed to CUDA through
UVA:

```text
NVMe -> pinned GB10 UMA slot -> CUDA UVA view -> FlashInfer CUTLASS
```

A runtime miss therefore does not repack the resident set. It updates only the
slot(s) that are missing and publishes a 384-entry logical-to-physical map.

### Prefill and decode scheduling

A wide prefill may route to more experts than fit in the resident arena.
Experts are frequency-ranked for that prompt, partitioned into cache-sized
groups and executed as exact partial routed sums. Every group computes only
tokens that actually select one of its experts. The hottest cache-sized group
runs last so it remains resident for the transition to decode.

The next group's first bounded read batch is submitted while the current group
runs. For ordinary single-group decode, a mixed hit/miss step similarly starts
cold reads while the resident routed contribution is computed. All-hit and
all-miss steps keep the simpler single-kernel path.

Disk Engram uses the original FP8 rows and ue8m0 scales from safetensors.
Hash-selected rows are submitted before decoder execution and consumed at the
Engram layer, avoiding the upstream roughly 200 GB pinned-table residency.

### Memory and I/O bounds

A released V4.1 routed expert is roughly 19 MiB. At 64 slots across 40 routed
layers the explicit expert cache is therefore on the order of 47 GiB before
other model/runtime allocations. Look-ahead prefetch is capped by
`VLLM_DSV41_NVME_EXPERT_READ_BATCH` (8 by default), rather than buffering an
entire cache-sized group.

The expert I/O workers are process-wide
(`VLLM_DSV41_NVME_IO_WORKERS=32` by default), not one thread pool per layer.
Buffered expert and Engram reads advise the kernel to drop consumed pages by
default to avoid a duplicate page-cache working set on unified memory.

## Runtime-layout store

```
first build:
checkpoint w1/w2/w3 + scales
  -> one-expert FlashInfer layout conversion
  -> persistent NVMe runtime record

runtime miss:
NVMe record -> one resident slot -> expert_map update -> fused MoE
```

There is no all-slot B12X repack on a miss. Wide prefills whose per-layer
expert union exceeds resident capacity are split into exact expert groups:
non-group top-k weights are zeroed and the partial outputs are summed. This
turns a large prefill into a small number of cache-sized MoE passes instead of
LRU-thrashing token by token.
