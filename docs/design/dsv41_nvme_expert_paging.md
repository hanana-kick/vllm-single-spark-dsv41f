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

The first correctness path is intentionally opt-in and GB10/B12X-specific:

```bash
export VLLM_DSV41_NVME_EXPERT_STORE_DIR=/fast-nvme/dsv41-experts
export VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS=64
export VLLM_DSV41_NVME_DIRECT_IO=1

# Force the SM12x MoE implementation used by the paging baseline.
# Other model/parallel configurations fail closed when the env var is set.
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --moe-backend b12x
```

This is not yet a performance configuration. On every cache miss it:

1. reads the selected raw MXFP4 expert record into one pinned staging buffer;
2. copies it into a fixed raw slot;
3. asks the existing B12X `prepare_weights()` implementation to repack the
   complete resident slot array; and
4. remaps the original global expert ids to physical slots only at the B12X
   expert-kernel boundary.

Repacking all slots on a miss is deliberately expensive. It avoids inventing a
second B12X packed format and gives the project a correctness oracle before
incremental packed-slot updates, asynchronous I/O, and batch miss coalescing.

### Why B12X, not FlashInfer TRTLLM

DGX Spark is SM121 (CUDA capability family 12.x). B12X explicitly supports
capability family 12.x. The current TRTLLM MXFP4 experts backend is gated to
capability family 10.x, so it is not the execution path for this target.

### Remaining blockers before a real single-Spark boot

- Engram still defaults to roughly 200 GB of pinned host memory upstream. On
  GB10 that is the same unified memory pool, so a disk-backed Engram reader is
  required before the full model can fit.
- The first creation of the expert store still consumes checkpoint tensors one
  at a time through the ordinary weight iterator. Reusing a completed store
  should later skip those payload reads entirely.
- CUDA graphs are not a target for the synchronous pager. Dynamic residency
  must first become correct in eager execution.
- The B12X repack-on-miss path is a correctness baseline only.


## Performance path update

The DGX Spark path now targets `flashinfer_cutlass`, not B12X. SM121 is
accepted by vLLM's FlashInfer CUTLASS MXFP4/MXFP8 experts implementation, and
that kernel consumes the converted expert tensors directly.

The disk store is therefore a runtime-layout cache:

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
