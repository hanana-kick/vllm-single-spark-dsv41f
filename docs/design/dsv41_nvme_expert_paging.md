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
