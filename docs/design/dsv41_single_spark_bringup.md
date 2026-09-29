# Single-Spark bring-up

This is the first hardware-validation profile for the experimental DeepSeek
V4.1 NVMe paging branch. It intentionally optimizes for debuggability, not
speed.

## Preconditions

- checkout `feature/dsv41-nvme-expert-pager`;
- one DGX Spark / GB10;
- the official DeepSeek-V4.1-Flash checkpoint on fast local NVMe;
- enough free NVMe for the per-layer routed-expert stores (roughly the routed
  expert payload size again);
- no other large GPU/unified-memory workload.

## First boot

```bash
export MODEL_DIR=/path/to/DeepSeek-V4.1-Flash
export EXPERT_STORE_DIR=/fast-nvme/dsv41-expert-store

# Conservative first pass: 64 runtime-layout expert slots per layer, text only,
# one sequence, 8K context, eager mode, buffered disk I/O.
bash scripts/dsv41_single_spark_nvme.sh
```

Why 64 slots rather than 64: the correctness baseline retains both the raw
slot tensors and FlashInfer CUTLASS's prepared/packed representation. One original V4.1
routed expert is roughly 19 MB across its tensors; 64 slots x 40 routed
layers is roughly 18 GB of raw cache before the FlashInfer CUTLASS packed copy and runtime
buffers. 64 slots per layer is too aggressive as a default on a 128 GB UMA
machine.

The environment variable
`VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS` is mandatory in the engine itself when
paging is enabled, so an omitted tuning value fails fast rather than silently
reserving a dangerous cache.

## Expected first-start behavior

The first start builds one fixed-stride disk store per routed MoE layer while
the ordinary checkpoint iterator visits the expert tensors. Each expert is
converted once into FlashInfer CUTLASS MXFP8xMXFP4 runtime layout before being
persisted. Runtime cache misses therefore perform only NVMe read + one slot
copy; there is no resident-set repack. The Engram tables are not copied into
pinned host memory; hash-selected rows are read directly from their safetensors
shards.

The first store-build pass can be slow. The goal is:

1. model reaches API readiness without allocating all routed experts or the
   approximately 200 GB Engram tables;
2. one short greedy/text request completes coherently;
3. repeating the same request reuses the expert disk stores;
4. process memory stays comfortably below the GB10 unified-memory ceiling.

## Do not enable yet

For the first correctness run, keep all of these off:

- CUDA graphs;
- DSpark/speculative decoding;
- vision;
- multiple active requests;
- large prefill chunks;
- O_DIRECT.

After the single-request output is validated, raise in this order:

1. context to 32K then 128K/262K;
2. `MAX_SEQS=2` then 4/8 while keeping small batched-token limits;
3. vision;
4. asynchronous/batched expert reads;
5. incremental FlashInfer CUTLASS packed-slot updates;
6. DSpark;
7. CUDA graphs.

## Useful knobs

```bash
SERVED_MODEL_NAME=big
PORT=8000
CTX=8192
MAX_SEQS=1
MAX_BATCHED_TOKENS=8192
EXPERT_CACHE_SLOTS=64
ENGRAM_THREADS=32
GPU_MEMORY_UTILIZATION=0.70
```

If a prefill batch routes to more unique experts in one layer than the
configured slot count, the current correctness provider fails explicitly.
Lower `MAX_BATCHED_TOKENS` or raise `EXPERT_CACHE_SLOTS`; do not interpret
that failure as a model-quality issue.


## GB10 UVA slots

The performance profile enables `VLLM_DSV41_NVME_UVA_SLOTS=1` by default.
Expert slot tensors are backed by pinned CPU memory and exposed to CUDA using
vLLM's existing UVA device-view helper. On GB10 this is the same coherent
LPDDR5X pool.

For a cache miss, buffered `preadv` writes the four expert fields directly
into the actual slot backing:

```
NVMe -> pinned UMA expert slot -> CUDA UVA view -> FlashInfer CUTLASS
```

There is no staging-to-device copy in this mode. Slot overwrites are protected
by a CUDA event recorded after the previous MoE launch. Cache-hit-only decode
does not wait on that event.

A/B fallback:

```bash
VLLM_DSV41_NVME_UVA_SLOTS=0 bash scripts/dsv41_single_spark_nvme.sh
```

The fallback keeps CUDA-resident slots and uses aligned pinned staging plus
batched async H2D.


## Pager performance counters

Set:

```bash
export VLLM_DSV41_NVME_STATS_EVERY=128
```

to emit per-layer cumulative statistics every 128 pager calls:

```text
DSV4.1 NVMe layer=... hit=...% reads=... (... GiB) read=... GB/s slots=... UVA=...
```

This separates three important causes of slow decode: insufficient resident
hit rate, low effective NVMe throughput, and compute outside the pager.
