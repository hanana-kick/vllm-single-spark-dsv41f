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

# Conservative first pass: 24 raw expert slots per layer, text only,
# one sequence, 8K context, eager mode, buffered disk I/O.
bash scripts/dsv41_single_spark_nvme.sh
```

Why 24 slots rather than 64: the correctness baseline retains both the raw
slot tensors and B12X's prepared/packed representation. One original V4.1
routed expert is roughly 19 MB across its tensors; 24 slots x 40 routed
layers is roughly 18 GB of raw cache before the B12X packed copy and runtime
buffers. 64 slots per layer is too aggressive as a default on a 128 GB UMA
machine.

The environment variable
`VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS` is mandatory in the engine itself when
paging is enabled, so an omitted tuning value fails fast rather than silently
reserving a dangerous cache.

## Expected first-start behavior

The first start builds one fixed-stride disk store per routed MoE layer while
the ordinary checkpoint iterator visits the expert tensors. The Engram tables
are not copied into pinned host memory; hash-selected rows are read directly
from their safetensors shards.

This first pass can be slow. The goal is:

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
5. incremental B12X packed-slot updates;
6. DSpark;
7. CUDA graphs.

## Useful knobs

```bash
SERVED_MODEL_NAME=big
PORT=8000
CTX=8192
MAX_SEQS=1
MAX_BATCHED_TOKENS=8
EXPERT_CACHE_SLOTS=24
ENGRAM_THREADS=32
GPU_MEMORY_UTILIZATION=0.70
```

If a prefill batch routes to more unique experts in one layer than the
configured slot count, the current correctness provider fails explicitly.
Lower `MAX_BATCHED_TOKENS` or raise `EXPERT_CACHE_SLOTS`; do not interpret
that failure as a model-quality issue.
