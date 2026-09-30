#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the local DeepSeek-V4.1-Flash snapshot}"
: "${EXPERT_STORE_DIR:?Set EXPERT_STORE_DIR to a fast local NVMe directory}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-big}"
PORT="${PORT:-8000}"
CTX="${CTX:-8192}"
MAX_SEQS="${MAX_SEQS:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
EXPERT_CACHE_SLOTS="${EXPERT_CACHE_SLOTS:-64}"
ENGRAM_THREADS="${ENGRAM_THREADS:-32}"

mkdir -p "${EXPERT_STORE_DIR}"

export VLLM_DSV41_NVME_EXPERT_STORE_DIR="${EXPERT_STORE_DIR}"
export VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS="${EXPERT_CACHE_SLOTS}"
# Aligned GB10 UVA slots support direct NVMe reads. Unsupported filesystems
# or alignment automatically fall back to buffered preadv.
export VLLM_DSV41_NVME_DIRECT_IO="${VLLM_DSV41_NVME_DIRECT_IO:-1}"
export VLLM_DSV41_NVME_EXPERT_READ_BATCH="${VLLM_DSV41_NVME_EXPERT_READ_BATCH:-8}"
export VLLM_DSV41_NVME_IO_WORKERS="${VLLM_DSV41_NVME_IO_WORKERS:-32}"
export VLLM_DSV41_NVME_SPLIT_MISS_MIN_TOKENS="${VLLM_DSV41_NVME_SPLIT_MISS_MIN_TOKENS:-64}"
export VLLM_DSV41_NVME_IO_WORKERS="${VLLM_DSV41_NVME_IO_WORKERS:-32}"
export VLLM_DSV41_NVME_DROP_PAGE_CACHE="${VLLM_DSV41_NVME_DROP_PAGE_CACHE:-1}"
export VLLM_DSV41_NVME_STATS_EVERY="${VLLM_DSV41_NVME_STATS_EVERY:-0}"
# GB10 UMA: let NVMe reads write the pinned backing that FlashInfer reads
# through a CUDA UVA view. Set to 0 for the conventional staged-CUDA path.
export VLLM_DSV41_NVME_UVA_SLOTS="${VLLM_DSV41_NVME_UVA_SLOTS:-1}"

export VLLM_DSV41_ENGRAM_DISK=1
export VLLM_DSV41_ENGRAM_DISK_DIR="${MODEL_DIR}"
export VLLM_DSV41_ENGRAM_DISK_THREADS="${ENGRAM_THREADS}"
export VLLM_DSV41_ENGRAM_IO_WORKERS="${VLLM_DSV41_ENGRAM_IO_WORKERS:-32}"
export VLLM_DSV41_ENGRAM_STAGE_WORKERS="${VLLM_DSV41_ENGRAM_STAGE_WORKERS:-4}"
export VLLM_DSV41_ENGRAM_STAGE_WORKERS="${VLLM_DSV41_ENGRAM_STAGE_WORKERS:-4}"
export VLLM_DSV41_ENGRAM_DROP_PAGE_CACHE="${VLLM_DSV41_ENGRAM_DROP_PAGE_CACHE:-1}"

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --moe-backend flashinfer_cutlass \
  --enforce-eager \
  --language-model-only \
  --max-model-len "${CTX}" \
  --max-num-seqs "${MAX_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41
