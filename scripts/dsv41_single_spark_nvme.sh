#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the local DeepSeek-V4.1-Flash snapshot}"
: "${EXPERT_STORE_DIR:?Set EXPERT_STORE_DIR to a fast local NVMe directory}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-big}"
PORT="${PORT:-8000}"
CTX="${CTX:-8192}"
MAX_SEQS="${MAX_SEQS:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
EXPERT_CACHE_SLOTS="${EXPERT_CACHE_SLOTS:-64}"
ENGRAM_THREADS="${ENGRAM_THREADS:-32}"

mkdir -p "${EXPERT_STORE_DIR}"

export VLLM_DSV41_NVME_EXPERT_STORE_DIR="${EXPERT_STORE_DIR}"
export VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS="${EXPERT_CACHE_SLOTS}"
# Buffered reads first. Enable O_DIRECT only after correctness is established.
export VLLM_DSV41_NVME_DIRECT_IO="${VLLM_DSV41_NVME_DIRECT_IO:-0}"
export VLLM_DSV41_NVME_EXPERT_READ_BATCH="${VLLM_DSV41_NVME_EXPERT_READ_BATCH:-8}"
# GB10 UMA: let NVMe reads write the pinned backing that FlashInfer reads
# through a CUDA UVA view. Set to 0 for the conventional staged-CUDA path.
export VLLM_DSV41_NVME_UVA_SLOTS="${VLLM_DSV41_NVME_UVA_SLOTS:-1}"

export VLLM_DSV41_ENGRAM_DISK=1
export VLLM_DSV41_ENGRAM_DISK_DIR="${MODEL_DIR}"
export VLLM_DSV41_ENGRAM_DISK_THREADS="${ENGRAM_THREADS}"

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
