#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_stage1_256p.yaml}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TOKENIZERS_PARALLELISM=false

torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m vieth3.training.train \
  --config "${CONFIG}" \
  "${@:2}"
