#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw_videos}"
WORK_DIR="${2:-data}"
MODEL_CFG="${3:-configs/model_vieth3_mini_500m.yaml}"
BUCKET_STAGE="${4:-480p}"
VLM_API="${VLM_API:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

echo "[1/4] Splitting raw videos into 24 fps H3-grid (17n + 5) clips..."
python -m vieth3.data.scene_split \
  --input-dir "${RAW_DIR}" \
  --output-dir "${WORK_DIR}/clips" \
  --fps 24 \
  --grid h3 \
  --target-frames 56 73 90 124

echo "[2/4] Filtering clips (resolution, sharpness, motion, watermark/hardsub)..."
python -m vieth3.data.filters \
  --input "${WORK_DIR}/clips/manifest_clips.jsonl" \
  --output "${WORK_DIR}/manifest_filtered.jsonl" \
  --min-width 256 \
  --min-height 256 \
  --min-frames 22

echo "[3/4] Recaptioning clips in native Vietnamese..."
python -m vieth3.data.recaption_vi \
  --manifest "${WORK_DIR}/manifest_filtered.jsonl" \
  --output "${WORK_DIR}/manifest_captioned.jsonl" \
  --api-base "${VLM_API}" \
  --model "${VLM_MODEL}"

echo "[4/4] Caching VAE latents & Qwen Vietnamese text embeddings..."
python -m vieth3.data.cache_latents \
  --manifest "${WORK_DIR}/manifest_captioned.jsonl" \
  --output-dir "${WORK_DIR}/cached_${BUCKET_STAGE}" \
  --model-config "${MODEL_CFG}" \
  --vae-type h3 \
  --text-encoder qwen25vl \
  --bucket-stage "${BUCKET_STAGE}" \
  --device cuda \
  --dtype bfloat16

echo "Done! Cache ready at ${WORK_DIR}/cached_${BUCKET_STAGE}/index.jsonl"
