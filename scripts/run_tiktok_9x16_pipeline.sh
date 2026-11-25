#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw_tiktok}"
WORK_DIR="${2:-data/tiktok_work}"
MODEL_CFG="${3:-configs/model_vieth3_tiktok_9x16_mini.yaml}"
BUCKET_STAGE="${4:-480p}" # 480p or 720p
WITH_POSE="${WITH_POSE:-0}"
VLM_API="${VLM_API:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

echo "[1/6] Ingesting & deduplicating raw TikTok videos..."
python -m vieth3.data.ingest_tiktok \
  --local-dir "${RAW_DIR}" \
  --output-dir "${WORK_DIR}/ingested"

echo "[2/6] Splitting into 24 fps Wan2.1-grid (4n + 1) clips..."
python -m vieth3.data.scene_split \
  --input-dir "${RAW_DIR}" \
  --output-dir "${WORK_DIR}/clips" \
  --fps 24 \
  --grid wan21 \
  --target-frames 49 65 81 121

echo "[3/6] Filtering for 9:16 portrait, tripod-locked background, zero flicker/glow..."
python -m vieth3.data.filters \
  --input "${WORK_DIR}/clips/manifest_clips.jsonl" \
  --output "${WORK_DIR}/manifest_filtered.jsonl" \
  --min-width 480 \
  --min-height 832 \
  --min-frames 21 \
  --require-portrait \
  --max-bg-motion 0.25 \
  --min-fg-motion 1.0 \
  --max-flicker 12.0 \
  --max-overlay-score 0.18

FILTERED_MANIFEST="${WORK_DIR}/manifest_filtered.jsonl"

if [ "${WITH_POSE}" = "1" ]; then
  echo "[4/6] Extracting MediaPipe body + hand pose skeletons..."
  python -m vieth3.data.pose_extract \
    --manifest "${FILTERED_MANIFEST}" \
    --output-dir "${WORK_DIR}/poses" \
    --output-manifest "${WORK_DIR}/manifest_posed.jsonl" \
    --backend mediapipe
  FILTERED_MANIFEST="${WORK_DIR}/manifest_posed.jsonl"
else
  echo "[4/6] Skipping pose extraction (set WITH_POSE=1 to enable)."
fi

echo "[5/6] Recaptioning dance clips in native Vietnamese..."
python -m vieth3.data.recaption_vi \
  --manifest "${FILTERED_MANIFEST}" \
  --output "${WORK_DIR}/manifest_captioned.jsonl" \
  --api-base "${VLM_API}" \
  --model "${VLM_MODEL}"

echo "[6/6] Caching Wan2.1-VAE (float32, deterministic) + Qwen Vietnamese embeddings..."
python -m vieth3.data.cache_latents \
  --manifest "${WORK_DIR}/manifest_captioned.jsonl" \
  --output-dir "${WORK_DIR}/cached_${BUCKET_STAGE}" \
  --model-config "${MODEL_CFG}" \
  --vae-type wan21 \
  --vae-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --text-encoder qwen25vl \
  --bucket-stage "${BUCKET_STAGE}" \
  --device cuda \
  --dtype bfloat16

echo "Done! Cached TikTok dataset ready at ${WORK_DIR}/cached_${BUCKET_STAGE}/index.jsonl"
