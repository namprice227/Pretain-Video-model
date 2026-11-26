#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw_tiktok

echo "[1/4] Downloading curated TikTok URLs from data/urls_tiktok.txt..."
yt-dlp \
  --ignore-errors \
  --no-playlist \
  --sleep-interval 2 \
  --max-sleep-interval 5 \
  --download-archive data/raw_tiktok/downloaded_archive.txt \
  -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b" \
  -o "data/raw_tiktok/%(uploader_id)s_%(id)s.%(ext)s" \
  -a data/urls_tiktok.txt || true

MP4_COUNT=$(find data/raw_tiktok -maxdepth 1 -type f -name "*.mp4" | wc -l)
if [ "${MP4_COUNT}" -eq 0 ]; then
  echo "ERROR: 0 MP4 files found in data/raw_tiktok after yt-dlp." >&2
  exit 1
fi
echo "Ready: ${MP4_COUNT} raw MP4 files in data/raw_tiktok"

echo "[2/4] Preparing scenes, local Vietnamese Qwen2.5-VL captions, and Wan2.1 latents..."
SKIP_FILTER=1 RECAPTION_BACKEND=local bash scripts/run_tiktok_9x16_pipeline.sh \
  data/raw_tiktok \
  data/tiktok_work \
  configs/model_vieth3_tiktok_9x16_mini.yaml \
  480p

echo "[3/4] Running 2x H100 pre-flight VRAM & environment check..."
python scripts/h100_preflight.py \
  --config configs/model_vieth3_tiktok_9x16_mini.yaml \
  --gpus 2

echo "[4/4] Launching 2x H100 FSDP training..."
bash scripts/run_2xh100_train.sh configs/train_tiktok_9x16_480p.yaml
