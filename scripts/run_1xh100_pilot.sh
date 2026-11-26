#!/usr/bin/env bash
set -euo pipefail

PILOT_STEPS="${1:-3000}"

mkdir -p data/raw_tiktok

echo "[1/5] Downloading curated TikTok URLs from data/urls_tiktok.txt..."
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

echo "[2/5] Preparing scenes, local Vietnamese Qwen2.5-VL captions, and Wan2.1 latents..."
SKIP_FILTER=1 RECAPTION_BACKEND=local bash scripts/run_tiktok_9x16_pipeline.sh \
  data/raw_tiktok \
  data/tiktok_work \
  configs/model_vieth3_tiktok_9x16_mini.yaml \
  480p

echo "[3/5] Running 1x H100 pre-flight & 10-minute verification gate..."
python scripts/h100_preflight.py \
  --config configs/model_vieth3_tiktok_9x16_mini.yaml \
  --gpus 1
NUM_GPUS=1 bash scripts/verify_pipeline.sh data/cached_tiktok_480p/index.jsonl 300

echo "[4/5] Launching 1x H100 pilot training (${PILOT_STEPS} steps)..."
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TOKENIZERS_PARALLELISM=false \
python -m vieth3.training.train \
  --config configs/train_tiktok_9x16_480p.yaml \
  --strategy single \
  --max-steps "${PILOT_STEPS}"

echo "[5/5] Generating final pilot sample from latest checkpoint..."
CKPT=$(cat outputs/tiktok_9x16_480p/checkpoints/latest)
META_VALUES=$(python - << 'PY'
import json
from pathlib import Path
m = json.loads(Path("outputs/verify/verify_meta.json").read_text(encoding="utf-8"))
print(f"{m['height']}\t{m['width']}\t{m['frames']}\t{m['prompt']}")
PY
)
IFS=$'\t' read -r V_H V_W V_FRAMES V_PROMPT <<< "${META_VALUES}"

python -m vieth3.inference.sample \
  --model-config configs/model_vieth3_tiktok_9x16_mini.yaml \
  --checkpoint "${CKPT}" \
  --prompt "${V_PROMPT}" \
  --output outputs/tiktok_9x16_480p/pilot_sample.mp4 \
  --height "${V_H}" \
  --width "${V_W}" \
  --frames "${V_FRAMES}" \
  --steps 25 \
  --cfg-scale 5.0 \
  --flow-shift 5.0 \
  --vae-type wan21 \
  --vae-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --text-encoder qwen25vl \
  --device cuda

echo "1x H100 pilot complete!"
echo "  VAE check     : outputs/verify/vae_roundtrip.mp4"
echo "  Overfit check : outputs/verify/overfit_sample.mp4"
echo "  Pilot sample  : outputs/tiktok_9x16_480p/pilot_sample.mp4"
