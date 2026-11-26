#!/usr/bin/env bash
set -euo pipefail

CACHE_INDEX="${1:-data/cached_tiktok_480p/index.jsonl}"
STEPS="${2:-300}"
MAX_T_LAT="${3:-13}"
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"

if [ ! -f "${CACHE_INDEX}" ]; then
  echo "ERROR: Cache index not found at ${CACHE_INDEX}" >&2
  exit 1
fi

mkdir -p outputs/verify

echo "[1/3] Checking Wan2.1-VAE latent round-trip & selecting 4-clip verification bucket..."
python - "${CACHE_INDEX}" "${MAX_T_LAT}" << 'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from vieth3.inference.sample import write_video_mp4
from vieth3.models.vae_text import load_vae

idx_path = Path(sys.argv[1]).resolve()
max_t_lat = int(sys.argv[2])
rows = [json.loads(l) for l in idx_path.read_text(encoding="utf-8").splitlines() if l.strip()]
if not rows:
    raise RuntimeError(f"Empty cache index: {idx_path}")

# Pick the largest (t_lat, h_lat, w_lat) bucket so batch_size=2 across 2 GPUs always aligns cleanly.
bucket_counts = Counter((min(int(r["t_lat"]), max_t_lat), int(r["h_lat"]), int(r["w_lat"])) for r in rows)
target_bucket = bucket_counts.most_common(1)[0][0]
bucket_rows = [r for r in rows if (min(int(r["t_lat"]), max_t_lat), int(r["h_lat"]), int(r["w_lat"])) == target_bucket]
selected = [bucket_rows[i % len(bucket_rows)] for i in range(4)]

subset_path = Path("outputs/verify/verify_4clip_index.jsonl")
with open(subset_path, "w", encoding="utf-8") as f:
    for r in selected:
        item = dict(r)
        item["shard"] = str((idx_path.parent / r["shard"]).resolve())
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

first = selected[0]
t_lat = min(int(first["t_lat"]), max_t_lat)
h_lat, w_lat = int(first["h_lat"]), int(first["w_lat"])
meta = {
    "prompt": str(first.get("caption") or "Cô gái Việt Nam nhảy trong phòng sáng rõ nét."),
    "height": h_lat * 8,
    "width": w_lat * 8,
    "frames": 1 if t_lat == 1 else (t_lat - 1) * 4 + 1,
    "source_path": str(first.get("source_path", "")),
}
Path("outputs/verify/verify_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

with safe_open(str(idx_path.parent / first["shard"]), framework="pt", device="cpu") as f:
    latent = f.get_tensor(f"{first['key']}.latent")[:, :max_t_lat].unsqueeze(0)

vae = load_vae(
    "wan21",
    model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    device="cuda",
    dtype=torch.bfloat16,
)
with torch.no_grad():
    video = vae.decode(latent.to(device="cuda", dtype=torch.bfloat16))
video_u8 = (
    ((video.squeeze(0).float().clamp(-1.0, 1.0) + 1.0) * 127.5)
    .permute(1, 2, 3, 0)
    .cpu()
    .numpy()
    .astype(np.uint8)
)
out_mp4 = write_video_mp4(video_u8, "outputs/verify/vae_roundtrip.mp4", fps=24)
print(f"VAE round-trip saved to {out_mp4} (source={meta['source_path']})")
PY

echo "[2/3] Running ${STEPS}-step 4-clip overfit on ${NUM_GPUS} GPU(s) (max_t_lat=${MAX_T_LAT}, BF16)..."
if [ "${NUM_GPUS}" -le 1 ]; then
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" PYTHONUNBUFFERED=1 python -m vieth3.training.train \
    --config configs/train_tiktok_9x16_480p.yaml \
    --strategy single \
    --cache-index outputs/verify/verify_4clip_index.jsonl \
    --output-dir outputs/verify_overfit \
    --max-steps "${STEPS}" \
    --warmup-steps 20 \
    --max-t-lat "${MAX_T_LAT}" \
    --batch-size 2 \
    --grad-accum 1
else
  PYTHONUNBUFFERED=1 bash scripts/run_2xh100_train.sh configs/train_tiktok_9x16_480p.yaml \
    --cache-index outputs/verify/verify_4clip_index.jsonl \
    --output-dir outputs/verify_overfit \
    --max-steps "${STEPS}" \
    --warmup-steps 20 \
    --max-t-lat "${MAX_T_LAT}" \
    --batch-size 2 \
    --grad-accum 1
fi

echo "[3/3] Sampling from overfit checkpoint..."
CKPT=$(cat outputs/verify_overfit/checkpoints/latest)
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
  --output outputs/verify/overfit_sample.mp4 \
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

echo "Verification finished!"
echo "  1. Inspect outputs/verify/vae_roundtrip.mp4 (should closely match original clip)"
echo "  2. Inspect outputs/verify/overfit_sample.mp4 (should show human silhouette/scene structure)"
echo "  3. If both look clean, launch full 2x H100 training:"
echo "     bash scripts/run_2xh100_train.sh configs/train_tiktok_9x16_480p.yaml"
