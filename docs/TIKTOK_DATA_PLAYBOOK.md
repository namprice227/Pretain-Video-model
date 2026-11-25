# Vietnamese 9:16 TikTok Dance Data Playbook

Clean training data determines 80% of final video stability. Follow this checklist before spending H100 hours.

---

## 1. "Gold" vs. "Poison" TikTok Clips

### Gold Clips (Keep)
- **9:16 portrait (`>= 540×960` or `720×1280`)**, single dancer, tripod-locked phone.
- **Static indoor/outdoor background** (bedroom, studio, café, street corner) with `bg_motion <= 0.25 px/frame`.
- **Natural even lighting** (daylight window or steady softbox).
- **Clean full-body or waist-up framing** where arms/hands stay inside the frame.

### Poison Clips (Reject Automatically or Manually)
- **Handheld / selfie walking / zoom edits:** Causes background walls to warp (`shaky_background`).
- **CapCut sparkles, flashing ring lights, RGB strobes:** Teaches the model to hallucinate "weird glowy moving things" (`flicker_or_glow`).
- **Heavy Vietnamese lyrics / floating stickers / split-screen duets:** Causes text artifacts (`watermark_or_hardsub`).
- **Hard jump-cuts every 0.8s:** `scene_split.py` cuts these out automatically (`min_frames: 21`).

---

## 2. Dataset Sizing Milestones

| Milestone | Clip Count | Goal | Expected Budget |
| :--- | :--- | :--- | :--- |
| **Pilot Stage 0** | **50 – 200 clips** | Prove Frame-0 background stays 100% still and body motion converges | **< \$20** (a few hours) |
| **Stage 1 (`480p`)** | **2,000 – 10,000 clips** | Generalize across Vietnamese rooms, outfits, and dance choreography | **\$300 – \$1,000** |
| **Stage 2 (`720p`)** | **5,000 – 20,000 clips** | Crisp `704×1248` (`9:16`) hands, faces, and high-frequency textures | **\$500 – \$2,000** |

---

## 3. Exact 2× H100 Workflow

1. **Pre-flight check on your rented 2× H100 node:**
   ```bash
   pip install -e ".[dev,data,hf,cuda]"
   python scripts/h100_preflight.py --config configs/model_vieth3_tiktok_9x16_mini.yaml --gpus 2 --check-hf
   ```
2. **Run the 9:16 Wan2.1-VAE data pipeline** (with local `Qwen2.5-VL-7B-Instruct` vLLM server on port `8000`):
   ```bash
   bash scripts/run_tiktok_9x16_pipeline.sh data/raw_tiktok data/tiktok_work configs/model_vieth3_tiktok_9x16_mini.yaml 480p
   ```
3. **Train Stage 1 (`480p`) → Stage 2 (`720p`):**
   ```bash
   bash scripts/run_2xh100_train.sh configs/train_tiktok_9x16_480p.yaml
   bash scripts/run_2xh100_train.sh configs/train_tiktok_9x16_720p.yaml --resume outputs/tiktok_9x16_480p/checkpoints/step_030000
   ```
4. **Generate a `9:16` dance video anchored to a starting frame:**
   ```bash
   python -m vieth3.inference.sample \
     --model-config configs/model_vieth3_tiktok_9x16_mini.yaml \
     --checkpoint outputs/tiktok_9x16_720p/checkpoints/step_020000 \
     --ref-image my_room_dancer_frame0.png \
     --prompt "Cô gái Việt Nam nhảy TikTok nhịp nhàng trong phòng sáng tự nhiên, máy quay cố định, phông nền đứng yên." \
     --height 1248 --width 704 --frames 49 --steps 30 --cfg-scale 5.0 --flow-shift 7.0 \
     --vae-type wan21 --vae-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers --text-encoder qwen25vl \
     --output outputs/tiktok_dance_720p.mp4
   ```
