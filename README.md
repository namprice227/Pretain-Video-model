# VietH3 — Vietnamese Text-to-Video Pretraining on 2× H100 GPUs

**VietH3** is a clean-room, MiniMax-H3-inspired single-stream Omni-DiT video generation codebase engineered specifically to train on **Vietnamese video + image data** using **2× NVIDIA H100 (80GB)** GPUs within a **$1K–$3K pilot budget**.

---

## 1. MiniMax-H3 → VietH3 Architecture Mapping

| Component | MiniMax-H3 (33B) | VietH3 Pilot (`mini_500m`) | VietH3 Scale-Up (`base_1_5b`) |
| :--- | :--- | :--- | :--- |
| **Backbone** | Single-stream Omni-DiT (`hidden=5376, L=50`) | Single-stream DiT (`hidden=1152, L=16`, ~0.57B) | Single-stream DiT (`hidden=1536, L=24`, ~1.45B) |
| **Modality Fusion** | Unified `[text, video]` sequence + dual AdaLN-Zero | Identical dual AdaLN-Zero + shared QKV/FFN | Identical dual AdaLN-Zero + shared QKV/FFN |
| **Text Refiner** | 2-layer refiner + 1D RoPE | 2-layer `TextTokenRefinerBlock` + 1D RoPE | 2-layer `TextTokenRefinerBlock` + 1D RoPE |
| **Position Encoding** | 3D Multimodal RoPE `(t, h, w)` | 3D RoPE `(32, 48, 48)` | 3D RoPE `(32, 48, 48)` |
| **Norm & FFN** | RMSNorm + QK-Norm + SwiGLU | RMSNorm + QK-Norm + SwiGLU | RMSNorm + QK-Norm + SwiGLU |
| **Patchification** | `[1, 2, 2]` | `[1, 2, 2]` | `[1, 2, 2]` |
| **3D Visual VAE** | `H3-VisualVAE` (`f16t4d24`, `17n + 5` @ 24fps) | `H3-VisualVAE` (or Apache-2.0 `Wan2.1-VAE`) | `H3-VisualVAE` (or Apache-2.0 `Wan2.1-VAE`) |
| **Vietnamese Text Encoder** | Qwen3-VL layer 50 (offline cached) | Qwen2.5-VL-7B / Qwen3-VL (offline cached) | Qwen2.5-VL-7B / Qwen3-VL (offline cached) |
| **Training Objective** | Rectified Flow Matching + logit-normal shift `3.0` | Rectified Flow Matching + shift `3.0` | Rectified Flow Matching + shift `3.0` |

> **Why offline caching matters:** By pre-computing VAE latents and Qwen Vietnamese embeddings once into sharded `.safetensors` files, the frozen VAE + text encoder consume **0 GB VRAM** during DiT training.

---

## 2. 2× H100 (80GB) Memory & Budget Calculator

Run the exact meta-tensor memory & cloud budget calculator locally:

```bash
python -m vieth3.training.memory_budget --config configs/model_vieth3_mini_500m.yaml --gpus 2 --height 480 --width 832 --frames 56 --batch-size 2
python -m vieth3.training.memory_budget --config configs/model_vieth3_base_1_5b.yaml --gpus 2 --height 480 --width 832 --frames 56 --batch-size 2
```

- Both configurations fit comfortably inside **2× H100 80GB** using PyTorch **FSDP + BF16 + Gradient Checkpointing + FlashAttention/SDPA**.
- **Pilot cost guide (2× H100):**
  - Vast.ai marketplace (~$1.75/GPU-hr = **$3.50/hr** per 2× H100 node)
  - RunPod / Lambda dedicated (~$3.80/GPU-hr = **$7.60/hr** per 2× H100 node)
  - A full progressive pilot (`Stage 1: 256p` + `Stage 2: 480p`) fits comfortably in a **$1K–$3K** budget.

---

## 3. Local Setup & Verification

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,data]"

# Run unit & integration tests
.venv/Scripts/pytest -v

# Run end-to-end synthetic Vietnamese data -> train -> MP4 smoke test
.venv/Scripts/python scripts/run_local_smoke_test.py
```

> **Windows Smart App Control note:** If Windows Application Control (`WinError 4551`) blocks `torch_python.dll` on native Windows, run the exact same local verification commands inside **WSL `Ubuntu`** (which matches the Linux 2× H100 target).

---

## 4. Cloud Deployment (2× H100 Node)

1. Rent a **2× H100 80GB SXM** instance (Ubuntu 22.04/24.04, CUDA 12.x, >= 1 TB NVMe disk) and clone your repo.
2. Run the 1-command Conda cluster setup (`source ~/miniconda/bin/activate` → `conda create` → CUDA PyTorch + extras + `pytest` + H100 preflight):
   ```bash
   bash scripts/setup_cluster.sh
   # Optional flags: WITH_POSE=1 WITH_VLLM=1 bash scripts/setup_cluster.sh
   ```
3. Prepare your Vietnamese dataset (`data/raw_videos/`):
   ```bash
   # Optional: serve Qwen2.5-VL-7B-Instruct locally with vLLM on port 8000 for Vietnamese recaptioning
   bash scripts/run_data_pipeline.sh data/raw_videos data configs/model_vieth3_mini_500m.yaml 256p
   bash scripts/run_data_pipeline.sh data/raw_videos data configs/model_vieth3_mini_500m.yaml 480p
   ```
4. Progressive training (`Stage 0 → Stage 1 → Stage 2`):
   - **Stage 0 (Sanity overfit on 8 clips):**
     ```bash
     python -m vieth3.training.train --config configs/train_stage0_overfit.yaml --device cuda
     ```
   - **Stage 1 (256p joint Image + Video pretraining on 2× H100 FSDP):**
     ```bash
     bash scripts/run_2xh100_train.sh configs/train_stage1_256p.yaml
     ```
   - **Stage 2 (480p high-res continuation from Stage 1 checkpoint):**
     ```bash
     bash scripts/run_2xh100_train.sh configs/train_stage2_480p.yaml --resume outputs/stage1_256p/checkpoints/step_030000
     ```
5. Generate Vietnamese Text-to-Video MP4s:
   ```bash
   python -m vieth3.inference.sample \
     --model-config configs/model_vieth3_mini_500m.yaml \
     --checkpoint outputs/stage2_480p/checkpoints/step_020000 \
     --prompt "Một cô gái mặc áo dài trắng đi bộ trên phố cổ Hội An rực rỡ đèn lồng lúc hoàng hôn, ánh sáng vàng ấm, góc máy lia chậm." \
     --height 480 --width 832 --frames 56 --steps 30 --cfg-scale 5.0 \
     --vae-type h3 --text-encoder qwen25vl \
     --output outputs/hoian.mp4
   ```

---

## 5. Planned Architectural Improvements & TikTok 9:16 Playbook

- **Upgrade Roadmap:** See [docs/ARCHITECTURE_IMPROVEMENTS.md](docs/ARCHITECTURE_IMPROVEMENTS.md) for the prioritized A/B upgrade roadmap (**REPA**, **Shared AdaLN**, **3L1G Sparse Attention**, **Timestep MoE**).
- **9:16 TikTok Dance Data & Wan2.1-VAE Workflow:** See [docs/TIKTOK_DATA_PLAYBOOK.md](docs/TIKTOK_DATA_PLAYBOOK.md) and run `python scripts/h100_preflight.py --config configs/model_vieth3_tiktok_9x16_mini.yaml` on your 2× H100 node before training.
