# VietH3 Architectural Improvements & A/B Experiment Roadmap

This document tracks the **high-ROI architectural upgrades** to test sequentially on **2× NVIDIA H100 (80GB)** GPUs after establishing the baseline run.

> **Baseline Reference (Verified):**
> - `VietH3-Mini`: `configs/model_vieth3_mini_500m.yaml` — **566,335,968 params (0.566B)**, ~8.04 GB/GPU peak (`FSDP + Grad Ckpt`)
> - `VietH3-Base`: `configs/model_vieth3_base_1_5b.yaml` — **1,457,792,096 params (1.458B)**, ~15.65 GB/GPU peak (`FSDP + Grad Ckpt`)
> - Core modules: `src/vieth3/models/dit.py`, `src/vieth3/models/rope3d.py`, `src/vieth3/models/flow_matching.py`

---

## 1. High-ROI Upgrades (Ranked for 2× H100 Budget)

### Upgrade #1 — REPA / Video-REPA (Representation Alignment)
- **Where:** Mid-network DiT hidden states (`layer L // 2`) during training only.
- **What to change:**
  - Pre-cache or compute frozen visual teacher features (e.g., **DINOv2** per-frame spatial features or **V-JEPA 2** spatio-temporal features) on clean clips.
  - Attach a lightweight 2-layer MLP projector (`hidden_size → teacher_dim`) at `blocks[num_layers // 2]`.
  - Add an auxiliary cosine similarity loss: `loss = flow_mse_loss + repa_weight * (1 - cos_sim(proj(h_mid), teacher_feat))`.
  - Discard the projector completely at inference (**0 extra inference VRAM or latency**).
- **Planned config flags:** `repa_enabled: true`, `repa_layer_idx: 8`, `repa_weight: 0.5`, `repa_teacher: "dinov2_vitb14"`.
- **Expected payoff:** 5×–15× faster convergence in early pretraining because the DiT receives dense semantic structure rather than learning visual concepts purely from noisy velocity MSE.

---

### Upgrade #2 — Shared Time-MLP + Per-Block Bias (`AdaLN-Single`) — **IMPLEMENTED**
- **Status:** Implemented in `src/vieth3/models/dit.py` (`shared_adaln: true`).
- **What changed:** Replaces per-block dual `Linear(H → 6H)` layers (~37% of parameters) with one global `shared_adaLN_txt` / `shared_adaLN_vid` + learnable per-block `6H` biases, enabling **24 layers** instead of 16 at smaller total parameter count (`configs/model_vieth3_tiktok_9x16_dense.yaml`).

---

### Upgrade #2B — Learnable Register Tokens + Symmetric Long Skip Connections (`U-ViT / Skip-DiT`) — **IMPLEMENTED**
- **Status:** Implemented in `src/vieth3/models/dit.py` (`num_register_tokens: 8`, `use_long_skips: true`).
- **What changed:**
  - Appends `8` learnable `[REG]` tokens into the unified stream to absorb global outlier features away from low-texture background walls (reduces glowing artifacts).
  - Adds zero-initialized symmetric `encoder[i] → decoder[L - 1 - i]` skip projections to preserve crisp Frame-0 background details through 24 layers.

---

### Upgrade #2C — Dispersive Loss Regularization — **IMPLEMENTED**
- **Status:** Implemented in `src/vieth3/models/flow_matching.py` and `src/vieth3/training/train.py` (`dispersive_weight: 0.1`, `dispersive_temp: 0.5`).
- **What changed:** Penalizes feature collapse on `blocks[L // 2 - 1]` across batch items with zero external teacher and zero inference cost.

---

### Upgrade #3 — Hybrid Local + Global / Sparse 3D Attention (FastH3 / VSA) & MicroDiT Patch-Mixer — *Next Phase*
- **Where:** `VietH3SingleStreamBlock` attention in `src/vieth3/models/dit.py`.
- **What to change:**
  - Interleave local 3D spatio-temporal window attention (each video token attends to its local `(t, h, w)` window + **all** text tokens) with periodic global 3D full-sequence attention blocks (`3L1G`).
  - Optionally pair with MicroDiT deferred patch masking (drop 50–75% of video patches after 2 mixer blocks during early pretraining for 4×–6× throughput at `720p`).
- **Planned config flags:** `sparse_attn_pattern: "3L1G"`, `local_window_size: [4, 8, 8]`, `patch_mask_ratio: 0.5`.
- **Expected payoff:** 2×–6× attention speedup at `480p`–`720p`, unlocking longer `90f–121f` clips cleanly on 2× H100.

---

### Upgrade #4 — Timestep-Routed Experts / MoE (Wan 2.2 Trick)
- **Where:** FFN / block routing across diffusion timesteps `t ∈ [0, 1]`.
- **What to change:**
  - Split or route experts by noise regime:
    - **High-noise expert (`t >= 0.5`):** specializes in global scene layout, object trajectories, and camera motion.
    - **Low-noise expert (`t < 0.5`):** specializes in sharp Vietnamese textures, faces, *áo dài* fabric, and lighting.
  - Because only one expert runs per timestep (during sampling) or per bucketed batch, active FLOPs stay constant.
- **Planned config flags:** `timestep_moe: true`, `moe_split_t: 0.5`.
- **Expected payoff:** ~2× model capacity with **0% extra inference FLOPs**.

---

## 2. Components to Keep Frozen (Low ROI to Change)

Do **not** spend experiment cycles modifying these unless a specific bug requires it:
- **Frozen 3D VAE & Qwen Text Encoder** (`src/vieth3/models/vae_text.py`) — offline cached (`0 GB` training VRAM).
- **Text Token Refiner & Flow Shift (`3.0`)** (`src/vieth3/models/dit.py`, `src/vieth3/models/flow_matching.py`) — already aligned with MiniMax-H3 / SD3 baselines.
- **Micro-block primitives:** `RMSNorm`, `QK-Norm`, `SwiGLU`, and base Rectified Flow velocity MSE.
- **3D RoPE axes `(t, h, w)`** (`src/vieth3/models/rope3d.py`) — keep standard unless extending inference beyond trained durations (then add NTK/YaRN frequency scaling).

---

## 3. Sequential A/B Testing Protocol

Always test **one change at a time** against the identical cached `256p` Vietnamese dataset (`configs/train_stage1_256p.yaml`) for a fixed short run (e.g., `5,000` steps on 2× H100):

| Run ID | Variant | Primary Metrics to Record |
| :--- | :--- | :--- |
| **Exp 0** | **Baseline (`VietH3-Mini 0.566B`)** | Validation flow MSE (`t ∈ {0.2, 0.5, 0.8}`), `sec/step`, peak VRAM/GPU, fixed 10-prompt Vietnamese visual grid |
| **Exp 1** | Baseline + **Upgrade #1 (REPA)** | Convergence speedup (steps to match Exp 0 val loss & prompt alignment) |
| **Exp 2** | Best of Exp 1 + **Upgrade #2 (Shared AdaLN + deeper L)** | Quality/motion improvement at matched ~0.57B param count |
| **Exp 3** | Best of Exp 2 + **Upgrade #3 (3L1G Sparse Attn)** | Step time & VRAM savings on `480p` (`56f–90f`) clips |
| **Exp 4** | Best of Exp 3 + **Upgrade #4 (Timestep MoE)** | High-frequency texture sharpness & motion coherence at `480p` |
