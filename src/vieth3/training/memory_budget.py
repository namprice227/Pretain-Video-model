"""Exact VRAM and 2x H100 cloud budget estimator for VietH3 configurations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel


@dataclass
class MemoryBudgetReport:
    model_name: str
    exact_params: int
    gpus: int
    tokens_per_sample: int
    static_ddp_gb_per_gpu: float
    static_fsdp_gb_per_gpu: float
    activation_ckpt_gb_per_gpu: float
    activation_no_ckpt_gb_per_gpu: float
    total_fsdp_ckpt_gb_per_gpu: float
    total_ddp_ckpt_gb_per_gpu: float
    fits_80gb_h100: bool
    sec_per_step_est: float
    cost_50k_steps_vast_usd: float
    cost_50k_steps_runpod_usd: float


def estimate_memory_and_budget(
    config: VietH3Config,
    gpus: int = 2,
    height: int = 480,
    width: int = 832,
    frames: int = 56,
    batch_size: int = 2,
    grad_accum: int = 4,
    pilot_steps: int = 50_000,
) -> MemoryBudgetReport:
    """Compute exact meta-tensor parameter count, per-GPU VRAM, and H100 cost."""
    with torch.device("meta"):
        meta_model = VietH3Transformer3DModel(config)
        exact_params = meta_model.count_parameters()

    t_lat = 1 if frames == 1 else (frames - 1) // config.vae_temporal_stride + 1
    h_lat = height // config.vae_spatial_stride
    w_lat = width // config.vae_spatial_stride
    pt, ph, pw = config.patch_size
    l_vid = (t_lat // pt) * (h_lat // ph) * (w_lat // pw)
    l_total = config.max_text_len + l_vid

    gb = 1024**3
    # 16 bytes/param: 2B BF16 weights + 2B BF16 grads + 4B FP32 master + 8B AdamW m/v
    static_ddp_gb = (exact_params * 16.0) / gb
    static_fsdp_gb = (exact_params * 16.0 / max(1, gpus)) / gb

    # Activation memory with SDPA FlashAttention (O(L) attention memory):
    # - With gradient checkpointing: boundary hidden states across blocks + 1 active block
    b_l_h = batch_size * l_total * config.hidden_size
    boundary_bytes = config.num_layers * b_l_h * 2.0 * 2.0  # text+video residual streams
    active_block_bytes = b_l_h * (14.0 * 2.0) + batch_size * l_total * config.ffn_dim * 6.0
    act_ckpt_gb = (boundary_bytes + active_block_bytes) / gb

    # Without gradient checkpointing: all layers store full intermediate activations
    act_no_ckpt_gb = (config.num_layers * active_block_bytes) / gb

    total_fsdp_ckpt = static_fsdp_gb + act_ckpt_gb + 2.5  # +2.5 GB CUDA/NCCL overhead
    total_ddp_ckpt = static_ddp_gb + act_ckpt_gb + 2.5

    # FLOPs estimate per optimizer step (6 * N_params * tokens_per_global_step)
    global_tokens = batch_size * grad_accum * gpus * l_total
    flops_per_step = 6.0 * exact_params * global_tokens
    # H100 SXM BF16 peak = 989 TFLOPs; realistic 45% MFU with grad ckpt -> ~445 TFLOPs/GPU
    effective_flops = gpus * 445e12
    sec_per_step = flops_per_step / effective_flops

    total_hours = (pilot_steps * sec_per_step) / 3600.0
    vast_usd = total_hours * gpus * 1.75
    runpod_usd = total_hours * gpus * 3.80

    return MemoryBudgetReport(
        model_name=config.name,
        exact_params=exact_params,
        gpus=gpus,
        tokens_per_sample=l_total,
        static_ddp_gb_per_gpu=static_ddp_gb,
        static_fsdp_gb_per_gpu=static_fsdp_gb,
        activation_ckpt_gb_per_gpu=act_ckpt_gb,
        activation_no_ckpt_gb_per_gpu=act_no_ckpt_gb,
        total_fsdp_ckpt_gb_per_gpu=total_fsdp_ckpt,
        total_ddp_ckpt_gb_per_gpu=total_ddp_ckpt,
        fits_80gb_h100=total_fsdp_ckpt < 76.0,
        sec_per_step_est=sec_per_step,
        cost_50k_steps_vast_usd=vast_usd,
        cost_50k_steps_runpod_usd=runpod_usd,
    )


def format_report(r: MemoryBudgetReport) -> str:
    status = "FITS COMFORTABLY on 80GB H100" if r.fits_80gb_h100 else "EXCEEDS 80GB H100"
    return (
        f"=== VietH3 Memory & Budget Report ({r.model_name}) ===\n"
        f"Exact Parameters          : {r.exact_params:,} ({r.exact_params / 1e9:.3f}B)\n"
        f"GPUs                      : {r.gpus}x H100 80GB\n"
        f"Tokens per sample         : {r.tokens_per_sample:,}\n"
        f"Static VRAM/GPU (DDP)     : {r.static_ddp_gb_per_gpu:.2f} GB\n"
        f"Static VRAM/GPU (FSDP)    : {r.static_fsdp_gb_per_gpu:.2f} GB\n"
        f"Activations (Grad Ckpt)   : {r.activation_ckpt_gb_per_gpu:.2f} GB\n"
        f"Activations (No Ckpt)     : {r.activation_no_ckpt_gb_per_gpu:.2f} GB\n"
        f"Total Peak/GPU (FSDP+Ckpt): {r.total_fsdp_ckpt_gb_per_gpu:.2f} GB  [{status}]\n"
        f"Total Peak/GPU (DDP+Ckpt) : {r.total_ddp_ckpt_gb_per_gpu:.2f} GB\n"
        f"Est. Step Time            : {r.sec_per_step_est:.2f} s/step\n"
        f"Est. 50K-Step Pilot Cost  : ${r.cost_50k_steps_vast_usd:,.0f} (Vast.ai @$1.75/hr/GPU) | "
        f"${r.cost_50k_steps_runpod_usd:,.0f} (RunPod/Lambda @$3.80/hr/GPU)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate VietH3 VRAM and 2x H100 budget.")
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--pilot-steps", type=int, default=50_000)
    args = parser.parse_args()

    cfg = VietH3Config.from_yaml(args.config)
    report = estimate_memory_and_budget(
        cfg,
        gpus=args.gpus,
        height=args.height,
        width=args.width,
        frames=args.frames,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        pilot_steps=args.pilot_steps,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
