"""1-command H100 (or local CPU --dry-run) pre-flight check for VietH3."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

import torch

from vieth3.data.scene_split import find_ffmpeg, find_ffprobe
from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel
from vieth3.models.flow_matching import RectifiedFlowScheduler
from vieth3.training.memory_budget import estimate_memory_and_budget, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight environment & VRAM check for VietH3.")
    parser.add_argument("--config", default="configs/model_vieth3_tiktok_9x16_mini.yaml")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Run CPU smoke verification without CUDA GPUs.")
    parser.add_argument("--check-hf", action="store_true", help="Test Wan2.1-VAE + Qwen2.5-VL round-trip.")
    args = parser.parse_args()

    print("=== VietH3 Pre-Flight Environment Check ===")
    print(f"Python  : {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch : {torch.__version__}")

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()
    print(f"FFmpeg  : {ffmpeg}")
    print(f"FFprobe : {ffprobe}")

    disk = shutil.disk_usage(Path.cwd())
    free_gb = disk.free / (1024**3)
    print(f"Disk    : {free_gb:.1f} GB free in {Path.cwd()}")

    cfg = VietH3Config.from_yaml(args.config)
    report = estimate_memory_and_budget(
        cfg,
        gpus=args.gpus,
        height=args.height,
        width=args.width,
        frames=args.frames,
        batch_size=args.batch_size,
    )
    print("\n" + format_report(report))
    if not report.fits_80gb_h100:
        raise RuntimeError(
            f"Config {cfg.name} estimated at {report.total_fsdp_ckpt_gb_per_gpu:.2f} GB/GPU (>76 GB limit)."
        )

    if args.dry_run:
        print("[DRY-RUN] Running tiny CPU DiT + I2V + SDPA forward/backward smoke test...")
        tiny = VietH3Config(
            name="preflight-tiny",
            in_channels=cfg.in_channels,
            patch_size=cfg.patch_size,
            hidden_size=128,
            num_layers=2,
            num_refiner_layers=1,
            num_attention_heads=4,
            attention_head_dim=32,
            ffn_dim=256,
            text_dim=64,
            max_text_len=8,
            use_i2v_cond=cfg.use_i2v_cond,
            use_pose_cond=cfg.use_pose_cond,
            pose_channels=cfg.pose_channels,
            shared_adaln=cfg.shared_adaln,
            num_register_tokens=cfg.num_register_tokens,
            use_long_skips=cfg.use_long_skips,
        )
        m = VietH3Transformer3DModel(tiny)
        x = torch.randn(1, tiny.in_channels, 2, 8, 6)
        txt = torch.randn(1, 8, tiny.text_dim)
        mask = torch.ones(1, 8, dtype=torch.bool)
        f0 = torch.randn(1, tiny.in_channels, 1, 8, 6) if tiny.use_i2v_cond else None
        i2v_m = torch.ones(1) if tiny.use_i2v_cond else None
        pose = torch.randn(1, tiny.pose_channels, 2, 8, 6) if tiny.use_pose_cond else None
        loss, _ = RectifiedFlowScheduler.training_loss(
            m, x, txt, text_mask=mask, frame0_latent=f0, i2v_mask=i2v_m, pose_latent=pose
        )
        loss.backward()
        print(f"[DRY-RUN] OK (loss={loss.item():.4f})")
        print("PRE-FLIGHT PASSED (CPU dry-run)!")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --dry-run on CPU machines.")

    gpu_count = torch.cuda.device_count()
    print(f"CUDA GPUs detected: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    if gpu_count < args.gpus:
        raise RuntimeError(f"Expected {args.gpus} GPUs, found {gpu_count}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this CUDA device.")

    print("[CUDA] Running 2-step BF16 forward/backward check on real configured model...")
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    model = VietH3Transformer3DModel(cfg).to(device=device, dtype=torch.bfloat16)
    model.enable_gradient_checkpointing(True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    t_lat = 1 if args.frames == 1 else (args.frames - 1) // cfg.vae_temporal_stride + 1
    h_lat = args.height // cfg.vae_spatial_stride
    w_lat = args.width // cfg.vae_spatial_stride
    x = torch.randn(args.batch_size, cfg.in_channels, t_lat, h_lat, w_lat, device=device, dtype=torch.bfloat16)
    txt = torch.randn(args.batch_size, cfg.max_text_len, cfg.text_dim, device=device, dtype=torch.bfloat16)
    mask = torch.ones(args.batch_size, cfg.max_text_len, device=device, dtype=torch.bool)
    f0 = x[:, :, :1].clone() if cfg.use_i2v_cond else None
    i2v_m = torch.ones(args.batch_size, device=device, dtype=torch.bfloat16) if cfg.use_i2v_cond else None
    pose = torch.randn(args.batch_size, cfg.pose_channels, t_lat, h_lat, w_lat, device=device, dtype=torch.bfloat16) if cfg.use_pose_cond else None

    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = RectifiedFlowScheduler.training_loss(
                model, x, txt, text_mask=mask, frame0_latent=f0, i2v_mask=i2v_m, pose_latent=pose
            )
        loss.backward()
        opt.step()
        print(f"  Step {step + 1}/2 loss={loss.item():.5f}")

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"[CUDA] Peak single-GPU allocated VRAM (no FSDP shard): {peak_gb:.2f} GB")

    if args.check_hf:
        from vieth3.models.vae_text import load_text_encoder, load_vae

        print("[HF] Testing Wan2.1-VAE & Qwen2.5-VL round-trip...")
        vae = load_vae(
            vae_type=cfg.vae_type,
            latent_channels=cfg.in_channels,
            spatial_stride=cfg.vae_spatial_stride,
            temporal_stride=cfg.vae_temporal_stride,
            device=device,
            dtype=torch.bfloat16,
        )
        pix = torch.zeros((1, 3, 5, 128, 128), device=device, dtype=torch.float32)
        lat = vae.encode(pix)
        rec = vae.decode(lat)
        assert rec.shape == pix.shape, f"VAE round-trip shape mismatch: {rec.shape}"

        enc = load_text_encoder(
            encoder_type="qwen25vl",
            text_dim=cfg.text_dim,
            max_length=cfg.max_text_len,
            device=device,
            dtype=torch.bfloat16,
        )
        emb, m = enc.encode(["Cô gái mặc áo dài nhảy TikTok trong phòng sáng tự nhiên."])
        assert emb.shape[-1] == cfg.text_dim and m.any()
        print("[HF] Wan2.1-VAE + Qwen2.5-VL round-trip OK!")

    print("PRE-FLIGHT PASSED!")


if __name__ == "__main__":
    main()
