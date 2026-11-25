"""Inference CLI to generate Vietnamese text-to-video MP4s from VietH3 checkpoints."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from vieth3.data.cache_latents import load_media_tensor
from vieth3.data.frame_grid import GridType, snap_frame_count, snap_resolution
from vieth3.data.scene_split import find_ffmpeg
from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel
from vieth3.models.flow_matching import RectifiedFlowScheduler
from vieth3.models.vae_text import load_text_encoder, load_vae


def write_video_mp4(frames_rgb: np.ndarray, output_path: str | Path, fps: int = 24) -> Path:
    """Write uint8 (T, H, W, 3) RGB frames to an H.264 MP4 file using ffmpeg."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames_rgb.shape
    if t == 1:
        Image.fromarray(frames_rgb[0]).save(out)
        return out

    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out),
    ]
    proc = subprocess.run(
        cmd,
        input=frames_rgb.tobytes(),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _ = proc
    return out


@torch.no_grad()
def generate_video(
    model_config: str | Path,
    checkpoint: str | Path,
    prompt: str,
    output: str | Path,
    neg_prompt: str = "",
    height: int = 256,
    width: int = 448,
    frames: int = 22,
    steps: int = 20,
    cfg_scale: float = 5.0,
    flow_shift: float = 3.0,
    solver: Literal["euler", "heun"] = "euler",
    vae_type: str = "dummy",
    text_encoder_type: str = "dummy",
    vae_id: Optional[str] = None,
    text_encoder_id: Optional[str] = None,
    use_ema: bool = False,
    seed: int = 42,
    device: str = "cpu",
    ref_image: Optional[str | Path] = None,
    pose_video: Optional[str | Path] = None,
) -> Path:
    """Generate a video from a native Vietnamese prompt and save it as MP4."""
    cfg = VietH3Config.from_yaml(model_config)
    dev = torch.device(device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    model = VietH3Transformer3DModel(cfg)
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        candidate = ckpt_path / ("ema.safetensors" if use_ema else "model.safetensors")
        if not candidate.exists():
            candidate = ckpt_path / "model.safetensors"
        ckpt_path = candidate

    state = load_file(str(ckpt_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device=dev, dtype=dtype)
    model.eval()

    grid: GridType = "wan21" if cfg.vae_type == "wan21" else "h3"
    snapped_frames = snap_frame_count(frames, grid=grid, allow_image=True)
    stride_multiple = max(32, cfg.vae_spatial_stride * cfg.patch_size[1])
    snapped_h, snapped_w = snap_resolution(height, width, multiple=stride_multiple)

    t_lat = 1 if snapped_frames == 1 else (snapped_frames - 1) // cfg.vae_temporal_stride + 1
    h_lat = snapped_h // cfg.vae_spatial_stride
    w_lat = snapped_w // cfg.vae_spatial_stride

    txt_enc = load_text_encoder(
        encoder_type=text_encoder_type,
        text_dim=cfg.text_dim,
        max_length=cfg.max_text_len,
        model_id=text_encoder_id,
        device=dev,
        dtype=dtype,
    )
    txt_emb, txt_mask = txt_enc.encode([prompt])
    neg_emb, neg_mask = txt_enc.encode([neg_prompt])

    vae = load_vae(
        vae_type=vae_type,
        latent_channels=cfg.in_channels,
        spatial_stride=cfg.vae_spatial_stride,
        temporal_stride=cfg.vae_temporal_stride,
        model_id=vae_id,
        device=dev,
        dtype=dtype,
    )

    frame0_lat: Optional[torch.Tensor] = None
    i2v_mask: Optional[torch.Tensor] = None
    if cfg.use_i2v_cond and ref_image is not None:
        img_ten = load_media_tensor(ref_image, snapped_h, snapped_w, grid=grid).to(
            device=dev, dtype=dtype
        )
        frame0_lat = vae.encode(img_ten)[:, :, :1]
        i2v_mask = torch.ones((1, 1, 1, 1, 1), device=dev, dtype=dtype)

    pose_lat: Optional[torch.Tensor] = None
    if cfg.use_pose_cond and pose_video is not None:
        pose_ten = load_media_tensor(pose_video, snapped_h, snapped_w, grid=grid).to(
            device=dev, dtype=dtype
        )
        pose_lat = vae.encode(pose_ten)

    gen = torch.Generator(device=dev).manual_seed(seed)
    latents = RectifiedFlowScheduler.sample(
        model=model,
        shape=(1, cfg.in_channels, t_lat, h_lat, w_lat),
        text_embeds=txt_emb,
        text_mask=txt_mask,
        neg_text_embeds=neg_emb,
        neg_text_mask=neg_mask,
        num_steps=steps,
        cfg_scale=cfg_scale,
        shift=flow_shift,
        solver=solver,
        generator=gen,
        device=dev,
        dtype=dtype,
        frame0_latent=frame0_lat,
        i2v_mask=i2v_mask,
        pose_latent=pose_lat,
    )
    video = vae.decode(latents)  # (1, 3, T, H, W) in [-1, 1]
    video_u8 = (
        ((video.squeeze(0).float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return write_video_mp4(video_u8, output, fps=24)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Vietnamese T2V video with VietH3.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--neg-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--vae-type", choices=["dummy", "h3", "wan21"], default="dummy")
    parser.add_argument("--text-encoder", choices=["dummy", "qwen", "qwen25vl", "qwen3vl"], default="dummy")
    parser.add_argument("--vae-id", default=None)
    parser.add_argument("--text-encoder-id", default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ref-image", default=None, help="Optional Frame-0 anchor image for I2V.")
    parser.add_argument("--pose-video", default=None, help="Optional aligned pose skeleton MP4.")
    args = parser.parse_args()

    out = generate_video(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        neg_prompt=args.neg_prompt,
        output=args.output,
        height=args.height,
        width=args.width,
        frames=args.frames,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        solver=args.solver,
        vae_type=args.vae_type,
        text_encoder_type=args.text_encoder,
        vae_id=args.vae_id,
        text_encoder_id=args.text_encoder_id,
        use_ema=args.use_ema,
        seed=args.seed,
        device=args.device,
        ref_image=args.ref_image,
        pose_video=args.pose_video,
    )
    print(f"Generated video saved to {out}")


if __name__ == "__main__":
    main()
