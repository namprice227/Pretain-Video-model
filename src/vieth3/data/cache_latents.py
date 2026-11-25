"""Offline latent & Vietnamese text embedding caching into sharded .safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import Tensor

from vieth3.data.frame_grid import GridType, select_aspect_bucket, snap_frame_count
from vieth3.models.config import VietH3Config
from vieth3.models.vae_text import load_text_encoder, load_vae

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _center_crop_resize(frame_rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w, _ = frame_rgb.shape
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = max(0, (new_h - target_h) // 2)
    x0 = max(0, (new_w - target_w) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def load_media_tensor(
    path: str | Path,
    target_height: int,
    target_width: int,
    grid: GridType = "h3",
    max_frames: int = 243,
) -> Tensor:
    """Load an image or video as (1, 3, T, H, W) float32 tensor in [-1, 1]."""
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXTENSIONS:
        img = Image.open(p).convert("RGB")
        arr = _center_crop_resize(np.array(img), target_height, target_width)
        ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
        return ten.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open media: {p}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(_center_crop_resize(rgb, target_height, target_width))
    cap.release()
    if not frames:
        raise RuntimeError(f"Empty video: {p}")

    snapped = snap_frame_count(len(frames), grid=grid, allow_image=True, max_frames=max_frames)
    if len(frames) >= snapped:
        frames = frames[:snapped]
    else:
        while len(frames) < snapped:
            frames.append(frames[-1])

    arr = np.stack(frames, axis=0)  # (T, H, W, 3)
    ten = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    return ten.unsqueeze(0)  # (1, 3, T, H, W)


def _probe_hw(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(path) as img:
            w, h = img.size
            return h, w
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return max(32, h), max(32, w)


def cache_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: str | Path,
    vae_type: str = "dummy",
    text_encoder_type: str = "dummy",
    vae_id: Optional[str] = None,
    text_encoder_id: Optional[str] = None,
    bucket_stage: str = "480p",
    shard_size: int = 64,
    device: str = "cpu",
    dtype_str: str = "float32",
) -> Path:
    """Pre-encode media and Vietnamese captions into sharded .safetensors files."""
    cfg = VietH3Config.from_yaml(model_config)
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    grid: GridType = "wan21" if cfg.vae_type == "wan21" else "h3"

    vae = load_vae(
        vae_type=vae_type,
        latent_channels=cfg.in_channels,
        spatial_stride=cfg.vae_spatial_stride,
        temporal_stride=cfg.vae_temporal_stride,
        model_id=vae_id,
        device=device,
        dtype=dtype,
    )
    txt_enc = load_text_encoder(
        encoder_type=text_encoder_type,
        text_dim=cfg.text_dim,
        max_length=cfg.max_text_len,
        model_id=text_encoder_id,
        device=device,
        dtype=dtype,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    index_rows: list[dict] = []
    shard_tensors: dict[str, Tensor] = {}
    shard_idx = 0
    in_shard = 0

    def _flush_shard() -> None:
        nonlocal shard_tensors, shard_idx, in_shard
        if not shard_tensors:
            return
        shard_file = shards_dir / f"shard_{shard_idx:05d}.safetensors"
        save_file(shard_tensors, str(shard_file))
        shard_tensors = {}
        shard_idx += 1
        in_shard = 0

    for item_idx, entry in enumerate(entries):
        media_path = Path(entry["path"])
        caption = entry.get("caption")
        if not caption:
            txt_p = media_path.with_suffix(".txt")
            caption = txt_p.read_text(encoding="utf-8").strip() if txt_p.exists() else ""

        raw_h, raw_w = _probe_hw(media_path)
        stride_multiple = max(32, cfg.vae_spatial_stride * cfg.patch_size[1])
        tgt_h, tgt_w = select_aspect_bucket(
            raw_h, raw_w, stage=bucket_stage, multiple=stride_multiple
        )

        video_ten = load_media_tensor(media_path, tgt_h, tgt_w, grid=grid).to(
            device=device, dtype=dtype
        )
        pose_path_str = entry.get("pose_path")
        with torch.no_grad():
            latent = vae.encode(video_ten).squeeze(0).cpu()
            txt_emb, txt_mask = txt_enc.encode([caption])
            txt_emb = txt_emb.squeeze(0).cpu()
            txt_mask = txt_mask.squeeze(0).to(torch.uint8).cpu()
            pose_lat: Optional[Tensor] = None
            if pose_path_str and Path(pose_path_str).exists():
                pose_ten = load_media_tensor(pose_path_str, tgt_h, tgt_w, grid=grid).to(
                    device=device, dtype=dtype
                )
                pose_lat = vae.encode(pose_ten).squeeze(0).cpu()

        key = f"sample_{item_idx:07d}"
        shard_name = f"shards/shard_{shard_idx:05d}.safetensors"
        shard_tensors[f"{key}.latent"] = latent.contiguous()
        shard_tensors[f"{key}.frame0_latent"] = latent[:, :1].contiguous()
        shard_tensors[f"{key}.text_embeds"] = txt_emb.contiguous()
        shard_tensors[f"{key}.text_mask"] = txt_mask.contiguous()
        if pose_lat is not None:
            shard_tensors[f"{key}.pose_latent"] = pose_lat.contiguous()

        c_lat, t_lat, h_lat, w_lat = latent.shape
        index_rows.append(
            {
                "key": key,
                "shard": shard_name,
                "t_lat": int(t_lat),
                "h_lat": int(h_lat),
                "w_lat": int(w_lat),
                "channels": int(c_lat),
                "is_image": bool(t_lat == 1),
                "has_pose": bool(pose_lat is not None),
                "caption": caption,
                "source_path": str(media_path.resolve()),
            }
        )
        in_shard += 1
        if in_shard >= shard_size:
            _flush_shard()

    _flush_shard()
    index_path = out_path / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache VAE latents & Vietnamese text embeddings.")
    parser.add_argument("--manifest", required=True, help="Captioned JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Cache output directory.")
    parser.add_argument("--model-config", required=True, help="Model YAML config.")
    parser.add_argument("--vae-type", choices=["dummy", "h3", "wan21"], default="dummy")
    parser.add_argument("--text-encoder", choices=["dummy", "qwen", "qwen25vl", "qwen3vl"], default="dummy")
    parser.add_argument("--vae-id", default=None)
    parser.add_argument("--text-encoder-id", default=None)
    parser.add_argument("--bucket-stage", default="480p")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    idx_path = cache_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_config=args.model_config,
        vae_type=args.vae_type,
        text_encoder_type=args.text_encoder,
        vae_id=args.vae_id,
        text_encoder_id=args.text_encoder_id,
        bucket_stage=args.bucket_stage,
        shard_size=args.shard_size,
        device=args.device,
        dtype_str=args.dtype,
    )
    print(f"Cached latents index written to {idx_path}")


if __name__ == "__main__":
    main()
