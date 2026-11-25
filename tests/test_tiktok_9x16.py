"""Tests for 9:16 TikTok dance I2V background lock, Pose conditioning, and tripod filters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from vieth3.data.cache_latents import cache_dataset
from vieth3.data.filters import compute_clip_metrics
from vieth3.data.pose_extract import extract_pose_video
from vieth3.data.scene_split import find_ffmpeg, find_ffprobe
from vieth3.inference.sample import generate_video
from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel
from vieth3.models.flow_matching import RectifiedFlowScheduler
from vieth3.training.train import train


def _write_mp4(frames_rgb: np.ndarray, path: Path, fps: int = 24) -> None:
    ffmpeg = find_ffmpeg()
    _t, h, w, _ = frames_rgb.shape
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
        "16",
        str(path),
    ]
    subprocess.run(
        cmd, input=frames_rgb.tobytes(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _draw_static_room_bg(h: int, w: int) -> np.ndarray:
    img = np.full((h, w, 3), 45, dtype=np.uint8)
    # Subtle full-frame room texture (low enough not to trigger Canny overlay detector, rich enough for optical flow)
    for y in range(0, h, 24):
        cv2.line(img, (0, y), (w, y), (62, 66, 74), 2)
    for x in range(0, w, 24):
        cv2.line(img, (x, 0), (x, h), (58, 62, 70), 2)
    cv2.rectangle(img, (18, int(h * 0.2)), (58, int(h * 0.65)), (95, 82, 68), 2)
    cv2.rectangle(img, (w - 58, int(h * 0.2)), (w - 18, int(h * 0.65)), (95, 82, 68), 2)
    return img


def test_i2v_and_pose_conditioning_forward_and_sample() -> None:
    cfg = VietH3Config(
        name="tiktok-debug",
        in_channels=8,
        patch_size=(1, 2, 2),
        hidden_size=128,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=32,
        ffn_dim=256,
        text_dim=64,
        max_text_len=8,
        use_i2v_cond=True,
        use_pose_cond=True,
        pose_channels=8,
    )
    assert cfg.patch_in_channels == 8 + 9 + 8
    model = VietH3Transformer3DModel(cfg)
    torch.nn.init.normal_(model.final_proj.weight, std=0.02)

    x = torch.randn(2, 8, 2, 10, 6)
    t = torch.tensor([0.3, 0.7])
    txt = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, dtype=torch.bool)
    f0 = torch.randn(2, 8, 1, 10, 6, requires_grad=True)
    i2v_m = torch.tensor([1.0, 0.0])
    pose = torch.randn(2, 8, 2, 10, 6, requires_grad=True)

    loss, _ = RectifiedFlowScheduler.training_loss(
        model, x, txt, text_mask=mask, frame0_latent=f0, i2v_mask=i2v_m, pose_latent=pose
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert f0.grad is not None and f0.grad.abs().sum() > 0
    assert pose.grad is not None and pose.grad.abs().sum() > 0

    sampled = RectifiedFlowScheduler.sample(
        model,
        shape=(1, 8, 2, 10, 6),
        text_embeds=txt[:1],
        text_mask=mask[:1],
        neg_text_embeds=torch.zeros_like(txt[:1]),
        neg_text_mask=mask[:1],
        num_steps=3,
        cfg_scale=5.0,
        shift=7.0,
        frame0_latent=f0[:1].detach(),
        i2v_mask=torch.ones(1),
        pose_latent=pose[:1].detach(),
    )
    assert sampled.shape == (1, 8, 2, 10, 6)
    assert torch.isfinite(sampled).all()


def test_tiktok_9x16_tripod_and_flicker_filters(tmp_path: Path) -> None:
    h, w, n_frames = 512, 288, 22
    bg = _draw_static_room_bg(h, w)

    # 1. Good static-background TikTok dance clip
    good_frames = []
    for f in range(n_frames):
        frame = bg.copy()
        cx = int(w * 0.5 + 34 * np.sin(f * 0.45))
        cy = int(h * 0.48 + 20 * np.cos(f * 0.35))
        cv2.circle(frame, (cx, cy - 55), 22, (235, 195, 165), -1)
        cv2.rectangle(frame, (cx - 26, cy - 30), (cx + 26, cy + 55), (240, 240, 250), -1)
        arm_y = int(cy - 10 + 35 * np.sin(f * 0.6))
        cv2.line(frame, (cx - 26, cy - 15), (cx - 56, arm_y), (235, 195, 165), 7)
        cv2.line(frame, (cx + 26, cy - 15), (cx + 56, arm_y), (235, 195, 165), 7)
        good_frames.append(frame)
    good_path = tmp_path / "good_dance.mp4"
    _write_mp4(np.stack(good_frames), good_path)

    m_good = compute_clip_metrics(
        good_path,
        min_width=256,
        min_height=256,
        min_frames=22,
        min_sharpness=20.0,
        min_motion=0.1,
        max_overlay_score=0.25,
        require_portrait=True,
        max_bg_motion=0.25,
        min_fg_motion=0.6,
        max_flicker=12.0,
    )
    assert m_good.passed, f"Expected good clip to pass, got {m_good}"
    assert m_good.bg_motion_score <= 0.25
    assert m_good.fg_motion_score >= 0.6

    # 2. Shaky handheld camera clip (entire scene shifts horizontally)
    shaky_frames = []
    for f in range(n_frames):
        dx = 14 if f % 2 == 0 else -14
        M = np.float32([[1, 0, dx], [0, 1, 6 if f % 2 == 0 else -6]])
        shaky = cv2.warpAffine(good_frames[f], M, (w, h), borderMode=cv2.BORDER_REFLECT)
        shaky_frames.append(shaky)
    shaky_path = tmp_path / "shaky_cam.mp4"
    _write_mp4(np.stack(shaky_frames), shaky_path)

    m_shaky = compute_clip_metrics(
        shaky_path,
        min_width=256,
        min_height=256,
        min_frames=22,
        min_sharpness=10.0,
        min_motion=0.1,
        max_overlay_score=0.35,
        require_portrait=True,
        max_bg_motion=0.25,
    )
    assert not m_shaky.passed
    assert any(r.startswith("shaky_background") for r in m_shaky.reject_reasons)

    # 3. Flashing / glowy sparkle clip
    flicker_frames = []
    for f in range(n_frames):
        frame = good_frames[f].copy()
        if f % 2 == 1:
            frame = np.clip(frame.astype(np.int32) + 180, 0, 255).astype(np.uint8)
        flicker_frames.append(frame)
    flicker_path = tmp_path / "flicker.mp4"
    _write_mp4(np.stack(flicker_frames), flicker_path)

    m_flicker = compute_clip_metrics(
        flicker_path,
        min_width=256,
        min_height=256,
        min_frames=22,
        min_sharpness=10.0,
        min_motion=0.1,
        max_overlay_score=0.35,
        require_portrait=True,
        max_flicker=12.0,
    )
    assert not m_flicker.passed
    assert any(r.startswith("flicker_or_glow") for r in m_flicker.reject_reasons)


def test_i2v_pose_end_to_end_train_and_sample(tmp_path: Path) -> None:
    h, w, n_frames = 256, 160, 22
    bg = _draw_static_room_bg(h, w)
    frames = []
    for f in range(n_frames):
        frame = bg.copy()
        cx = int(w * 0.5 + 20 * np.sin(f * 0.4))
        cv2.circle(frame, (cx, int(h * 0.45)), 18, (240, 180, 90), -1)
        frames.append(frame)
    arr = np.stack(frames)
    clip_path = tmp_path / "dance_9x16.mp4"
    _write_mp4(arr, clip_path)

    ref_img = tmp_path / "frame0.png"
    Image.fromarray(arr[0]).save(ref_img)

    pose_path = tmp_path / "dance_9x16_pose.mp4"
    extract_pose_video(clip_path, pose_path, backend="dummy")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "path": str(clip_path),
                "pose_path": str(pose_path),
                "caption": "Cô gái nhảy TikTok trong phòng sáng tự nhiên, phông nền đứng yên.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = VietH3Config(
        name="tiktok-9x16-smoke",
        in_channels=8,
        patch_size=(1, 2, 2),
        hidden_size=128,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=32,
        ffn_dim=256,
        text_dim=64,
        max_text_len=16,
        vae_type="dummy",
        use_i2v_cond=True,
        use_pose_cond=True,
        pose_channels=8,
    )
    cfg_path = tmp_path / "model.yaml"
    cfg.to_yaml(cfg_path)

    cache_idx = cache_dataset(
        manifest_path=manifest_path,
        output_dir=tmp_path / "cache",
        model_config=cfg_path,
        vae_type="dummy",
        text_encoder_type="dummy",
        bucket_stage="native",
        device="cpu",
    )

    summary = train(
        {
            "model_config": str(cfg_path),
            "cache_index": str(cache_idx),
            "output_dir": str(tmp_path / "out"),
            "strategy": "single",
            "bf16": False,
            "batch_size": 1,
            "max_steps": 8,
            "warmup_steps": 2,
            "lr": 1e-3,
            "cfg_dropout": 0.0,
            "i2v_dropout": 0.0,
            "pose_dropout": 0.0,
            "use_ema": False,
            "log_every": 4,
            "save_every": 8,
            "device": "cpu",
        }
    )
    assert summary["final_step"] == 8

    out_mp4 = generate_video(
        model_config=cfg_path,
        checkpoint=summary["checkpoint_dir"],
        prompt="Cô gái mặc áo trắng nhảy TikTok trong phòng, nền cố định.",
        output=tmp_path / "out_9x16.mp4",
        height=256,
        width=160,
        frames=22,
        steps=3,
        cfg_scale=3.0,
        flow_shift=7.0,
        vae_type="dummy",
        text_encoder_type="dummy",
        device="cpu",
        ref_image=ref_img,
        pose_video=pose_path,
    )
    assert out_mp4.exists()

    ffprobe = find_ffprobe()
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(out_mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert int(stream["width"]) == 160 and int(stream["height"]) == 256
