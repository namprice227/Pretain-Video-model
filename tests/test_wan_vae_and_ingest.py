"""Tests for Wan2.1 per-channel latent normalization, TikTok ingestion, and H100 preflight."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from vieth3.data.filters import ClipMetrics
from vieth3.data.ingest_tiktok import audit_filter_results, ingest_local_videos
from vieth3.data.scene_split import find_ffmpeg
from vieth3.models.vae_text import denormalize_latents, normalize_latents


def _make_clip(path: Path, color: tuple[int, int, int], width: int = 160, height: int = 288, frames: int = 12, crf: int = 18) -> None:
    ffmpeg = find_ffmpeg()
    arr = np.zeros((frames, height, width, 3), dtype=np.uint8)
    for f in range(frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        cx = int(width * 0.5 + 24 * np.sin(f * 0.5))
        cv2.circle(frame, (cx, height // 2), 20, (250, 220, 120), -1)
        arr[f] = frame
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "24", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path),
    ]
    subprocess.run(cmd, input=arr.tobytes(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_wan21_per_channel_normalization_roundtrip() -> None:
    torch.manual_seed(0)
    z = torch.randn(2, 16, 4, 8, 6)
    mean = torch.linspace(-0.5, 0.5, 16).view(1, 16, 1, 1, 1)
    std = torch.linspace(0.8, 2.2, 16).view(1, 16, 1, 1, 1)

    z_norm = normalize_latents(z, mean, std)
    z_rec = denormalize_latents(z_norm, mean, std)
    assert torch.allclose(z, z_rec, atol=1e-5)

    # Verify channel-0 shift & scale explicitly
    expected_c0 = (z[:, 0] - mean[0, 0]) / std[0, 0]
    assert torch.allclose(z_norm[:, 0], expected_c0, atol=1e-6)


def test_ingest_deduplicates_exact_and_reencoded_clips(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "ingested"

    clip_a = raw_dir / "a.mp4"
    clip_a_copy = raw_dir / "a_exact_copy.mp4"
    clip_a_reenc = raw_dir / "a_reencoded.mp4"
    clip_b = raw_dir / "b_distinct.mp4"

    _make_clip(clip_a, color=(30, 50, 90), crf=16)
    shutil.copy2(clip_a, clip_a_copy)
    # Re-encode identical visual content at different CRF -> different SHA-256, same dHash
    _make_clip(clip_a_reenc, color=(30, 50, 90), crf=28)
    # Distinct video with different scene layout
    _make_clip(clip_b, color=(120, 40, 60), width=192, height=288, crf=16)
    # Add a contrasting bar on clip_b so dHash differs clearly
    cap_frames = []
    cap = cv2.VideoCapture(str(clip_b))
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cv2.rectangle(rgb, (0, 0), (75, 288), (240, 240, 240), -1)
        cv2.line(rgb, (0, 0), (192, 288), (20, 220, 40), 12)
        cap_frames.append(rgb)
    cap.release()
    ffmpeg = find_ffmpeg()
    arr_b = np.stack(cap_frames)
    subprocess.run(
        [
            ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "192x288", "-r", "24", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(clip_b),
        ],
        input=arr_b.tobytes(),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    manifest = ingest_local_videos(raw_dir, out_dir, copy_files=False)
    report = json.loads((out_dir / "ingest_report.json").read_text(encoding="utf-8"))

    assert report["total_scanned"] == 4
    assert report["unique_kept"] == 2
    assert report["duplicates_skipped"] == 2
    assert report["portrait_count"] == 2
    assert len(manifest) == 2


def test_audit_filter_results_aggregation() -> None:
    sample = [
        ClipMetrics("a.mp4", 720, 1280, 49, 24.0, 120.0, 2.0, 0.05, True, []),
        ClipMetrics("b.mp4", 720, 1280, 49, 24.0, 100.0, 5.0, 0.05, False, ["shaky_background:1.80"]),
        ClipMetrics("c.mp4", 720, 1280, 49, 24.0, 100.0, 5.0, 0.05, False, ["flicker_or_glow:22.00", "shaky_background:1.20"]),
    ]
    summary = audit_filter_results(sample)
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["rejected"] == 2
    assert summary["reject_reason_counts"]["shaky_background"] == 2
    assert summary["reject_reason_counts"]["flicker_or_glow"] == 1


def test_h100_preflight_dry_run_script() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "h100_preflight.py"),
            "--dry-run",
            "--config",
            str(repo / "configs" / "model_vieth3_tiktok_9x16_mini.yaml"),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PRE-FLIGHT PASSED (CPU dry-run)!" in proc.stdout
