"""Tests for the tiktok-dance-dataset-crawler workspace skill and helper script."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from vieth3.data.scene_split import find_ffmpeg


def _write_mp4(frames_rgb: np.ndarray, path: Path, fps: int = 24) -> None:
    ffmpeg = find_ffmpeg()
    _t, h, w, _ = frames_rgb.shape
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path),
    ]
    subprocess.run(cmd, input=frames_rgb.tobytes(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _draw_bg(h: int, w: int) -> np.ndarray:
    img = np.full((h, w, 3), 45, dtype=np.uint8)
    for y in range(0, h, 24):
        cv2.line(img, (0, y), (w, y), (62, 66, 74), 2)
    for x in range(0, w, 24):
        cv2.line(img, (x, 0), (x, h), (58, 62, 70), 2)
    return img


def test_skill_frontmatter_and_references() -> None:
    repo = Path(__file__).resolve().parents[1]
    skill_dir = repo / ".agents" / "skills" / "tiktok-dance-dataset-crawler"
    skill_md = skill_dir / "SKILL.md"
    req_md = skill_dir / "references" / "REQUIREMENTS.md"
    script_py = skill_dir / "scripts" / "crawl_and_filter_tiktok.py"

    assert skill_md.exists()
    assert req_md.exists()
    assert script_py.exists()

    text = skill_md.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) >= 3
    meta = yaml.safe_load(parts[1])
    assert meta["name"] == "tiktok-dance-dataset-crawler"
    assert "9:16" in meta["description"]
    assert "./references/REQUIREMENTS.md" in text
    assert "./scripts/crawl_and_filter_tiktok.py" in text
    assert "send_message" in text
    assert "CAPTCHA" in text
    assert "Anti-AI" in text
    req_text = req_md.read_text(encoding="utf-8")
    assert "Anti-AI / Real-Human Creator Verification Gate" in req_text
    assert "Real-Creator Snowball" in req_text


def test_crawl_and_filter_script_end_to_end(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script_py = repo / ".agents" / "skills" / "tiktok-dance-dataset-crawler" / "scripts" / "crawl_and_filter_tiktok.py"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"

    h, w, n_frames = 512, 288, 25
    bg = _draw_bg(h, w)

    # 1. Good static-background solo dance clip
    good_frames = []
    for f in range(n_frames):
        frame = bg.copy()
        cx = int(w * 0.5 + 34 * np.sin(f * 0.45))
        cy = int(h * 0.48 + 20 * np.cos(f * 0.35))
        cv2.circle(frame, (cx, cy - 55), 22, (235, 195, 165), -1)
        cv2.rectangle(frame, (cx - 26, cy - 30), (cx + 26, cy + 55), (240, 240, 250), -1)
        good_frames.append(frame)
    good_arr = np.stack(good_frames)
    good_path = raw_dir / "good_dance.mp4"
    _write_mp4(good_arr, good_path)
    shutil.copy2(good_path, raw_dir / "good_dance_dup.mp4")

    # 2. Shaky handheld clip
    shaky_frames = []
    for f in range(n_frames):
        dx = 14 if f % 2 == 0 else -14
        M = np.float32([[1, 0, dx], [0, 1, 6 if f % 2 == 0 else -6]])
        shaky_frames.append(cv2.warpAffine(good_frames[f], M, (w, h), borderMode=cv2.BORDER_REFLECT))
    _write_mp4(np.stack(shaky_frames), raw_dir / "shaky_cam.mp4")

    # 3. Flashing / glowy clip
    flicker_frames = []
    for f in range(n_frames):
        fr = good_frames[f].copy()
        if f % 2 == 1:
            fr = np.clip(fr.astype(np.int32) + 180, 0, 255).astype(np.uint8)
        flicker_frames.append(fr)
    _write_mp4(np.stack(flicker_frames), raw_dir / "flicker_glow.mp4")

    proc = subprocess.run(
        [
            sys.executable,
            str(script_py),
            "--local-dir", str(raw_dir),
            "--output-dir", str(out_dir),
            "--target-frames", "21",
            "--min-width", "256",
            "--min-height", "480",
            "--min-sharpness", "15",
            "--min-motion", "0.1",
            "--min-fg-motion", "0.6",
            "--max-overlay-score", "0.35",
            "--max-bg-motion", "0.25",
            "--max-flicker", "12.0",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Filter results: 1/3 clips passed" in proc.stdout

    summary = json.loads((out_dir / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["ingest"]["total_scanned"] == 4
    assert summary["ingest"]["unique_kept"] == 3
    assert summary["ingest"]["duplicates_skipped"] == 1
    assert summary["filter_audit"]["passed"] == 1
    assert summary["filter_audit"]["rejected"] == 2
    assert summary["filter_audit"]["reject_reason_counts"]["shaky_background"] == 1
    assert summary["filter_audit"]["reject_reason_counts"]["flicker_or_glow"] == 1
    assert len(summary["contact_sheets"]) == 1
    sheet_path = Path(summary["contact_sheets"][0])
    assert sheet_path.exists() and sheet_path.stat().st_size > 1000
