"""Pose skeleton extraction for 9:16 TikTok dance conditioning."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from vieth3.data.scene_split import find_ffmpeg


def _render_dummy_pose_frame(bgr: np.ndarray) -> np.ndarray:
    """Zero-dependency edge/contour skeleton map on a clean black canvas."""
    h, w, _ = bgr.shape
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[edges > 0] = (0, 255, 180)
    return canvas


def extract_pose_video(
    video_path: str | Path,
    output_path: str | Path,
    backend: Literal["dummy", "mediapipe"] = "dummy",
) -> Path:
    """Extract an aligned RGB pose skeleton video at matching resolution & fps."""
    src = Path(video_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src}")
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 24.0))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_ctx = None
    mp_draw = None
    mp_pose = None
    if backend == "mediapipe":
        try:
            import mediapipe as mp  # type: ignore[import-untyped]
        except ImportError as exc:
            cap.release()
            raise ImportError(
                "mediapipe is required for backend='mediapipe'. Install with: pip install mediapipe"
            ) from exc
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose_ctx = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    frames_rgb: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if backend == "dummy":
                frames_rgb.append(_render_dummy_pose_frame(bgr))
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = pose_ctx.process(rgb)  # type: ignore[union-attr]
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                if res.pose_landmarks:
                    mp_draw.draw_landmarks(  # type: ignore[union-attr]
                        canvas, res.pose_landmarks, mp_pose.POSE_CONNECTIONS  # type: ignore[union-attr]
                    )
                frames_rgb.append(canvas)
    finally:
        cap.release()
        if pose_ctx is not None:
            pose_ctx.close()

    if not frames_rgb:
        raise RuntimeError(f"No frames read from {src}")

    arr = np.stack(frames_rgb, axis=0)
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
        "16",
        str(out),
    ]
    subprocess.run(cmd, input=arr.tobytes(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def extract_poses_for_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    output_manifest: str | Path,
    backend: Literal["dummy", "mediapipe"] = "dummy",
) -> list[dict]:
    """Run pose extraction for every clip in a manifest and add `pose_path`."""
    entries: list[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    pose_dir = Path(output_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)
    out_m = Path(output_manifest)
    out_m.parent.mkdir(parents=True, exist_ok=True)

    updated_rows: list[dict] = []
    with open(out_m, "w", encoding="utf-8") as out_f:
        for entry in entries:
            clip = Path(entry["path"])
            pose_out = pose_dir / f"{clip.stem}_pose.mp4"
            extract_pose_video(clip, pose_out, backend=backend)
            row = {**entry, "pose_path": str(pose_out.resolve())}
            updated_rows.append(row)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return updated_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract aligned pose skeleton videos.")
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory to store pose MP4s.")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL with pose_path.")
    parser.add_argument("--backend", choices=["dummy", "mediapipe"], default="dummy")
    args = parser.parse_args()

    rows = extract_poses_for_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        backend=args.backend,
    )
    print(f"Extracted {len(rows)} pose videos -> {args.output_manifest}")


if __name__ == "__main__":
    main()
