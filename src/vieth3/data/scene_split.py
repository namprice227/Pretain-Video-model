"""Scene-cut video splitter that exports 24 fps clips snapped to the H3 frame grid."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import cv2

from vieth3.data.frame_grid import GridType, snap_frame_count

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _find_binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        winget_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        if winget_root.exists():
            matches = list(winget_root.glob(f"**/{name}.exe"))
            if matches:
                return str(matches[0])
    raise FileNotFoundError(
        f"'{name}' executable not found on PATH. Please install FFmpeg and ensure it is on PATH."
    )


def find_ffmpeg() -> str:
    return _find_binary("ffmpeg")


def find_ffprobe() -> str:
    return _find_binary("ffprobe")


def detect_scenes(
    video_path: str | Path,
    threshold: float = 27.0,
    min_scene_len: int = 22,
) -> list[tuple[int, int]]:
    """Detect shot boundaries returning [(start_frame, end_frame), ...]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        return []

    try:
        from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore[import-untyped]

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        manager.detect_scenes(video, show_progress=False)
        scene_list = manager.get_scene_list()
        if not scene_list:
            return [(0, total_frames)]
        return [(s[0].get_frames(), s[1].get_frames()) for s in scene_list]
    except Exception:
        return [(0, total_frames)]


def split_videos_to_clips(
    input_dir: str | Path,
    output_dir: str | Path,
    fps: int = 24,
    grid: GridType = "h3",
    target_frames: Iterable[int] = (56, 73, 90, 124),
    threshold: float = 27.0,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> list[dict]:
    """Split raw videos into clean 24 fps clips snapped to valid H3 frame counts."""
    ffmpeg = find_ffmpeg()
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    clips_dir = out_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    sorted_targets = sorted(int(x) for x in target_frames)
    min_target = sorted_targets[0]

    videos = sorted(
        p for p in in_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    manifest: list[dict] = []

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
        cap.release()

        scenes = detect_scenes(vid, threshold=threshold, min_scene_len=min_target)
        for s_idx, (start_f, end_f) in enumerate(scenes):
            dur_sec = (end_f - start_f) / max(1.0, src_fps)
            avail_frames_at_fps = int(dur_sec * fps)
            if avail_frames_at_fps < 5:
                continue

            # Pick largest allowed target frame count that fits in scene, else snap
            fitting = [t for t in sorted_targets if t <= avail_frames_at_fps]
            n_frames = (
                fitting[-1]
                if fitting
                else snap_frame_count(avail_frames_at_fps, grid=grid, allow_image=False)
            )
            if n_frames > avail_frames_at_fps:
                continue

            start_sec = start_f / max(1.0, src_fps)
            clip_name = f"{vid.stem}_s{s_idx:03d}_{n_frames}f.mp4"
            clip_out = clips_dir / clip_name

            vf_parts = [f"fps={fps}"]
            if target_height is not None and target_width is not None:
                vf_parts.append(
                    f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
                    f"crop={target_width}:{target_height}"
                )

            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{start_sec:.4f}",
                "-i",
                str(vid),
                "-vf",
                ",".join(vf_parts),
                "-frames:v",
                str(n_frames),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(clip_out),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            manifest.append(
                {
                    "path": str(clip_out.resolve()),
                    "source": str(vid.resolve()),
                    "scene_index": s_idx,
                    "num_frames": n_frames,
                    "fps": fps,
                    "grid": grid,
                }
            )

    manifest_path = out_path / "manifest_clips.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split raw Vietnamese videos into H3-grid clips.")
    parser.add_argument("--input-dir", required=True, help="Directory of raw videos.")
    parser.add_argument("--output-dir", required=True, help="Output directory for clips & manifest.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--grid", choices=["h3", "wan21"], default="h3")
    parser.add_argument("--target-frames", type=int, nargs="+", default=[56, 73, 90, 124])
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    args = parser.parse_args()

    clips = split_videos_to_clips(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        grid=args.grid,
        target_frames=args.target_frames,
        threshold=args.threshold,
        target_height=args.height,
        target_width=args.width,
    )
    print(f"Exported {len(clips)} clips to {args.output_dir}")


if __name__ == "__main__":
    main()
