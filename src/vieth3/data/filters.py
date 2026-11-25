"""Quality filtering pipeline for Vietnamese training video clips."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class ClipMetrics:
    path: str
    width: int
    height: int
    num_frames: int
    fps: float
    laplacian_var: float
    motion_score: float
    overlay_score: float
    passed: bool
    reject_reasons: list[str]
    bg_motion_score: float = 0.0
    fg_motion_score: float = 0.0
    flicker_score: float = 0.0


def _estimate_overlay_clutter(gray: np.ndarray) -> float:
    """Heuristic detecting dense high-contrast text/watermarks in subtitle & corner bands."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 100, 220) > 0

    # Subtitle band (bottom 20%) and top corner watermark bands (top 12%, left/right 25%)
    bottom_band = edges[int(h * 0.80) :, :]
    top_left = edges[: int(h * 0.12), : int(w * 0.25)]
    top_right = edges[: int(h * 0.12), int(w * 0.75) :]

    scores = [
        float(bottom_band.mean()) if bottom_band.size else 0.0,
        float(top_left.mean()) if top_left.size else 0.0,
        float(top_right.mean()) if top_right.size else 0.0,
    ]
    return max(scores)


def compute_clip_metrics(
    video_path: str | Path,
    min_width: int = 256,
    min_height: int = 256,
    min_frames: int = 22,
    min_sharpness: float = 40.0,
    min_motion: float = 0.25,
    max_motion: float = 25.0,
    max_overlay_score: float = 0.22,
    require_portrait: bool = False,
    max_bg_motion: Optional[float] = None,
    min_fg_motion: Optional[float] = None,
    max_flicker: Optional[float] = None,
    ocr_hook: Optional[Callable[[list[np.ndarray]], float]] = None,
) -> ClipMetrics:
    """Evaluate sharpness, optical-flow motion, and watermark/hardsub clutter."""
    path_str = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return ClipMetrics(
            path=path_str,
            width=0,
            height=0,
            num_frames=0,
            fps=0.0,
            laplacian_var=0.0,
            motion_score=0.0,
            overlay_score=1.0,
            passed=False,
            reject_reasons=["unreadable_video"],
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)

    sample_count = min(8, max(2, num_frames))
    indices = np.linspace(0, max(0, num_frames - 1), sample_count, dtype=int)
    frames_gray: list[np.ndarray] = []
    frames_rgb: list[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames_rgb.append(frame)
        frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(frames_gray) < 2:
        return ClipMetrics(
            path=path_str,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            laplacian_var=0.0,
            motion_score=0.0,
            overlay_score=0.0,
            passed=False,
            reject_reasons=["insufficient_decoded_frames"],
        )

    sharpness_vals = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in frames_gray]
    laplacian_var = float(np.mean(sharpness_vals))

    # Downsample for fast optical flow + foreground/background split + flicker check
    flow_mags: list[float] = []
    bg_mags: list[float] = []
    fg_mags: list[float] = []
    flicker_vals: list[float] = []
    for i in range(len(frames_gray) - 1):
        g0 = cv2.resize(frames_gray[i], (160, 160))
        g1 = cv2.resize(frames_gray[i + 1], (160, 160))
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None, pyr_scale=0.5, levels=2, winsize=11, iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_mags.append(float(mag.mean()))

        border = np.concatenate(
            [mag[:16, :].ravel(), mag[-16:, :].ravel(), mag[:, :18].ravel(), mag[:, -18:].ravel()]
        )
        bg_mags.append(max(float(np.median(border)), float(np.percentile(mag, 30))))
        fg_box = mag[24:144, 24:136]
        fg_mags.append(float(np.percentile(fg_box, 85)))

        luma_jump = abs(float(g1.mean()) - float(g0.mean()))
        sparkle_jump = abs(float((g1 > 245).mean()) - float((g0 > 245).mean())) * 100.0
        flicker_vals.append(luma_jump + sparkle_jump)

    motion_score = float(np.mean(flow_mags))
    bg_motion_score = float(np.mean(bg_mags))
    fg_motion_score = float(np.mean(fg_mags))
    flicker_score = float(np.max(flicker_vals)) if flicker_vals else 0.0

    overlay_score = float(np.mean([_estimate_overlay_clutter(g) for g in frames_gray]))
    if ocr_hook is not None:
        overlay_score = max(overlay_score, float(ocr_hook(frames_rgb)))

    reasons: list[str] = []
    if width < min_width or height < min_height:
        reasons.append(f"low_resolution:{width}x{height}")
    if require_portrait and height <= width:
        reasons.append(f"not_portrait:{width}x{height}")
    if num_frames < min_frames:
        reasons.append(f"too_short:{num_frames}")
    if laplacian_var < min_sharpness:
        reasons.append(f"blurry:{laplacian_var:.1f}")
    if motion_score < min_motion:
        reasons.append(f"too_static:{motion_score:.2f}")
    if motion_score > max_motion:
        reasons.append(f"excessive_motion:{motion_score:.2f}")
    if overlay_score > max_overlay_score:
        reasons.append(f"watermark_or_hardsub:{overlay_score:.2f}")
    if max_bg_motion is not None and bg_motion_score > max_bg_motion:
        reasons.append(f"shaky_background:{bg_motion_score:.2f}")
    if min_fg_motion is not None and fg_motion_score < min_fg_motion:
        reasons.append(f"static_subject:{fg_motion_score:.2f}")
    if max_flicker is not None and flicker_score > max_flicker:
        reasons.append(f"flicker_or_glow:{flicker_score:.2f}")

    return ClipMetrics(
        path=path_str,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        laplacian_var=laplacian_var,
        motion_score=motion_score,
        overlay_score=overlay_score,
        passed=len(reasons) == 0,
        reject_reasons=reasons,
        bg_motion_score=bg_motion_score,
        fg_motion_score=fg_motion_score,
        flicker_score=flicker_score,
    )


def filter_clips(
    input_source: str | Path,
    output_manifest: str | Path,
    min_width: int = 256,
    min_height: int = 256,
    min_frames: int = 22,
    min_sharpness: float = 40.0,
    min_motion: float = 0.25,
    max_motion: float = 25.0,
    max_overlay_score: float = 0.22,
    require_portrait: bool = False,
    max_bg_motion: Optional[float] = None,
    min_fg_motion: Optional[float] = None,
    max_flicker: Optional[float] = None,
) -> list[ClipMetrics]:
    """Filter clips from a JSONL manifest or directory and write passed entries."""
    src = Path(input_source)
    entries: list[dict] = []
    if src.is_file() and src.suffix == ".jsonl":
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    elif src.is_dir():
        for p in sorted(src.rglob("*.mp4")):
            entries.append({"path": str(p.resolve())})
    else:
        raise FileNotFoundError(f"Input source not found: {src}")

    results: list[ClipMetrics] = []
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for entry in entries:
            metrics = compute_clip_metrics(
                entry["path"],
                min_width=min_width,
                min_height=min_height,
                min_frames=min_frames,
                min_sharpness=min_sharpness,
                min_motion=min_motion,
                max_motion=max_motion,
                max_overlay_score=max_overlay_score,
                require_portrait=require_portrait,
                max_bg_motion=max_bg_motion,
                min_fg_motion=min_fg_motion,
                max_flicker=max_flicker,
            )
            results.append(metrics)
            if metrics.passed:
                merged = {**entry, **asdict(metrics)}
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Vietnamese video clips by quality.")
    parser.add_argument("--input", required=True, help="manifest_clips.jsonl or clip directory.")
    parser.add_argument("--output", required=True, help="Output filtered JSONL path.")
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument("--min-frames", type=int, default=22)
    parser.add_argument("--min-sharpness", type=float, default=40.0)
    parser.add_argument("--min-motion", type=float, default=0.25)
    parser.add_argument("--max-motion", type=float, default=25.0)
    parser.add_argument("--max-overlay-score", type=float, default=0.22)
    parser.add_argument("--require-portrait", action="store_true")
    parser.add_argument("--max-bg-motion", type=float, default=None)
    parser.add_argument("--min-fg-motion", type=float, default=None)
    parser.add_argument("--max-flicker", type=float, default=None)
    args = parser.parse_args()

    res = filter_clips(
        input_source=args.input,
        output_manifest=args.output,
        min_width=args.min_width,
        min_height=args.min_height,
        min_frames=args.min_frames,
        min_sharpness=args.min_sharpness,
        min_motion=args.min_motion,
        max_motion=args.max_motion,
        max_overlay_score=args.max_overlay_score,
        require_portrait=args.require_portrait,
        max_bg_motion=args.max_bg_motion,
        min_fg_motion=args.min_fg_motion,
        max_flicker=args.max_flicker,
    )
    passed = sum(1 for m in res if m.passed)
    print(f"Filtered {len(res)} clips: {passed} passed, {len(res) - passed} rejected.")


if __name__ == "__main__":
    main()
