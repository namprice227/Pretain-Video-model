"""Crawl, deduplicate, split, filter, and render visual QA sheets for 9:16 TikTok dance clips."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vieth3.data.filters import filter_clips
from vieth3.data.ingest_tiktok import audit_filter_results, ingest_local_videos
from vieth3.data.scene_split import split_videos_to_clips


def _find_ytdlp() -> str:
    sibling = Path(sys.executable).parent / "yt-dlp"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("yt-dlp")
    if found:
        return found
    raise FileNotFoundError(
        "yt-dlp executable not found. Install with: pip install yt-dlp"
    )


def _download_urls_polite(
    urls_file: Path,
    download_dir: Path,
    max_per_source: int = 25,
    sleep_min: float = 2.0,
    sleep_max: float = 5.0,
) -> list[Path]:
    ytdlp = _find_ytdlp()
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_file = download_dir / "downloaded_archive.txt"
    failed_file = download_dir / "failed_urls.txt"
    raw_urls = [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    urls = list(dict.fromkeys(raw_urls))
    if not urls:
        raise ValueError(f"No URLs found in {urls_file}")

    failed_entries: list[str] = []
    for url in urls:
        cmd = [
            ytdlp,
            "--no-warnings",
            "--download-archive",
            str(archive_file),
            "--playlist-end",
            str(max_per_source),
            "--sleep-interval",
            str(sleep_min),
            "--max-sleep-interval",
            str(sleep_max),
            "--match-filter",
            "duration >= 4 & duration <= 60",
            "-f",
            "bv*[ext=mp4]/bv*",
            "--merge-output-format",
            "mp4",
            "-o",
            str(download_dir / "%(uploader_id|unknown)s_%(id)s.%(ext)s"),
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            err = msg.lower()
            if any(k in err for k in ("captcha", "login", "sign in", "403", "429", "blocked")):
                if failed_entries:
                    failed_file.write_text("\n".join(failed_entries) + "\n", encoding="utf-8")
                raise RuntimeError(
                    f"Download blocked/gated for {url} (CAPTCHA/login/rate-limit). "
                    "Do not bypass; complete verification in a browser or skip this source.\n"
                    f"yt-dlp output: {msg}"
                )
            failed_entries.append(f"{url}\t{msg.splitlines()[-1] if msg else 'unknown_error'}")

    if failed_entries:
        failed_file.write_text("\n".join(failed_entries) + "\n", encoding="utf-8")

    return sorted(download_dir.glob("*.mp4"))


def render_contact_sheet(video_path: Path, output_jpg: Path, num_frames: int = 6) -> Path:
    """Render a labeled 2x3 contact sheet for fast visual QA of a passed clip."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip for contact sheet: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)

    tiles: list[np.ndarray] = []
    tile_w, tile_h = 216, 384
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            bgr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            bgr = cv2.resize(bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        t_sec = idx / max(1.0, fps)
        cv2.rectangle(bgr, (4, 4), (94, 28), (0, 0, 0), -1)
        cv2.putText(
            bgr,
            f"f{idx} {t_sec:.2f}s",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(bgr)
    cap.release()

    row1 = np.concatenate(tiles[:3], axis=1)
    row2 = np.concatenate(tiles[3:6], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_jpg), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return output_jpg


def run_pipeline(
    output_dir: Path,
    local_dir: Optional[Path] = None,
    urls_file: Optional[Path] = None,
    max_per_source: int = 25,
    target_frames: tuple[int, ...] = (49, 65, 81, 121),
    min_width: int = 540,
    min_height: int = 960,
    min_frames: int = 21,
    min_sharpness: float = 40.0,
    min_motion: float = 0.35,
    max_motion: float = 18.0,
    max_overlay_score: float = 0.18,
    max_bg_motion: float = 0.25,
    min_fg_motion: float = 1.0,
    max_flicker: float = 12.0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    src_dir = local_dir
    if urls_file is not None:
        dl_dir = output_dir / "raw_downloads"
        _download_urls_polite(urls_file, dl_dir, max_per_source=max_per_source)
        if src_dir is None:
            src_dir = dl_dir

    if src_dir is None:
        raise ValueError("Provide --local-dir and/or --urls-file")

    # 1. Deduplicate raw sources into ingested/raw so duplicates are never scene-split twice
    ingested_dir = output_dir / "ingested"
    ingest_local_videos(src_dir, ingested_dir, copy_files=True)
    ingest_report = json.loads((ingested_dir / "ingest_report.json").read_text(encoding="utf-8"))

    # 2. Split deduplicated clips onto 24 fps Wan2.1 4n+1 frame grid
    clips_dir = output_dir / "clips"
    split_videos_to_clips(
        input_dir=ingested_dir / "raw",
        output_dir=clips_dir,
        fps=24,
        grid="wan21",
        target_frames=target_frames,
    )

    # 3. Filter strictly for portrait 9:16, static tripod background, active dance, zero flicker/glow
    filtered_manifest = output_dir / "manifest_filtered.jsonl"
    metrics = filter_clips(
        input_source=clips_dir / "manifest_clips.jsonl",
        output_manifest=filtered_manifest,
        min_width=min_width,
        min_height=min_height,
        min_frames=min_frames,
        min_sharpness=min_sharpness,
        min_motion=min_motion,
        max_motion=max_motion,
        max_overlay_score=max_overlay_score,
        require_portrait=True,
        max_bg_motion=max_bg_motion,
        min_fg_motion=min_fg_motion,
        max_flicker=max_flicker,
    )
    audit = audit_filter_results(metrics)

    # 4. Render 6-frame contact sheets for passed clips for human/agent visual QA
    sheets_dir = output_dir / "audit_sheets"
    sheet_paths: list[str] = []
    for m in metrics:
        if m.passed:
            clip_p = Path(m.path)
            sheet_p = sheets_dir / f"{clip_p.stem}_sheet.jpg"
            render_contact_sheet(clip_p, sheet_p)
            sheet_paths.append(str(sheet_p.resolve()))

    summary = {
        "ingest": ingest_report,
        "filter_audit": audit,
        "filtered_manifest": str(filtered_manifest.resolve()),
        "contact_sheets": sheet_paths,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl & filter 9:16 Vietnamese TikTok dance clips.")
    parser.add_argument("--local-dir", default=None, help="Existing folder of raw TikTok MP4s.")
    parser.add_argument("--urls-file", default=None, help="UTF-8 txt file of public TikTok video/creator URLs.")
    parser.add_argument("--output-dir", required=True, help="Output workspace for filtered clips & QA sheets.")
    parser.add_argument("--max-per-source", type=int, default=25)
    parser.add_argument("--target-frames", type=int, nargs="+", default=[49, 65, 81, 121])
    parser.add_argument("--min-width", type=int, default=540)
    parser.add_argument("--min-height", type=int, default=960)
    parser.add_argument("--min-frames", type=int, default=21)
    parser.add_argument("--min-sharpness", type=float, default=40.0)
    parser.add_argument("--min-motion", type=float, default=0.35)
    parser.add_argument("--max-motion", type=float, default=18.0)
    parser.add_argument("--max-overlay-score", type=float, default=0.18)
    parser.add_argument("--max-bg-motion", type=float, default=0.25)
    parser.add_argument("--min-fg-motion", type=float, default=1.0)
    parser.add_argument("--max-flicker", type=float, default=12.0)
    args = parser.parse_args()

    summary = run_pipeline(
        output_dir=Path(args.output_dir),
        local_dir=Path(args.local_dir) if args.local_dir else None,
        urls_file=Path(args.urls_file) if args.urls_file else None,
        max_per_source=args.max_per_source,
        target_frames=tuple(args.target_frames),
        min_width=args.min_width,
        min_height=args.min_height,
        min_frames=args.min_frames,
        min_sharpness=args.min_sharpness,
        min_motion=args.min_motion,
        max_motion=args.max_motion,
        max_overlay_score=args.max_overlay_score,
        max_bg_motion=args.max_bg_motion,
        min_fg_motion=args.min_fg_motion,
        max_flicker=args.max_flicker,
    )
    fa = summary["filter_audit"]
    print(
        f"Crawled/ingested {summary['ingest']['unique_kept']} unique videos "
        f"(dupes skipped: {summary['ingest']['duplicates_skipped']}).\n"
        f"Filter results: {fa['passed']}/{fa['total']} clips passed "
        f"(pass_rate={fa['pass_rate']:.1%}). Reasons: {fa['reject_reason_counts']}\n"
        f"Contact sheets: {len(summary['contact_sheets'])} written to {args.output_dir}/audit_sheets/"
    )


if __name__ == "__main__":
    main()
