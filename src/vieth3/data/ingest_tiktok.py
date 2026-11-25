"""TikTok 9:16 video ingestion, SHA-256 + perceptual dHash deduplication, and audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from vieth3.data.filters import ClipMetrics

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _dhash_frame(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = "".join("1" if b else "0" for b in diff.flatten())
    return f"{int(bits, 2):016x}"


def compute_video_fingerprint(video_path: str | Path) -> tuple[str, str]:
    """Compute (sha256_file_hash, 3-frame perceptual dHash) to catch exact & re-encoded dupes."""
    p = Path(video_path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    sha256_hex = h.hexdigest()

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return sha256_hex, ""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), 3, dtype=int)
    hashes: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            hashes.append(_dhash_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    cap.release()
    return sha256_hex, "-".join(hashes)


def ingest_local_videos(
    input_dir: str | Path,
    output_dir: str | Path,
    copy_files: bool = False,
) -> list[dict]:
    """Scan local videos, deduplicate by SHA-256 & perceptual dHash, and write manifest + report."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_copy_dir = out_path / "raw"
    if copy_files:
        raw_copy_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in in_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    seen_sha: set[str] = set()
    seen_dhash: set[str] = set()

    manifest: list[dict] = []
    dupes_skipped = 0
    portrait_count = 0
    landscape_count = 0
    total_sec = 0.0

    for vid in videos:
        sha, dhash = compute_video_fingerprint(vid)
        if sha in seen_sha or (dhash and dhash in seen_dhash):
            dupes_skipped += 1
            continue
        seen_sha.add(sha)
        if dhash:
            seen_dhash.add(dhash)

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        cap.release()
        if w <= 0 or h <= 0 or n_frames <= 0:
            continue

        dur = n_frames / max(1.0, fps)
        total_sec += dur
        if h > w:
            portrait_count += 1
        else:
            landscape_count += 1

        final_path = vid.resolve()
        if copy_files:
            dst = raw_copy_dir / f"{sha[:12]}_{vid.name}"
            shutil.copy2(vid, dst)
            final_path = dst.resolve()

        manifest.append(
            {
                "path": str(final_path),
                "sha256": sha,
                "dhash": dhash,
                "width": w,
                "height": h,
                "num_frames": n_frames,
                "fps": fps,
                "duration_sec": round(dur, 3),
                "is_portrait": bool(h > w),
            }
        )

    manifest_path = out_path / "manifest_ingested.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "total_scanned": len(videos),
        "unique_kept": len(manifest),
        "duplicates_skipped": dupes_skipped,
        "portrait_count": portrait_count,
        "landscape_count": landscape_count,
        "total_hours": round(total_sec / 3600.0, 4),
    }
    (out_path / "ingest_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def download_urls_ytdlp(urls_file: str | Path, output_dir: str | Path) -> Path:
    """Download user-provided video URLs via yt-dlp into output_dir."""
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise FileNotFoundError(
            "yt-dlp executable not found on PATH. Install with: pip install yt-dlp"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    urls = [
        line.strip()
        for line in Path(urls_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for url in urls:
        cmd = [
            ytdlp,
            "-f",
            "bv*[ext=mp4]/bv*",
            "--merge-output-format",
            "mp4",
            "-o",
            str(out / "%(id)s.%(ext)s"),
            url,
        ]
        subprocess.run(cmd, check=True)
    return out


def audit_filter_results(metrics_list: Iterable[ClipMetrics]) -> dict:
    """Aggregate pass/reject counts and rejection reason breakdown."""
    items = list(metrics_list)
    passed = sum(1 for m in items if m.passed)
    rejected = len(items) - passed
    reason_counts: Counter[str] = Counter()
    for m in items:
        for r in m.reject_reasons:
            prefix = r.split(":", 1)[0]
            reason_counts[prefix] += 1
    return {
        "total": len(items),
        "passed": passed,
        "rejected": rejected,
        "pass_rate": round(passed / len(items), 4) if items else 0.0,
        "reject_reason_counts": dict(reason_counts),
        "sample_rejected": [asdict(m) for m in items if not m.passed][:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest & deduplicate 9:16 TikTok videos.")
    parser.add_argument("--local-dir", default=None, help="Directory containing raw TikTok MP4s.")
    parser.add_argument("--urls-file", default=None, help="Optional txt file of URLs for yt-dlp.")
    parser.add_argument("--output-dir", required=True, help="Output directory for manifest & report.")
    parser.add_argument("--copy-files", action="store_true")
    args = parser.parse_args()

    src_dir: Optional[Path] = Path(args.local_dir) if args.local_dir else None
    if args.urls_file:
        dl_dir = Path(args.output_dir) / "downloaded"
        download_urls_ytdlp(args.urls_file, dl_dir)
        if src_dir is None:
            src_dir = dl_dir

    if src_dir is None:
        raise ValueError("Provide --local-dir and/or --urls-file")

    rows = ingest_local_videos(src_dir, args.output_dir, copy_files=args.copy_files)
    print(f"Ingested {len(rows)} unique clips -> {args.output_dir}/manifest_ingested.jsonl")


if __name__ == "__main__":
    main()
