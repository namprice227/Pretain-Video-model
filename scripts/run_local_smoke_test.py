"""End-to-end local CPU smoke test for VietH3:
1. Synthesize 4 crisp moving-shape MP4 videos with Vietnamese captions in a temp dir.
2. Run scene_split -> filters -> recaption_vi -> cache_latents (dummy VAE & text encoder).
3. Train VietH3 debug model for 15 steps and assert loss decreases.
4. Sample a Vietnamese prompt to MP4 and verify with ffprobe.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from vieth3.data.cache_latents import cache_dataset
from vieth3.data.filters import filter_clips
from vieth3.data.recaption_vi import recaption_manifest
from vieth3.data.scene_split import find_ffmpeg, find_ffprobe, split_videos_to_clips
from vieth3.inference.sample import generate_video
from vieth3.training.train import train

VIETNAMESE_PROMPTS = [
    "Một chiếc đèn lồng đỏ đung đưa nhẹ nhàng trên phố cổ Hội An vào buổi tối.",
    "Người phụ nữ mặc áo dài trắng đi bộ bên hồ Hoàn Kiếm dưới ánh nắng dịu.",
    "Những con thuyền nhỏ lướt trên mặt nước xanh biếc ở vịnh Hạ Long.",
    "Xe máy di chuyển qua con phố rợp bóng cây xanh ở trung tâm Sài Gòn.",
]


def _make_synthetic_video(path: Path, seed: int, width: int = 256, height: int = 256, frames: int = 36) -> None:
    ffmpeg = find_ffmpeg()
    rng = np.random.default_rng(seed)
    bg = rng.integers(30, 80, size=(3,), dtype=np.uint8)
    raw_frames: list[np.ndarray] = []
    for f in range(frames):
        img = np.tile(bg, (height, width, 1)).astype(np.uint8)
        # Sharp checkerboard background border-free in center for high Laplacian variance
        cx = int(48 + (width - 96) * (f / max(1, frames - 1)))
        cy = int(height // 2 + 40 * np.sin(f * 0.25 + seed))
        cv2.circle(img, (cx, cy), 28, (240, 180, 50), -1)
        cv2.rectangle(img, (cx - 18, cy - 18), (cx + 18, cy + 18), (40, 220, 220), 3)
        cv2.line(img, (32, 32), (cx, cy), (255, 255, 255), 2)
        raw_frames.append(img)
    arr = np.stack(raw_frames, axis=0)
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "24", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path),
    ]
    subprocess.run(cmd, input=arr.tobytes(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_cfg = repo_root / "configs" / "model_vieth3_debug_20m.yaml"
    ffprobe = find_ffprobe()

    with tempfile.TemporaryDirectory(prefix="vieth3_smoke_") as tmp:
        tmp_dir = Path(tmp)
        raw_dir = tmp_dir / "raw"
        raw_dir.mkdir()

        for i in range(4):
            _make_synthetic_video(raw_dir / f"vn_clip_{i}.mp4", seed=10 + i)

        print("[1/5] Scene splitting to H3 grid (22 frames)...")
        clips = split_videos_to_clips(
            input_dir=raw_dir,
            output_dir=tmp_dir / "split",
            fps=24,
            grid="h3",
            target_frames=(22,),
        )
        assert len(clips) == 4, f"Expected 4 split clips, got {len(clips)}"

        print("[2/5] Filtering clips...")
        metrics = filter_clips(
            input_source=tmp_dir / "split" / "manifest_clips.jsonl",
            output_manifest=tmp_dir / "manifest_filtered.jsonl",
            min_width=256,
            min_height=256,
            min_frames=22,
            min_sharpness=20.0,
            min_motion=0.1,
            max_overlay_score=0.30,
        )
        passed = [m for m in metrics if m.passed]
        assert len(passed) == 4, f"Expected 4 passed clips, got {len(passed)}: {metrics}"

        # Attach Vietnamese captions for each clip
        filtered_lines = [
            json.loads(line)
            for line in (tmp_dir / "manifest_filtered.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for idx, row in enumerate(filtered_lines):
            row["caption"] = VIETNAMESE_PROMPTS[idx % len(VIETNAMESE_PROMPTS)]
        (tmp_dir / "manifest_filtered.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in filtered_lines) + "\n",
            encoding="utf-8",
        )

        print("[3/5] Writing Vietnamese captions...")
        captioned = recaption_manifest(
            manifest_path=tmp_dir / "manifest_filtered.jsonl",
            output_manifest=tmp_dir / "manifest_captioned.jsonl",
            dry_run=True,
        )
        assert len(captioned) == 4

        print("[4/5] Caching latents & Vietnamese embeddings...")
        cache_idx = cache_dataset(
            manifest_path=tmp_dir / "manifest_captioned.jsonl",
            output_dir=tmp_dir / "cache",
            model_config=model_cfg,
            vae_type="dummy",
            text_encoder_type="dummy",
            bucket_stage="native",
            device="cpu",
        )

        print("[5/5] Running 15-step training + Vietnamese T2V inference...")
        summary = train(
            {
                "model_config": str(model_cfg),
                "cache_index": str(cache_idx),
                "output_dir": str(tmp_dir / "train_out"),
                "strategy": "single",
                "bf16": False,
                "gradient_checkpointing": False,
                "batch_size": 2,
                "grad_accum": 1,
                "max_steps": 15,
                "warmup_steps": 2,
                "lr": 5e-4,
                "cfg_dropout": 0.0,
                "use_ema": True,
                "log_every": 5,
                "save_every": 15,
                "device": "cpu",
                "seed": 42,
            }
        )
        print("Training summary:", summary)
        assert summary["final_loss"] < summary["initial_loss"], (
            f"Expected loss to drop, got initial={summary['initial_loss']} final={summary['final_loss']}"
        )

        sample_out = generate_video(
            model_config=model_cfg,
            checkpoint=summary["checkpoint_dir"],
            prompt="Phố cổ Hội An rực rỡ đèn lồng vào buổi tối.",
            output=tmp_dir / "sample_vi.mp4",
            height=128,
            width=128,
            frames=22,
            steps=4,
            cfg_scale=2.0,
            solver="euler",
            vae_type="dummy",
            text_encoder_type="dummy",
            device="cpu",
        )
        assert sample_out.exists() and sample_out.stat().st_size > 1000

        probe = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,codec_name",
                "-of", "json", str(sample_out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probe_data = json.loads(probe.stdout)
        stream = probe_data["streams"][0]
        print("ffprobe verified output:", stream)
        assert int(stream["width"]) == 128 and int(stream["height"]) == 128
        print("SMOKE TEST PASSED!")


if __name__ == "__main__":
    main()
