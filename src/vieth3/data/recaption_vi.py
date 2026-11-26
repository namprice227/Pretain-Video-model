"""Native Vietnamese VLM dense recaptioning for video clips."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np

VIETNAMESE_CAPTION_SYSTEM_PROMPT = (
    "Bạn là chuyên gia mô tả dữ liệu huấn luyện video AI bằng tiếng Việt tự nhiên, "
    "giàu chi tiết điện ảnh. Hãy quan sát chuỗi khung hình theo đúng trình tự thời gian "
    "và viết một đoạn mô tả súc tích (2 đến 4 câu) bằng tiếng Việt có dấu chuẩn xác, bao gồm:\n"
    "1. Bối cảnh Việt Nam (kiến trúc, đường phố, thiên nhiên, thời tiết nếu nhận diện được).\n"
    "2. Chủ thể và trang phục (ngoại hình, áo dài, nón lá, trang phục đời thường).\n"
    "3. Hành động và diễn biến chuyển động từ đầu đến cuối đoạn video.\n"
    "4. Góc máy, chuyển động camera (quay cận, lia máy, theo dấu, tĩnh), ánh sáng và tông màu.\n"
    "TUYỆT ĐỐI KHÔNG mở đầu bằng các cụm từ thừa như 'Video này cho thấy', "
    "'Đoạn clip mô tả', hay 'Trong các khung hình'."
)


def _sample_keyframes_bgr(
    video_path: str | Path,
    num_frames: int = 6,
    max_side: Optional[int] = None,
) -> list[np.ndarray]:
    """Sample evenly spaced BGR frames from a video clip, optionally downscaling."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), max(1, num_frames), dtype=int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if max_side is not None and max(frame.shape[:2]) > max_side:
            scale = max_side / float(max(frame.shape[:2]))
            new_w = max(1, int(round(frame.shape[1] * scale)))
            new_h = max(1, int(round(frame.shape[0] * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def sample_keyframes_base64(video_path: str | Path, num_frames: int = 6) -> list[str]:
    """Sample evenly spaced JPEG base64 frames from a video clip."""
    encoded: list[str] = []
    for frame in _sample_keyframes_bgr(video_path, num_frames=num_frames):
        ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok_jpg:
            encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return encoded


def sample_keyframes_pil(video_path: str | Path, num_frames: int = 6, max_side: int = 640):
    """Sample evenly spaced RGB PIL Images from a video clip."""
    from PIL import Image

    images = []
    for frame in _sample_keyframes_bgr(video_path, num_frames=num_frames, max_side=max_side):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    return images


class _LocalQwenVLCaptioner:
    """Lazy-loaded local Transformers Qwen2.5-VL captioner for offline cluster runs."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def caption(self, video_path: str | Path, num_frames: int = 6) -> str:
        """Caption one video clip using local Qwen2.5-VL inference."""
        import torch

        frames = sample_keyframes_pil(video_path, num_frames=num_frames, max_side=640)
        if not frames:
            raise RuntimeError(f"No frames extracted from {video_path}")

        user_content: list[dict] = [{"type": "image"} for _ in frames]
        user_content.append(
            {"type": "text", "text": "Hãy mô tả chi tiết đoạn video này theo hướng dẫn."}
        )
        messages = [
            {"role": "system", "content": VIETNAMESE_CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=frames, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        return str(self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]).strip()


def caption_clip_vlm(
    video_path: str | Path,
    api_base: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    api_key: str = "EMPTY",
    num_frames: int = 6,
    timeout: float = 120.0,
) -> str:
    """Query an OpenAI-compatible VLM endpoint to produce a native Vietnamese caption."""
    frames_b64 = sample_keyframes_base64(video_path, num_frames=num_frames)
    if not frames_b64:
        raise RuntimeError(f"No frames extracted from {video_path}")

    content: list[dict] = [
        {"type": "text", "text": "Hãy mô tả chi tiết đoạn video này theo hướng dẫn."}
    ]
    for b64 in frames_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VIETNAMESE_CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{api_base.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def recaption_manifest(
    manifest_path: str | Path,
    output_manifest: str | Path,
    api_base: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    api_key: str = "EMPTY",
    num_frames: int = 6,
    dry_run: bool = False,
    backend: str = "api",
) -> list[dict]:
    """Generate native Vietnamese captions for every clip in a JSONL manifest."""
    entries: list[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    captioned: list[dict] = []
    local_captioner: Optional[_LocalQwenVLCaptioner] = None
    total = len(entries)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for idx, entry in enumerate(entries, start=1):
            clip_path = Path(entry["path"])
            txt_path = clip_path.with_suffix(".txt")
            if entry.get("caption"):
                caption = str(entry["caption"]).strip()
            elif txt_path.exists():
                caption = txt_path.read_text(encoding="utf-8").strip()
            elif dry_run:
                caption = (
                    "Cảnh quay đường phố Việt Nam vào ban ngày với chuyển động tự nhiên, "
                    "ánh sáng dịu nhẹ và góc máy ổn định."
                )
            elif backend == "local":
                if local_captioner is None:
                    local_captioner = _LocalQwenVLCaptioner(model_id=model)
                caption = local_captioner.caption(clip_path, num_frames=num_frames)
            else:
                caption = caption_clip_vlm(
                    clip_path,
                    api_base=api_base,
                    model=model,
                    api_key=api_key,
                    num_frames=num_frames,
                )

            txt_path.write_text(caption, encoding="utf-8")
            updated = {**entry, "caption": caption, "caption_path": str(txt_path.resolve())}
            captioned.append(updated)
            out_f.write(json.dumps(updated, ensure_ascii=False) + "\n")
            out_f.flush()
            if idx % 10 == 0 or idx == total:
                print(f"[recaption_vi] {idx}/{total} clips captioned", flush=True)

    return captioned


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese VLM recaptioning for video clips.")
    parser.add_argument("--manifest", required=True, help="Input filtered JSONL manifest.")
    parser.add_argument("--output", required=True, help="Output captioned JSONL manifest.")
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = recaption_manifest(
        manifest_path=args.manifest,
        output_manifest=args.output,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        num_frames=args.num_frames,
        dry_run=args.dry_run,
        backend=args.backend,
    )
    print(f"Captioned {len(rows)} clips -> {args.output}")


if __name__ == "__main__":
    main()
