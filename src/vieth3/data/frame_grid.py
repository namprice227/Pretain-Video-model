"""Frame-count grid and spatial bucketing rules for VietH3.

MiniMax-H3's VisualVAE (f16t4d24) enforces the 17n + 5 frame rule at 24 fps
(e.g., 5, 22, 39, 56, 73, 90, 107, 124, ..., 243).
Wan2.1-VAE (f8t4d16) uses the 4n + 1 frame rule (1, 5, 9, 13, ...).
Images (T = 1) are also supported for joint Image + Video pretraining.
"""

from __future__ import annotations

from typing import Literal

GridType = Literal["h3", "wan21"]

STANDARD_BUCKETS: dict[str, list[tuple[int, int]]] = {
    "128p": [(128, 192), (192, 128), (128, 128)],
    "256p": [(256, 448), (448, 256), (256, 256)],
    "480p": [(480, 832), (832, 480), (480, 480)],
    "720p": [(704, 1248), (1248, 704), (768, 768)],
}


def valid_h3_frames(max_frames: int = 243) -> list[int]:
    """Return all valid H3 frame counts 17n + 5 up to max_frames."""
    counts: list[int] = []
    n = 0
    while True:
        val = 17 * n + 5
        if val > max_frames:
            break
        counts.append(val)
        n += 1
    return counts


def valid_wan_frames(max_frames: int = 241) -> list[int]:
    """Return all valid Wan2.1 frame counts 4n + 1 up to max_frames."""
    counts: list[int] = []
    n = 0
    while True:
        val = 4 * n + 1
        if val > max_frames:
            break
        counts.append(val)
        n += 1
    return counts


def snap_frame_count(
    n_frames: int,
    grid: GridType = "h3",
    allow_image: bool = True,
    max_frames: int = 243,
) -> int:
    """Snap a raw frame count down to the nearest valid grid length."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if allow_image and n_frames == 1:
        return 1
    valid = valid_h3_frames(max_frames) if grid == "h3" else valid_wan_frames(max_frames)
    candidates = [v for v in valid if v <= n_frames]
    if candidates:
        return candidates[-1]
    return valid[0]


def snap_resolution(height: int, width: int, multiple: int = 32) -> tuple[int, int]:
    """Snap height and width down to the nearest multiple (for VAE + [1,2,2] patchify)."""
    h = max(multiple, (height // multiple) * multiple)
    w = max(multiple, (width // multiple) * multiple)
    return h, w


def select_aspect_bucket(
    height: int,
    width: int,
    stage: str = "480p",
    multiple: int = 32,
) -> tuple[int, int]:
    """Pick the closest aspect-ratio bucket for a resolution stage."""
    if stage == "native":
        return snap_resolution(height, width, multiple=multiple)
    if stage not in STANDARD_BUCKETS:
        raise ValueError(f"Unknown bucket stage {stage!r}; choose from {list(STANDARD_BUCKETS)}")
    aspect = width / max(1, height)
    buckets = STANDARD_BUCKETS[stage]
    return min(buckets, key=lambda hw: abs((hw[1] / hw[0]) - aspect))
