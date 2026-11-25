"""Model configuration for VietH3 single-stream Omni-DiT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VietH3Config:
    """Configuration for VietH3Transformer3DModel.

    Mirrors MiniMax-H3's single-stream Omni-DiT design (unified text+video sequence,
    2-layer text token refiner, dual AdaLN-Zero modulation, 3D RoPE, SwiGLU FFN,
    and [1, 2, 2] latent patchification), scaled for 2x H100 (80GB) training.
    """

    name: str = "vieth3-mini-500m"
    in_channels: int = 24
    patch_size: tuple[int, int, int] = (1, 2, 2)
    hidden_size: int = 1536
    num_layers: int = 20
    num_refiner_layers: int = 2
    num_attention_heads: int = 12
    attention_head_dim: int = 128
    ffn_dim: int = 4096
    text_dim: int = 3584
    max_text_len: int = 256
    rope_theta: float = 10000.0
    qk_norm: bool = True
    vae_type: str = "h3"
    vae_spatial_stride: int = 16
    vae_temporal_stride: int = 4
    use_i2v_cond: bool = False
    use_pose_cond: bool = False
    pose_channels: int = 24
    shared_adaln: bool = False
    num_register_tokens: int = 0
    use_long_skips: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(int(x) for x in self.patch_size)  # type: ignore[assignment]
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 dims (pt, ph, pw), got {self.patch_size}")
        if self.attention_head_dim % 2 != 0:
            raise ValueError(f"attention_head_dim must be even, got {self.attention_head_dim}")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def patch_in_channels(self) -> int:
        channels = self.in_channels
        if self.use_i2v_cond:
            channels += self.in_channels + 1
        if self.use_pose_cond:
            channels += self.pose_channels
        return channels

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VietH3Config:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VietH3Config:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if "model" in raw and isinstance(raw["model"], dict):
            raw = raw["model"]
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["patch_size"] = list(self.patch_size)
        return d

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    def estimate_num_params(self) -> int:
        """Analytic parameter count matching VietH3Transformer3DModel."""
        h = self.hidden_size
        inner = self.inner_dim
        ffn = self.ffn_dim
        pt, ph, pw = self.patch_size
        patch_in_dim = pt * ph * pw * self.patch_in_channels
        patch_out_dim = pt * ph * pw * self.in_channels

        # Patch embedding + text projection + timestep embedder
        params = patch_in_dim * h + h  # Conv3d weight + bias
        params += self.text_dim * h + 2 * h  # text_proj (weight+bias) + text_norm
        params += 256 * h + h + h * h + h  # TimestepEmbedder (freq=256 -> h -> h)

        # Text refiner blocks (AdaLN + QKV + O + QK-norm + SwiGLU FFN)
        qk_norm_p = 2 * self.attention_head_dim if self.qk_norm else 0
        refiner_per_layer = (
            (6 * h * h + 6 * h)  # adaLN
            + (h * 3 * inner + inner * h)  # QKV + O
            + qk_norm_p
            + (3 * h * ffn)  # SwiGLU w_gate, w_up, w_down
            + 2 * h  # 2x RMSNorm
        )
        params += self.num_refiner_layers * refiner_per_layer

        # Main single-stream blocks (dual AdaLN for text & video + shared attn + shared FFN)
        if self.shared_adaln:
            params += 2 * (6 * h * h + 6 * h)  # global shared_adaLN_txt & shared_adaLN_vid
            adaln_per_layer = 2 * (6 * h)  # per-block biases
        else:
            adaln_per_layer = 2 * (6 * h * h + 6 * h)

        block_per_layer = (
            adaln_per_layer
            + (h * 3 * inner + inner * h)  # QKV + O
            + qk_norm_p
            + (3 * h * ffn)  # SwiGLU
            + 4 * h  # norm1/norm2 for text & video
        )
        params += self.num_layers * block_per_layer

        if self.num_register_tokens > 0:
            params += self.num_register_tokens * h
        if self.use_long_skips:
            n_half = self.num_layers // 2
            params += n_half * (2 * h * h + h)

        # Final AdaLN + final projection
        params += (2 * h * h + 2 * h) + (h * patch_out_dim + patch_out_dim)
        return params
