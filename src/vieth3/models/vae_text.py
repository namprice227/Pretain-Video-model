"""Pluggable Video VAE and native Vietnamese Text Encoder wrappers.

Supports:
- DummyVideoVAE / DummyVietnameseTextEncoder for zero-download local testing
- HuggingFaceVideoVAE (Wan2.1-VAE / H3-VisualVAE) via lazy diffusers import
- QwenTextEncoder (Qwen2.5-VL / Qwen3 / Qwen3-VL) via lazy transformers import
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseVideoVAE(ABC):
    """Abstract interface for 3D Video VAEs."""

    latent_channels: int
    spatial_stride: int
    temporal_stride: int

    @abstractmethod
    def encode(self, video: Tensor) -> Tensor:
        """Encode pixel video (B, 3, T, H, W) in [-1, 1] -> latent (B, C, T_lat, H_lat, W_lat)."""

    @abstractmethod
    def decode(self, latents: Tensor) -> Tensor:
        """Decode latent (B, C, T_lat, H_lat, W_lat) -> pixel video (B, 3, T, H, W) in [-1, 1]."""


class DummyVideoVAE(BaseVideoVAE):
    """Deterministic lightweight 3D encoder-decoder for offline testing.

    Projects RGB patches via a fixed orthonormal basis into `latent_channels`
    and reconstructs back to RGB without downloading weights.
    """

    def __init__(
        self,
        latent_channels: int = 24,
        spatial_stride: int = 16,
        temporal_stride: int = 4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.latent_channels = latent_channels
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.device = torch.device(device)

        gen = torch.Generator(device="cpu").manual_seed(42)
        proj = torch.randn(latent_channels, 3, generator=gen) / (3.0**0.5)
        pinv = torch.linalg.pinv(proj)  # (3, latent_channels)
        self._enc_proj = proj.to(self.device)
        self._dec_proj = pinv.to(self.device)

    def encode(self, video: Tensor) -> Tensor:
        b, c, t, h, w = video.shape
        if c != 3:
            raise ValueError(f"Expected 3 RGB channels, got {c}")
        t_lat = max(1, (t - 1) // self.temporal_stride + 1)
        h_lat = max(1, h // self.spatial_stride)
        w_lat = max(1, w // self.spatial_stride)
        pooled = F.adaptive_avg_pool3d(video.to(torch.float32), (t_lat, h_lat, w_lat))
        # (B, 3, T_lat, H_lat, W_lat) -> (B, C_lat, T_lat, H_lat, W_lat)
        lat = torch.einsum("cd,bdthw->bcthw", self._enc_proj.to(pooled.device), pooled)
        return lat.to(video.dtype)

    def decode(self, latents: Tensor) -> Tensor:
        b, c, t_lat, h_lat, w_lat = latents.shape
        if c != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {c}")
        t_out = 1 if t_lat == 1 else (t_lat - 1) * self.temporal_stride + 1
        h_out = h_lat * self.spatial_stride
        w_out = w_lat * self.spatial_stride
        rgb_low = torch.einsum("dc,bcthw->bdthw", self._dec_proj.to(latents.device), latents.float())
        video = F.interpolate(rgb_low, size=(t_out, h_out, w_out), mode="trilinear", align_corners=False)
        return video.clamp(-1.0, 1.0).to(latents.dtype)


def normalize_latents(z: Tensor, mean: Optional[Tensor], std: Optional[Tensor]) -> Tensor:
    """Apply Wan2.1 per-channel (z - mean) / std latent normalization."""
    if mean is None or std is None:
        return z
    m = mean.to(device=z.device, dtype=torch.float32)
    s = std.to(device=z.device, dtype=torch.float32).clamp(min=1e-6)
    return ((z.to(torch.float32) - m) / s).to(z.dtype)


def denormalize_latents(z: Tensor, mean: Optional[Tensor], std: Optional[Tensor]) -> Tensor:
    """Invert Wan2.1 per-channel normalization: z * std + mean."""
    if mean is None or std is None:
        return z
    m = mean.to(device=z.device, dtype=torch.float32)
    s = std.to(device=z.device, dtype=torch.float32)
    return (z.to(torch.float32) * s + m).to(z.dtype)


class HuggingFaceVideoVAE(BaseVideoVAE):
    """Wrapper around diffusers 3D Video VAEs (Wan2.1 AutoencoderKLWan in float32)."""

    def __init__(
        self,
        model_id: str,
        subfolder: str = "vae",
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        latent_channels: int = 16,
        spatial_stride: int = 8,
        temporal_stride: int = 4,
    ) -> None:
        try:
            from diffusers import AutoencoderKLWan  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "diffusers>=0.31.0 is required for HuggingFaceVideoVAE. "
                "Install with: pip install -e '.[hf]'"
            ) from exc

        self.device = torch.device(device)
        self.dtype = dtype
        # Run Wan2.1 VAE in float32 to prevent BF16 banding/shimmer on static backgrounds
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder=subfolder, torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.latent_channels = getattr(self.vae.config, "z_dim", latent_channels)
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride

        raw_mean = getattr(self.vae.config, "latents_mean", None)
        raw_std = getattr(self.vae.config, "latents_std", None)
        if raw_mean is not None and raw_std is not None:
            self.latents_mean: Optional[Tensor] = torch.tensor(
                raw_mean, device=self.device, dtype=torch.float32
            ).view(1, -1, 1, 1, 1)
            self.latents_std: Optional[Tensor] = torch.tensor(
                raw_std, device=self.device, dtype=torch.float32
            ).view(1, -1, 1, 1, 1)
        else:
            self.latents_mean = None
            self.latents_std = None

    @torch.no_grad()
    def encode(self, video: Tensor, deterministic: bool = True) -> Tensor:
        x = video.to(device=self.device, dtype=torch.float32)
        enc = self.vae.encode(x)
        dist = enc.latent_dist if hasattr(enc, "latent_dist") else enc[0]
        raw_latents = dist.mode() if deterministic else dist.sample()
        normed = normalize_latents(raw_latents, self.latents_mean, self.latents_std)
        return normed.to(self.dtype)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        denormed = denormalize_latents(
            latents.to(device=self.device, dtype=torch.float32),
            self.latents_mean,
            self.latents_std,
        )
        dec = self.vae.decode(denormed)
        sample = dec.sample if hasattr(dec, "sample") else dec[0]
        return sample.clamp(-1.0, 1.0).to(self.dtype)


class BaseTextEncoder(ABC):
    """Abstract interface for native Vietnamese text encoders."""

    text_dim: int
    max_length: int

    @abstractmethod
    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        """Encode prompts -> (text_embeds (B, L, text_dim), text_mask (B, L) bool)."""


class DummyVietnameseTextEncoder(BaseTextEncoder):
    """Deterministic UTF-8 Vietnamese n-gram text encoder for CPU testing.

    Produces distinct, reproducible embeddings and masks for any Vietnamese UTF-8
    prompt without downloading external checkpoints.
    """

    def __init__(
        self,
        text_dim: int = 3584,
        max_length: int = 64,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.text_dim = text_dim
        self.max_length = max_length
        self.device = torch.device(device)
        self.dtype = dtype

    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        batch_embeds = torch.zeros(
            (len(prompts), self.max_length, self.text_dim),
            device=self.device,
            dtype=self.dtype,
        )
        batch_mask = torch.zeros(
            (len(prompts), self.max_length),
            device=self.device,
            dtype=torch.bool,
        )
        for idx, text in enumerate(prompts):
            words = text.strip().split()
            if not words:
                batch_mask[idx, 0] = True
                continue
            n_tok = min(len(words), self.max_length)
            batch_mask[idx, :n_tok] = True
            for tok_i in range(n_tok):
                digest = hashlib.sha256(words[tok_i].encode("utf-8")).digest()
                seed = int.from_bytes(digest[:8], "little")
                gen = torch.Generator(device="cpu").manual_seed(seed)
                vec = torch.randn(self.text_dim, generator=gen, dtype=torch.float32)
                vec = vec / (vec.norm() + 1e-6)
                batch_embeds[idx, tok_i] = vec.to(device=self.device, dtype=self.dtype)
        return batch_embeds, batch_mask


class QwenTextEncoder(BaseTextEncoder):
    """Qwen2.5-VL / Qwen3 / Qwen3-VL text encoder extracting intermediate hidden states."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        text_dim: int = 3584,
        max_length: int = 256,
        layer_index: int = -2,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.45.0 is required for QwenTextEncoder. "
                "Install with: pip install -e '.[hf]'"
            ) from exc

        self.text_dim = text_dim
        self.max_length = max_length
        self.layer_index = layer_index
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[self.layer_index]
        mask = tokens["attention_mask"].to(torch.bool)
        return hidden.to(self.dtype), mask


def load_vae(
    vae_type: str = "dummy",
    latent_channels: int = 24,
    spatial_stride: int = 16,
    temporal_stride: int = 4,
    model_id: Optional[str] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BaseVideoVAE:
    """Factory to load either DummyVideoVAE or a HuggingFace 3D Video VAE."""
    if vae_type == "dummy":
        return DummyVideoVAE(
            latent_channels=latent_channels,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            device=device,
        )
    if vae_type in {"h3", "wan21"}:
        default_id = (
            "MiniMaxAI/MiniMax-H3" if vae_type == "h3" else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        )
        return HuggingFaceVideoVAE(
            model_id=model_id or default_id,
            dtype=dtype,
            device=device,
            latent_channels=latent_channels,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
        )
    raise ValueError(f"Unknown vae_type: {vae_type!r}")


def load_text_encoder(
    encoder_type: str = "dummy",
    text_dim: int = 3584,
    max_length: int = 256,
    model_id: Optional[str] = None,
    layer_index: int = -2,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BaseTextEncoder:
    """Factory to load DummyVietnameseTextEncoder or QwenTextEncoder."""
    if encoder_type == "dummy":
        return DummyVietnameseTextEncoder(
            text_dim=text_dim,
            max_length=max_length,
            device=device,
            dtype=dtype,
        )
    if encoder_type in {"qwen", "qwen25vl", "qwen3vl"}:
        return QwenTextEncoder(
            model_id=model_id or "Qwen/Qwen2.5-VL-7B-Instruct",
            text_dim=text_dim,
            max_length=max_length,
            layer_index=layer_index,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
