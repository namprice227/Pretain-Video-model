"""3D Multimodal Rotary Position Embedding (RoPE) utilities for VietH3."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_rope_axes(head_dim: int) -> tuple[int, int, int]:
    """Split head_dim into even axis dimensions (d_t, d_h, d_w) summing to head_dim."""
    if head_dim < 6 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even and >= 6, got {head_dim}")
    d_t = ((head_dim // 4) // 2) * 2
    d_t = max(d_t, 2)
    rem = head_dim - d_t
    d_h = ((rem // 2) // 2) * 2
    d_w = rem - d_h
    if d_w % 2 != 0 or d_w < 2:
        raise ValueError(f"Cannot evenly split head_dim={head_dim} into (d_t, d_h, d_w)")
    return d_t, d_h, d_w


def build_1d_rope_freqs(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build 1D RoPE (cos, sin) tables of shape (seq_len, dim)."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE axis dim must be even, got {dim}")
    half = dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)  # (seq_len, half)
    freqs = torch.cat([angles, angles], dim=-1)  # (seq_len, dim)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def build_3d_rope_freqs(
    t_len: int,
    h_len: int,
    w_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build flattened 3D (t, h, w) RoPE (cos, sin) of shape (t_len * h_len * w_len, head_dim)."""
    d_t, d_h, d_w = compute_rope_axes(head_dim)
    cos_t, sin_t = build_1d_rope_freqs(t_len, d_t, theta=theta, device=device, dtype=torch.float32)
    cos_h, sin_h = build_1d_rope_freqs(h_len, d_h, theta=theta, device=device, dtype=torch.float32)
    cos_w, sin_w = build_1d_rope_freqs(w_len, d_w, theta=theta, device=device, dtype=torch.float32)

    cos_t_grid = cos_t[:, None, None, :].expand(t_len, h_len, w_len, d_t)
    cos_h_grid = cos_h[None, :, None, :].expand(t_len, h_len, w_len, d_h)
    cos_w_grid = cos_w[None, None, :, :].expand(t_len, h_len, w_len, d_w)

    sin_t_grid = sin_t[:, None, None, :].expand(t_len, h_len, w_len, d_t)
    sin_h_grid = sin_h[None, :, None, :].expand(t_len, h_len, w_len, d_h)
    sin_w_grid = sin_w[None, None, :, :].expand(t_len, h_len, w_len, d_w)

    cos = torch.cat([cos_t_grid, cos_h_grid, cos_w_grid], dim=-1).reshape(-1, head_dim)
    sin = torch.cat([sin_t_grid, sin_h_grid, sin_w_grid], dim=-1).reshape(-1, head_dim)
    return cos.to(dtype), sin.to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to x of shape (B, H, L, D).

    Computes rotation in float32 for numerical stability and casts back to x.dtype.
    """
    orig_dtype = x.dtype
    cos_b = cos.unsqueeze(0).unsqueeze(0).to(torch.float32)
    sin_b = sin.unsqueeze(0).unsqueeze(0).to(torch.float32)
    x_f32 = x.to(torch.float32)
    out = x_f32 * cos_b + _rotate_half(x_f32) * sin_b
    return out.to(orig_dtype)
