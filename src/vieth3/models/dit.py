"""VietH3 Single-Stream Omni-DiT Transformer architecture.

Inspired by MiniMax-H3:
- Unified single-stream sequence over [text_tokens, video_tokens]
- 2-layer TextTokenRefiner with 1D RoPE
- Dual AdaLN-Zero modulation (separate text/video modulation inside shared blocks)
- RMSNorm + QK-Norm + 3D Multimodal RoPE (t, h, w) on video tokens
- SwiGLU Feed-Forward Network
- [1, 2, 2] latent patchification
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from vieth3.models.config import VietH3Config
from vieth3.models.rope3d import apply_rope, build_1d_rope_freqs, build_3d_rope_freqs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        x_f32 = x.to(torch.float32)
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_f32 * rms).to(x.dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer SiLU MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timesteps: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps.to(torch.float32)[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, timesteps: Tensor, dtype: torch.dtype) -> Tensor:
        t_freq = self.timestep_embedding(timesteps, self.freq_dim).to(dtype)
        return self.mlp(t_freq)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TextTokenRefinerBlock(nn.Module):
    """Timestep-aware transformer refiner for conditioning text tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.adaLN = nn.Linear(hidden_size, 6 * hidden_size)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

        self.to_qkv = nn.Linear(hidden_size, 3 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.ffn = SwiGLUFFN(hidden_size, ffn_dim)

    def forward(
        self,
        x: Tensor,
        temb: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
            F.silu(temb)
        ).chunk(6, dim=-1)

        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        b, l, _ = h.shape
        qkv = self.to_qkv(h).view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_rope(self.q_norm(q.transpose(1, 2)), rope_cos, rope_sin)
        k = apply_rope(self.k_norm(k.transpose(1, 2)), rope_cos, rope_sin)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(b, l, -1)
        x = x + gate_msa.unsqueeze(1) * self.to_out(attn_out)

        h_mlp = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(h_mlp)
        return x


class VietH3SingleStreamBlock(nn.Module):
    """Single-stream joint attention + FFN block with dual AdaLN-Zero for text & video."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        qk_norm: bool = True,
        shared_adaln: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.shared_adaln = shared_adaln
        inner_dim = num_heads * head_dim

        self.norm1_txt = RMSNorm(hidden_size)
        self.norm2_txt = RMSNorm(hidden_size)
        self.norm1_vid = RMSNorm(hidden_size)
        self.norm2_vid = RMSNorm(hidden_size)

        if shared_adaln:
            self.adaLN_bias_txt = nn.Parameter(torch.zeros(1, 6 * hidden_size))
            self.adaLN_bias_vid = nn.Parameter(torch.zeros(1, 6 * hidden_size))
        else:
            self.adaLN_txt = nn.Linear(hidden_size, 6 * hidden_size)
            self.adaLN_vid = nn.Linear(hidden_size, 6 * hidden_size)
            nn.init.zeros_(self.adaLN_txt.weight)
            nn.init.zeros_(self.adaLN_txt.bias)
            nn.init.zeros_(self.adaLN_vid.weight)
            nn.init.zeros_(self.adaLN_vid.bias)

        self.to_qkv = nn.Linear(hidden_size, 3 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.ffn = SwiGLUFFN(hidden_size, ffn_dim)

    def forward(
        self,
        text_h: Tensor,
        video_h: Tensor,
        temb: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        attn_mask: Optional[Tensor] = None,
        shared_mod_txt: Optional[Tensor] = None,
        shared_mod_vid: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if self.shared_adaln:
            assert shared_mod_txt is not None and shared_mod_vid is not None
            mod_txt = shared_mod_txt + self.adaLN_bias_txt
            mod_vid = shared_mod_vid + self.adaLN_bias_vid
        else:
            t_act = F.silu(temb)
            mod_txt = self.adaLN_txt(t_act)
            mod_vid = self.adaLN_vid(t_act)

        s_msa_t, sc_msa_t, g_msa_t, s_mlp_t, sc_mlp_t, g_mlp_t = mod_txt.chunk(6, dim=-1)
        s_msa_v, sc_msa_v, g_msa_v, s_mlp_v, sc_mlp_v, g_mlp_v = mod_vid.chunk(6, dim=-1)

        txt_mod = _modulate(self.norm1_txt(text_h), s_msa_t, sc_msa_t)
        vid_mod = _modulate(self.norm1_vid(video_h), s_msa_v, sc_msa_v)

        l_txt = text_h.shape[1]
        x_cat = torch.cat([txt_mod, vid_mod], dim=1)
        b, l_all, _ = x_cat.shape

        qkv = self.to_qkv(x_cat).view(b, l_all, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)

        # Apply 3D RoPE only to video tokens; text tokens stay at neutral (0, 0, 0)
        q_txt, q_vid = q[:, :, :l_txt, :], q[:, :, l_txt:, :]
        k_txt, k_vid = k[:, :, :l_txt, :], k[:, :, l_txt:, :]
        q = torch.cat([q_txt, apply_rope(q_vid, rope_cos, rope_sin)], dim=2)
        k = torch.cat([k_txt, apply_rope(k_vid, rope_cos, rope_sin)], dim=2)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = self.to_out(attn_out.transpose(1, 2).reshape(b, l_all, -1))
        attn_txt, attn_vid = attn_out[:, :l_txt, :], attn_out[:, l_txt:, :]

        text_h = text_h + g_msa_t.unsqueeze(1) * attn_txt
        video_h = video_h + g_msa_v.unsqueeze(1) * attn_vid

        txt_mlp_in = _modulate(self.norm2_txt(text_h), s_mlp_t, sc_mlp_t)
        vid_mlp_in = _modulate(self.norm2_vid(video_h), s_mlp_v, sc_mlp_v)
        ffn_cat = self.ffn(torch.cat([txt_mlp_in, vid_mlp_in], dim=1))
        ffn_txt, ffn_vid = ffn_cat[:, :l_txt, :], ffn_cat[:, l_txt:, :]

        text_h = text_h + g_mlp_t.unsqueeze(1) * ffn_txt
        video_h = video_h + g_mlp_v.unsqueeze(1) * ffn_vid
        return text_h, video_h


class VietH3Transformer3DModel(nn.Module):
    """Single-stream 3D Diffusion Transformer for Vietnamese Text-to-Video."""

    def __init__(self, config: VietH3Config) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        pt, ph, pw = config.patch_size
        patch_vol = pt * ph * pw

        self.patch_embed = nn.Conv3d(
            config.patch_in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.text_proj = nn.Linear(config.text_dim, config.hidden_size)
        self.text_norm = RMSNorm(config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)

        self.text_refiner = nn.ModuleList(
            [
                TextTokenRefinerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    ffn_dim=config.ffn_dim,
                    qk_norm=config.qk_norm,
                )
                for _ in range(config.num_refiner_layers)
            ]
        )

        if config.shared_adaln:
            self.shared_adaLN_txt = nn.Linear(config.hidden_size, 6 * config.hidden_size)
            self.shared_adaLN_vid = nn.Linear(config.hidden_size, 6 * config.hidden_size)
            nn.init.zeros_(self.shared_adaLN_txt.weight)
            nn.init.zeros_(self.shared_adaLN_txt.bias)
            nn.init.zeros_(self.shared_adaLN_vid.weight)
            nn.init.zeros_(self.shared_adaLN_vid.bias)

        if config.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, config.num_register_tokens, config.hidden_size) * 0.02
            )

        self.blocks = nn.ModuleList(
            [
                VietH3SingleStreamBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    ffn_dim=config.ffn_dim,
                    qk_norm=config.qk_norm,
                    shared_adaln=config.shared_adaln,
                )
                for _ in range(config.num_layers)
            ]
        )

        if config.use_long_skips:
            n_half = config.num_layers // 2
            self.long_skip_projs = nn.ModuleList(
                [nn.Linear(2 * config.hidden_size, config.hidden_size) for _ in range(n_half)]
            )
            for proj in self.long_skip_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

        self.final_norm = RMSNorm(config.hidden_size, elementwise_affine=False)
        self.final_adaLN = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

        self.final_proj = nn.Linear(config.hidden_size, patch_vol * config.in_channels)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _build_masks(
        self,
        text_mask: Optional[Tensor],
        batch_size: int,
        l_txt: int,
        l_vid: int,
        device: torch.device,
        num_reg: int = 0,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if text_mask is None:
            return None, None
        txt_valid = text_mask.to(dtype=torch.bool, device=device)
        # Ensure at least one valid key per batch item to avoid NaN in softmax
        empty_rows = ~txt_valid.any(dim=1)
        if empty_rows.any():
            txt_valid = txt_valid.clone()
            txt_valid[empty_rows, 0] = True
        txt_attn_mask = txt_valid[:, None, None, :]  # (B, 1, 1, L_txt)
        parts = [txt_valid]
        if num_reg > 0:
            parts.append(torch.ones((batch_size, num_reg), dtype=torch.bool, device=device))
        parts.append(torch.ones((batch_size, l_vid), dtype=torch.bool, device=device))
        full_attn_mask = torch.cat(parts, dim=1)[:, None, None, :]
        return txt_attn_mask, full_attn_mask

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        text_embeds: Tensor,
        text_mask: Optional[Tensor] = None,
        frame0_latent: Optional[Tensor] = None,
        i2v_mask: Optional[Tensor] = None,
        pose_latent: Optional[Tensor] = None,
        return_mid_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass predicting Rectified Flow velocity.

        Args:
            x: Noisy video latent of shape (B, C, T, H, W).
            timesteps: Flow timesteps in [0, 1] of shape (B,).
            text_embeds: Text encoder hidden states of shape (B, L_txt, text_dim).
            text_mask: Optional boolean mask of shape (B, L_txt), True = valid token.
            frame0_latent: Optional clean Frame-0 anchor latent (B, C, 1 or T, H, W).
            i2v_mask: Optional conditioning mask (B,) or (B, 1, 1 or T, H, W).
            pose_latent: Optional aligned pose latent (B, pose_channels, T, H, W).

        Returns:
            Predicted velocity of shape (B, C, T, H, W).
        """
        b, c, t, h, w = x.shape
        pt, ph, pw = self.config.patch_size
        if t % pt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(
                f"Latent shape {(t, h, w)} must be divisible by patch_size {self.config.patch_size}"
            )

        patch_in = x
        if self.config.use_i2v_cond:
            if frame0_latent is None:
                f0 = torch.zeros((b, c, t, h, w), device=x.device, dtype=x.dtype)
                m_default = 0.0
            else:
                f0 = frame0_latent.to(device=x.device, dtype=x.dtype)
                if f0.shape[2] == 1 and t > 1:
                    f0 = f0.expand(b, c, t, h, w)
                m_default = 1.0
            if i2v_mask is None:
                m = torch.full((b, 1, t, h, w), m_default, device=x.device, dtype=x.dtype)
            else:
                m = i2v_mask.to(device=x.device, dtype=x.dtype)
                if m.ndim == 1:
                    m = m.view(b, 1, 1, 1, 1)
                m = m.expand(b, 1, t, h, w)
            patch_in = torch.cat([patch_in, f0, m], dim=1)

        if self.config.use_pose_cond:
            if pose_latent is None:
                pose = torch.zeros(
                    (b, self.config.pose_channels, t, h, w), device=x.device, dtype=x.dtype
                )
            else:
                pose = pose_latent.to(device=x.device, dtype=x.dtype)
                if pose.shape[2] == 1 and t > 1:
                    pose = pose.expand(b, self.config.pose_channels, t, h, w)
            patch_in = torch.cat([patch_in, pose], dim=1)

        # Patchify video (+ optional I2V/Pose channels) -> (B, L_vid, hidden_size)
        vid = self.patch_embed(patch_in)
        t_p, h_p, w_p = vid.shape[2], vid.shape[3], vid.shape[4]
        video_h = vid.flatten(2).transpose(1, 2).contiguous()
        l_vid = video_h.shape[1]

        # Project text embeddings -> (B, L_txt, hidden_size)
        text_h = self.text_norm(self.text_proj(text_embeds.to(x.dtype)))
        l_txt = text_h.shape[1]

        temb = self.t_embedder(timesteps, dtype=x.dtype)
        txt_attn_mask, full_attn_mask = self._build_masks(
            text_mask, b, l_txt, l_vid, device=x.device, num_reg=self.config.num_register_tokens
        )

        # Refine text tokens with 1D RoPE
        txt_cos, txt_sin = build_1d_rope_freqs(
            l_txt,
            self.config.attention_head_dim,
            theta=self.config.rope_theta,
            device=x.device,
        )
        for ref_block in self.text_refiner:
            if self.gradient_checkpointing and self.training:
                text_h = checkpoint(
                    ref_block,
                    text_h,
                    temb,
                    txt_cos,
                    txt_sin,
                    txt_attn_mask,
                    use_reentrant=False,
                )
            else:
                text_h = ref_block(text_h, temb, txt_cos, txt_sin, txt_attn_mask)

        # Zero out masked padded text tokens before entering main single-stream blocks
        if text_mask is not None:
            text_h = text_h * text_mask.to(dtype=text_h.dtype, device=x.device).unsqueeze(-1)

        # Append learnable register tokens to absorb global outlier features away from blank walls
        if self.config.num_register_tokens > 0:
            reg = self.register_tokens.expand(b, -1, -1).to(dtype=text_h.dtype)
            text_h = torch.cat([text_h, reg], dim=1)

        # Build 3D RoPE for video tokens
        rope_cos, rope_sin = build_3d_rope_freqs(
            t_p,
            h_p,
            w_p,
            self.config.attention_head_dim,
            theta=self.config.rope_theta,
            device=x.device,
        )

        shared_mod_txt: Optional[Tensor] = None
        shared_mod_vid: Optional[Tensor] = None
        if self.config.shared_adaln:
            t_act = F.silu(temb)
            shared_mod_txt = self.shared_adaLN_txt(t_act)
            shared_mod_vid = self.shared_adaLN_vid(t_act)

        n_layers = len(self.blocks)
        n_half = n_layers // 2
        dec_start = (n_layers + 1) // 2
        mid_idx = max(0, n_half - 1)
        encoder_skips: list[Tensor] = []
        mid_video_h: Optional[Tensor] = None

        for idx, block in enumerate(self.blocks):
            if self.config.use_long_skips and idx < n_half:
                encoder_skips.append(video_h)

            if self.gradient_checkpointing and self.training:
                text_h, video_h = checkpoint(
                    block,
                    text_h,
                    video_h,
                    temb,
                    rope_cos,
                    rope_sin,
                    full_attn_mask,
                    shared_mod_txt,
                    shared_mod_vid,
                    use_reentrant=False,
                )
            else:
                text_h, video_h = block(
                    text_h,
                    video_h,
                    temb,
                    rope_cos,
                    rope_sin,
                    full_attn_mask,
                    shared_mod_txt,
                    shared_mod_vid,
                )

            if self.config.use_long_skips and idx >= dec_start:
                skip_h = encoder_skips[n_layers - 1 - idx]
                proj = self.long_skip_projs[idx - dec_start]
                video_h = video_h + proj(torch.cat([video_h, skip_h], dim=-1))

            if return_mid_features and idx == mid_idx:
                mid_video_h = video_h.mean(dim=1)

        shift, scale = self.final_adaLN(F.silu(temb)).chunk(2, dim=-1)
        out = _modulate(self.final_norm(video_h), shift, scale)
        out = self.final_proj(out)  # (B, t_p*h_p*w_p, pt*ph*pw*C)

        # Unpatchify back to (B, C, T, H, W)
        out = out.view(b, t_p, h_p, w_p, c, pt, ph, pw)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        v_pred = out.view(b, c, t, h, w)
        if return_mid_features:
            assert mid_video_h is not None
            return v_pred, mid_video_h
        return v_pred
