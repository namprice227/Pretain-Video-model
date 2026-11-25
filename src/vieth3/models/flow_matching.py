"""Rectified Flow Matching scheduler and ODE sampler for VietH3.

Convention used uniformly across training and inference:
- x_0 = clean latent, x_1 ~ N(0, I) = Gaussian noise, t in [0, 1]
- Interpolation: x_t = (1 - t) * x_0 + t * x_1
- Target velocity: v_target = x_1 - x_0
- ODE sampling integrates from t = 1.0 down to t = 0.0 (dt < 0):
    x <- x + dt * v_pred
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RectifiedFlowScheduler:
    """Rectified Flow Matching training objective and inference ODE sampler."""

    @staticmethod
    def apply_timestep_shift(u: Tensor, shift: float = 3.0) -> Tensor:
        """SD3 / H3 resolution-dependent flow timestep shift."""
        if shift == 1.0:
            return u
        return (shift * u) / (1.0 + (shift - 1.0) * u)

    @classmethod
    def sample_timesteps(
        cls,
        batch_size: int,
        device: torch.device,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample logit-normal timesteps in (0, 1) with SD3/H3 shift."""
        z = torch.randn(batch_size, device=device, dtype=torch.float32) * logit_std + logit_mean
        u = torch.sigmoid(z)
        t = cls.apply_timestep_shift(u, shift=shift)
        return t.clamp(1e-5, 1.0 - 1e-5).to(dtype)

    @staticmethod
    def add_noise(x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Linear interpolation x_t = (1 - t) * x_0 + t * x_1."""
        t_view = t.view(-1, *([1] * (x0.ndim - 1))).to(dtype=x0.dtype, device=x0.device)
        return (1.0 - t_view) * x0 + t_view * noise

    @staticmethod
    def dispersive_loss(features: Tensor, temperature: float = 0.5) -> Tensor:
        """Self-contained representation dispersion regularizer (arXiv:2506.09027).

        Penalizes representation collapse across batch items without external teachers.
        Returns 0.0 when batch size < 2.
        """
        b = features.shape[0]
        if b < 2:
            return features.new_zeros(())
        z = F.normalize(features.to(torch.float32), dim=-1)
        dist_sq = torch.cdist(z, z, p=2.0).pow(2)
        eye = torch.eye(b, dtype=torch.bool, device=features.device)
        return torch.exp(-dist_sq / max(1e-4, temperature))[~eye].mean()

    @classmethod
    def training_loss(
        cls,
        model: nn.Module,
        x0: Tensor,
        text_embeds: Tensor,
        text_mask: Optional[Tensor] = None,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        noise: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        frame0_latent: Optional[Tensor] = None,
        i2v_mask: Optional[Tensor] = None,
        pose_latent: Optional[Tensor] = None,
        dispersive_weight: float = 0.0,
        dispersive_temp: float = 0.5,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute Rectified Flow velocity MSE loss (+ optional Dispersive Loss)."""
        if noise is None:
            noise = torch.randn_like(x0)
        if timesteps is None:
            timesteps = cls.sample_timesteps(
                x0.shape[0],
                device=x0.device,
                shift=shift,
                logit_mean=logit_mean,
                logit_std=logit_std,
                dtype=torch.float32,
            )

        xt = cls.add_noise(x0, noise, timesteps)
        v_target = (noise - x0).to(torch.float32)
        cond_kwargs: dict[str, Tensor] = {}
        if frame0_latent is not None:
            cond_kwargs["frame0_latent"] = frame0_latent
        if i2v_mask is not None:
            cond_kwargs["i2v_mask"] = i2v_mask
        if pose_latent is not None:
            cond_kwargs["pose_latent"] = pose_latent

        disp_val = 0.0
        if dispersive_weight > 0.0:
            v_pred_raw, mid_feat = model(
                xt,
                timesteps.to(xt.dtype),
                text_embeds,
                text_mask=text_mask,
                return_mid_features=True,
                **cond_kwargs,
            )
            v_pred = v_pred_raw.to(torch.float32)
            flow_loss = F.mse_loss(v_pred, v_target)
            d_loss = cls.dispersive_loss(mid_feat, temperature=dispersive_temp)
            loss = flow_loss + dispersive_weight * d_loss
            disp_val = float(d_loss.detach().item())
        else:
            v_pred = model(
                xt, timesteps.to(xt.dtype), text_embeds, text_mask=text_mask, **cond_kwargs
            ).to(torch.float32)
            loss = F.mse_loss(v_pred, v_target)

        metrics = {
            "loss": float(loss.detach().item()),
            "disp_loss": disp_val,
            "t_mean": float(timesteps.detach().mean().item()),
        }
        return loss, metrics

    @classmethod
    @torch.no_grad()
    def sample(
        cls,
        model: nn.Module,
        shape: tuple[int, ...],
        text_embeds: Tensor,
        text_mask: Optional[Tensor] = None,
        neg_text_embeds: Optional[Tensor] = None,
        neg_text_mask: Optional[Tensor] = None,
        num_steps: int = 30,
        cfg_scale: float = 5.0,
        shift: float = 3.0,
        solver: Literal["euler", "heun"] = "euler",
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        frame0_latent: Optional[Tensor] = None,
        i2v_mask: Optional[Tensor] = None,
        pose_latent: Optional[Tensor] = None,
    ) -> Tensor:
        """Integrate flow ODE from t = 1.0 (noise) to t = 0.0 (clean latent)."""
        dev = device or text_embeds.device
        dt_type = dtype or text_embeds.dtype
        b = shape[0]

        x = torch.randn(shape, generator=generator, device=dev, dtype=torch.float32).to(dt_type)
        u_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=dev, dtype=torch.float32)
        t_grid = cls.apply_timestep_shift(u_grid, shift=shift)

        use_cfg = cfg_scale != 1.0 and neg_text_embeds is not None

        def _cond_kwargs(dup: bool) -> dict[str, Tensor]:
            kw: dict[str, Tensor] = {}
            if frame0_latent is not None:
                kw["frame0_latent"] = (
                    torch.cat([frame0_latent, frame0_latent], dim=0) if dup else frame0_latent
                )
            if i2v_mask is not None:
                kw["i2v_mask"] = torch.cat([i2v_mask, i2v_mask], dim=0) if dup else i2v_mask
            if pose_latent is not None:
                kw["pose_latent"] = (
                    torch.cat([pose_latent, pose_latent], dim=0) if dup else pose_latent
                )
            return kw

        def _predict_v(x_cur: Tensor, t_scalar: Tensor) -> Tensor:
            t_vec = t_scalar.expand(b).to(x_cur.dtype)
            if not use_cfg:
                return model(x_cur, t_vec, text_embeds, text_mask=text_mask, **_cond_kwargs(False))
            x_in = torch.cat([x_cur, x_cur], dim=0)
            t_in = torch.cat([t_vec, t_vec], dim=0)
            txt_in = torch.cat([neg_text_embeds, text_embeds], dim=0)
            if text_mask is not None and neg_text_mask is not None:
                mask_in = torch.cat([neg_text_mask, text_mask], dim=0)
            elif text_mask is not None:
                mask_in = torch.cat(
                    [torch.ones_like(text_mask), text_mask], dim=0
                )
            else:
                mask_in = None
            v_uncond, v_cond = model(
                x_in, t_in, txt_in, text_mask=mask_in, **_cond_kwargs(True)
            ).chunk(2, dim=0)
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        for i in range(num_steps):
            t_cur = t_grid[i]
            t_next = t_grid[i + 1]
            dt = (t_next - t_cur).to(torch.float32)

            v1 = _predict_v(x, t_cur).to(torch.float32)
            if solver == "heun" and i < num_steps - 1:
                x_euler = (x.to(torch.float32) + dt * v1).to(dt_type)
                v2 = _predict_v(x_euler, t_next).to(torch.float32)
                x = (x.to(torch.float32) + dt * 0.5 * (v1 + v2)).to(dt_type)
            else:
                x = (x.to(torch.float32) + dt * v1).to(dt_type)

        return x
