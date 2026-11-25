"""Exponential Moving Average (EMA) weights tracker."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EMAModel:
    """Maintains float32 shadow weights updated with exponential moving average."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
        warmup_steps: int = 100,
    ) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        raw_model = getattr(model, "module", model)
        self.shadow: dict[str, Tensor] = {
            name: param.detach().to(dtype=torch.float32, device="cpu").clone()
            for name, param in raw_model.named_parameters()
            if param.requires_grad
        }

    def get_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = self.get_decay()
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            p_cpu = param.detach().to(dtype=torch.float32, device="cpu")
            self.shadow[name].mul_(d).add_(p_cpu, alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        raw_model = getattr(model, "module", model)
        for name, param in raw_model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    def state_dict(self) -> dict[str, Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(dtype=torch.float32, device="cpu"))
