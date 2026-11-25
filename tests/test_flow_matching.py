"""Tests for RectifiedFlowScheduler and 2x H100 memory/budget estimator."""

from pathlib import Path

import torch

from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel
from vieth3.models.flow_matching import RectifiedFlowScheduler
from vieth3.training.memory_budget import estimate_memory_and_budget


def _tiny_model() -> tuple[VietH3Config, VietH3Transformer3DModel]:
    cfg = VietH3Config(
        name="tiny-flow",
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=64,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=2,
        attention_head_dim=32,
        ffn_dim=128,
        text_dim=32,
        max_text_len=8,
    )
    return cfg, VietH3Transformer3DModel(cfg)


def test_flow_matching_endpoints_and_loss() -> None:
    cfg, model = _tiny_model()
    x0 = torch.randn(2, cfg.in_channels, 2, 4, 4)
    noise = torch.randn_like(x0)

    xt_0 = RectifiedFlowScheduler.add_noise(x0, noise, torch.zeros(2))
    xt_1 = RectifiedFlowScheduler.add_noise(x0, noise, torch.ones(2))
    assert torch.allclose(xt_0, x0)
    assert torch.allclose(xt_1, noise)

    txt = torch.randn(2, 8, cfg.text_dim)
    mask = torch.ones(2, 8, dtype=torch.bool)
    loss, metrics = RectifiedFlowScheduler.training_loss(model, x0, txt, text_mask=mask)
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert 0.0 < metrics["t_mean"] < 1.0
    loss.backward()


def test_flow_euler_and_heun_samplers() -> None:
    cfg, model = _tiny_model()
    txt = torch.randn(1, 8, cfg.text_dim)
    neg = torch.zeros_like(txt)
    mask = torch.ones(1, 8, dtype=torch.bool)

    for solver in ("euler", "heun"):
        out = RectifiedFlowScheduler.sample(
            model,
            shape=(1, cfg.in_channels, 2, 4, 4),
            text_embeds=txt,
            text_mask=mask,
            neg_text_embeds=neg,
            neg_text_mask=mask,
            num_steps=4,
            cfg_scale=3.0,
            solver=solver,  # type: ignore[arg-type]
        )
        assert out.shape == (1, cfg.in_channels, 2, 4, 4)
        assert torch.isfinite(out).all()


def test_memory_budget_configs_fit_2xh100() -> None:
    repo = Path(__file__).resolve().parents[1]
    for yaml_name in ("model_vieth3_mini_500m.yaml", "model_vieth3_base_1_5b.yaml"):
        cfg = VietH3Config.from_yaml(repo / "configs" / yaml_name)
        report = estimate_memory_and_budget(cfg, gpus=2, height=480, width=832, frames=56, batch_size=2)
        assert report.exact_params > 500_000_000
        assert report.fits_80gb_h100, f"{yaml_name} exceeded 80GB: {report.total_fsdp_ckpt_gb_per_gpu}"
