"""Tests for Shared AdaLN, Register Tokens, Long Skip Connections, and Dispersive Loss."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel
from vieth3.models.flow_matching import RectifiedFlowScheduler
from vieth3.training.train import train


def test_shared_adaln_reduces_params_and_matches_estimate() -> None:
    base_cfg = VietH3Config(
        name="base",
        in_channels=16,
        hidden_size=256,
        num_layers=8,
        num_refiner_layers=2,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_dim=512,
        text_dim=128,
        shared_adaln=False,
    )
    shared_cfg = VietH3Config(
        name="shared",
        in_channels=16,
        hidden_size=256,
        num_layers=8,
        num_refiner_layers=2,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_dim=512,
        text_dim=128,
        shared_adaln=True,
        num_register_tokens=8,
        use_long_skips=True,
    )
    m_base = VietH3Transformer3DModel(base_cfg)
    m_shared = VietH3Transformer3DModel(shared_cfg)

    p_base = m_base.count_parameters()
    p_shared = m_shared.count_parameters()
    assert p_base == base_cfg.estimate_num_params()
    assert p_shared == shared_cfg.estimate_num_params()
    assert p_shared < p_base * 0.72


def test_register_tokens_preserve_padding_mask_invariance() -> None:
    torch.manual_seed(0)
    cfg = VietH3Config(
        name="reg-mask-test",
        in_channels=8,
        hidden_size=128,
        num_layers=4,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=32,
        ffn_dim=256,
        text_dim=64,
        max_text_len=8,
        shared_adaln=True,
        num_register_tokens=8,
        use_long_skips=True,
    )
    model = VietH3Transformer3DModel(cfg)
    torch.nn.init.normal_(model.final_proj.weight, std=0.02)
    torch.nn.init.normal_(model.shared_adaLN_txt.weight, std=0.02)
    torch.nn.init.normal_(model.shared_adaLN_vid.weight, std=0.02)

    x = torch.randn(1, 8, 2, 4, 4)
    t = torch.tensor([0.5])
    mask = torch.tensor([[True, True, True, False, False, False]])
    txt_a = torch.randn(1, 6, 64, requires_grad=True)
    txt_b = txt_a.detach().clone()
    txt_b[:, 3:, :] = 123.0

    out_a = model(x, t, txt_a, text_mask=mask)
    out_b = model(x, t, txt_b, text_mask=mask)
    assert out_a.shape == x.shape
    assert torch.allclose(out_a, out_b, atol=1e-5)

    out_a.sum().backward()
    assert txt_a.grad is not None
    assert torch.allclose(txt_a.grad[:, 3:, :], torch.zeros_like(txt_a.grad[:, 3:, :]), atol=1e-7)
    assert txt_a.grad[:, :3, :].abs().sum() > 0


def test_long_skips_even_and_odd_layers_receive_gradients() -> None:
    for n_layers in (4, 5):
        cfg = VietH3Config(
            name=f"skips-{n_layers}",
            in_channels=8,
            hidden_size=128,
            num_layers=n_layers,
            num_refiner_layers=1,
            num_attention_heads=4,
            attention_head_dim=32,
            ffn_dim=256,
            text_dim=64,
            use_long_skips=True,
        )
        model = VietH3Transformer3DModel(cfg)
        torch.nn.init.normal_(model.final_proj.weight, std=0.02)
        for b in model.blocks:
            torch.nn.init.normal_(b.adaLN_vid.weight, std=0.02)

        x = torch.randn(2, 8, 2, 4, 4)
        t = torch.tensor([0.3, 0.7])
        txt = torch.randn(2, 6, 64)
        out = model(x, t, txt)
        assert out.shape == x.shape
        out.pow(2).mean().backward()
        assert len(model.long_skip_projs) == n_layers // 2
        for proj in model.long_skip_projs:
            assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0


def test_dispersive_loss_behavior_and_training_integration(tmp_path: Path) -> None:
    # B=1 -> 0.0
    assert RectifiedFlowScheduler.dispersive_loss(torch.randn(1, 32)).item() == 0.0

    # Collapsed identical embeddings -> higher penalty than orthogonal embeddings
    collapsed = torch.ones(4, 32)
    orthogonal = torch.eye(4, 32)
    loss_col = RectifiedFlowScheduler.dispersive_loss(collapsed, temperature=0.5).item()
    loss_orth = RectifiedFlowScheduler.dispersive_loss(orthogonal, temperature=0.5).item()
    assert loss_col > loss_orth

    cfg = VietH3Config(
        name="dense-smoke",
        in_channels=8,
        patch_size=(1, 2, 2),
        hidden_size=128,
        num_layers=4,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=32,
        ffn_dim=256,
        text_dim=64,
        max_text_len=8,
        vae_type="dummy",
        use_i2v_cond=True,
        shared_adaln=True,
        num_register_tokens=8,
        use_long_skips=True,
    )
    cfg_path = tmp_path / "model.yaml"
    cfg.to_yaml(cfg_path)

    cache_dir = tmp_path / "cache"
    shards_dir = cache_dir / "shards"
    shards_dir.mkdir(parents=True)
    torch.manual_seed(7)
    base_lat = torch.randn(8, 2, 4, 4) * 0.4
    tensors = {}
    rows = []
    for i in range(4):
        k = f"s_{i}"
        lat = base_lat + 0.1 * i
        tensors[f"{k}.latent"] = lat
        tensors[f"{k}.frame0_latent"] = lat[:, :1].clone()
        tensors[f"{k}.text_embeds"] = torch.randn(8, 64)
        tensors[f"{k}.text_mask"] = torch.ones(8, dtype=torch.uint8)
        rows.append(
            {"key": k, "shard": "shards/shard_00000.safetensors", "t_lat": 2, "h_lat": 4, "w_lat": 4}
        )
    save_file(tensors, str(shards_dir / "shard_00000.safetensors"))
    idx_path = cache_dir / "index.jsonl"
    idx_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary = train(
        {
            "model_config": str(cfg_path),
            "cache_index": str(idx_path),
            "output_dir": str(tmp_path / "out"),
            "strategy": "single",
            "bf16": False,
            "batch_size": 2,
            "max_steps": 15,
            "warmup_steps": 2,
            "lr": 1e-3,
            "cfg_dropout": 0.0,
            "i2v_dropout": 0.0,
            "dispersive_weight": 0.1,
            "dispersive_temp": 0.5,
            "use_ema": True,
            "log_every": 5,
            "save_every": 15,
            "device": "cpu",
        }
    )
    assert summary["final_step"] == 15
    assert summary["final_loss"] < summary["initial_loss"]
