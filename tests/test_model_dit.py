"""Unit tests for VietH3Transformer3DModel and 3D RoPE."""

import torch

from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel
from vieth3.models.rope3d import build_3d_rope_freqs, compute_rope_axes


def _debug_config() -> VietH3Config:
    return VietH3Config(
        name="test-debug",
        in_channels=8,
        patch_size=(1, 2, 2),
        hidden_size=128,
        num_layers=2,
        num_refiner_layers=2,
        num_attention_heads=4,
        attention_head_dim=32,
        ffn_dim=256,
        text_dim=64,
        max_text_len=16,
    )


def test_rope3d_axes_and_shapes() -> None:
    for head_dim in (32, 64, 128):
        dt, dh, dw = compute_rope_axes(head_dim)
        assert dt + dh + dw == head_dim
        assert dt % 2 == 0 and dh % 2 == 0 and dw % 2 == 0
    cos, sin = build_3d_rope_freqs(3, 4, 5, head_dim=64)
    assert cos.shape == (60, 64)
    assert sin.shape == (60, 64)


def test_dit_forward_image_and_video_shapes() -> None:
    cfg = _debug_config()
    model = VietH3Transformer3DModel(cfg)
    model.eval()

    txt = torch.randn(2, 10, cfg.text_dim)
    mask = torch.ones(2, 10, dtype=torch.bool)
    t = torch.tensor([0.2, 0.8])

    # Image batch (T = 1)
    x_img = torch.randn(2, cfg.in_channels, 1, 8, 8)
    out_img = model(x_img, t, txt, text_mask=mask)
    assert out_img.shape == x_img.shape

    # Video batch (T = 6)
    x_vid = torch.randn(2, cfg.in_channels, 6, 8, 12)
    out_vid = model(x_vid, t, txt, text_mask=mask)
    assert out_vid.shape == x_vid.shape


def test_padding_mask_invariance_and_zero_grad() -> None:
    torch.manual_seed(0)
    cfg = _debug_config()
    model = VietH3Transformer3DModel(cfg)
    # Re-init final_proj non-zero so outputs & gradients are non-trivial
    torch.nn.init.normal_(model.final_proj.weight, std=0.02)
    for b in model.blocks:
        torch.nn.init.normal_(b.adaLN_txt.weight, std=0.02)
        torch.nn.init.normal_(b.adaLN_vid.weight, std=0.02)
    for r in model.text_refiner:
        torch.nn.init.normal_(r.adaLN.weight, std=0.02)

    x = torch.randn(1, cfg.in_channels, 2, 4, 4)
    t = torch.tensor([0.5])
    mask = torch.tensor([[True, True, True, False, False, False]])

    txt_a = torch.randn(1, 6, cfg.text_dim, requires_grad=True)
    txt_b = txt_a.detach().clone()
    txt_b[:, 3:, :] = 999.0  # radically mutate masked padding positions

    out_a = model(x, t, txt_a, text_mask=mask)
    out_b = model(x, t, txt_b, text_mask=mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)

    out_a.sum().backward()
    assert txt_a.grad is not None
    assert torch.allclose(txt_a.grad[:, 3:, :], torch.zeros_like(txt_a.grad[:, 3:, :]), atol=1e-7)
    assert txt_a.grad[:, :3, :].abs().sum() > 0


def test_gradient_checkpointing_matches_standard() -> None:
    torch.manual_seed(7)
    cfg = _debug_config()
    model = VietH3Transformer3DModel(cfg)
    torch.nn.init.normal_(model.final_proj.weight, std=0.02)
    model.train()

    x = torch.randn(1, cfg.in_channels, 2, 4, 4)
    t = torch.tensor([0.4])
    txt = torch.randn(1, 6, cfg.text_dim)
    mask = torch.ones(1, 6, dtype=torch.bool)

    model.enable_gradient_checkpointing(False)
    out_std = model(x, t, txt, text_mask=mask)

    model.enable_gradient_checkpointing(True)
    out_ckpt = model(x, t, txt, text_mask=mask)
    assert torch.allclose(out_std, out_ckpt, atol=1e-6)
