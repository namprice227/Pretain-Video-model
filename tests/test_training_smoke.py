"""Integration test for VietH3 training, checkpointing, EMA, and resume."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vieth3.models.config import VietH3Config
from vieth3.training.train import train


def test_training_loss_decreases_and_resumes(tmp_path: Path) -> None:
    model_cfg = VietH3Config(
        name="smoke-test",
        in_channels=8,
        patch_size=(1, 2, 2),
        hidden_size=128,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=32,
        ffn_dim=256,
        text_dim=64,
        max_text_len=8,
        vae_type="dummy",
    )
    cfg_path = tmp_path / "model.yaml"
    model_cfg.to_yaml(cfg_path)

    cache_dir = tmp_path / "cache"
    shards_dir = cache_dir / "shards"
    shards_dir.mkdir(parents=True)

    torch.manual_seed(123)
    tensors = {}
    rows = []
    # Smooth low-variance structured latents + fixed evaluation probe
    base_lat = torch.randn(8, 2, 4, 4) * 0.5
    for i in range(4):
        key = f"s_{i}"
        tensors[f"{key}.latent"] = base_lat + 0.1 * i
        tensors[f"{key}.text_embeds"] = torch.randn(8, 64)
        tensors[f"{key}.text_mask"] = torch.ones(8, dtype=torch.uint8)
        rows.append(
            {
                "key": key,
                "shard": "shards/shard_00000.safetensors",
                "t_lat": 2,
                "h_lat": 4,
                "w_lat": 4,
            }
        )
    save_file(tensors, str(shards_dir / "shard_00000.safetensors"))
    index_path = cache_dir / "index.jsonl"
    index_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "train_out"
    res1 = train(
        {
            "model_config": str(cfg_path),
            "cache_index": str(index_path),
            "output_dir": str(out_dir),
            "strategy": "single",
            "bf16": False,
            "batch_size": 2,
            "max_steps": 25,
            "warmup_steps": 3,
            "lr": 2e-3,
            "cfg_dropout": 0.0,
            "use_ema": True,
            "log_every": 5,
            "save_every": 25,
            "device": "cpu",
        }
    )
    assert res1["final_step"] == 25
    assert res1["final_loss"] < res1["initial_loss"]
    ckpt_dir = Path(res1["checkpoint_dir"])
    assert (ckpt_dir / "model.safetensors").exists()
    assert (ckpt_dir / "ema.safetensors").exists()

    # Resume for 4 more steps
    res2 = train(
        {
            "model_config": str(cfg_path),
            "cache_index": str(index_path),
            "output_dir": str(out_dir),
            "strategy": "single",
            "bf16": False,
            "batch_size": 2,
            "max_steps": 29,
            "warmup_steps": 3,
            "lr": 2e-3,
            "cfg_dropout": 0.0,
            "use_ema": True,
            "log_every": 2,
            "save_every": 29,
            "resume": str(ckpt_dir),
            "device": "cpu",
        }
    )
    assert res2["final_step"] == 29
