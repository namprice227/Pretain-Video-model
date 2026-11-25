"""Tests for H3 frame grid, Vietnamese text encoder, and bucketed dataset."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vieth3.data.dataset import BucketBatchSampler, CachedLatentDataset
from vieth3.data.frame_grid import (
    select_aspect_bucket,
    snap_frame_count,
    snap_resolution,
    valid_h3_frames,
)
from vieth3.models.vae_text import DummyVietnameseTextEncoder, DummyVideoVAE


def test_h3_frame_grid_and_buckets() -> None:
    grid = valid_h3_frames(130)
    assert grid[:7] == [5, 22, 39, 56, 73, 90, 107]
    assert snap_frame_count(1, grid="h3", allow_image=True) == 1
    assert snap_frame_count(60, grid="h3") == 56
    assert snap_frame_count(243, grid="h3") == 243

    assert snap_resolution(490, 850, multiple=32) == (480, 832)
    assert select_aspect_bucket(1080, 1920, stage="480p") == (480, 832)
    assert select_aspect_bucket(1920, 1080, stage="480p") == (832, 480)


def test_dummy_vae_and_vietnamese_text_encoder() -> None:
    vae = DummyVideoVAE(latent_channels=24, spatial_stride=16, temporal_stride=4)
    vid = torch.randn(1, 3, 22, 128, 128).clamp(-1, 1)
    lat = vae.encode(vid)
    assert lat.shape == (1, 24, 6, 8, 8)
    recon = vae.decode(lat)
    assert recon.shape == (1, 3, 21, 128, 128) or recon.shape[3:] == (128, 128)

    enc = DummyVietnameseTextEncoder(text_dim=128, max_length=16)
    emb, mask = enc.encode(["Phở bò Hà Nội buổi sáng", "Áo dài phố cổ Hội An"])
    assert emb.shape == (2, 16, 128)
    assert mask[0, :6].all() and not mask[0, 6:].any()
    assert not torch.allclose(emb[0], emb[1])


def test_bucket_batch_sampler_mixed_image_and_video(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    tensors = {}
    rows = []
    # 3 images (t_lat=1) and 3 videos (t_lat=6)
    for i in range(6):
        is_img = i < 3
        t_lat = 1 if is_img else 6
        key = f"sample_{i:04d}"
        tensors[f"{key}.latent"] = torch.randn(8, t_lat, 8, 8)
        tensors[f"{key}.text_embeds"] = torch.randn(12, 64)
        tensors[f"{key}.text_mask"] = torch.ones(12, dtype=torch.uint8)
        rows.append(
            {
                "key": key,
                "shard": "shards/shard_00000.safetensors",
                "t_lat": t_lat,
                "h_lat": 8,
                "w_lat": 8,
            }
        )
    save_file(tensors, str(shards_dir / "shard_00000.safetensors"))
    index_path = tmp_path / "index.jsonl"
    index_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ds = CachedLatentDataset(index_path, cfg_dropout=0.0)
    sampler = BucketBatchSampler(ds, batch_size=2, shuffle=False, num_replicas=1, rank=0)
    batches = list(sampler)
    # 2 image batches ([0,1], [2]) + 2 video batches ([3,4], [5]) = 4 batches
    assert len(batches) == 4
    for batch in batches:
        shapes = {ds[idx]["latent"].shape for idx in batch}
        assert len(shapes) == 1  # zero padding within batch!

    # Distributed rank split check
    sampler_r0 = BucketBatchSampler(ds, batch_size=2, shuffle=False, num_replicas=2, rank=0)
    sampler_r1 = BucketBatchSampler(ds, batch_size=2, shuffle=False, num_replicas=2, rank=1)
    assert len(list(sampler_r0)) == 2 and len(list(sampler_r1)) == 2
