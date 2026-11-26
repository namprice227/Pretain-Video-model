"""Cached latent dataset and resolution/frame bucket batch sampler."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import torch
from safetensors import safe_open
from torch import Tensor
from torch.utils.data import Dataset, Sampler


class CachedLatentDataset(Dataset):
    """Loads pre-cached VAE latents and Vietnamese text embeddings from .safetensors shards."""

    def __init__(
        self,
        index_path: str | Path,
        cfg_dropout: float = 0.1,
        i2v_dropout: float = 0.1,
        pose_dropout: float = 0.15,
        seed: int = 42,
        max_t_lat: Optional[int] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.root_dir = self.index_path.parent
        self.cfg_dropout = cfg_dropout
        self.i2v_dropout = i2v_dropout
        self.pose_dropout = pose_dropout
        self.max_t_lat = int(max_t_lat) if max_t_lat is not None else None
        self.rng = random.Random(seed)
        self.samples: list[dict] = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError(f"Empty latent cache index: {self.index_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_bucket_key(self, idx: int) -> tuple[int, int, int]:
        s = self.samples[idx]
        t_lat = int(s["t_lat"])
        if self.max_t_lat is not None:
            t_lat = min(t_lat, self.max_t_lat)
        return (t_lat, int(s["h_lat"]), int(s["w_lat"]))

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = self.samples[idx]
        shard_path = self.root_dir / item["shard"]
        key = item["key"]
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            avail_keys = set(f.keys())
            latent = f.get_tensor(f"{key}.latent")
            frame0_latent = (
                f.get_tensor(f"{key}.frame0_latent")
                if f"{key}.frame0_latent" in avail_keys
                else latent[:, :1].clone()
            )
            text_embeds = f.get_tensor(f"{key}.text_embeds")
            text_mask = f.get_tensor(f"{key}.text_mask").to(torch.bool)
            pose_latent = (
                f.get_tensor(f"{key}.pose_latent")
                if f"{key}.pose_latent" in avail_keys
                else None
            )

        if self.max_t_lat is not None and latent.shape[1] > self.max_t_lat:
            latent = latent[:, : self.max_t_lat].contiguous()
            if pose_latent is not None and pose_latent.shape[1] > self.max_t_lat:
                pose_latent = pose_latent[:, : self.max_t_lat].contiguous()

        if self.cfg_dropout > 0.0 and self.rng.random() < self.cfg_dropout:
            text_embeds = torch.zeros_like(text_embeds)
            text_mask = torch.zeros_like(text_mask)
            text_mask[0] = True

        if self.i2v_dropout > 0.0 and self.rng.random() < self.i2v_dropout:
            frame0_latent = torch.zeros_like(frame0_latent)
            i2v_mask = torch.zeros((1, 1, 1, 1), dtype=latent.dtype)
        else:
            i2v_mask = torch.ones((1, 1, 1, 1), dtype=latent.dtype)

        out: dict[str, Tensor] = {
            "latent": latent,
            "frame0_latent": frame0_latent,
            "i2v_mask": i2v_mask,
            "text_embeds": text_embeds,
            "text_mask": text_mask,
        }
        if pose_latent is not None:
            if self.pose_dropout > 0.0 and self.rng.random() < self.pose_dropout:
                pose_latent = torch.zeros_like(pose_latent)
            out["pose_latent"] = pose_latent
        return out


class BucketBatchSampler(Sampler[list[int]]):
    """Groups samples by (t_lat, h_lat, w_lat) so batches require zero latent padding."""

    def __init__(
        self,
        dataset: CachedLatentDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            self.buckets[dataset.get_bucket_key(i)].append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_all_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        all_batches: list[list[int]] = []
        for key in sorted(self.buckets.keys()):
            indices = list(self.buckets[key])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        all_batches = self._build_all_batches()
        if self.num_replicas > 1:
            usable = (len(all_batches) // self.num_replicas) * self.num_replicas
            if usable == 0 and all_batches:
                # Pad by repeating batches so distributed ranks never deadlock
                while len(all_batches) < self.num_replicas:
                    all_batches.extend(all_batches)
                usable = self.num_replicas
            all_batches = all_batches[:usable]
            rank_batches = all_batches[self.rank :: self.num_replicas]
            return iter(rank_batches)
        return iter(all_batches)

    def __len__(self) -> int:
        all_batches = self._build_all_batches()
        if self.num_replicas > 1:
            if len(all_batches) < self.num_replicas and all_batches:
                return 1
            return len(all_batches) // self.num_replicas
        return len(all_batches)
