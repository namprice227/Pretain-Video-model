"""Distributed Rectified Flow training loop for VietH3 (CPU, single GPU, DDP, FSDP)."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

from vieth3.data.dataset import BucketBatchSampler, CachedLatentDataset
from vieth3.models.config import VietH3Config
from vieth3.models.dit import VietH3Transformer3DModel
from vieth3.models.flow_matching import RectifiedFlowScheduler
from vieth3.training.ema import EMAModel


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= total_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _setup_distributed(device_arg: Optional[str] = None) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    is_dist = world_size > 1
    if is_dist and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend=backend)
    return is_dist, rank, local_rank, world_size, device


def _save_checkpoint(
    out_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMAModel],
    strategy: str,
    rank: int,
) -> Path:
    ckpt_dir = out_dir / "checkpoints" / f"step_{step:06d}"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            state_dict = model.state_dict()
        if rank == 0:
            save_file(
                {k: v.contiguous() for k, v in state_dict.items()},
                str(ckpt_dir / "model.safetensors"),
            )
    else:
        if rank == 0:
            raw_model = getattr(model, "module", model)
            state_dict = {
                k: v.detach().cpu().contiguous() for k, v in raw_model.state_dict().items()
            }
            save_file(state_dict, str(ckpt_dir / "model.safetensors"))

    if rank == 0:
        if ema is not None:
            save_file(ema.state_dict(), str(ckpt_dir / "ema.safetensors"))
        torch.save(
            {"step": step, "optimizer": optimizer.state_dict()},
            ckpt_dir / "training_state.pt",
        )
        (out_dir / "checkpoints" / "latest").write_text(str(ckpt_dir.resolve()), encoding="utf-8")
    return ckpt_dir


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the VietH3 training loop and return final training summary."""
    is_dist, rank, _local_rank, world_size, device = _setup_distributed(cfg.get("device"))
    seed = int(cfg.get("seed", 42)) + rank
    torch.manual_seed(seed)

    out_dir = Path(cfg["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = VietH3Config.from_yaml(cfg["model_config"])
    model = VietH3Transformer3DModel(model_cfg)

    use_bf16 = bool(cfg.get("bf16", device.type == "cuda")) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    if bool(cfg.get("gradient_checkpointing", False)):
        model.enable_gradient_checkpointing(True)

    strategy = str(cfg.get("strategy", "ddp")).lower()
    if strategy == "fsdp" and is_dist and device.type == "cuda":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools
        from vieth3.models.dit import VietH3SingleStreamBlock

        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={VietH3SingleStreamBlock},
        )
        model = FSDP(
            model.to(device),
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            device_id=device,
            use_orig_params=True,
        )
    else:
        strategy = "ddp" if is_dist else "single"
        model = model.to(device)
        if is_dist:
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    base_lr = float(cfg.get("lr", 1e-4))
    min_lr = float(cfg.get("min_lr", base_lr * 0.1))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    ema: Optional[EMAModel] = None
    if bool(cfg.get("use_ema", True)) and strategy != "fsdp":
        ema = EMAModel(
            model,
            decay=float(cfg.get("ema_decay", 0.9995)),
            warmup_steps=int(cfg.get("ema_warmup", 100)),
        )

    start_step = 0
    resume_path = cfg.get("resume")
    if resume_path:
        r_dir = Path(resume_path)
        raw_model = getattr(model, "module", model)
        raw_model.load_state_dict(load_file(str(r_dir / "model.safetensors"), device=str(device)))
        state = torch.load(r_dir / "training_state.pt", map_location="cpu", weights_only=True)
        start_step = int(state["step"])
        optimizer.load_state_dict(state["optimizer"])
        if ema is not None and (r_dir / "ema.safetensors").exists():
            ema.load_state_dict(load_file(str(r_dir / "ema.safetensors")))

    dataset = CachedLatentDataset(
        cfg["cache_index"],
        cfg_dropout=float(cfg.get("cfg_dropout", 0.1)),
        i2v_dropout=float(cfg.get("i2v_dropout", 0.1)),
        pose_dropout=float(cfg.get("pose_dropout", 0.15)),
        seed=seed,
    )
    batch_size = int(cfg.get("batch_size", 1))
    sampler = BucketBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_replicas=world_size,
        rank=rank,
        seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=int(cfg.get("num_workers", 0)))

    max_steps = int(cfg.get("max_steps", 100))
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    warmup_steps = int(cfg.get("warmup_steps", min(50, max_steps // 10)))
    log_every = max(1, int(cfg.get("log_every", 10)))
    save_every = int(cfg.get("save_every", max_steps))
    flow_shift = float(cfg.get("flow_shift", 3.0))
    dispersive_weight = float(cfg.get("dispersive_weight", 0.0))
    dispersive_temp = float(cfg.get("dispersive_temp", 0.5))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    metrics_file = out_dir / "metrics.jsonl"
    loss_history: list[float] = []

    model.train()
    step = start_step
    epoch = 0
    micro_step = 0
    accum_loss = 0.0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            x0 = batch["latent"].to(device=device, dtype=amp_dtype)
            txt_emb = batch["text_embeds"].to(device=device, dtype=amp_dtype)
            txt_mask = batch["text_mask"].to(device=device)
            frame0_lat = (
                batch["frame0_latent"].to(device=device, dtype=amp_dtype)
                if model_cfg.use_i2v_cond and "frame0_latent" in batch
                else None
            )
            i2v_mask = (
                batch["i2v_mask"].to(device=device, dtype=amp_dtype)
                if model_cfg.use_i2v_cond and "i2v_mask" in batch
                else None
            )
            pose_lat = (
                batch["pose_latent"].to(device=device, dtype=amp_dtype)
                if model_cfg.use_pose_cond and "pose_latent" in batch
                else None
            )

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and use_bf16)
            ):
                loss, _ = RectifiedFlowScheduler.training_loss(
                    model,
                    x0=x0,
                    text_embeds=txt_emb,
                    text_mask=txt_mask,
                    shift=flow_shift,
                    frame0_latent=frame0_lat,
                    i2v_mask=i2v_mask,
                    pose_latent=pose_lat,
                    dispersive_weight=dispersive_weight,
                    dispersive_temp=dispersive_temp,
                )
                scaled_loss = loss / grad_accum

            scaled_loss.backward()
            accum_loss += float(loss.detach().item()) / grad_accum
            micro_step += 1

            if micro_step % grad_accum == 0:
                lr = _cosine_lr(step, max_steps, warmup_steps, base_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if strategy == "fsdp":
                    model.clip_grad_norm_(max_grad_norm)  # type: ignore[operator]
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

                step += 1
                loss_history.append(accum_loss)

                if rank == 0 and (step % log_every == 0 or step == 1 or step == max_steps):
                    elapsed = time.perf_counter() - t0
                    row = {
                        "step": step,
                        "loss": round(accum_loss, 6),
                        "lr": lr,
                        "elapsed_s": round(elapsed, 2),
                    }
                    with open(metrics_file, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(row) + "\n")
                    print(f"[step {step:06d}/{max_steps}] loss={accum_loss:.5f} lr={lr:.2e} ({elapsed:.1f}s)")

                if save_every > 0 and (step % save_every == 0 or step == max_steps):
                    _save_checkpoint(out_dir, step, model, optimizer, ema, strategy, rank)

                accum_loss = 0.0
                if step >= max_steps:
                    break
        epoch += 1

    if is_dist:
        dist.barrier()

    return {
        "final_step": step,
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "checkpoint_dir": str((out_dir / "checkpoints" / f"step_{step:06d}").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VietH3 Video Diffusion Transformer.")
    parser.add_argument("--config", required=True, help="Training YAML config path.")
    parser.add_argument("--cache-index", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--strategy", choices=["ddp", "fsdp", "single"], default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for key in [
        "cache_index",
        "model_config",
        "output_dir",
        "max_steps",
        "batch_size",
        "grad_accum",
        "lr",
        "strategy",
        "resume",
        "device",
    ]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    train(cfg)


if __name__ == "__main__":
    main()
