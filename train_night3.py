"""Fresh warm-start heal of the compact-lane flagship, DDP, batched windows.

OPENLY OFF-CONTRACT: feasibility run of the 2026-07-20 compact-lane Package B
(gdnx_0720 tree). Reuses the campaign trainer (QwenHealTrainer,
distributed=True) so loss semantics, nonfinite guards, and accounting match
campaign runs, but uses a custom budget, batch size, band, and LR schedule.
Outputs live under ~/runs/night3 and must never feed promotion.

Phase 1 (fresh start, T=4096, B=4/rank):
  cd ~/gdnx_0720 && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=~/gdnx_0720 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/gdnx/.venv/bin/torchrun --nproc_per_node=2 --master_port=29531 \
      train_night3.py --updates 1000 --batch 4 --seq-len 4096 \
      --band ~/attn_data/data/gdnx_night3_band --out ~/runs/night3/phase1

Phase 2 (resume, T=32768, B=2/rank, judged retrieval band):
  ... train_night3.py --updates 100 --batch 2 --seq-len 32768 \
      --band-tensor ~/attn_data/data/retrieval/judged_blocks_32k_v1/train_semantic_retrieval_32k_judged_v1.pt \
      --resume ~/runs/night3/phase1/ckpt-latest.pt --lr 1e-4 --warmup 0 \
      --checkpoint-every 16 --out ~/runs/night3/phase2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

MODEL = str(Path.home() / "models" / "qwen3_5_0_8b")


def log(msg: str) -> None:
    if dist.get_rank() == 0:
        print(f"[night3] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True, help="windows per rank per update")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--band", type=Path, default=None,
                        help="band dir with shard_main_*.pt + example_ids.json")
    parser.add_argument("--band-tensor", type=Path, default=None,
                        help="single [N, T] tensor file (ids derived from row index)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint from a previous phase (fresh start if omitted)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-cache", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--smoke", type=int, default=0, help="stop after N updates")
    parser.add_argument("--abort-ratio", type=float, default=1.5,
                        help="stop if smoothed total loss exceeds best smoothed "
                             "loss by this factor (0 disables)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    # Hybrid conversion requires exact R1 natives (campaign loader pins the
    # same); must be set before the model modules are constructed.
    os.environ["GDN3_KMD2_NATIVE"] = "1"
    os.environ["GDN3_KMD2_ROUT"] = "1"

    from transformers import AutoModelForCausalLM
    from gdn3.gdn3_upgrade import GDN3UpgradeManager
    from research.kmd2_ablation.qwen_architecture import build_maximum_control_architecture
    from research.kmd2_ablation.qwen_training import (
        QwenHealTrainer, QwenHealTrainingConfig, build_qwen_heal_optimizer,
    )

    # --- model: pretrained -> upgrade -> compact-lane flagship conversion
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    manager = GDN3UpgradeManager(model, model.config)
    indices = manager.apply_upgrade()
    for i in indices:
        model.model.layers[i].linear_attn = build_maximum_control_architecture(
            model.model.layers[i].linear_attn, "package-b-hola-w64").to(torch.bfloat16)
    log(f"converted {len(indices)} layers to package-b-hola-w64 (compact lanes)")

    # --- trainables: exactly the converted layers, backbone frozen
    prefixes = tuple(f"model.layers.{i}.linear_attn." for i in indices)
    trainable = tuple(sorted(
        name for name, _ in model.named_parameters() if name.startswith(prefixes)
    ))
    trainable_set = set(trainable)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable_set)
    cache_names = tuple(n for n in trainable
                        if ".hola." in n or n.endswith(".components.cache_gate_logit"))
    memory_names = tuple(n for n in trainable if n not in set(cache_names))
    n_param = sum(p.numel() for n, p in model.named_parameters() if n in trainable_set)
    log(f"trainables: {len(trainable)} tensors, {n_param/1e6:.2f}M params "
        f"({len(cache_names)} cache-group)")

    resume_meta = {}
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:3]}"
        assert set(ckpt["optimizer_parameter_names"]) == trainable_set, \
            "resume checkpoint trainable set does not match this build"
        resume_meta = dict(ckpt["metadata"])
        log(f"resumed model from {args.resume}")
    model = model.to(device)

    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    teacher.eval()

    # --- config mirrors the campaign heal (ce/kl/lw 0.1/1.0/0.1, temp 2)
    tokens_per_update = world * args.batch * args.seq_len
    config = QwenHealTrainingConfig(
        objective="language_model_heal", ce_weight=0.1, kl_weight=1.0,
        layerwise_weight=0.1, temperature=2.0, accumulation_steps=1,
        max_updates=args.updates, max_tokens=args.updates * tokens_per_update,
        gradient_checkpointing=True, specialization_updates=8,
        lambda_spec=0.001, lambda_gate=0.001,
    )

    optimizer = build_qwen_heal_optimizer(
        model, memory_parameter_names=memory_names, cache_parameter_names=cache_names,
        learning_rate=args.lr, lr_cache=args.lr_cache,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
    )
    if args.resume is not None:
        # Keep Adam moments but override the checkpoint's param-group LRs:
        # load_state_dict restores them, and LambdaLR would bake those stale
        # values in as base_lrs (resuming at the LR we are trying to change).
        desired_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for group, desired in zip(optimizer.param_groups, desired_lrs):
            group["lr"] = desired
            group.pop("initial_lr", None)
        log(f"optimizer state restored, lrs overridden to {desired_lrs}")

    # warmup -> constant plateau (the cosine-to-zero tail strangled the last
    # run; horizon is intentionally open)
    warmup = max(0, args.warmup)
    def multiplier(step: int) -> float:
        return (step + 1) / warmup if warmup and step < warmup else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index], broadcast_buffers=True)

    # --- data
    if args.band is not None:
        shards = sorted(args.band.glob("shard_main_*.pt"))
        blocks = torch.cat([torch.load(p, map_location="cpu", weights_only=True) for p in shards])
        example_ids = json.loads((args.band / "example_ids.json").read_text())
    else:
        blocks = torch.load(args.band_tensor, map_location="cpu", weights_only=True)
        example_ids = [f"n3p2-{i:05d}" for i in range(blocks.shape[0])]
    assert blocks.shape[1] == args.seq_len, f"band is T={blocks.shape[1]}, want {args.seq_len}"
    need = args.updates * world * args.batch
    n_rows = blocks.shape[0]
    if n_rows < need:
        log(f"band has {n_rows} rows < {need}; cycling with per-epoch id suffixes")
    log(f"band: {n_rows} x {args.seq_len} rows for {need} window slots")

    def slot(index: int) -> tuple[torch.Tensor, str]:
        epoch, row = divmod(index, n_rows)
        eid = example_ids[row] if epoch == 0 else f"{example_ids[row]}-e{epoch}"
        return blocks[row], eid

    expected_windows = tuple(
        tuple(slot(u * world * args.batch + rank * args.batch + b)[1]
              for b in range(args.batch))
        for u in range(args.updates)
    )
    trainer = QwenHealTrainer(
        model=ddp, teacher=teacher, optimizer=optimizer, scheduler=scheduler,
        config=config, job_id="d0" * 32, pairing_id="d1" * 32, arm="native",
        expected_example_windows=expected_windows, teacher_device=device,
        distributed=True,
    )

    out = args.out
    (out / "live").mkdir(parents=True, exist_ok=True)
    live = (out / "live" / f"rank{rank}.jsonl").open("a")
    if rank == 0:
        (out / "run_config.json").write_text(json.dumps({
            "tree": "gdnx_0720", "arch": "package-b-hola-w64-compact",
            "updates": args.updates, "batch_per_rank": args.batch, "world": world,
            "seq_len": args.seq_len, "tokens_per_update": tokens_per_update,
            "lr": args.lr, "lr_cache": args.lr_cache, "warmup": warmup,
            "band": str(args.band or args.band_tensor),
            "resume": str(args.resume) if args.resume else None,
            "resume_meta": resume_meta,
        }, indent=1))

    stop_at = args.smoke if args.smoke else args.updates
    t0 = time.time()
    # Unattended divergence tripwire: EMA of total loss vs its own best.
    ema, best_ema, tripped = None, None, 0
    aborted = False
    for update in range(stop_at):
        base = update * world * args.batch + rank * args.batch
        rows, ids = zip(*(slot(base + b) for b in range(args.batch)))
        window = torch.stack(rows).to(device=device, dtype=torch.long)
        batch = {"input_ids": window, "labels": window.clone(), "example_ids": tuple(ids)}
        step_log = trainer.train_update([batch])
        # Trainer losses are rank-local; average them so every rank feeds the
        # same value into the tripwire and the abort decision cannot desync.
        total_t = torch.tensor(float(step_log.losses["total"]), device=device)
        dist.all_reduce(total_t, op=dist.ReduceOp.AVG)
        total = float(total_t)
        ema = total if ema is None else 0.95 * ema + 0.05 * total
        if update >= 20:  # let the EMA settle past warmup noise
            best_ema = ema if best_ema is None else min(best_ema, ema)
            tripped = tripped + 1 if (
                args.abort_ratio and ema > best_ema * args.abort_ratio
            ) else 0
        if tripped >= 10:
            aborted = True
        if rank == 0:
            row = {"update": update + 1, "tokens_seen": trainer.tokens_seen,
                   "losses": step_log.losses, "lr": scheduler.get_last_lr()[0],
                   "ema": round(ema, 4),
                   "seconds_per_update": (time.time() - t0) / (update + 1)}
            live.write(json.dumps(row) + "\n"); live.flush()
        if aborted or (update + 1) % args.checkpoint_every == 0 or update + 1 == stop_at:
            dist.barrier()
            if rank == 0:
                state = {
                    "model_state": {k: v for k, v in ddp.module.state_dict().items()
                                    if k in trainable_set},
                    "optimizer_state": optimizer.state_dict(),
                    "optimizer_parameter_names": trainable,
                    "metadata": {**resume_meta, "run": "night3",
                                 "tree": "gdnx_0720", "update": update + 1,
                                 "seq_len": args.seq_len,
                                 "global_batch": world * args.batch, "lr": args.lr},
                    "schema_version": "night3-1",
                }
                tmp = out / ".ckpt.tmp"
                torch.save(state, tmp)
                tmp.replace(out / f"ckpt-{update + 1:05d}.pt")
                latest = out / "ckpt-latest.pt"
                if latest.is_symlink() or latest.exists():
                    latest.unlink()
                latest.symlink_to(out / f"ckpt-{update + 1:05d}.pt")
                log(f"checkpoint at update {update + 1}")
            dist.barrier()
        if aborted:
            log(f"DIVERGENCE ABORT at update {update + 1}: smoothed loss "
                f"{ema:.3f} > {args.abort_ratio}x best {best_ema:.3f} for 10 "
                f"consecutive updates; checkpointed and stopping")
            break
    log(f"done: {(update + 1) if aborted else stop_at} updates, "
        f"{trainer.tokens_seen} tokens, {trainer.skipped_steps} skipped")
    dist.destroy_process_group()
    if aborted:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
