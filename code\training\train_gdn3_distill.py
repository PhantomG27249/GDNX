"""
GDN3 <- Qwen3.5 Self-Distillation Trainer
=========================================

Retrofits Qwen3.5-0.8B's linear-attention (GatedDeltaNet) layers with the
GDN3 Kronecker-Residual MIMO recurrence and trains the student to match the
*original* Qwen3.5 (teacher) via KL self-distillation on the materialised
data mix.  The Qwen backbone (embeddings, MLPs, full-attention layers,
lm_head) is frozen; only the 18 GDN3 layers are trained.

  Teacher : original Qwen3.5-0.8B (GatedDeltaNet), frozen, GPU:teacher_dev
  Student : GDN3-upgraded Qwen3.5-0.8B, GPU:student_dev
  Loss    : w_kl * KL(teacher||student)/T^2-scaled  +  w_ce * CE(next-token)

Per-mechanism learning rates (3 groups):
  memory     : W_w, W_b, W_decay, router_proj, _agg_proj      (new state/gates)
  coproduct  : W_q_a..W_v_b, coprod_mix_*, coprod_strength_*   (Hopf channels)
  preserved  : in_proj_qkv/z/a/b, conv1d, norm, out_proj       (warm-started)

Discord logging via BOT_TOKEN / CHANNEL_ID files at the package root.

Usage:
  python train_gdn3_distill.py --steps 1500 --seq-len 512 --batch-size 1
  python train_gdn3_distill.py --smoke        # 3-step smoke test
"""
from __future__ import annotations
import sys, os, time, json, math, argparse, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integration"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

MODEL_SNAP = "/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"


# --------------------------------------------------------------------------- #
# Discord logging (best-effort, non-blocking)
# --------------------------------------------------------------------------- #
class Discord:
    def __init__(self, root: Path, enabled=True):
        self.enabled = enabled
        self.token = self.chan = None
        try:
            self.token = (root / "BOT_TOKEN").read_text().strip()
            self.chan = (root / "CHANNEL_ID").read_text().strip()
        except Exception as e:
            print(f"[discord] disabled: {e}"); self.enabled = False

    def send(self, msg: str):
        print(msg, flush=True)
        if not self.enabled or not self.token or not self.chan:
            return
        def _post():
            try:
                import requests
                requests.post(
                    f"https://discord.com/api/v10/channels/{self.chan}/messages",
                    headers={"Authorization": f"Bot {self.token}"},
                    json={"content": msg[:1990]}, timeout=10)
            except Exception as e:
                print(f"[discord] send failed: {e}")
        threading.Thread(target=_post, daemon=True).start()


# --------------------------------------------------------------------------- #
# Data: 512-token windows sampled from the 2048-token materialised blocks
# --------------------------------------------------------------------------- #
class WindowedMix(torch.utils.data.Dataset):
    def __init__(self, path, seq_len, seed=0):
        from data.data_mix import MaterializedMix
        self.base = MaterializedMix(path)
        self.seq_len = seq_len
        self.block_len = self.base.seq_len
        self.n_win = max(1, self.block_len // seq_len)
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        block = self.base[i]  # [block_len] long
        if self.block_len > self.seq_len:
            max_off = self.block_len - self.seq_len
            off = int(torch.randint(0, max_off + 1, (1,), generator=self.g).item())
        else:
            off = 0
        return block[off:off + self.seq_len]


# --------------------------------------------------------------------------- #
# Param grouping for per-mechanism LR
# --------------------------------------------------------------------------- #
MEMORY_KEYS = ("W_w", "W_b", "W_decay", "router_proj", "_agg_proj")
COPROD_KEYS = ("W_q_a", "W_q_b", "W_k_a", "W_k_b", "W_v_a", "W_v_b",
               "coprod_mix", "coprod_strength")
PRESERVED_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                  "conv1d", "norm", "out_proj")


def classify(name: str) -> str:
    if any(k in name for k in MEMORY_KEYS):
        return "memory"
    if any(k in name for k in COPROD_KEYS):
        return "coproduct"
    if any(k in name for k in PRESERVED_KEYS):
        return "preserved"
    return "memory"  # default: treat unknown GDN3 params as memory


def build_param_groups(model, upgraded_layers, lrs):
    buckets = {"memory": [], "coproduct": [], "preserved": []}
    for idx in upgraded_layers:
        for name, p in model.model.layers[idx].linear_attn.named_parameters():
            if p.requires_grad:
                buckets[classify(name)].append(p)
    groups, counts = [], {}
    for k, params in buckets.items():
        if params:
            groups.append({"params": params, "lr": lrs[k], "name": k})
            counts[k] = sum(p.numel() for p in params)
    return groups, counts


def lr_lambda_factory(warmup, total, floor=0.1):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def distill_loss(student_logits, teacher_logits, input_ids, w_kl, w_ce, tau):
    B, T, Vsz = student_logits.shape
    s = student_logits.float()
    t = teacher_logits.float()
    # KL(teacher || student) with temperature, scaled by tau^2
    t_logp = F.log_softmax(t / tau, dim=-1)
    s_logp = F.log_softmax(s / tau, dim=-1)
    kl = F.kl_div(s_logp, t_logp, reduction="batchmean", log_target=True) * (tau * tau) / T
    # next-token CE on the real tokens
    ce = F.cross_entropy(s[:, :-1].reshape(-1, Vsz), input_ids[:, 1:].reshape(-1))
    return w_kl * kl + w_ce * ce, kl.detach(), ce.detach()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-memory", type=float, default=6e-4)
    ap.add_argument("--lr-coproduct", type=float, default=4e-4)
    ap.add_argument("--lr-preserved", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--w-ce", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--student-dev", default="cuda:0")
    ap.add_argument("--teacher-dev", default="cuda:1")
    ap.add_argument("--data", default=str(ROOT / "data" / "mix_v1"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "gdn3_distill"))
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.log_every, args.ckpt_every = 3, 1, 999999

    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dc = Discord(ROOT, enabled=not args.no_discord)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from integration.gdn3_upgrade import GDN3UpgradeManager

    sdev = torch.device(args.student_dev)
    tdev = torch.device(args.teacher_dev if torch.cuda.device_count() > 1 else args.student_dev)

    dc.send(f"🚀 **GDN3 self-distillation starting**\n"
            f"steps={args.steps} seq_len={args.seq_len} bs={args.batch_size} "
            f"accum={args.grad_accum}\nLR mem={args.lr_memory} coprod={args.lr_coproduct} "
            f"preserved={args.lr_preserved} | tau={args.tau} w_kl={args.w_kl} w_ce={args.w_ce}\n"
            f"student={sdev} teacher={tdev}")

    # ---- teacher (frozen, original Qwen3.5) ----
    print("[load] teacher ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_SNAP, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    teacher.config.use_cache = False
    teacher.eval().to(tdev)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---- student (GDN3-upgraded) ----
    print("[load] student ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    mgr = GDN3UpgradeManager(student)
    mgr.apply_upgrade()
    upgraded = mgr.upgraded_layers
    # freeze everything, then unfreeze GDN3 layers
    for p in student.parameters():
        p.requires_grad_(False)
    for idx in upgraded:
        for p in student.model.layers[idx].linear_attn.parameters():
            p.requires_grad_(True)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.to(sdev)
    student.train()

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in student.parameters())
    print(f"[student] total {n_tot/1e6:.1f}M | trainable {n_train/1e6:.1f}M | "
          f"upgraded {len(upgraded)} layers", flush=True)

    # ---- optimizer + schedule ----
    lrs = {"memory": args.lr_memory, "coproduct": args.lr_coproduct,
           "preserved": args.lr_preserved}
    groups, counts = build_param_groups(student, upgraded, lrs)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(args.warmup, args.steps))
    dc.send("📊 trainable params by mechanism: " +
            ", ".join(f"{k}={v/1e6:.1f}M" for k, v in counts.items()))

    # ---- data ----
    ds = WindowedMix(args.data, args.seq_len, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=2, pin_memory=True)
    print(f"[data] {len(ds)} blocks, seq_len={args.seq_len}", flush=True)

    def batches():
        while True:
            for b in dl:
                yield b

    bit = batches()
    Vsz = student.config.vocab_size
    running = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
    t_start = time.time()
    step = 0
    opt.zero_grad(set_to_none=True)

    while step < args.steps:
        micro_loss = 0.0
        for _ in range(args.grad_accum):
            ids = next(bit).to(sdev, non_blocking=True).long()
            with torch.no_grad():
                t_logits = teacher(input_ids=ids.to(tdev)).logits.to(sdev)
            s_logits = student(input_ids=ids).logits
            loss, kl, ce = distill_loss(s_logits, t_logits, ids,
                                        args.w_kl, args.w_ce, args.tau)
            (loss / args.grad_accum).backward()
            micro_loss += loss.item() / args.grad_accum
            running["kl"] += kl.item() / args.grad_accum
            running["ce"] += ce.item() / args.grad_accum
            del t_logits, s_logits

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], args.clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        running["loss"] += micro_loss
        step += 1

        if step % args.log_every == 0:
            k = args.log_every
            el = time.time() - t_start
            sps = step / el
            eta_h = (args.steps - step) / max(sps, 1e-9) / 3600
            cur_lrs = {g["name"]: g["lr"] for g in opt.param_groups}
            msg = (f"step {step}/{args.steps} | loss {running['loss']/k:.4f} "
                   f"kl {running['kl']/k:.4f} ce {running['ce']/k:.4f} | "
                   f"gnorm {gnorm:.2f} | lr_mem {cur_lrs.get('memory',0):.2e} | "
                   f"{el/step:.1f}s/step eta {eta_h:.1f}h | "
                   f"mem {torch.cuda.max_memory_allocated(sdev)/1e9:.1f}G")
            dc.send(msg)
            running = {kk: 0.0 for kk in running}

        if step % args.ckpt_every == 0 and step < args.steps:
            _save(student, upgraded, out / f"step{step}", dc, step)

    _save(student, upgraded, out / "final", dc, step)
    dc.send(f"✅ **GDN3 distillation complete** — {step} steps in "
            f"{(time.time()-t_start)/3600:.2f}h. Saved to {out/'final'}")


def _save(student, upgraded, path: Path, dc, step):
    path.mkdir(parents=True, exist_ok=True)
    sd = {}
    for idx in upgraded:
        for name, p in student.model.layers[idx].linear_attn.named_parameters():
            sd[f"model.layers.{idx}.linear_attn.{name}"] = p.detach().cpu()
    torch.save(sd, path / "gdn3_layers.pt")
    (path / "meta.json").write_text(json.dumps(
        {"step": step, "upgraded_layers": upgraded}, indent=2))
    dc.send(f"💾 checkpoint @ step {step} -> {path}")


if __name__ == "__main__":
    main()
