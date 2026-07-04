#!/usr/bin/env python3
"""MQAR speedrun — the deterministic fitness function for GDN3 auto-research.

Trains the FULL 0.8B GDN3 model (backbone frozen, only GDN3 memory+coproduct
params trained) on synthetic multi-key associative recall with next-token CE,
and measures how fast retrieval EMERGES. Sub-hour, single-GPU (no teacher, so
two can run in parallel across the two 5060 Tis), one clean number out.

Why this proxy (see docs/HANDOFF_chunked_scan.md lineage + research/RESEARCH.md):
  * FULL model (all 18 GDN3 layers) -> captures inter-layer composition, which is
    what single-layer MQAR (the friend's test) missed.
  * dense synthetic recall + direct CE -> retrieval is forced to emerge in
    hundreds of steps, not the thousands the diluted natural-distill heal needed.
  * a config that can't learn MQAR fast here will never do RULER in a 14h heal;
    one that can is a promotion candidate for a real distill+RULER run.

Config knobs (JSON): arch  -> residual_rank(P), slow_decay, decay_clamp  (env)
                     optim -> lr_memory, lr_coproduct, warmup, clip
                     task  -> steps, seq_len, n_keys, eval_every, batch, grad_accum

Output JSON: config + recall_curve + final_recall + emergence_step + skip_rate +
             final_ce + wall_s + status  (status="ok" | "diverged" | "error:...").

Usage: proxy_mqar.py --config cfg.json --out result.json [--device cuda:0]
"""
from __future__ import annotations
import os, sys, json, time, argparse, random, traceback

ROOT = "/home/dev/gdn3_two_timescale_release"
sys.path.insert(0, ROOT)
SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
PRESERVED = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "norm", "out_proj")

_ADJ = "amber brisk cobalt dusky ember frost glim hazel ivory jade lunar mossy nimble ochre pewter quartz russet slate umber vivid".split()
_NOUN = "lantern harbor cipher meadow falcon anchor beacon cedar willow marble canyon thicket pylon zephyr cobble trellis onyx pyre ridge vault".split()


def make_mqar(rng, tok, n_keys, seq_len):
    """One MQAR episode: n_keys distinct 'code for K is V' bindings (shuffled),
    then a query for one key. Returns (ids, ans_start, gold_ids). CE/eval score
    only the answer tokens. Pads toward seq_len with extra distractor bindings."""
    def rand_key():
        return f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}"
    keys, vals, seen = [], [], set()
    target = max(n_keys, 4)
    while len(keys) < target * 3:                     # generate a pool; distractors fill length
        k = rand_key()
        if k in seen:
            continue
        seen.add(k); keys.append(k); vals.append(rng.randint(1000, 9999))  # 4-digit: fewer answer tokens
    lines = [f"The code for {k} is {v}." for k, v in zip(keys, vals)]
    rng.shuffle(lines)
    qi = rng.randrange(len(keys))
    qk, qv = keys[qi], vals[qi]
    # grow context until ~seq_len tokens, always keeping the queried binding in
    ctx_lines = [f"The code for {qk} is {qv}."]
    for ln in lines:
        if ln.startswith(f"The code for {qk} "):
            continue
        ctx_lines.append(ln)
    rng.shuffle(ctx_lines)
    prefix = " ".join(ctx_lines) + f"\nWhat is the code for {qk}? The code for {qk} is:"
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    a_ids = tok(f" {qv}", add_special_tokens=False).input_ids
    # trim from the front (keep query + answer at the end) if over length
    total = len(p_ids) + len(a_ids)
    if total > seq_len:
        p_ids = p_ids[total - seq_len:]
    ids = p_ids + a_ids
    return ids, len(p_ids), a_ids


def post_discord(result, out_path):
    """Best-effort one-line experiment summary to the research Discord channel.
    Reads BOT_TOKEN / CHANNEL_ID from the release root. Never raises."""
    try:
        tok = open(os.path.join(ROOT, "BOT_TOKEN")).read().strip()
        ch = open(os.path.join(ROOT, "CHANNEL_ID")).read().strip()
        if not (tok and ch):
            return
        import requests
    except Exception:
        return
    cfg = result.get("config", {})
    exp = os.path.splitext(os.path.basename(out_path))[0]
    st = result.get("status", "?")
    icon = "🧪" if st == "ok" else ("⚠️" if st == "diverged" else "❌")
    name = cfg.get("name", "?")
    hyp = (cfg.get("hypothesis") or "")[:220]
    if str(st).startswith("error"):
        body = f"{icon} **{exp}** · {name}  [{st}]\n{str(result.get('error',''))[:250]}"
    else:
        emg = result.get("emergence_step")
        emg = f"emerge@{emg}" if emg is not None else "no-emerge"
        body = (f"{icon} **{exp}** · {name}  [{st}]\n"
                f"tok_acc **{result.get('final_tokacc', 0):.3f}** · recall {result.get('final_recall', 0):.2f} · {emg} · "
                f"skip {result.get('skip_rate', 0) * 100:.0f}% · ce {result.get('final_ce', '?')} · "
                f"{result.get('wall_s', 0) / 60:.0f}m\n"
                f"_{hyp}_\n"
                f"knobs: lr_m {cfg.get('lr_memory')} lr_c {cfg.get('lr_coproduct')} · "
                f"P{cfg.get('residual_rank', 16)} sd{cfg.get('slow_decay', 0.97)} "
                f"dc{cfg.get('decay_clamp', 0.999)} · nkeys{cfg.get('n_keys', 8)} steps{cfg.get('steps', 300)}")
    try:
        requests.post(f"https://discord.com/api/v10/channels/{ch}/messages",
                      headers={"Authorization": f"Bot {tok}"},
                      json={"content": body[:1990]}, timeout=10)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-discord", action="store_true", help="suppress the Discord result post")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    t0 = time.time()

    # arch knobs -> env, BEFORE importing gdn3 (read in GDN3LinearAttn.__init__)
    if "residual_rank" in cfg: os.environ["GDN3_P"] = str(cfg["residual_rank"])
    if "slow_decay" in cfg:    os.environ["GDN3_SLOW_DECAY"] = str(cfg["slow_decay"])
    if "decay_clamp" in cfg:   os.environ["GDN3_DECAY_CLAMP"] = str(cfg["decay_clamp"])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    result = {"config": cfg, "status": "error:init", "device": args.device}
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from gdn3.gdn3_upgrade import GDN3UpgradeManager

        dev = torch.device(args.device)
        seq_len = int(cfg.get("seq_len", 512)); n_keys = int(cfg.get("n_keys", 4))
        steps = int(cfg.get("steps", 300));     warmup = int(cfg.get("warmup", 40))
        eval_every = int(cfg.get("eval_every", 75)); accum = int(cfg.get("grad_accum", 2))
        clip = float(cfg.get("clip", 1.0)); seed = int(cfg.get("seed", 0))
        lr_mem = float(cfg.get("lr_memory", 2.5e-4)); lr_cop = float(cfg.get("lr_coproduct", 1.5e-4))
        torch.manual_seed(seed); rng = random.Random(seed); evalrng = random.Random(9999)

        tok = AutoTokenizer.from_pretrained(SNAP)
        model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
        for p in model.parameters(): p.requires_grad_(False)
        mem_params, cop_params = [], []
        for idx in upg:
            for n, p in model.model.layers[idx].linear_attn.named_parameters():
                if any(k in n for k in PRESERVED):
                    continue                                  # freeze warm-started projections (matches heal)
                p.requires_grad_(True)
                (cop_params if "coprod" in n or n.startswith(("W_q_", "W_k_", "W_v_")) else mem_params).append(p)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(dev).train()

        opt = torch.optim.AdamW([{"params": mem_params, "lr": lr_mem},
                                 {"params": cop_params, "lr": lr_cop}],
                                betas=(0.9, 0.95), weight_decay=0.01)
        def lr_at(step):
            return (step + 1) / max(1, warmup) if step < warmup else 1.0
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

        @torch.no_grad()
        def evaluate(nprobe=48):
            """Returns (exact_recall, tok_acc). tok_acc = fraction of answer tokens
            argmax-correct — a CONTINUOUS signal that discriminates configs long
            before all-or-nothing exact recall emerges within the sub-hour budget."""
            model.eval(); em = tok_c = tok_n = 0
            for _ in range(nprobe):
                ids, a0, gold = make_mqar(evalrng, tok, n_keys, seq_len)
                x = torch.tensor([ids], device=dev)
                h = model.model(input_ids=x).last_hidden_state
                pred = model.lm_head(h[0, a0 - 1:a0 - 1 + len(gold)]).argmax(-1)
                g = torch.tensor(gold, device=dev)
                nc = int((pred == g).sum())
                em += int(nc == len(gold)); tok_c += nc; tok_n += len(gold)
            model.train(); return em / max(1, nprobe), tok_c / max(1, tok_n)

        recall_curve, tokacc_curve, skipped = {}, {}, 0
        opt.zero_grad(set_to_none=True)
        last_ce = float("nan")
        for step in range(steps):
            bad = False
            for _ in range(accum):
                ids, a0, gold = make_mqar(rng, tok, n_keys, seq_len)
                x = torch.tensor([ids], device=dev)
                logits = model(input_ids=x).logits
                lg = logits[0, a0 - 1:a0 - 1 + len(gold)]
                loss = F.cross_entropy(lg.float(), torch.tensor(gold, device=dev))
                if not torch.isfinite(loss):
                    bad = True; break
                (loss / accum).backward(); last_ce = float(loss.detach())
            gnorm = torch.nn.utils.clip_grad_norm_(mem_params + cop_params, clip)
            if bad or not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True); sched.step(); skipped += 1; continue
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % eval_every == 0 or step == steps - 1:
                em, ta = evaluate()
                recall_curve[str(step)] = round(em, 4); tokacc_curve[str(step)] = round(ta, 4)

        ta_by_step = {int(k): v for k, v in tokacc_curve.items()}
        final_tokacc = ta_by_step[max(ta_by_step)] if ta_by_step else 0.0
        final_recall = recall_curve[str(max(int(k) for k in recall_curve))] if recall_curve else 0.0
        emerged = next((s for s in sorted(ta_by_step) if ta_by_step[s] >= 0.5), None)  # tok_acc>=0.5
        skip_rate = skipped / max(1, steps)
        result.update(status="diverged" if skip_rate > 0.5 else "ok",
                      tokacc_curve=tokacc_curve, recall_curve=recall_curve,
                      final_tokacc=round(final_tokacc, 4), final_recall=round(final_recall, 4),
                      emergence_step=emerged, skip_rate=round(skip_rate, 4),
                      final_ce=round(last_ce, 4), wall_s=round(time.time() - t0, 1))
    except Exception as e:
        result.update(status=f"error:{type(e).__name__}", error=str(e)[:400],
                      traceback=traceback.format_exc()[-1500:], wall_s=round(time.time() - t0, 1))
    json.dump(result, open(args.out, "w"), indent=2)
    if not args.no_discord:
        post_discord(result, args.out)
    print(json.dumps({k: result[k] for k in ("status", "final_tokacc", "final_recall",
                                             "emergence_step", "skip_rate", "wall_s") if k in result}))


if __name__ == "__main__":
    main()
