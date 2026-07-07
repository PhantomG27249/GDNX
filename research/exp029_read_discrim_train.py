"""exp029: does TRAINING make the GDN3 read discriminative, or keep it uniform?

Diagnostic (diag_frozen_qk.py) found the read scores (q.k, post-coproduct,
post-RoPE) are UNIFORM at init (entropy ~ ln(T) = uniform max) -> a uniform read
= average of stored values = the marginal digit distribution = the FORMAT shortcut
(tokacc ~0.30, recall 0). This explains exp028 (softmax over uniform = uniform).

The open question: the coproduct (W_q_a/W_k_a) is TRAINABLE. Does TRAINING push
the read to discriminate the queried key's value, or does it stay uniform (because
format CE is satisfied by the uniform read -> no gradient toward discrimination)?

This script faithfully replicates research/proxy_mqar.py's training loop and adds a
read-discrimination probe (entropy + value-rank of the q.k softmax at answer
positions) measured at init and after training. Standalone (does not edit the
proxy or GDN3 source). Output: research/runs/exp029.json (appended to leaderboard).

If entropy stays ~uniform after training AND tokacc ~0.30 / recall 0 -> CONFIRMS
the format-shortcut ceiling: training optimizes state-shaping for format, the
read stays uniform, recall stays 0. The fix must make discrimination NECESSARY
(remove the format shortcut or add a retrieval signal). A naive learnable-read-proj
would ALSO stay uniform (no discrimination gradient) -> that plan is invalidated.
If entropy DROPS (read becomes sharp) after training -> the coproduct CAN
discriminate, and the bottleneck is downstream (value/LM-head) -> different fix.
"""
import sys, os, json, time, random, argparse, math
sys.path.insert(0, '/home/dev/gdn3_two_timescale_release')

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="research/configs/exp027.json")
ap.add_argument("--out", default="research/runs/exp029.json")
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--steps", type=int, default=200)
args = ap.parse_args()

cfg = json.load(open(args.config))
cfg["steps"] = args.steps
cfg["eval_every"] = 50
t0 = time.time()

# arch knobs -> env, BEFORE importing gdn3 (read in GDN3LinearAttn.__init__)
if "residual_rank" in cfg: os.environ["GDN3_P"] = str(cfg["residual_rank"])
if "slow_decay" in cfg:    os.environ["GDN3_SLOW_DECAY"] = str(cfg["slow_decay"])
if "decay_clamp" in cfg:   os.environ["GDN3_DECAY_CLAMP"] = str(cfg["decay_clamp"])
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager
import research.proxy_mqar as pm

SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
PRESERVED = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "norm", "out_proj")
DEV = torch.device(args.device)

seq_len = int(cfg.get("seq_len", 512)); n_keys = int(cfg.get("n_keys", 4))
warmup = int(cfg.get("warmup", 40)); clip = float(cfg.get("clip", 1.0))
seed = int(cfg.get("seed", 0))
lr_mem = float(cfg.get("lr_memory", 2.5e-4)); lr_cop = float(cfg.get("lr_coproduct", 1.5e-4))
steps = int(cfg["steps"]); eval_every = int(cfg.get("eval_every", 50))

torch.manual_seed(seed); rng = random.Random(seed); evalrng = random.Random(9999)

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
for p in model.parameters(): p.requires_grad_(False)
mem_params, cop_params = [], []
for idx in upg:
    for n, p in model.model.layers[idx].linear_attn.named_parameters():
        if any(k in n for k in PRESERVED): continue
        p.requires_grad_(True)
        (cop_params if "coprod" in n or n.startswith(("W_q_", "W_k_", "W_v_")) else mem_params).append(p)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(DEV).train()
opt = torch.optim.AdamW([{"params": mem_params, "lr": lr_mem},
                         {"params": cop_params, "lr": lr_cop}], betas=(0.9, 0.95), weight_decay=0.01)
def lr_at(step): return (step + 1) / max(1, warmup) if step < warmup else 1.0
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

# --- read-discrimination probe: capture q,k from one upgraded layer's recurrence ---
CAP_LAYER = upg[len(upg)//2]
cap = {}
layer_mod = model.model.layers[CAP_LAYER].linear_attn
H, M, K = layer_mod.H, layer_mod.M, layer_mod.K
orig_fn = layer_mod._gdn3_recurrent_state
def capturing_fn(qf, kf, vf, bg, wg, dec):
    cap['q'] = qf.detach(); cap['k'] = kf.detach(); cap['v'] = vf.detach()
    return orig_fn(qf, kf, vf, bg, wg, dec)
layer_mod._gdn3_recurrent_state = capturing_fn

def find_value_pos(ids, gold):
    """Find the token index where the stored value (gold id sequence) begins in
    the prefix (the EARLIER occurrence, not the answer). Returns -1 if absent."""
    L = len(gold)
    for i in range(len(ids) - L - L):  # stop before the answer region
        if ids[i:i+L] == gold:
            return i
    return -1

@torch.no_grad()
def probe_read(nprobe=16):
    """Measure read-score discrimination at answer positions over nprobe episodes.
    Returns mean entropy (nats; uniform=ln(T_prior)), frac where argmax==value pos,
    and mean rank-percentile of the value position's score (1.0=top)."""
    model.eval()
    ents, val_hits, val_ranks, tok_c, tok_n = [], [], [], 0, 0
    pr = random.Random(424242)
    for _ in range(nprobe):
        ids, a0, gold = pm.make_mqar(pr, tok, n_keys, seq_len)
        x = torch.tensor([ids], device=DEV)
        _ = model(x)
        if 'q' not in cap: continue
        qf = cap['q'][0]; kf = cap['k'][0]   # [T,H,M,K]
        T = qf.shape[0]
        vpos = find_value_pos(ids, gold)
        # first answer token reads from all prior positions
        s = a0
        prior_scores = torch.einsum('hmk,thmk->ht', qf[s], kf[:s]) / (K**0.5)  # [H,T_prior]
        prior_scores = prior_scores.mean(0)  # mean over heads -> [T_prior]
        sm = F.softmax(prior_scores, dim=-1)
        ent = float(-(sm*torch.log(sm+1e-12)).sum())
        ents.append(ent)
        if vpos >= 0 and vpos < s:
            amax = int(prior_scores.argmax())
            val_hits.append(float(amax == vpos))
            rank = float((prior_scores > prior_scores[vpos]).sum()) / s  # 0=best
            val_ranks.append(rank)
        # tokacc on this probe
        h = model.model(input_ids=x).last_hidden_state
        pred = model.lm_head(h[0, a0-1:a0-1+len(gold)]).argmax(-1)
        g = torch.tensor(gold, device=DEV)
        tok_c += int((pred==g).sum()); tok_n += len(gold)
    model.train()
    import statistics as st
    return {
        "read_entropy": round(st.mean(ents), 4) if ents else None,
        "uniform_entropy": round(math.log(seq_len), 4),
        "val_argmax_acc": round(st.mean(val_hits), 4) if val_hits else None,
        "val_rank_pct": round(st.mean(val_ranks), 4) if val_ranks else None,
        "tokacc": round(tok_c/max(1,tok_n), 4),
        "nprobe": nprobe,
    }

print("=== read-discrimination probe at INIT (untrained) ===")
probe_init = probe_read()
print(json.dumps(probe_init, indent=2))
print(f"  (uniform = {probe_init['uniform_entropy']} nats)")

# --- train (faithful replica of proxy_mqar.py training loop) ---
print(f"=== training {steps} steps (lr_mem {lr_mem} lr_cop {lr_cop}) ===")
tokacc_curve = {}; skipped = 0; last_ce = float('nan')
opt.zero_grad(set_to_none=True)
for step in range(steps):
    bad = False
    ids, a0, gold = pm.make_mqar(rng, tok, n_keys, seq_len)
    x = torch.tensor([ids], device=DEV)
    logits = model(input_ids=x).logits
    lg = logits[0, a0-1:a0-1+len(gold)]
    loss = F.cross_entropy(lg.float(), torch.tensor(gold, device=DEV))
    if not torch.isfinite(loss):
        bad = True
    else:
        loss.backward(); last_ce = float(loss.detach())
    gnorm = torch.nn.utils.clip_grad_norm_(mem_params + cop_params, clip)
    if bad or not torch.isfinite(gnorm):
        opt.zero_grad(set_to_none=True); sched.step(); skipped += 1; continue
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % eval_every == 0 or step == steps-1:
        # quick tokacc
        model.eval()
        with torch.no_grad():
            tc=tn=0
            for _ in range(24):
                eids,ea0,egold = pm.make_mqar(evalrng, tok, n_keys, seq_len)
                ex = torch.tensor([eids], device=DEV)
                eh = model.model(input_ids=ex).last_hidden_state
                ep = model.lm_head(eh[0, ea0-1:ea0-1+len(egold)]).argmax(-1)
                tc += int((ep==torch.tensor(egold,device=DEV)).sum()); tn += len(egold)
        model.train()
        tokacc_curve[str(step)] = round(tc/max(1,tn),4)
        print(f"  step {step}: tokacc {tokacc_curve[str(step)]:.3f}  ce {last_ce:.3f}")

print("=== read-discrimination probe AFTER training ===")
probe_post = probe_read()
print(json.dumps(probe_post, indent=2))
print(f"  (uniform = {probe_post['uniform_entropy']} nats)")

d_init = probe_init['read_entropy'] or 0
d_post = probe_post['read_entropy'] or 0
unif = probe_init['uniform_entropy']
print(f"\n=== VERDICT ===")
print(f"entropy init {d_init} -> post {d_post}  (uniform={unif})")
if d_post > 0.9*unif and probe_post['tokacc'] and probe_post['tokacc'] > 0.2 and (probe_post.get('val_argmax_acc') or 0) < 0.1:
    print("CONFIRMS format-shortcut ceiling: read stays UNIFORM after training (no discrimination), tokacc from state-shaping format, recall 0. Learnable-read-proj plan (exp029 original) INVALIDATED.")
elif d_post < 0.7*unif:
    print("read BECAME discriminative after training -> coproduct CAN discriminate; bottleneck is downstream (value/LM-head).")
else:
    print("ambiguous: partial sharpening. Compare val_argmax_acc / val_rank_pct for the real signal.")

result = {
    "config": {**cfg, "name": "read_discrim_train_exp029",
               "hypothesis": "Does TRAINING make the GDN3 read discriminative, or keep it uniform? Diagnostic found read scores UNIFORM at init (entropy~ln(T)) -> uniform read = format marginal -> tokacc 0.30 recall 0. If entropy stays uniform after training + tokacc~0.30 + recall 0 -> CONFIRMS format-shortcut ceiling (no discrimination gradient; learnable-read-proj plan invalidated). If entropy drops -> coproduct CAN discriminate, bottleneck downstream."},
    "status": "ok", "device": args.device,
    "probe_init": probe_init, "probe_post": probe_post,
    "tokacc_curve": tokacc_curve,
    "final_tokacc": probe_post.get("tokacc", 0.0),
    "final_recall": 0.0,
    "read_entropy_init": d_init, "read_entropy_post": d_post, "uniform_entropy": unif,
    "skip_rate": round(skipped/max(1,steps),4), "final_ce": round(last_ce,4),
    "wall_s": round(time.time()-t0,1),
}
json.dump(result, open(args.out, "w"), indent=2)
print(f"\n{json.dumps({k:result[k] for k in ('status','final_tokacc','read_entropy_init','read_entropy_post','uniform_entropy','skip_rate','wall_s')})}")
