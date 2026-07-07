"""exp032: WRITE/binding-side decisive test — is the correct value even RETRIEVABLE?

All prior diagnostics (exp029/031) measured the read with the ACTUAL query. None
tested whether the correct value is retrievable with an IDEAL/oracle query. This
cleanly separates:
  - binding NOT stored (write-side lever: fix the coproduct binding / value encoding)
  - binding stored but query wrong (read/loss ceiling -> handoff airtight)

Three probes on the no-compaction single-chunk state (P=64, seq_len=64 -> the buffer
at the answer position = per-token writes 0..s-1, kron_q=0):
  (1) SELF-RETRIEVAL: query = k at the gold value's own position. Does softmax(k·k_j)
      peak at the gold value? (tests if stored values are distinguishable by their k)
  (2) KEY->VALUE BINDING: query = k at the gold key's mention. Does softmax over j
      peak at the gold value position? (tests the actual key->value association)
  (3) ACTUAL: query = q at the answer position. (= exp029's read; baseline.)

Decision:
  - (1) fails  -> values not distinguishable (frozen-v encoding ceiling -> human)
  - (1) ok, (2) fails -> key/value not bound (WRITE-side lever: coproduct binding)
  - (2) ok, (3) fails -> binding stored, query mismatch (read/loss ceiling, handoff)
  - (2) ok, (3) ok   -> should already have recall (contradiction -> recheck)

Standalone (no proxy/GDN3 source edits). Trains 200 steps (so the coproduct binding
forms), then probes. Output: research/runs/exp032.json (appended to leaderboard).
"""
import sys, os, json, time, random, argparse, math
sys.path.insert(0, '/home/dev/gdn3_two_timescale_release')

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="research/configs/exp027.json")
ap.add_argument("--out", default="research/runs/exp032.json")
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--steps", type=int, default=200)
args = ap.parse_args()

cfg = json.load(open(args.config)); cfg["steps"] = args.steps; cfg["eval_every"] = 50
t0 = time.time()
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
warmup = int(cfg.get("warmup", 40)); clip = float(cfg.get("clip", 1.0)); seed = int(cfg.get("seed", 0))
lr_mem = float(cfg.get("lr_memory", 2.5e-4)); lr_cop = float(cfg.get("lr_coproduct", 1.5e-4))
steps = int(cfg["steps"])
torch.manual_seed(seed); rng = random.Random(seed)

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
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/max(1,warmup) if s < warmup else 1.0)

CAP_LAYER = upg[len(upg)//2]; cap = {}; layer_mod = model.model.layers[CAP_LAYER].linear_attn
H, M, K = layer_mod.H, layer_mod.M, layer_mod.K
orig_fn = layer_mod._gdn3_recurrent_state
def capturing_fn(qf, kf, vf, bg, wg, dec):
    cap['q'] = qf.detach(); cap['k'] = kf.detach(); cap['v'] = vf.detach(); cap['dec'] = dec.detach()
    return orig_fn(qf, kf, vf, bg, wg, dec)
layer_mod._gdn3_recurrent_state = capturing_fn

def find_subseq(hay, needle):
    """All start indices where needle (list) appears in hay (list)."""
    L = len(needle); out = []
    for i in range(len(hay) - L + 1):
        if hay[i:i+L] == needle: out.append(i)
    return out

@torch.no_grad()
def probe(nprobe=24):
    """For each episode, find gold value pos + gold key pos, run 3 retrieval probes."""
    model.eval(); pr = random.Random(131313)
    res = {"self_peak": [], "self_rank": [], "bind_peak": [], "bind_rank": [],
           "actual_peak": [], "actual_rank": [], "v_cosine_same": [], "v_cosine_diff": []}
    for _ in range(nprobe):
        ids, a0, gold = pm.make_mqar(pr, tok, n_keys, seq_len)
        x = torch.tensor([ids], device=DEV); _ = model(x)
        if 'q' not in cap: continue
        qf = cap['q'][0]; kf = cap['k'][0]; vf = cap['v'][0]  # [T,H,M,K]
        T = qf.shape[0]
        # gold value write positions = FIRST occurrence of gold in prefix (the stored write)
        gold_positions = find_subseq(ids[:a0], gold)
        if not gold_positions: continue
        vpos = gold_positions[0]  # first digit of the stored gold value
        # gold key mention: "The code for KEY is VALUE" -> KEY is right before " is".
        # Find the KEY by decoding: the key word precedes " is" which precedes the value.
        # Simplest: the key mention is a few tokens before vpos. Find " is" before vpos.
        toks = [tok.decode([i]) for i in ids]
        # walk back from vpos to find the key word (the token before " is")
        # "The code for KEY is VALUE" -- " is" is right before vpos
        is_pos = None
        for j in range(vpos-1, max(0, vpos-4), -1):
            if toks[j].strip().lower() == "is": is_pos = j; break
        if is_pos is None: continue
        # KEY is the token(s) before " is" (after "for"). Take the 1-2 tokens before "is".
        key_pos = is_pos - 1  # last token of the key word
        if key_pos < 0: continue
        s = a0  # answer position (reads from 0..s-1)
        # mean over H,M for the query vectors
        k_all = kf[:s].mean(dim=(1,2))         # [s, K]  (per-token k, averaged over H,M)
        v_all = vf[:s].mean(dim=(1,2))         # [s, K]
        q_ans = qf[s-1].mean(dim=(0,1))        # [K]  (answer-position query; s-1 predicts first digit)
        k_key = kf[key_pos].mean(dim=(0,1))    # [K]  (key mention's k)
        k_val = kf[vpos].mean(dim=(0,1))       # [K]  (value's own k)
        scale = K ** 0.5
        # probe 1: self-retrieval (query = value's own k)
        sc_self = (k_all * k_val.unsqueeze(0)).sum(-1) / scale   # [s]
        # probe 2: key->value binding (query = key's k) -- expect peak at vpos
        sc_bind = (k_all * k_key.unsqueeze(0)).sum(-1) / scale
        # probe 3: actual query
        sc_act = (k_all * q_ans.unsqueeze(0)).sum(-1) / scale
        # ranks (0 = best) of vpos among 0..s-1
        def rank_of(scores, target):
            return float((scores[:s] > scores[target]).sum().item()) / s
        def peak(scores, target):
            return float(scores[:s].argmax().item() == target)
        res["self_peak"].append(peak(sc_self, vpos)); res["self_rank"].append(rank_of(sc_self, vpos))
        res["bind_peak"].append(peak(sc_bind, vpos)); res["bind_rank"].append(rank_of(sc_bind, vpos))
        res["actual_peak"].append(peak(sc_act, vpos)); res["actual_rank"].append(rank_of(sc_act, vpos))
        # value distinctness: cosine of v at vpos vs v at OTHER 4-digit-value starts
        v_gold = v_all[vpos]
        v_gold_n = v_gold / (v_gold.norm()+1e-9)
        # find OTHER 4-digit-value starts (a digit token followed by 3 more digit tokens,
        # not preceded by a digit) -- excludes same-value digits
        import re
        for j in range(s-4):
            if j == vpos: continue
            if not toks[j].strip().isdigit(): continue
            if not all(toks[j+d].strip().isdigit() for d in range(1,4)): continue
            if j > 0 and toks[j-1].strip().isdigit(): continue  # mid-value digit
            # confirm it's a DIFFERENT value (different 4 ids)
            if ids[j:j+4] == ids[vpos:vpos+4]: continue
            vj = v_all[j]; vj_n = vj/(vj.norm()+1e-9)
            res["v_cosine_diff"].append(float((v_gold_n*vj_n).sum().item()))
        # same value: other digits of the gold value (vpos+1..vpos+3)
        for d in range(1, min(4, len(gold))):
            if vpos+d < s:
                vd = v_all[vpos+d]; vd_n = vd/(vd.norm()+1e-9)
                res["v_cosine_same"].append(float((v_gold_n*vd_n).sum().item()))
    model.train()
    import statistics as st
    out = {}
    for k in ["self_peak","self_rank","bind_peak","bind_rank","actual_peak","actual_rank","v_cosine_same","v_cosine_diff"]:
        out[k] = round(st.mean(res[k]), 4) if res[k] else None
    out["nprobe"] = len(res["self_peak"])
    return out

print("=== probe at INIT ==="); p_init = probe(); print(json.dumps(p_init, indent=2))

print(f"=== training {steps} steps ==="); last_ce=float('nan')
opt.zero_grad(set_to_none=True)
for step in range(steps):
    ids, a0, gold = pm.make_mqar(rng, tok, n_keys, seq_len)
    x = torch.tensor([ids], device=DEV)
    logits = model(input_ids=x).logits
    lg = logits[0, a0-1:a0-1+len(gold)]
    loss = F.cross_entropy(lg.float(), torch.tensor(gold, device=DEV))
    if torch.isfinite(loss):
        loss.backward(); last_ce=float(loss.detach())
        gnorm = torch.nn.utils.clip_grad_norm_(mem_params+cop_params, clip)
        if torch.isfinite(gnorm):
            opt.step()
    sched.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0 or step == steps-1:
        print(f"  step {step}: ce {last_ce:.3f}")

print("=== probe AFTER training ==="); p_post = probe(); print(json.dumps(p_post, indent=2))
print()
print("=== DECISION ===")
sp, bp, ap_ = p_post["self_peak"], p_post["bind_peak"], p_post["actual_peak"]
print(f"self-retrieval peak (value dist. by own k):  {sp}")
print(f"key->value binding peak (key k -> value pos): {bp}")
print(f"actual query peak (q_ans -> value pos):       {ap_}")
if sp is not None and sp < 0.3:
    print("=> SELF-RETRIEVAL FAILS: values not distinguishable by their own k -> frozen-v encoding ceiling (HUMAN: unfreeze in_proj_qkv or value proj)")
elif bp is not None and sp is not None and sp > 0.5 and bp < 0.3:
    print("=> BINDING FAILS (self ok, key->value fails): key/value not associated -> WRITE-side lever (coproduct binding)")
elif bp is not None and bp > 0.5 and (ap_ is None or ap_ < 0.3):
    print("=> BINDING STORED but actual query misses -> read/loss ceiling, HANDOFF AIRTIGHT")
else:
    print("=> ambiguous/partial; compare ranks for nuance")

result = {
  "config": {**cfg, "name": "ideal_query_probe_exp032",
             "hypothesis": "WRITE/binding-side decisive test. Is the correct value RETRIEVABLE? Probe with (1) value's own k (self-retrieval: values distinguishable?), (2) key's k (key->value binding stored?), (3) actual q (exp029 baseline). Separates binding-not-stored (write-side lever) from binding-stored-query-wrong (read/loss ceiling, handoff airtight)."},
  "status": "ok", "device": args.device,
  "probe_init": p_init, "probe_post": p_post,
  "final_tokacc": 0.0, "final_recall": 0.0,
  "skip_rate": 0.0, "final_ce": round(last_ce,4), "wall_s": round(time.time()-t0,1),
}
json.dump(result, open(args.out,"w"), indent=2)
print(f"\n{json.dumps({k:result[k] for k in ('status','probe_post','wall_s')})}")
