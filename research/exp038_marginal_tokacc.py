"""exp038: is the tokacc ~0.30 PLATEAU pure format (marginal digits) or partial retrieval?

Unexplained: tokacc ~0.30 but the format shortcut (output any 4-digit number)
with a UNIFORM digit marginal gives tokacc ~0.10 (1/9 for first digit 1-9, 1/10
for others). The extra ~0.20 above marginal is unaccounted for. Two possibilities:
  (A) FORMAT (non-uniform marginal): the values 1000-9999 don't have a uniform
      digit distribution; outputting the per-position MARGINAL digit gives >0.10.
      If marginal-tokacc ~0.30, the model is JUST outputting the marginal (pure
      format, handoff final).
  (B) PARTIAL RETRIEVAL: the 5 inducing heads (exp036) contribute some correct
      digits beyond the marginal. If marginal-tokacc ~0.10 and model tokacc ~0.30,
      there IS a retrieval signal (reopens the editable side: the inducing heads
      ARE contributing; amplifying them might work after all).

We compute the per-position marginal digit distribution from many MQAR episodes
and its tokacc (= fraction correct if you always output the per-position argmax
digit). This is the FORMAT-ONLY baseline. Compare to the model's ~0.30.

Standalone (no training, no source edits, ~20s). Output: research/runs/exp038.json.
"""
import sys, os, json, time, random, collections
sys.path.insert(0, '/home/dev/gdn3_two_timescale_release')
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager
import research.proxy_mqar as pm

SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
DEV = "cuda:1"
t0 = time.time()

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade()
for p in model.parameters(): p.requires_grad_(False)
model.config.use_cache = False; model.to(DEV).eval()

# --- 1. Compute the per-position marginal digit distribution from MANY episodes ---
# The answer is ' VVVV' = [220 (space), d0, d1, d2, d3]. The model predicts each
# digit token (ids for '0'..'9' = 15..24). Per-position marginal = frequency of
# each digit at that position across episodes.
N_EP = 2000
rng = random.Random(2024)
digit_pos_counts = [collections.Counter() for _ in range(4)]  # 4 digit positions
gold_tokens = []  # list of (gold_ids) for the actual-eval comparison
for _ in range(N_EP):
    ids, a0, gold = pm.make_mqar(rng, tok, n_keys=4, seq_len=512)
    # gold = [220, d0, d1, d2, d3]; the 4 digits are gold[1:5]
    if len(gold) >= 5:
        for pos in range(4):
            digit_id = gold[1 + pos]
            digit_pos_counts[pos][digit_id] += 1

# per-position argmax digit (the marginal-optimal prediction)
marginal_argmax = [c.most_common(1)[0][0] for c in digit_pos_counts]
marginal_dist = [{d: c[d] / sum(c.values()) for d in c} for c in digit_pos_counts]
print("=== Per-position marginal digit distribution (from", N_EP, "episodes) ===")
for pos in range(4):
    top = digit_pos_counts[pos].most_common(3)
    print(f"  pos {pos}: argmax digit {tok.decode([marginal_argmax[pos]])!r} (p={marginal_dist[pos][marginal_argmax[pos]]:.3f})  top3={[(tok.decode([d]),round(p,3)) for d,p in top]}")

# --- 2. Marginal tokacc: if you always output the per-position argmax digit ---
rng2 = random.Random(999)
marginal_correct = 0; marginal_total = 0
for _ in range(500):
    ids, a0, gold = pm.make_mqar(rng2, tok, n_keys=4, seq_len=512)
    if len(gold) >= 5:
        for pos in range(4):
            marginal_correct += int(gold[1 + pos] == marginal_argmax[pos])
            marginal_total += 1
marginal_tokacc = marginal_correct / max(1, marginal_total)
print(f"\n=== Marginal (format-only) tokacc: {marginal_tokacc:.4f} ===")
print(f"    (always output the per-position most-common digit)")

# --- 3. Random tokacc (uniform) for reference ---
rng3 = random.Random(123)
digit_ids = [tok.encode(' ' + str(d), add_special_tokens=False)[1] for d in range(10)]  # '0'..'9' ids
random_correct = 0; random_total = 0
for _ in range(500):
    ids, a0, gold = pm.make_mqar(rng3, tok, n_keys=4, seq_len=512)
    if len(gold) >= 5:
        for pos in range(4):
            # uniform over the 10 digit ids
            guess = random.Random(hash((_,pos)) % 2**32).choice(digit_ids)
            random_correct += int(guess == gold[1 + pos])
            random_total += 1
random_tokacc = random_correct / max(1, random_total)
print(f"=== Random (uniform) tokacc:        {random_tokacc:.4f} ===")

print(f"\n=== VERDICT ===")
print(f"  random (uniform):   {random_tokacc:.3f}")
print(f"  marginal (format):  {marginal_tokacc:.3f}  <- format-only baseline")
print(f"  model (trained):    ~0.29-0.30  <- from leaderboard")
print(f"  gap (model-marginal): ~{0.30 - marginal_tokacc:.3f}")
if marginal_tokacc > 0.25:
    print("=> MARGINAL ~0.30: the model's 0.30 is PURE FORMAT (outputting the per-position")
    print("   most-common digit). NO partial retrieval. Handoff AIRTIGHT (final).")
elif marginal_tokacc < 0.15:
    print(f"=> MARGINAL ~{marginal_tokacc:.3f} but model ~0.30: there IS partial retrieval")
    print("   (model gets ~0.15-0.20 above marginal). The inducing heads (exp036) ARE")
    print("   contributing correct digits. REOPENS the editable side — amplifying them might work.")
else:
    print(f"=> MARGINAL ~{marginal_tokacc:.3f}: partial format + partial retrieval. Compare carefully.")

result = {
  "config": {"name":"marginal_tokacc_exp038","hypothesis":"Is the tokacc 0.30 plateau PURE FORMAT (marginal digits) or PARTIAL RETRIEVAL? Compute per-position marginal digit tokacc. If marginal~0.30 -> pure format, handoff final. If marginal~0.10 and model~0.30 -> partial retrieval exists, reopens editable side.","steps":0,"n_keys":4,"seq_len":512},
  "status":"ok","device":DEV,"n_episodes_marginal":N_EP,
  "marginal_tokacc": round(marginal_tokacc,4),
  "random_tokacc": round(random_tokacc,4),
  "model_tokacc_approx": 0.29,
  "marginal_argmax_digits": [tok.decode([marginal_argmax[p]]) for p in range(4)],
  "marginal_dist_top": [[(tok.decode([d]),round(p,3)) for d,p in digit_pos_counts[p].most_common(3)] for p in range(4)],
  "final_tokacc":0.0,"final_recall":0.0,"skip_rate":0.0,"final_ce":0.0,
  "wall_s":round(time.time()-t0,1),
}
json.dump(result, open("research/runs/exp038.json","w"), indent=2)
print(f"\n{json.dumps({k:result[k] for k in ('marginal_tokacc','random_tokacc','wall_s')})}")
