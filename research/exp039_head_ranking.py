"""exp039-step1: find the CONSISTENTLY-inducing heads across many episodes.

exp036 measured per-episode best-head induction gap (0.18) and that ~4.8/16 heads
induce per episode — but the INDUCING HEADS may DIFFER per episode. exp037 zeroed a
RANDOM half (not the inducing ones) -> 0.30/0.0. To do TARGETED head selection we
need the heads that induce CONSISTENTLY across episodes (high mean induction gap).

This ranks all 16 heads by mean induction gap over many episodes. The top-K are the
candidates to KEEP (zero _agg_proj for the rest). Standalone, read-only, ~30s.
Output: prints the head ranking + writes research/runs/exp039_heads.json.
"""
import sys, os, json, time, random, re
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
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
for p in model.parameters(): p.requires_grad_(False)
model.config.use_cache = False; model.to(DEV).eval()

CAP_LAYER = upg[len(upg)//2]; cap = {}; layer_mod = model.model.layers[CAP_LAYER].linear_attn
H, M, K = layer_mod.H, layer_mod.M, layer_mod.K
orig_gen = layer_mod._generate_coproduct_channels
def capturing_gen(q_dense, k_dense, v_dense, x):
    cap['q_dense'] = q_dense.detach(); cap['k_dense'] = k_dense.detach()
    return orig_gen(q_dense, k_dense, v_dense, x)
layer_mod._generate_coproduct_channels = capturing_gen
print(f"layer {CAP_LAYER}: H={H} M={M} K={K}")

def cos(a,b):
    an=a/(a.norm(dim=-1,keepdim=True)+1e-9); bn=b/(b.norm(dim=-1,keepdim=True)+1e-9)
    return (an*bn).sum(-1)

@torch.no_grad()
def head_induction_gaps(nprobe=200):
    """For each episode, compute per-head induction gap (match - mean distractor).
    Return [nprobe, H] tensor of per-head gaps."""
    pr = random.Random(314159); all_gaps = []
    for _ in range(nprobe):
        ids,a0,gold = pm.make_mqar(pr, tok, n_keys=4, seq_len=512)
        x = torch.tensor([ids], device=DEV); _ = model(x)
        if 'q_dense' not in cap: continue
        qd = cap['q_dense'][0]; kd = cap['k_dense'][0]  # [T,H,K]
        T = qd.shape[0]; toks = [tok.decode([i]) for i in ids]; text = ''.join(toks)
        cum = [0]*(T+1)
        for i,t in enumerate(toks): cum[i+1]=cum[i]+len(t)
        def char_to_tok(c):
            for i in range(T):
                if cum[i] <= c < cum[i+1]: return i
            return T-1
        bindings=[]
        for m in re.finditer(r'code for (.+?) is', text):
            ks=m.group(1); klt=char_to_tok(m.start(1)+len(ks)-1)
            vm=re.match(r' ?(\d{4})', text[m.end():m.end()+6])
            if vm: bindings.append({'key':ks,'klt':klt,'val':vm.group(1)})
        qm=re.search(r'What is the code for (.+?)\?', text)
        if qm is None: continue
        qkey=qm.group(1); qlt=char_to_tok(qm.start(1)+len(qkey)-1)
        match=None; distractors=[]
        for b in bindings:
            if b['key']==qkey and b['val']:
                if match is None: match=b
            elif b['val']:
                distractors.append(b)
        if match is None or not distractors: continue
        # per-head: cos(qd[qlt] , kd[match.klt])  vs  mean cos(qd[qlt], kd[distractor.klt])
        per_head_match = cos(qd[qlt], kd[match['klt']])  # [H]
        per_head_dist = torch.stack([cos(qd[qlt], kd[d['klt']]) for d in distractors]).mean(0)  # [H]
        all_gaps.append((per_head_match - per_head_dist))  # [H]
    return torch.stack(all_gaps)  # [n, H]

print("=== computing per-head induction gap over 200 episodes ===")
gaps = head_induction_gaps(200)
print(f"episodes: {gaps.shape[0]}")
mean_gap = gaps.mean(0)  # [H]
# also fraction of episodes each head induces (gap>0.05)
induce_frac = (gaps > 0.05).float().mean(0)  # [H]

print(f"\n{'head':>4} {'mean_gap':>10} {'induce_frac':>12}")
ranking = []
for h in range(H):
    ranking.append((h, float(mean_gap[h]), float(induce_frac[h])))
    print(f"{h:>4} {float(mean_gap[h]):>10.4f} {float(induce_frac[h]):>12.3f}")

# sort by mean gap
ranking.sort(key=lambda x: -x[1])
print("\n=== RANKED by mean induction gap ===")
for r in ranking:
    print(f"  head {r[0]:>2}: mean_gap {r[1]:.4f}  induce_frac {r[2]:.3f}")

topk = [r[0] for r in ranking[:5]]
print(f"\nTop-5 inducing heads: {topk}")
print(f"Their mean gaps: {[round(r[1],4) for r in ranking[:5]]}")
print(f"Bottom-11 mean gaps: {[round(r[1],4) for r in ranking[5:]]}")

# is the induction CONSISTENT (same heads across episodes) or random?
top5_gaps = gaps[:, topk].mean(1)  # per-episode mean of top-5
rest_gaps = gaps[:, [r[0] for r in ranking[5:]]].mean(1)
print(f"\nConsistency check: top-5 mean gap {float(top5_gaps.mean()):.4f} (std {float(top5_gaps.std()):.4f})")
print(f"                   rest-11 mean gap {float(rest_gaps.mean()):.4f} (std {float(rest_gaps.std()):.4f})")
print(f"top-5 > rest-11 in {float((top5_gaps > rest_gaps).float().mean())*100:.0f}% of episodes")

result = {
  "config": {"name":"head_ranking_exp039","hypothesis":"Rank heads by consistent induction gap to find the heads to KEEP for targeted head selection.","steps":0,"n_keys":4,"seq_len":512},
  "status":"ok","device":DEV,"n_episodes":int(gaps.shape[0]),
  "head_mean_gap": [round(float(mean_gap[h]),4) for h in range(H)],
  "head_induce_frac": [round(float(induce_frac[h]),3) for h in range(H)],
  "ranked_heads": [r[0] for r in ranking],
  "top5_heads": topk,
  "top5_mean_gaps": [round(r[1],4) for r in ranking[:5]],
  "top5_consistency_pct": round(float((top5_gaps > rest_gaps).float().mean())*100,1),
  "final_tokacc":0.0,"final_recall":0.0,"skip_rate":0.0,"final_ce":0.0,"wall_s":round(time.time()-t0,1),
}
json.dump(result, open("research/runs/exp039_heads.json","w"), indent=2)
print(f"\nwall_s {result['wall_s']}")
