"""exp029-diag: measure the ACTUAL GDN3 read scores (q·k) the recurrence sees,
to ground whether any read-side fix can break the universal recall=0.0.

Key discovery while planning exp029: the read q,k are NOT purely frozen — they're
blended with TRAINABLE coproduct channels (_generate_coproduct_channels, W_q_a/W_k_a
are trainable GDN3 params). So "frozen q·k uninformative" (exp028 diagnosis) is
INCOMPLETE — the model CAN shape q,k. The real question: do the ACTUAL read scores
(post-coproduct, post-RoPE) discriminate the queried key's stored value?

We monkeypatch ONE upgraded layer's _gdn3_recurrent_state to capture q_features,
k_features [B,T,H,M,K] (the exact read inputs), run one MQAR episode, and measure:
  at each ANSWER token position, the softmax(q·k / sqrt(K)) over all prior positions
  — does it peak at the stored VALUE (digit) position, the KEY position, or is it
  uniform (uninformative)?

This is read-only (no source edits, no training). Grounds the exp029 design: if
scores are uniform/uninformative even WITH the trainable coproduct at init, the
learnable-projection fix targets the right thing; if they DO discriminate at init
but recall is still 0, the bottleneck is downstream (value read / frozen LM head).
"""
import sys, os, random
sys.path.insert(0, '/home/dev/gdn3_two_timescale_release')
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager
import research.proxy_mqar as pm

SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
DEV = "cuda:1"
PRESERVED = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "norm", "out_proj")

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
for p in model.parameters():
    p.requires_grad_(False)
model.config.use_cache = False
model.to(DEV).eval()

# capture read inputs from one upgraded layer (a middle one)
CAP_LAYER = upg[len(upg)//2]
cap = {}
layer_mod = model.model.layers[CAP_LAYER].linear_attn
orig_fn = layer_mod._gdn3_recurrent_state
def capturing_fn(qf, kf, vf, bg, wg, dec):
    cap['q'] = qf.detach(); cap['k'] = kf.detach(); cap['v'] = vf.detach()
    cap['dec'] = dec.detach()
    return orig_fn(qf, kf, vf, bg, wg, dec)
layer_mod._gdn3_recurrent_state = capturing_fn
print(f"capturing layer {CAP_LAYER} (upgraded idx; H={layer_mod.H} M={layer_mod.M} K={layer_mod.K})")

rng = random.Random(0)
ids, a0, gold = pm.make_mqar(rng, tok, n_keys=4, seq_len=128)
ids_t = torch.tensor([ids], device=DEV)
with torch.no_grad():
    _ = model(ids_t)

qf = cap['q'][0]  # [T,H,M,K]  (drop batch)
kf = cap['k'][0]  # [T,H,M,K]
vf = cap['v'][0]  # [T,H,M,V]
T, H, M, K = qf.shape
print(f"qf {tuple(qf.shape)}  a0(answer start)={a0}  gold len={len(gold)}")
print(f"q norm mean {qf.norm(dim=-1).mean():.3f}  k norm mean {kf.norm(dim=-1).mean():.3f}")
print()

# tokenize to find structure
toks = [tok.decode([i]) for i in ids]
print("episode tail:", repr(''.join(toks[a0-12:a0+len(gold)+2])))
# find stored VALUE positions: the gold digits appear earlier in the statement
gold_str = tok.decode(gold).strip()
# scan for the gold value substring in the prefix
prefix_str = ''.join(toks[:a0])
print(f"gold value: {gold_str!r}")
# find character offset of the FIRST occurrence of gold_str in prefix (the stored write)
val_char = prefix_str.find(gold_str)
if val_char >= 0:
    # map char offset -> token index (approx, via cumulative token lengths)
    cum = 0
    val_tok = None
    for i, t in enumerate(toks[:a0]):
        if cum + len(t) > val_char:
            val_tok = i; break
        cum += len(t)
    print(f"stored VALUE '{gold_str}' at ~token {val_tok}: {repr(''.join(toks[val_tok:val_tok+6]))}")
else:
    val_tok = None
    print("(gold value not found in prefix as contiguous substring)")

# the queried KEY: last "code for KEY" before the answer
# answer prompt ends "...The code for KEY is:" — find the KEY word just before a0
tail = ''.join(toks[max(0,a0-15):a0])
print(f"query tail: {tail!r}")
print()

# === MEASUREMENT: at each answer position, softmax(q·k/sqrt(K)) over prior positions ===
# average over heads H and lanes M (flatten to [T,T] mean score)
scale = K ** 0.5
# scores[s,t] = mean over H,M of q[s]·k[t] / sqrt(K), for t < s (causal)
scores = torch.einsum('shmk,thmk->st', qf, kf) / scale / (H*M)   # [T,T]
# at answer positions a0..a0+len(gold)-1, look at the distribution over prior positions
print("=== read-score distribution at each answer token (mean over H,M) ===")
print(f"{'pos':>4} {'tok':>6} {'argmax_prior':>12} {'score@val':>10} {'score@max':>10} {'entropy':>8} {'uniform?':>9}")
for di in range(len(gold)):
    s = a0 + di
    if s >= T: break
    prior = scores[s, :s]                      # [s]
    sm = F.softmax(prior, dim=-1)
    argmax = int(prior.argmax())
    s_val = float(prior[val_tok]) if (val_tok is not None and val_tok < s) else float('nan')
    s_max = float(prior.max())
    ent = float(-(sm*torch.log(sm+1e-12)).sum())  # nats; uniform = ln(s)
    unif_ent = float(torch.log(torch.tensor(float(s))))
    is_uniform = "UNIFORM" if ent > 0.9*unif_ent else ""
    print(f"{s:>4} {repr(toks[s]):>6} {argmax:>12} {s_val:>10.3f} {s_max:>10.3f} {ent:>8.3f} {is_uniform:>9}")
print()

# === does the score at the stored value position stand out? ===
if val_tok is not None:
    print("=== discrim: score at stored-VALUE pos vs mean prior score ===")
    for di in range(len(gold)):
        s = a0 + di
        if s >= T: break
        prior = scores[s, :s]
        mean_sc = float(prior.mean()); std_sc = float(prior.std())
        val_sc = float(prior[val_tok])
        z = (val_sc - mean_sc) / (std_sc + 1e-9)
        print(f"  ans{di}: score@val={val_sc:.3f}  mean={mean_sc:.3f}  std={std_sc:.3f}  z={z:+.2f}  {'DISCRIM' if z>1 else 'no'}")
print()

# === best (H,M) chain: does ANY head discriminate the value? ===
if val_tok is not None:
    print("=== per-(H,M) z-score of value-token read, at first answer token ===")
    s = a0
    q_s = qf[s]  # [H,M,K]
    k_all = kf[:s]  # [s,H,M,K]
    sc = torch.einsum('hmk,thmk->ht', q_s, k_all) / scale  # [H,T_prior=s]
    mean = sc.mean(-1, keepdim=True); std = sc.std(-1, keepdim=True)
    zmap = (sc[:,:,None][:,:,0] - mean[:,:,0])/(std[:,:,0]+1e-9)  # [H,s]
    z_val = zmap[:, val_tok]  # [H]
    print(f"  z-score@val per head (best of M lanes shown):")
    for h in range(min(H,8)):
        # take the best lane per head
        best_m = int(z_val[h].argmax()) if z_val.dim()>1 else 0
        print(f"    head{h}: best-lane z@val = {float(z_val[h]):+.2f}")
    print(f"  max z@val over ALL (H,M): {float(z_val.max()):+.2f}  ({'DISCRIMINATES' if z_val.max()>2 else 'no discrimination'})")
