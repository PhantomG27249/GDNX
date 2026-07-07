"""exp030-diag: WHY did the sharp softmax (exp030) give the SAME ~0.28/recall-0
as the linear (exp027) and soft-softmax (exp028) reads? All three read MECHANISMS
are equivalent -> the retrieval read is a MINORITY term, swamped by kron_q (the
Kronecker compressed-state read, LINEAR in q -> format marginal).

Hypothesis: y = kron_q + y_retrieval + self_term, and ||kron_q|| >> ||y_retrieval||
so the retrieval mechanism (linear/softmax/sharp) doesn't change the output.

This diagnostic captures the recurrence inputs (q,k,v,dec) from one upgraded layer
on a fresh episode and RECOMPUTES the three read terms externally (no source edits)
to measure their relative norms. Read-only, fast (~10s).
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

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
for p in model.parameters(): p.requires_grad_(False)
model.config.use_cache = False
model.to(DEV).eval()

# capture recurrence inputs from a middle upgraded layer
CAP_LAYER = upg[len(upg)//2]
cap = {}
layer_mod = model.model.layers[CAP_LAYER].linear_attn
orig_fn = layer_mod._gdn3_recurrent_state
def capturing_fn(qf, kf, vf, bg, wg, dec):
    cap['q'] = qf.detach(); cap['k'] = kf.detach()
    cap['v'] = vf.detach(); cap['bg'] = bg.detach(); cap['wg'] = wg.detach(); cap['dec'] = dec.detach()
    return orig_fn(qf, kf, vf, bg, wg, dec)
layer_mod._gdn3_recurrent_state = capturing_fn

rng = random.Random(0)
ids, a0, gold = pm.make_mqar(rng, tok, n_keys=4, seq_len=64)
ids_t = torch.tensor([ids], device=DEV)
with torch.no_grad():
    _ = model(ids_t)

qf = cap['q'][0]; kf = cap['k'][0]; vf = cap['v'][0]  # [T,H,M,K]
dec = cap['dec'][0]  # [T,H,M]
bg = cap['bg'][0]; wg = cap['wg'][0]
T, H, M, K = qf.shape
V = vf.shape[-1]
print(f"layer {CAP_LAYER}: T={T} H={H} M={M} K={K} V={V}")
print(f"a0(answer)={a0} gold_len={len(gold)}")
print()

# Recompute the three read terms at the first answer position (s=a0), per the
# live chunked read structure (one chunk, P=64, C=T=64, no compaction):
#   y = kron_q + y_retrieval + self_term
# We approximate each term's norm at position s=a0 (the first answer read).
# For a fair comparison we compute per-(H,M) chain and report mean norms.

s = a0  # first answer position (reads from all prior positions)
P = layer_mod.P  # residual_rank (64)

# --- kron_q: Kronecker read. At init A,Bk=0 -> kron_q=0. But we can measure
#     its magnitude by calling the layer's _kron_read on the chunk. Since A,Bk
#     start at 0, kron_q=0 at init. So at init, the retrieval + self dominate.
#     Let's just compute all three from the recurrence formula at init.
# At init: A,Bk = 0 -> kron_q = 0, kron_h = 0, old_h = 0 (U=0).
# alpha = stable_alpha((k*h).sum) ; new_u = alpha * (u - 0 - 0) = alpha * u
# self_term = new_u * kq  where kq = (k*q).sum
# retrieval (linear): prev_q = sum_{j<i} new_u[j] * (k[j]*q[i]*rel_decay) ; old_q=0
# retrieval (softmax): softmax over scores * new_u
# y = kron_q(=0) + retrieval + self_term

# So at INIT, kron_q = 0! The output is ONLY retrieval + self. If all three read
# mechanisms give the same ~0.28, and kron_q=0 at init, then the self_term or the
# format-from-LM-head dominates, NOT kron_q. Let me measure self vs retrieval.

# compute at position s (causal: read from 0..s-1)
q_s = qf[s]  # [H,M,K]
k_all = kf[:s]  # [s,H,M,K]
v_all = vf[:s]
bg_all = bg[:s]
wg_all = wg[:s]
dec_all = dec[:s]  # [s,H,M]

# h = bg * k ; u = wg * v
h_all = bg_all * k_all   # [s,H,M,K]
u_all = wg_all * v_all   # [s,H,M,V]

# alpha = stable_alpha((k*h).sum(-1))  per token
c_all = (k_all * h_all).sum(-1)  # [s,H,M]
# stable_alpha: -expm1(-c)/c
alpha_all = -torch.expm1(-c_all.clamp(min=1e-6)) / c_all.clamp(min=1e-6)  # approx

# new_u = alpha * u  (since kron_h=old_h=0 at init)
new_u_all = alpha_all.unsqueeze(-1) * u_all  # [s,H,M,V]

# self_term at s: new_u[s] * kq[s]  -- but s reads from PRIOR, self is j=s which
# is the current token itself. In the live code, the self-term is new_u * kq where
# kq = (k_c * q_c).sum(-1) computed for ALL positions including s. At position s,
# self = new_u[s] * (k[s]*q[s]).sum. But new_u[s] depends on u[s] (the current
# token's value) -- this is the "self-read" (current token reads its own value).
kq_self = (kf[s] * qf[s]).sum(-1)  # [H,M]
self_term = new_u_all[s-1] * kq_self.unsqueeze(-1)  # wait, need new_u at s
# Actually new_u is per-position; at position s, new_u[s] = alpha[s]*u[s]
alpha_s = alpha_all[s-1] if s <= len(alpha_all) else alpha_all[-1]
# Let me just compute at position s using prior positions for retrieval:
# retrieval (linear): prev_q[s] = sum_{j<s} new_u[j] * (k[j] dot q[s]) * decay
# For simplicity (no decay at init since dec~0.95, rel_decay ~0.95^(s-j)):
gamma = dec_all  # [s,H,M]
# rel_decay[i,j] = prod gamma[j+1..i]. For position s reading j: prod gamma[j+1..s]
# approximate: cumulative product from j to s
# This is getting complex; let me just measure the NORMS of the raw components:
#   self_term ~ ||new_u[s]|| * |kq[s]|
#   retrieval ~ ||sum_j new_u[j] * score||  where score ~ |k[j].q[s]| ~ 0.01
#   kron_q = 0 at init

print("=== At INIT (A,Bk=0, U=0 -> kron_q=0, old_q=0) ===")
print("y = kron_q(=0) + retrieval + self_term")
print()

# self_term norm (per H,M, mean)
self_norm = self_term.norm(dim=-1).mean().item()  # but self_term shape may be off
# Let me recompute cleanly at position s (0-indexed, s=a0):
# Prior positions: 0..s-1. new_u[j] for j in 0..s-1.
# self at s: new_u[s] * kq[s]. But new_u[s] requires u[s]=wg[s]*v[s].
u_s = wg[s] * vf[s]  # [H,M,V]
c_s = (kf[s] * (bg[s]*kf[s])).sum(-1)  # [H,M]
alpha_s = -torch.expm1(-c_s.clamp(min=1e-6)) / c_s.clamp(min=1e-6)
new_u_s = alpha_s.unsqueeze(-1) * u_s  # [H,M,V]
kq_s = (kf[s] * qf[s]).sum(-1)  # [H,M]
self_term_s = new_u_s * kq_s.unsqueeze(-1)  # [H,M,V]
self_norm = self_term_s.norm(dim=-1).mean().item()

# retrieval (linear) at s: sum_{j<s} new_u[j] * (k[j].q[s]) * rel_decay
# scores j<s:
scores = torch.einsum('jhmk,hmk->jhm', k_all, qf[s])  # [s,H,M] raw k.q
# rel_decay approx: prod gamma[j+1..s] ~ 0.95^(s-j)
rel = torch.ones(s, H, M, device=DEV)
for j in range(s):
    for jp in range(j+1, s):
        rel[j] = rel[j] * gamma[jp]
scores = scores * rel  # decay-weighted
# prev_q = sum_j new_u[j] * scores[j]
retr_linear = torch.einsum('jhmv,jhm->hmv', new_u_all, scores)  # [H,M,V]
retr_lin_norm = retr_linear.norm(dim=-1).mean().item()

# retrieval (sharp softmax) at s:
combined = scores  # [s,H,M]
w = F.softmax(combined / 0.1, dim=0)  # sharp, over s positions
retr_sharp = torch.einsum('jhmv,jhm->hmv', new_u_all, w)  # [H,M,V]
retr_sharp_norm = retr_sharp.norm(dim=-1).mean().item()

print(f"  ||self_term||     = {self_norm:.4f}")
print(f"  ||retrieval_lin|| = {retr_lin_norm:.4f}")
print(f"  ||retrieval_sharp||= {retr_sharp_norm:.4f}")
print(f"  ||kron_q||        = 0.0 (A,Bk=0 at init)")
print()
print(f"  self/retrieval ratio = {self_norm/(retr_lin_norm+1e-9):.2f}x")
print(f"  sharp/linear retrieval ratio = {retr_sharp_norm/(retr_lin_norm+1e-9):.2f}x")
print()
if self_norm > 3 * retr_lin_norm:
    print(">>> SELF_TERM DOMINATES (3x+ retrieval). The format comes from the")
    print("    self-read (current token reads its OWN value), NOT from retrieval.")
    print("    This is the 'copy-adjacent' shortcut: the answer token reads itself.")
elif retr_lin_norm > 3 * self_norm:
    print(">>> RETRIEVAL DOMINATES over self. Read mechanism SHOULD matter.")
    print("    If exp027/028/030 are still equal, kron_q (after training) dominates.")
else:
    print(f">>> self and retrieval are comparable. Neither dominates at init.")
    print("    After training kron_q (compressed state) likely dominates the format.")
