"""exp040: is the frozen LM head ALIGNED to the GDN3 output subspace (for digits)?

The genuinely-untested question. exp038: the 0.30 plateau is partial retrieval
(3.5x format) -> some signal reaches the output. But does the GDN3 output (the
trainable _agg_proj output) at the answer position project onto the frozen LM head's
DIGIT-token rows? If the retrieval signal lives in a subspace orthogonal to the
digit rows, even a perfect retrieval signal couldn't produce digits -> a NEW ceiling
(LM-head alignment, distinct from all prior). If well-aligned, the signal reaches the
LM head and the loss is the only barrier (handoff final).

We measure: at the answer position, the model's final hidden state h[a0-1] feeds the
frozen LM head -> logits -> digit token. The digit logits = h @ W_digit^T. We ask:
how much of h's energy projects onto the digit-row subspace (span of W_digit rows)?
And: is the GDN3 contribution to h aligned with the digit rows?

Concretely (frozen model, no training):
  - h = final hidden state at answer position (post all layers, post GDN3 upgrade).
  - W_digit = LM head rows for the 10 digit tokens (ids for '0'..'9' = 15..24).
  - projection coefficient of h onto each digit row (the logit, pre-softmax).
  - fraction of ||h||^2 in the digit-row subspace (span of 10 digit rows in 1024-D).
  - compare to a RANDOM 10-row subspace (control): is the digit subspace special?

If the digit-subspace fraction is ~chance (10/1024 ~ 1%) and the digit logits are
near-uniform -> the LM head is NOT aligned to retrieve digits from GDN3's output ->
NEW ceiling. If the digit-subspace fraction is high and digit logits vary -> aligned,
loss is the barrier.

Standalone (no source edits, no training, ~15s). Output: research/runs/exp040.json.
"""
import sys, os, json, time, random
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

D = model.config.hidden_size  # 1024
W_lm = model.lm_head.weight    # [vocab, D]
print(f"hidden D={D}, lm_head shape {tuple(W_lm.shape)}, tied={model.config.tie_word_embeddings}")

# digit token ids (the ACTUAL answer tokens): ' 5267' = [220,20,17,21,22]; digits are 15..24
digit_ids = [tok.encode(' ' + str(d), add_special_tokens=False)[1] for d in range(10)]
print("digit ids:", digit_ids, "->", [tok.decode([i]) for i in digit_ids])
W_digit = W_lm[digit_ids]      # [10, D]
# normalize
Wd_n = W_digit / (W_digit.norm(dim=-1, keepdim=True) + 1e-9)   # [10, D]

# digit-row subspace: orthonormal basis via QR
Q_digit, _ = torch.linalg.qr(W_digit.T)   # Q_digit: [D, 10] orthonormal basis of digit-row span
# also a random 10-D subspace (control)
g = torch.Generator(device=DEV).manual_seed(0)
Q_rand, _ = torch.linalg.qr(torch.randn(D, 10, device=DEV, generator=g))

@torch.no_grad()
def probe(nprobe=64):
    pr = random.Random(555)
    h_norms=[]; digit_frac=[]; rand_frac=[]; digit_logit_std=[]; digit_logit_range=[]
    for _ in range(nprobe):
        ids, a0, gold = pm.make_mqar(pr, tok, n_keys=4, seq_len=512)
        x = torch.tensor([ids], device=DEV)
        h = model.model(input_ids=x).last_hidden_state[0, a0-1]   # [D]  (predicts first answer digit)
        h_n = h / (h.norm() + 1e-9)
        h_norms.append(float(h.norm()))
        # fraction of ||h||^2 in the digit subspace
        proj_digit = Q_digit @ (Q_digit.T @ h)   # projection onto digit subspace
        digit_frac.append(float((proj_digit.norm()**2) / (h.norm()**2 + 1e-12)))
        proj_rand = Q_rand @ (Q_rand.T @ h)
        rand_frac.append(float((proj_rand.norm()**2) / (h.norm()**2 + 1e-12)))
        # digit logits (pre-softmax) and their spread
        logits_digit = h @ W_digit.T   # [10]
        digit_logit_std.append(float(logits_digit.std()))
        digit_logit_range.append(float(logits_digit.max() - logits_digit.min()))
    import statistics as st
    return {
        "h_norm": round(st.mean(h_norms),4),
        "digit_subspace_frac": round(st.mean(digit_frac),4),
        "random_subspace_frac": round(st.mean(rand_frac),4),
        "digit_logit_std": round(st.mean(digit_logit_std),4),
        "digit_logit_range": round(st.mean(digit_logit_range),4),
        "nprobe": nprobe,
    }

print("\n=== probe (frozen model, answer-position hidden state) ===")
r = probe()
print(json.dumps(r, indent=2))
print()
# the digit subspace is 10-D out of 1024-D; chance fraction = 10/1024 = 0.0098
chance = 10.0 / D
print(f"chance subspace fraction (10/D): {chance:.4f}")
print(f"digit subspace fraction:        {r['digit_subspace_frac']:.4f}  ({r['digit_subspace_frac']/chance:.1f}x chance)")
print(f"random subspace fraction:       {r['random_subspace_frac']:.4f}  ({r['random_subspace_frac']/chance:.1f}x chance)")
print()
print(f"digit logit std: {r['digit_logit_std']:.4f}  range: {r['digit_logit_range']:.4f}")
print(f"  (if std~0, the 10 digit logits are nearly identical -> LM head can't discriminate digits from this h)")
print()
if r['digit_subspace_frac'] < 2 * chance and r['digit_logit_std'] < 0.5:
    print("=> LM head NOT aligned to GDN3 output for digits: NEW CEILING.")
    print("   The retrieval signal, even if perfect, lives mostly outside the digit-row subspace.")
    print("   Breaking this needs unfreezing the LM head (forbidden) OR a trainable output proj (editable!).")
elif r['digit_subspace_frac'] > 5 * chance and r['digit_logit_std'] > 1.0:
    print("=> LM head WELL-aligned: the signal reaches the digit rows. The loss is the only barrier (HANDOFF FINAL).")
else:
    print(f"=> partial alignment: digit-subspace {r['digit_subspace_frac']/chance:.1f}x chance, logit std {r['digit_logit_std']:.2f}.")
    print("   Some signal reaches digits; interpret with the numbers above.")

result = {
  "config": {"name":"lm_head_alignment_exp040","hypothesis":"Is the frozen LM head ALIGNED to the GDN3 output subspace for digits? If the retrieval signal lives orthogonal to the digit rows, even perfect retrieval can't produce digits -> NEW ceiling (LM-head alignment). If aligned -> loss is the only barrier (handoff final).","steps":0,"n_keys":4,"seq_len":512},
  "status":"ok","device":DEV, **r,
  "chance_subspace_frac": round(chance,4),
  "digit_subspace_x_chance": round(r['digit_subspace_frac']/chance,1),
  "final_tokacc":0.0,"final_recall":0.0,"skip_rate":0.0,"final_ce":0.0,"wall_s":round(time.time()-t0,1),
}
json.dump(result, open("research/runs/exp040.json","w"), indent=2)
print(f"\n{json.dumps({k:result[k] for k in ('digit_subspace_frac','random_subspace_frac','digit_subspace_x_chance','digit_logit_std','wall_s')})}")
