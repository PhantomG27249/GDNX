"""Probe: after training KMD-2 (horizon-fixed) for N steps, at the answer
position measure — per layer, per head:
  1. q·k alignment: does the query's q match the k slots written at the
     queried binding's tokens vs distractor bindings' tokens? (induction test)
  2. value retrievability: does reading the ACTUAL state S with the actual q
     produce something closer to the write at the gold binding than distractors?

Method: run the trained model on fresh MQAR episodes with forward hooks on each
KMD2LinearAttn capturing (q, K slots, Wg*V writes) per position. Compute, at the
final prompt position (the one that must emit the first answer token):
  match_score  = max over gold-binding positions of cos(q, k_slot)
  distract_score = max over distractor-binding positions of cos(q, k_slot)
  gap = match - distract  (positive => induction-style alignment forming)
Report per-layer mean over episodes/heads, plus best head.
"""
import os, sys, re, json, random, time
import torch, torch.nn.functional as F

sys.path.insert(0, "/home/dev/gdn3_fable")
sys.path.insert(0, "/home/dev/gdn3_fable/research")
os.environ["GDN3_KMD2"] = "1"
os.environ["GDN3_KMD2_DECAY_BIAS"] = "6.0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager
from gdn3.kmd2 import KMD2LinearAttn
from proxy_mqar import make_mqar, PRESERVED

DEV = "cuda:0"   # run with CUDA_VISIBLE_DEVICES=1
TRAIN_STEPS = int(os.environ.get("PROBE_TRAIN_STEPS", "200"))
N_KEYS, SEQ_LEN = 4, 512

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
for p in model.parameters(): p.requires_grad_(False)
train = []
for idx in upg:
    for n, p in model.model.layers[idx].linear_attn.named_parameters():
        if any(k in n for k in PRESERVED): continue
        p.requires_grad_(True); train.append(p)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(DEV).train()

opt = torch.optim.AdamW([{"params": train, "lr": 2.5e-4}], betas=(0.9, 0.95), weight_decay=0.01)
rng = random.Random(0)
print(f"training {TRAIN_STEPS} steps (same recipe as proxy)...", flush=True)
t0 = time.time()
for step in range(TRAIN_STEPS):
    ids, a0, gold = make_mqar(rng, tok, N_KEYS, SEQ_LEN)
    x = torch.tensor([ids], device=DEV)
    logits = model(input_ids=x).logits
    lg = logits[0, a0 - 1:a0 - 1 + len(gold)]
    loss = F.cross_entropy(lg.float(), torch.tensor(gold, device=DEV))
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True); continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(train, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        print(f"  step {step} ce={float(loss):.3f} ({time.time()-t0:.0f}s)", flush=True)
model.eval()

# ---- hooks: capture q and K slots (normalized, as used in the scan) ----
capt = {}
def mk_hook(li):
    def hook(mod, args, kwargs, out):
        x = args[0] if args else kwargs["hidden_states"]
        B, T, D = x.shape
        H, dk, r = mod.H, mod.dk, mod.r
        q = F.normalize(mod.q_proj(x).view(B, T, H, dk), p=2, dim=-1, eps=1e-6)
        K = F.normalize(mod.k_slots(x).view(B, T, H, r, dk), p=2, dim=-1, eps=1e-6)
        capt[li] = (q.detach(), K.detach())
    return hook
hooks = [model.model.layers[i].linear_attn.register_forward_hook(mk_hook(i), with_kwargs=True)
         for i in upg]

def episode_spans(prefix_text, ids_prefix, qk):
    """Char->token spans of each binding line; returns (gold_spans, distract_spans)."""
    offs = tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    gold, distract = [], []
    for m in re.finditer(r"The code for ([a-z]+-[a-z]+) is (\d+)\.", prefix_text):
        span = [i for i, (s, e) in enumerate(offs) if s < m.end() and e > m.start()]
        if not span: continue
        (gold if m.group(1) == qk else distract).append((span[0], span[-1]))
    return gold, distract

evalrng = random.Random(7777)
NEP = 24
layer_gap = {i: [] for i in upg}    # mean-over-heads gap per episode
layer_best = {i: [] for i in upg}   # best-head gap per episode
for ep in range(NEP):
    # rebuild episode text to locate spans (mirrors make_mqar internals)
    ids, a0, gold_ids = make_mqar(evalrng, tok, N_KEYS, SEQ_LEN)
    text = tok.decode(ids[:a0])
    m = re.search(r"The code for ([a-z]+-[a-z]+) is:$", text)
    if m is None: continue
    qk = m.group(1)
    gold_spans, distract_spans = episode_spans(text, ids[:a0], qk)
    if not gold_spans or not distract_spans: continue
    with torch.no_grad():
        model(input_ids=torch.tensor([ids[:a0]], device=DEV))
    qpos = a0 - 1
    for li in upg:
        q, K = capt[li]                       # q [1,T,H,dk], K [1,T,H,r,dk]
        qv = q[0, qpos]                       # [H,dk]
        def span_score(spans):
            # max over positions in spans and slots r of cos(q, k)  -> [H]
            best = None
            for (s, e) in spans:
                kk = K[0, s:e + 1]            # [L,H,r,dk]
                sc = torch.einsum('hd,lhrd->lhr', qv, kk).amax(dim=(0, 2))  # [H]
                best = sc if best is None else torch.maximum(best, sc)
            return best
        ms, ds = span_score(gold_spans), span_score(distract_spans)
        gap = ms - ds                          # [H]
        layer_gap[li].append(gap.mean().item())
        layer_best[li].append(gap.max().item())

for h in hooks: h.remove()
print(f"\n=== q->k alignment at answer position ({NEP} episodes, trained {TRAIN_STEPS} steps) ===")
print(f"{'layer':>5} {'mean gap':>9} {'best-head gap':>13}   (gap>0 => query prefers GOLD binding)")
tot = []
for li in upg:
    if not layer_gap[li]: continue
    mg = sum(layer_gap[li]) / len(layer_gap[li])
    bg = sum(layer_best[li]) / len(layer_best[li])
    tot.append(mg)
    print(f"{li:>5} {mg:>9.4f} {bg:>13.4f}")
print(f"\nALL-LAYER mean gap: {sum(tot)/len(tot):.4f}")
print("interpretation: ~0 => no induction alignment; >0.1 => real alignment forming")
