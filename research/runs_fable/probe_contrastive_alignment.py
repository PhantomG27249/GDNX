"""Decisive probe: is q->k alignment REPRESENTATIONALLY possible from the
frozen backbone's hidden states, independent of the CE loss?

Setup: run the UNTRAINED upgraded model (KMD-2 drop-ins at init; the layers
below any probe point are effectively the frozen Qwen stack + near-init drop-ins)
over many MQAR episodes. At each KMD-2 layer, capture the layer INPUT hidden
state x at (i) the answer position and (ii) every binding's key-name tokens.
Then, per layer, train a small projection pair (Wq, Wk) with InfoNCE so that
Wq·x_answer matches Wk·x_gold-binding against in-episode distractors + batch
negatives. Report held-out retrieval accuracy per layer.

Reading:
  acc >> chance  -> x DOES linearly encode the queried key at the answer
                    position; the ceiling is the CE gradient signal (loss-side).
  acc ~= chance  -> the frozen backbone never transports key identity to the
                    answer position; NO drop-in arch can pass this proxy.
Chance = 1/n_keys-ish (gold among distractors) for in-episode eval.
"""
import os, sys, re, random, time
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
from proxy_mqar import make_mqar

DEV = "cuda:0"          # launch with CUDA_VISIBLE_DEVICES=1
N_KEYS, SEQ_LEN = 4, 512
N_TRAIN_EP, N_EVAL_EP = 300, 100
PROJ_DIM = 64

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
model.config.use_cache = False
model.to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)

# capture layer INPUT (pre-forward hook) at each upgraded layer
capt = {}
def mk_pre(li):
    def pre(mod, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        capt[li] = x.detach()
    return pre
hooks = [model.model.layers[i].linear_attn.register_forward_pre_hook(mk_pre(i), with_kwargs=True)
         for i in upg]

BIND_RE = re.compile(r"The code for ([a-z]+-[a-z]+) is (\d+)\.")

MAX_BINDS = 16

def collect(rng, n_ep):
    """Returns per-layer stacks: anchors [E,D], cands [E,MAX_BINDS,D] (zero-padded),
    plus gold_idx [E] and valid-candidate mask [E,MAX_BINDS]."""
    anchors = {li: [] for li in upg}
    cands = {li: [] for li in upg}
    gold_idx, masks = [], []
    kept = 0
    for _ in range(n_ep):
        ids, a0, _ = make_mqar(rng, tok, N_KEYS, SEQ_LEN)
        text = tok.decode(ids[:a0])
        mq = re.search(r"The code for ([a-z]+-[a-z]+) is:$", text)
        if mq is None: continue
        qk = mq.group(1)
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
        binds = []   # (is_gold, tok_span)
        for m in BIND_RE.finditer(text):
            span = [i for i, (s, e) in enumerate(offs) if s < m.end() and e > m.start()]
            if span:
                binds.append((m.group(1) == qk, (span[0], span[-1])))
        binds = binds[:MAX_BINDS]
        if sum(g for g, _ in binds) != 1 or len(binds) < 3: continue
        with torch.no_grad():
            model(input_ids=torch.tensor([ids[:a0]], device=DEV))
        gi = next(i for i, (g, _) in enumerate(binds) if g)
        gold_idx.append(gi)
        mask = torch.zeros(MAX_BINDS, device=DEV); mask[:len(binds)] = 1.0
        masks.append(mask)
        for li in upg:
            x = capt[li][0]                       # [T, D]
            anchors[li].append(x[a0 - 1])
            pooled = [x[s:e + 1].mean(0) for _, (s, e) in binds]
            pooled += [torch.zeros_like(pooled[0])] * (MAX_BINDS - len(pooled))
            cands[li].append(torch.stack(pooled))
        kept += 1
    return anchors, cands, torch.tensor(gold_idx, device=DEV), torch.stack(masks), kept

print("collecting hidden states (untrained drop-ins; frozen backbone)...", flush=True)
t0 = time.time()
tr_anchor, tr_cand, tr_gold, tr_mask, ntr = collect(random.Random(0), N_TRAIN_EP)
ev_anchor, ev_cand, ev_gold, ev_mask, nev = collect(random.Random(5555), N_EVAL_EP)
print(f"kept train={ntr} eval={nev} episodes ({time.time()-t0:.0f}s)", flush=True)
print(f"mean bindings/episode: {tr_mask.sum(1).mean().item():.1f} "
      f"(chance = 1/that = {1.0/tr_mask.sum(1).mean().item():.3f})", flush=True)
for h in hooks: h.remove()

D = model.config.hidden_size
print(f"\n=== per-layer contrastive alignment probe (InfoNCE, {PROJ_DIM}-dim) ===")
print(f"{'layer':>5} {'train_acc':>9} {'eval_acc':>9}")
results = {}
for li in upg:
    A_tr = torch.stack(tr_anchor[li]); C_tr = torch.stack(tr_cand[li])   # [E,D],[E,MB,D]
    A_ev = torch.stack(ev_anchor[li]); C_ev = torch.stack(ev_cand[li])
    Wq = torch.zeros(D, PROJ_DIM, device=DEV, requires_grad=True)
    Wk = torch.zeros(D, PROJ_DIM, device=DEV, requires_grad=True)
    torch.nn.init.normal_(Wq, std=0.02); torch.nn.init.normal_(Wk, std=0.02)
    opt = torch.optim.AdamW([Wq, Wk], lr=1e-3, weight_decay=0.01)
    neg_inf = torch.finfo(torch.float32).min
    for it in range(300):
        qz = F.normalize(A_tr @ Wq, dim=-1)                    # [E,P]
        kz = F.normalize(C_tr @ Wk, dim=-1)                    # [E,MB,P]
        logits = torch.einsum('ep,ebp->eb', qz, kz) / 0.1      # [E,MB]
        logits = logits.masked_fill(tr_mask == 0, neg_inf)     # hide padded slots
        loss = F.cross_entropy(logits, tr_gold)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        def acc(A_, C_, g_, m_):
            qz = F.normalize(A_ @ Wq, dim=-1); kz = F.normalize(C_ @ Wk, dim=-1)
            lg = torch.einsum('ep,ebp->eb', qz, kz).masked_fill(m_ == 0, neg_inf)
            return (lg.argmax(-1) == g_).float().mean().item()
        ta = acc(A_tr, C_tr, tr_gold, tr_mask)
        ea = acc(A_ev, C_ev, ev_gold, ev_mask)
    results[li] = (ta, ea)
    print(f"{li:>5} {ta:>9.3f} {ea:>9.3f}", flush=True)

best = max(results.items(), key=lambda kv: kv[1][1])
print(f"\nBEST layer {best[0]}: eval_acc {best[1][1]:.3f}")
print("verdict: eval_acc >> chance -> representation FINE, ceiling is the CE loss signal;")
print("         eval_acc ~= chance -> frozen backbone never moves key identity to the answer position.")
