"""Train per-layer (Wq, Wk) InfoNCE alignment probes on the frozen backbone's
hidden states and SAVE them as an init for KMD-2's q_proj/k_slots.

Same method as probe_contrastive_alignment.py (95% held-out gold-binding
accuracy) but: (1) keys are PER-TOKEN (not span-pooled) with the span's max
similarity used in the loss, matching how the recurrence actually writes one
k-slot per token; (2) probes for all layers are saved to qk_probe_init.pt.
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

DEV = "cuda:0"
N_KEYS, SEQ_LEN = 4, 512
N_TRAIN_EP, N_EVAL_EP = 300, 100
PROJ_DIM = 64
MAX_BINDS, MAX_SPAN = 16, 12
OUT = "/home/dev/gdn3_fable/research/runs_fable/qk_probe_init.pt"

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
model.config.use_cache = False
model.to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)

capt = {}
def mk_pre(li):
    def pre(mod, args, kwargs):
        capt[li] = (args[0] if args else kwargs["hidden_states"]).detach()
    return pre
hooks = [model.model.layers[i].linear_attn.register_forward_pre_hook(mk_pre(i), with_kwargs=True)
         for i in upg]

BIND_RE = re.compile(r"The code for ([a-z]+-[a-z]+) is (\d+)\.")

def collect(rng, n_ep):
    """anchors[li] [E,D]; tokens[li] [E,MAX_BINDS,MAX_SPAN,D] zero-padded;
    tok_mask [E,MAX_BINDS,MAX_SPAN]; bind_mask [E,MAX_BINDS]; gold [E]."""
    anchors = {li: [] for li in upg}; tokens = {li: [] for li in upg}
    tok_masks, bind_masks, gold_idx = [], [], []
    kept = 0
    for _ in range(n_ep):
        ids, a0, _ = make_mqar(rng, tok, N_KEYS, SEQ_LEN)
        text = tok.decode(ids[:a0])
        mq = re.search(r"The code for ([a-z]+-[a-z]+) is:$", text)
        if mq is None: continue
        qk = mq.group(1)
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
        binds = []
        for m in BIND_RE.finditer(text):
            span = [i for i, (s, e) in enumerate(offs) if s < m.end() and e > m.start()]
            if span:
                binds.append((m.group(1) == qk, span[:MAX_SPAN]))
        binds = binds[:MAX_BINDS]
        if sum(g for g, _ in binds) != 1 or len(binds) < 3: continue
        with torch.no_grad():
            model(input_ids=torch.tensor([ids[:a0]], device=DEV))
        gold_idx.append(next(i for i, (g, _) in enumerate(binds) if g))
        tm = torch.zeros(MAX_BINDS, MAX_SPAN, device=DEV)
        bm = torch.zeros(MAX_BINDS, device=DEV)
        for bi, (_, span) in enumerate(binds):
            tm[bi, :len(span)] = 1.0; bm[bi] = 1.0
        tok_masks.append(tm); bind_masks.append(bm)
        for li in upg:
            x = capt[li][0]
            anchors[li].append(x[a0 - 1])
            T = torch.zeros(MAX_BINDS, MAX_SPAN, x.shape[-1], device=DEV)
            for bi, (_, span) in enumerate(binds):
                T[bi, :len(span)] = x[span]
            tokens[li].append(T)
        kept += 1
    return (anchors, tokens, torch.stack(tok_masks), torch.stack(bind_masks),
            torch.tensor(gold_idx, device=DEV), kept)

print("collecting hidden states...", flush=True)
t0 = time.time()
trA, trT, trTM, trBM, trG, ntr = collect(random.Random(0), N_TRAIN_EP)
evA, evT, evTM, evBM, evG, nev = collect(random.Random(5555), N_EVAL_EP)
print(f"kept train={ntr} eval={nev} ({time.time()-t0:.0f}s)", flush=True)
for h in hooks: h.remove()

D = model.config.hidden_size
neg_inf = torch.finfo(torch.float32).min
save = {}
print(f"\n{'layer':>5} {'train_acc':>9} {'eval_acc':>9}   (per-token keys, span max-pool)")
for li in upg:
    A_tr = torch.stack(trA[li]); T_tr = torch.stack(trT[li])   # [E,D],[E,MB,MS,D]
    A_ev = torch.stack(evA[li]); T_ev = torch.stack(evT[li])
    Wq = torch.randn(D, PROJ_DIM, device=DEV) * 0.02; Wq.requires_grad_(True)
    Wk = torch.randn(D, PROJ_DIM, device=DEV) * 0.02; Wk.requires_grad_(True)
    opt = torch.optim.AdamW([Wq, Wk], lr=1e-3, weight_decay=0.01)
    def logits_of(A_, T_, tm, bm):
        qz = F.normalize(A_ @ Wq, dim=-1)                        # [E,P]
        kz = F.normalize(T_ @ Wk, dim=-1)                        # [E,MB,MS,P]
        sim = torch.einsum('ep,ebsp->ebs', qz, kz)
        sim = sim.masked_fill(tm == 0, neg_inf).amax(-1)         # [E,MB] max over span tokens
        return sim.masked_fill(bm == 0, neg_inf)
    for it in range(300):
        loss = F.cross_entropy(logits_of(A_tr, T_tr, trTM, trBM) / 0.1, trG)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ta = (logits_of(A_tr, T_tr, trTM, trBM).argmax(-1) == trG).float().mean().item()
        ea = (logits_of(A_ev, T_ev, evTM, evBM).argmax(-1) == evG).float().mean().item()
    save[li] = {"Wq": Wq.detach().cpu(), "Wk": Wk.detach().cpu(),
                "train_acc": ta, "eval_acc": ea}
    print(f"{li:>5} {ta:>9.3f} {ea:>9.3f}", flush=True)

torch.save(save, OUT)
print(f"\nsaved per-layer probes -> {OUT}")
print("mean eval_acc:", sum(v["eval_acc"] for v in save.values()) / len(save))
