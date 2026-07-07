"""exp033: does the Kronecker coproduct binding form WITH compaction?

exp032 was confounded: no-compaction config (P=64,seq_len=64) has A,Bk=0 -> the GDN3
coproduct binding (which writes into A,Bk) is OFF. This tests the ACTUAL GDN3 binding
mechanism on a COMPACTING config (seq_len=512, P=16 -> 32 compactions, A,Bk populated).

Method: hook one upgraded layer to capture q,k,v,bg,wg,dec [B,T,H,M,K/V]. Replicate the
reference recurrence loop (parity-matched) to rebuild the state (A,Bk,U,Vb) at the
answer position. Then probe:
  (1) ACTUAL query: read(state, q_ans) -> y_actual. cos(y_actual, v@value_pos)?
  (2) IDEAL key query: read(state, k_key) -> y_ideal. cos(y_ideal, v@value_pos)?
  (3) RANDOM query (control): read(state, rand) -> y_rand. cos(y_rand, v@value_pos)?

Decision:
  - y_ideal >> y_rand AND y_ideal ~ v@value -> binding FORMS (Kronecker state has key->value).
    Then if y_actual << y_ideal -> the actual query doesn't match the key -> read/loss ceiling.
  - y_ideal ~ y_rand (no better than random) -> binding DOESN'T form with compaction ->
    editable WRITE-side lever (strengthen coproduct binding).
  - Also isolate: Kron-only (A,Bk, no buffer) vs full (A,Bk+U,Vb) to separate compressed
    binding from exact-buffer retrieval.

Standalone (no source edits). Trains 200 steps, probes. Output: research/runs/exp033.json.
"""
import sys, os, json, time, random, argparse, math
sys.path.insert(0, '/home/dev/gdn3_two_timescale_release')

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="research/configs/exp001.json")  # seq_len=512, P=16 (compacting)
ap.add_argument("--out", default="research/runs/exp033.json")
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
H, M, K, V = layer_mod.H, layer_mod.M, layer_mod.K, layer_mod.V
R, P = layer_mod.R, layer_mod.P
a_k, b_k, a_v, b_v = layer_mod.a_k, layer_mod.b_k, layer_mod.a_v, layer_mod.b_v
orig_fn = layer_mod._gdn3_recurrent_state
def capturing_fn(qf, kf, vf, bg, wg, dec):
    cap['q'] = qf.detach(); cap['k'] = kf.detach(); cap['v'] = vf.detach()
    cap['bg'] = bg.detach(); cap['wg'] = wg.detach(); cap['dec'] = dec.detach()
    return orig_fn(qf, kf, vf, bg, wg, dec)
layer_mod._gdn3_recurrent_state = capturing_fn
print(f"layer {CAP_LAYER}: H={H} M={M} K={K} V={V} R={R} P={P}  (seq_len={seq_len}, compactions~{seq_len//P})")

def find_subseq(hay, needle):
    L = len(needle); out = []
    for i in range(len(hay) - L + 1):
        if hay[i:i+L] == needle: out.append(i)
    return out

@torch.no_grad()
def rebuild_state_at(qf, kf, vf, bg, wg, dec, target_t):
    """Replicate the reference recurrence up to target_t, return state (A,Bk,U,Vb) there.
    Inputs are [B,T,H,M,K/V]; flattened to [T,N,dim] like the reference."""
    B, T, _, _, _ = qf.shape
    N = B * H * M
    def _flat(x, d):
        return x.permute(1, 0, 2, 3, 4).reshape(T, N, d).to(torch.float32)
    q = _flat(qf, K); k = _flat(kf, K); v = _flat(vf, V)
    bgf = _flat(bg, K); wgf = _flat(wg, V)
    decf = dec.permute(1, 0, 2, 3).reshape(T, N).to(torch.float32)
    device = qf.device
    A = torch.zeros(N, R, a_v, a_k, dtype=torch.float32, device=device)
    Bk = torch.zeros(N, R, b_v, b_k, dtype=torch.float32, device=device)
    U = torch.zeros(N, V, P, dtype=torch.float32, device=device)
    Vb = torch.zeros(N, K, P, dtype=torch.float32, device=device)
    p = 0
    for t in range(target_t):
        gamma = decf[t].clamp(0.0, 1.0)
        A = A * gamma.view(N, 1, 1, 1); Bk = Bk * gamma.view(N, 1, 1, 1); Vb = Vb * gamma.view(N, 1, 1)
        k_t, q_t = k[t], q[t]
        h = bgf[t] * k_t; u = wgf[t] * v[t]
        s_h = layer_mod._kron_read_vec(A, Bk, U, Vb, h)
        r = u - s_h
        c = (k_t * h).sum(-1); alpha = layer_mod._stable_alpha_vec(c)
        new_u = (alpha.unsqueeze(-1) * r)
        U = torch.cat([U[:, :, :p], new_u.unsqueeze(-1), U[:, :, p+1:]], dim=2)
        Vb = torch.cat([Vb[:, :, :p], k_t.unsqueeze(-1), Vb[:, :, p+1:]], dim=2)
        p += 1
        if p >= P:
            A, Bk, U, Vb, _ = layer_mod._compact_vec(A, Bk, U, Vb, layer_mod.slow_decay)
            p = 0
    return A, Bk, U, Vb

@torch.no_grad()
def probe(nprobe=16):
    model.eval(); pr = random.Random(271828)
    res = {"actual_full": [], "actual_kron": [], "ideal_full": [], "ideal_kron": [],
           "rand_full": [], "rand_kron": [], "v_norm": []}
    for _ in range(nprobe):
        ids, a0, gold = pm.make_mqar(pr, tok, n_keys, seq_len)
        x = torch.tensor([ids], device=DEV); _ = model(x)
        if 'q' not in cap: continue
        qf, kf, vf, bgf, wgf, dec = cap['q'], cap['k'], cap['v'], cap['bg'], cap['wg'], cap['dec']
        T = qf.shape[1]
        gold_positions = find_subseq(ids[:a0], gold)
        if not gold_positions: continue
        vpos = gold_positions[0]
        toks = [tok.decode([i]) for i in ids]
        is_pos = None
        for j in range(vpos-1, max(0, vpos-4), -1):
            if toks[j].strip().lower() == "is": is_pos = j; break
        if is_pos is None: continue
        key_pos = is_pos - 1
        if key_pos < 0: continue
        s = a0  # answer position
        # rebuild state at the answer position (the read happens at s, state from 0..s-1)
        A, Bk, U, Vb = rebuild_state_at(qf, kf, vf, bgf, wgf, dec, s)
        # queries (flattened to [N,K]): actual answer q, key's k, random
        def flat_q(vec):  # vec is [H,M,K] -> [N=H*M, K]
            return vec.reshape(-1, vec.shape[-1]).to(torch.float32)
        q_ans = flat_q(qf[0, s-1])       # [N,K]  (s-1 predicts first answer digit)
        k_key = flat_q(kf[0, key_pos])   # [N,K]
        v_val = vf[0, vpos].reshape(-1, V).to(torch.float32)  # [N,V]
        k_val_pos = flat_q(kf[0, vpos])  # [N,K] the value's own k (self-retrieval)
        rand_q = torch.randn_like(q_ans)
        # reads: full (kron + buffer) and kron-only (inline, no buffer)
        def read_full(qq): return layer_mod._kron_read_vec(A, Bk, U, Vb, qq)
        def read_kron(qq):
            # kron part of _kron_read_vec (no buffer): (A_r (x) B_r) x
            X = qq.reshape(qq.shape[0], a_k, b_k)
            AX = torch.einsum('nrvk,nkb->nrvb', A, X)
            AXB = torch.einsum('nrvb,nrwb->nrvw', AX, Bk)
            return AXB.sum(1).reshape(qq.shape[0], a_v * b_v)
        def cos(a, b):
            an = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
            bn = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
            return (an * bn).sum(-1)  # [N]
        v_n = v_val / (v_val.norm(dim=-1, keepdim=True) + 1e-9)
        for tag, qq in [("actual", q_ans), ("ideal", k_key), ("rand", rand_q)]:
            yf = read_full(qq); yk = read_kron(qq)
            res[f"{tag}_full"].append(cos(yf, v_val).mean().item())
            res[f"{tag}_kron"].append(cos(yk, v_val).mean().item())
        # also self-retrieval (value's own k -> should find its value if stored)
        ys = read_full(k_val_pos)
        res["v_norm"].append(v_val.norm(dim=-1).mean().item())
    model.train()
    import statistics as st
    out = {}
    for k in res:
        out[k] = round(st.mean(res[k]), 4) if res[k] else None
    out["nprobe"] = len(res["actual_full"])
    return out

print("=== probe at INIT ==="); p_init = probe(); print(json.dumps(p_init, indent=2))

print(f"=== training {steps} steps (seq_len={seq_len}, P={P}) ==="); last_ce=float('nan')
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
        if torch.isfinite(gnorm): opt.step()
    sched.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0 or step == steps-1: print(f"  step {step}: ce {last_ce:.3f}")

print("=== probe AFTER training ==="); p_post = probe(); print(json.dumps(p_post, indent=2))
print()
print("=== DECISION (cosine of read output with stored value v) ===")
for tag in ["actual","ideal","rand"]:
    ff = p_post.get(f"{tag}_full"); kk = p_post.get(f"{tag}_kron")
    print(f"  {tag:>7}: full={ff}  kron-only={kk}")
if p_post.get("ideal_full") is not None and p_post.get("rand_full") is not None:
    if p_post["ideal_full"] > p_post["rand_full"] + 0.1:
        print("=> IDEAL query retrieves value BETTER than random -> binding FORMS in Kron state.")
        if p_post.get("actual_full",0) < p_post["ideal_full"] - 0.05:
            print("   actual query << ideal -> query mismatch -> read/loss ceiling (handoff stands)")
        else:
            print("   actual ~ ideal -> should have recall (contradiction -> recheck)")
    else:
        print("=> IDEAL query NOT better than random -> binding DOESN'T form -> WRITE-side lever!")

result = {
  "config": {**cfg, "name": "kron_binding_compact_exp033",
             "hypothesis": "Does the Kronecker coproduct binding form WITH compaction (seq_len=512, P=16)? exp032 was confounded (no-compaction, A,Bk=0). Rebuild state at answer pos, probe with actual q vs key's k vs random. If ideal>>random -> binding forms (then query mismatch=loss ceiling). If ideal~random -> binding doesn't form -> editable WRITE-side lever."},
  "status": "ok", "device": args.device, "probe_init": p_init, "probe_post": p_post,
  "final_tokacc": 0.0, "final_recall": 0.0, "skip_rate": 0.0,
  "final_ce": round(last_ce,4), "wall_s": round(time.time()-t0,1),
}
json.dump(result, open(args.out,"w"), indent=2)
print(f"\n{json.dumps({k:result[k] for k in ('status','probe_post','wall_s')})}")
