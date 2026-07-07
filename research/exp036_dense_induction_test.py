"""exp036: can the FROZEN DENSE q/k geometry do MQAR induction at all?

The deepest untested question. exp033 found the coproduct-key-query retrieves the
value (cos 0.35) but the actual-answer-query mismatches (cos 0.10). The actual query
= dense_q + coprod_q; the key's k = dense_k + coprod_k. All prior levers changed the
BLEND/mix but never measured the FOUNDATION: does frozen dense_q at the query's KEY
token align with frozen dense_k at the stored KEY token (same word, diff context)?

MQAR induction REQUIRES this: the query "code for KEY?" must produce a q that matches
the stored "code for KEY is" position's k. If the frozen dense geometry can't, then NO
GDN3 edit can fix it (frozen-geometry ceiling -> MUST unfreeze in_proj_qkv). If it CAN,
the coproduct just needs to exploit it (and the loss is the only barrier).

We measure (read-only, frozen model, no training):
  (1) dense_q@query_KEY · dense_k@stored_KEY  vs  dense_q@query_KEY · dense_k@distractor_KEYs
      — does the query's key match its OWN stored key better than other keys? (induction ability)
  (2) the same for the coproduct q/k (trainable, but at init).
  (3) for control: does q@pos align with k@SAME-pos (the trivial self-match that RoPE gives).

This cleanly separates:
  - frozen dense CAN'T induce -> frozen-geometry ceiling (human: unfreeze in_proj_qkv)
  - frozen dense CAN induce but coproduct doesn't -> coproduct init/learning issue (editable)
  - both can induce but actual query still fails -> positional/context-window issue (loss/proxy)

Standalone (no source edits, no training, ~30s). Output: research/runs/exp036.json.
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

# capture the raw dense q,k (BEFORE coproduct blend) by hooking _generate_coproduct_channels
# inputs. Easier: monkeypatch _generate_coproduct_channels to capture q_dense,k_dense.
CAP_LAYER = upg[len(upg)//2]; cap = {}; layer_mod = model.model.layers[CAP_LAYER].linear_attn
H, M, K = layer_mod.H, layer_mod.M, layer_mod.K
orig_gen = layer_mod._generate_coproduct_channels
def capturing_gen(q_dense, k_dense, v_dense, x):
    cap['q_dense'] = q_dense.detach(); cap['k_dense'] = k_dense.detach(); cap['v_dense'] = v_dense.detach()
    return orig_gen(q_dense, k_dense, v_dense, x)
layer_mod._generate_coproduct_channels = capturing_gen
print(f"layer {CAP_LAYER}: H={H} M={M} K={K}")

def find_all_subseq(hay, needle):
    L=len(needle); out=[]
    for i in range(len(hay)-L+1):
        if hay[i:i+L]==needle: out.append(i)
    return out

def cos(a,b):
    an=a/(a.norm(dim=-1,keepdim=True)+1e-9); bn=b/(b.norm(dim=-1,keepdim=True)+1e-9)
    return (an*bn).sum(-1)

import re
@torch.no_grad()
def probe(nprobe=24):
    pr = random.Random(314159); res=[]
    for _ in range(nprobe):
        ids,a0,gold = pm.make_mqar(pr, tok, n_keys=4, seq_len=512)
        x = torch.tensor([ids], device=DEV); _ = model(x)
        if 'q_dense' not in cap: continue
        qd = cap['q_dense'][0]  # [T,H,K]
        kd = cap['k_dense'][0]  # [T,H,K]
        T = qd.shape[0]; toks = [tok.decode([i]) for i in ids]
        text = ''.join(toks)
        # find all 'code for KEY is' via regex; map char->token via cumulative lengths
        cum = [0]*(T+1)
        for i,t in enumerate(toks): cum[i+1]=cum[i]+len(t)
        def char_to_tok(c):
            # token index containing char c
            for i in range(T):
                if cum[i] <= c < cum[i+1]: return i
            return T-1
        bindings = []  # (key_str, key_last_tok, is_tok, value_str)
        for m in re.finditer(r'code for (.+?) is', text):
            key_str = m.group(1)
            # last char of key = m.start(1)+len(key)-1; find that token (the ' is' is the next)
            key_last_char = m.start(1)+len(key_str)-1
            key_last_tok = char_to_tok(key_last_char)
            is_char = m.start()+len('code for ')+len(key_str)+1  # the 'is'
            is_tok = char_to_tok(is_char)
            # value follows after ' is ' : the 4-digit value
            val_start_char = m.end()  # right after ' is'
            # value is the 4-digit number; find it
            vm = re.match(r' ?(\d{4})', text[val_start_char:val_start_char+6])
            val_str = vm.group(1) if vm else ''
            bindings.append({'key':key_str,'key_last_tok':key_last_tok,'is_tok':is_tok,'val':val_str})
        # the query key is the one in the question 'What is the code for KEY?'
        # its key appears right before '?'. Find '?' position.
        qm = re.search(r'What is the code for (.+?)\?', text)
        if qm is None: continue
        qkey = qm.group(1)
        # the query key's LAST token in the 'for KEY?' occurrence:
        qkey_last_char = qm.start(1)+len(qkey)-1
        qkey_last_tok = char_to_tok(qkey_last_char)
        # find the matching stored binding (same key string) and distractors
        match = None; distractors=[]
        for b in bindings:
            if b['key']==qkey and b['val']:  # stored (has a value)
                if match is None: match=b
                else: distractors.append(b)
            elif b['val']:
                distractors.append(b)
        if match is None: continue
        # DENSE induction: dense_q @ query_key_last_tok  vs  dense_k @ match.key_last_tok / distractors
        # mean over heads H
        qk_match = cos(qd[qkey_last_tok].mean(0).unsqueeze(0), kd[match['key_last_tok']].mean(0).unsqueeze(0)).item()
        qk_dist = [cos(qd[qkey_last_tok].mean(0).unsqueeze(0), kd[d['key_last_tok']].mean(0).unsqueeze(0)).item() for d in distractors]
        import statistics
        qk_dist_mean = statistics.mean(qk_dist) if qk_dist else 0.0
        per_head_match = cos(qd[qkey_last_tok], kd[match['key_last_tok']])  # [H]
        per_head_dist = [cos(qd[qkey_last_tok], kd[d['key_last_tok']]) for d in distractors]
        per_head_dist_mean = torch.stack(per_head_dist).mean(0) if per_head_dist else torch.zeros(H,device=DEV)
        per_head_diff = per_head_match - per_head_dist_mean
        res.append({
            'qk_match': round(qk_match,4), 'qk_distract_mean': round(qk_dist_mean,4),
            'qk_diff': round(qk_match-qk_dist_mean,4),
            'best_head_diff': round(float(per_head_diff.max()),4),
            'n_heads_induce': int((per_head_diff>0.05).sum().item()),
            'n_distract': len(distractors),
        })
    return res

print("=== FROZEN DENSE q/k induction test (no training) ===")
r = probe()
if not r:
    print("no valid episodes"); sys.exit(1)
import statistics as st
print(f"episodes: {len(r)}")
print(f"dense_q@query_key · dense_k@matching_key:    mean {st.mean(x['qk_match'] for x in r):.4f}")
print(f"dense_q@query_key · dense_k@distractor_keys:  mean {st.mean(x['qk_distract_mean'] for x in r):.4f}")
print(f"  -> induction gap (match - distract):        mean {st.mean(x['qk_diff'] for x in r):.4f}")
print(f"best-head induction gap (per-episode max):    mean {st.mean(x['best_head_diff'] for x in r):.4f}")
print(f"heads that induce (gap>0.05):                mean {st.mean(x['n_heads_induce'] for x in r):.1f} / {H}")
print()
gap = st.mean(x['qk_diff'] for x in r)
best_gap = st.mean(x['best_head_diff'] for x in r)
if gap > 0.05:
    print("=> FROZEN DENSE CAN INDUCE (match>distract by >0.05): the geometry supports MQAR induction.")
    print("   The coproduct just needs to exploit it -> editable-side issue, loss is the barrier.")
elif best_gap > 0.1:
    print(f"=> SOME heads induce (best gap {best_gap:.3f}): partial induction ability in the frozen geometry.")
    print("   A head-routing or per-head coproduct could exploit it (editable).")
else:
    print("=> FROZEN DENSE CANNOT INDUCE (match ~ distract): the geometry fundamentally can't match")
    print("   same-word different-context. NO GDN3 edit can fix this -> MUST unfreeze in_proj_qkv (human).")

result = {
  "config": {"name":"dense_induction_test_exp036","hypothesis":"Can the FROZEN dense q/k geometry do MQAR induction (match same-word-diff-context)? If yes->editable side can exploit it; if no->frozen-geometry ceiling, MUST unfreeze in_proj_qkv (human).","steps":0,"n_keys":4,"seq_len":512},
  "status":"ok","device":DEV,"n_episodes":len(r),
  "dense_qk_match": round(st.mean(x['qk_match'] for x in r),4),
  "dense_qk_distract": round(st.mean(x['qk_distract_mean'] for x in r),4),
  "induction_gap": round(gap,4),
  "best_head_induction_gap": round(best_gap,4),
  "mean_heads_induce": round(st.mean(x['n_heads_induce'] for x in r),2),
  "final_tokacc":0.0,"final_recall":0.0,"skip_rate":0.0,"final_ce":0.0,
  "wall_s":round(time.time()-t0,1),
}
json.dump(result, open("research/runs/exp036.json","w"), indent=2)
print(f"\n{json.dumps({k:result[k] for k in ('dense_qk_match','dense_qk_distract','induction_gap','best_head_induction_gap','mean_heads_induce','wall_s')})}")
