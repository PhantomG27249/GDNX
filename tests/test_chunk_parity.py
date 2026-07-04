"""Parity + timing gate for the GDN3 chunked-scan rewrite.

The invariant the rewrite MUST satisfy: the (new) live
`GDN3LinearAttn._gdn3_recurrent_state` reproduces the FROZEN sequential
reference (`gdn3._reference_recurrence.reference_recurrent_state`) within fp32
tolerance, on both forward outputs and input gradients.

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/dev/gdn3_qwen35_package/.venv/bin/python -m tests.test_chunk_parity

Right now (before the rewrite) the live method IS the reference, so this passes
trivially — that's the green baseline. After you replace `_gdn3_recurrent_state`
with the chunked version, this becomes the real correctness gate.

Exit code 0 = parity holds. Non-zero = mismatch (prints max abs err).
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, "/home/dev/gdn3_two_timescale_release")

import torch
from transformers import AutoConfig
from gdn3.gdn3_upgrade import GDN3LinearAttn
from gdn3._reference_recurrence import reference_recurrent_state

SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
DEV = "cuda:0"
ATOL, RTOL = 1e-4, 1e-3


def make_layer():
    cfg = AutoConfig.from_pretrained(SNAP).get_text_config()
    return GDN3LinearAttn(cfg, layer_idx=0).to(DEV)


def make_inputs(layer, B, T, seed=0, scale=0.1, requires_grad=False):
    """Small-magnitude, decay-heavy inputs to the recurrence (bypasses the
    forward's projections/gating). Kept modest so the reference stays finite
    over T steps — parity, not realism, is the point here."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, M, K, V = layer.H, layer.M, layer.K, layer.V
    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=DEV) * scale
    q = rnd(B, T, H, M, K); k = rnd(B, T, H, M, K); v = rnd(B, T, H, M, V)
    bg = torch.sigmoid(rnd(B, T, H, M, K) * 5)     # erase gate in (0,1)
    wg = torch.sigmoid(rnd(B, T, H, M, V) * 5)     # write gate in (0,1)
    dec = 0.9 + 0.09 * torch.rand(B, T, H, M, generator=g, device=DEV)  # gamma in [0.9,0.99)
    tensors = [q, k, v, bg, wg, dec]
    if requires_grad:
        for t in tensors:
            t.requires_grad_(True)
    return tensors


def parity_forward(B=1, T=64, seed=0):
    layer = make_layer()
    ins = make_inputs(layer, B, T, seed)
    with torch.no_grad():
        ref = reference_recurrent_state(layer, *ins)
        new = layer._gdn3_recurrent_state(*ins)
    err = (ref - new).abs().max().item()
    ok = torch.allclose(ref, new, atol=ATOL, rtol=RTOL) and torch.isfinite(new).all()
    print(f"[fwd ] B={B} T={T}: max|ref-new|={err:.2e}  finite={torch.isfinite(new).all().item()}  -> {'OK' if ok else 'FAIL'}")
    return ok


def parity_backward(B=1, T=64, seed=1):
    """Gradients through one compaction window must also match (compaction runs
    under no_grad in both, so grads flow only within a window — that's expected
    truncated BPTT; the rewrite must preserve the SAME truncation)."""
    layer = make_layer()
    ok_all = True
    for tag, fn in [("ref", reference_recurrent_state),
                    ("new", layer._gdn3_recurrent_state)]:
        pass
    ins_ref = make_inputs(layer, B, T, seed, requires_grad=True)
    ins_new = [t.detach().clone().requires_grad_(True) for t in ins_ref]
    ref = reference_recurrent_state(layer, *ins_ref)
    new = layer._gdn3_recurrent_state(*ins_new)
    gref = torch.autograd.grad(ref.pow(2).mean(), ins_ref, allow_unused=True)
    gnew = torch.autograd.grad(new.pow(2).mean(), ins_new, allow_unused=True)
    for nm, a, b in zip(["q", "k", "v", "bg", "wg", "dec"], gref, gnew):
        if a is None and b is None:
            continue
        if a is None or b is None:
            print(f"[bwd ] grad[{nm}]: one side None -> FAIL"); ok_all = False; continue
        e = (a - b).abs().max().item()
        good = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
        ok_all &= good
        print(f"[bwd ] grad[{nm}]: max|ref-new|={e:.2e} -> {'OK' if good else 'FAIL'}")
    return ok_all


def timing(B=1, T=512, iters=3):
    layer = make_layer()
    ins = make_inputs(layer, B, T, seed=2, requires_grad=True)
    def run(fn):
        for _ in range(2):  # warmup
            y = fn(layer, *ins) if fn is reference_recurrent_state else fn(*ins)
            torch.autograd.grad(y.pow(2).mean(), ins, allow_unused=True)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            y = fn(layer, *ins) if fn is reference_recurrent_state else fn(*ins)
            torch.autograd.grad(y.pow(2).mean(), ins, allow_unused=True)
        torch.cuda.synchronize(); return (time.time() - t0) / iters
    t_ref = run(reference_recurrent_state)
    t_new = run(layer._gdn3_recurrent_state)
    print(f"[time] T={T} fwd+bwd/layer: ref={t_ref*1e3:.0f}ms  new={t_new*1e3:.0f}ms  "
          f"speedup={t_ref/max(t_new,1e-9):.1f}x  (x18 layers: ref {t_ref*18:.1f}s new {t_new*18:.1f}s /step)")


if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True
    ok &= parity_forward(B=1, T=64)
    ok &= parity_forward(B=2, T=48)     # multiple chains, non-multiple-of-P tail
    ok &= parity_backward(B=1, T=64)
    timing(B=1, T=512)
    print("\nRESULT:", "PARITY OK ✅" if ok else "PARITY FAILED ❌")
    sys.exit(0 if ok else 1)
