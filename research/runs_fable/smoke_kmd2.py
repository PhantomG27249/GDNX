"""KMD-2 smoke test: build the upgraded model, run one fwd+bwd MQAR step,
report finiteness, grad flow, trainable param count, and per-step timing."""
import os, sys, time, torch, torch.nn.functional as F, random
sys.path.insert(0, "/home/dev/gdn3_fable")
sys.path.insert(0, "/home/dev/gdn3_fable/research")
os.environ["GDN3_KMD2"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SNAP = ("/home/dev/.cache/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17")
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn3.gdn3_upgrade import GDN3UpgradeManager
from proxy_mqar import make_mqar, PRESERVED

dev = "cuda:0"  # launched with CUDA_VISIBLE_DEVICES to pick the physical GPU
tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mgr = GDN3UpgradeManager(model); mgr.apply_upgrade(); upg = mgr.upgraded_layers
for p in model.parameters(): p.requires_grad_(False)
train = []
for idx in upg:
    for n, p in model.model.layers[idx].linear_attn.named_parameters():
        if any(k in n for k in PRESERVED):
            continue
        p.requires_grad_(True); train.append((n, p))
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(dev).train()
ntrain = sum(p.numel() for _, p in train)
print(f"upgraded={len(upg)}  trainable KMD-2={ntrain/1e6:.1f}M  names={sorted(set(n for n,_ in train))}")

rng = random.Random(0)
for trial in range(3):
    ids, a0, gold = make_mqar(rng, tok, 4, 512)
    x = torch.tensor([ids], device=dev)
    torch.cuda.synchronize(); t0 = time.time()
    out = model(input_ids=x)
    lg = out.logits[0, a0 - 1:a0 - 1 + len(gold)]
    loss = F.cross_entropy(lg.float(), torch.tensor(gold, device=dev))
    loss.backward()
    torch.cuda.synchronize(); dt = time.time() - t0
    if trial == 0:
        gnz = sum(1 for _, p in train if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"seq={len(ids)} loss={float(loss):.3f} finite={bool(torch.isfinite(loss))} "
              f"grad_nonzero={gnz}/{len(train)} logits_finite={bool(torch.isfinite(out.logits).all())}")
    for _, p in train: p.grad = None
    print(f"  trial {trial}: fwd+bwd {dt:.2f}s  -> est {dt*400/60:.0f} min for 400 steps")
print(f"peak GPU mem: {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB")
