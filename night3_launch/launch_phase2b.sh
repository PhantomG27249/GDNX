#!/usr/bin/env bash
# Night-3 phase 2 relaunch: B=1/rank (OOM fix: full-vocab heal logits at
# B=2 x 32768 needed 30 GiB), 160 updates to keep the ~10.5M-token budget.
set -uo pipefail

TREE=~/gdnx_0720
PY=~/gdnx/.venv/bin/python
TORCHRUN=~/gdnx/.venv/bin/torchrun
OUT=~/runs/night3
JUDGED=~/attn_data/data/retrieval/judged_blocks_32k_v1/train_semantic_retrieval_32k_judged_v1.pt

post() {
    "$PY" - "$1" <<'PYEOF'
import json, sys, urllib.request
from pathlib import Path
url = Path.home().joinpath("gdnx", "webhook").read_text().strip()
body = json.dumps({"content": sys.argv[1][:1900]}).encode()
req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=20)
except Exception as err:
    print(f"webhook post failed: {err}", flush=True)
PYEOF
}

cd "$TREE"
export PYTHONPATH="$TREE"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

post "GDN-X night3 phase 2 relaunch: B=1/rank x 160 updates @ T=32768 (OOM fix), resuming from phase1b ckpt-00672, LR 2.5e-5."

"$TORCHRUN" --nproc_per_node=2 --master_port=29535 train_night3.py \
    --updates 160 --batch 1 --seq-len 32768 \
    --band-tensor "$JUDGED" \
    --resume "$OUT/phase1b/ckpt-latest.pt" \
    --lr 2.5e-5 --warmup 10 --checkpoint-every 16 \
    --out "$OUT/phase2b" >> "$OUT/phase2b.log" 2>&1
P2=$?
if [ $P2 -eq 3 ]; then
    post "GDN-X night3 phase 2 TRIPWIRE abort - long-context loss diverged; phase1b result is safe. GPUs released."
    exit 3
elif [ $P2 -ne 0 ]; then
    post "GDN-X night3 phase 2 exited with code $P2 - see ~/runs/night3/phase2b.log"
    exit $P2
fi
post "GDN-X night3 phase 2 complete: 160 updates @ 32k = 10.5M long-context tokens. GPUs released."
