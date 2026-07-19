#!/usr/bin/env bash
# Night-3 restart: resume phase 1 from the pre-divergence minimum
# (phase1/ckpt-00128, best loss 1.29) at conservative LR 5e-5 constant.
# Divergence tripwire aborts+checkpoints+posts if smoothed loss blows up.
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

post ":arrows_counterclockwise: **GDN-X night3** RESTART — 2e-4 plateau diverged after upd ~150 (min 1.29 @ 128). Resuming from ckpt-00128 at LR 5e-5 constant, 672 updates to go. Tripwire armed: auto-abort+checkpoint if smoothed loss exceeds 1.5x its best."

"$TORCHRUN" --nproc_per_node=2 --master_port=29533 train_night3.py \
    --updates 672 --batch 4 --seq-len 4096 \
    --band ~/attn_data/data/gdnx_night3_band \
    --resume "$OUT/phase1/ckpt-00128.pt" \
    --lr 5e-5 --warmup 25 --checkpoint-every 64 \
    --out "$OUT/phase1b" >> "$OUT/phase1b.log" 2>&1
P1=$?

if [ $P1 -eq 3 ]; then
    post ":octagonal_sign: **GDN-X night3** phase 1b TRIPWIRE abort — loss diverged even at 5e-5. Checkpointed before stopping; GPUs released. Needs a human decision in the morning."
    exit 3
elif [ $P1 -ne 0 ]; then
    post ":x: **GDN-X night3** phase 1b exited with code $P1 — see ~/runs/night3/phase1b.log (tail: $(tail -c 400 "$OUT/phase1b.log" | tr '\n' ' ' | tr -d '"'))"
    exit $P1
fi
post ":white_check_mark: **GDN-X night3** phase 1b complete (672 updates @ 5e-5, ~22M new tokens). Starting phase 2 (T=32768, judged retrieval band)."

"$TORCHRUN" --nproc_per_node=2 --master_port=29534 train_night3.py \
    --updates 80 --batch 2 --seq-len 32768 \
    --band-tensor "$JUDGED" \
    --resume "$OUT/phase1b/ckpt-latest.pt" \
    --lr 2.5e-5 --warmup 10 --checkpoint-every 16 \
    --out "$OUT/phase2" >> "$OUT/phase2.log" 2>&1
P2=$?
if [ $P2 -eq 3 ]; then
    post ":octagonal_sign: **GDN-X night3** phase 2 TRIPWIRE abort — long-context phase diverged; phase-1b result is safe. GPUs released."
    exit 3
elif [ $P2 -ne 0 ]; then
    post ":x: **GDN-X night3** phase 2 exited with code $P2 — see ~/runs/night3/phase2.log"
    exit $P2
fi
post ":checkered_flag: **GDN-X night3** phase 2 complete (80 updates @ 32k, ~10.5M long-context tokens). Both GPUs released."
