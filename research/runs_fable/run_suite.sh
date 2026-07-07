#!/bin/bash
# KMD-2 testbed suite v2: conv + pos-emb fixed. Controls + MIMO/kron/compaction ladder.
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/runs_fable/../testbed_mqar.py "$@" --steps 6000 --eval_every 500 \
     --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "^step|final_tokacc"
}
run tb2_attn       --arch attn
run tb2_r4_noconv  --r 4 --no_conv
run tb2_r1         --r 1
run tb2_r4         --r 4
run tb2_r4_kron    --r 4 --dk 36 --kron
run tb2_r4_P32R8   --r 4 --compact_P 32 --compact_R 8
run tb2_r4_P32R4   --r 4 --compact_P 32 --compact_R 4
run tb2_r1_P32R8   --r 1 --compact_P 32 --compact_R 8
echo "SUITE DONE"
