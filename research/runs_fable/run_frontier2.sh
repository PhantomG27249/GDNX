#!/bin/bash
# Frontier 2: (a) slot-ortho rescue of r=4 under pressure; (b) compaction rank law.
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/testbed_mqar.py "$@" --steps 6000 --eval_every 1000 --keys_vocab 128 \
     --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "^step|final_tokacc"
}
# (a) does slot orthogonality rescue MIMO r=4?
run fr2_np64_r4_ortho     --n_pairs 64 --r 4 --slot_ortho 0.1
run fr2_np32_r4dk16_ortho --n_pairs 32 --r 4 --dk 16 --slot_ortho 0.1
# (b) compaction rank ladder at load 64 (r=1, the stronger base)
run fr2_np64_r1_P32R16    --n_pairs 64 --r 1 --compact_P 32 --compact_R 16
run fr2_np64_r1_P32R32    --n_pairs 64 --r 1 --compact_P 32 --compact_R 32
# (b') larger P (compact less often) at moderate R
run fr2_np64_r1_P64R16    --n_pairs 64 --r 1 --compact_P 64 --compact_R 16
echo "SUITE DONE"
