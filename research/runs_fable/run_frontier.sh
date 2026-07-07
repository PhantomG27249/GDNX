#!/bin/bash
# Capacity/interference frontier: where do r=1 and r=4 separate? (fable_idea §5 iso-state)
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/testbed_mqar.py "$@" --steps 6000 --eval_every 1000 --keys_vocab 128 \
     --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "^step|final_tokacc"
}
# load ladder at full state (dk=32)
run fr_np32_r1        --n_pairs 32 --r 1
run fr_np32_r4        --n_pairs 32 --r 4
run fr_np64_r1        --n_pairs 64 --r 1
run fr_np64_r4        --n_pairs 64 --r 4
# iso-small state (dk=16) — interference regime
run fr_np32_r1_dk16   --n_pairs 32 --r 1 --dk 16
run fr_np32_r4_dk16   --n_pairs 32 --r 4 --dk 16
# compaction under load
run fr_np64_r1_P32R4  --n_pairs 64 --r 1 --compact_P 32 --compact_R 4
run fr_np64_r4_P32R4  --n_pairs 64 --r 4 --compact_P 32 --compact_R 4
echo "SUITE DONE"
