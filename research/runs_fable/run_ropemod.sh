#!/bin/bash
# Data-dependent RoPE (fixed ladder x learned rate): plain + under compaction, 2 seeds.
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/testbed_mqar.py "$@" --steps 6000 --eval_every 3000 --keys_vocab 128 \
     --n_pairs 64 --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "final_tokacc"
}
for s in 0 1; do
  run rm${s}_np64     --r 1 --rope_mod --seed $s
  run rm${s}_np64_ste --r 1 --rope_mod --compact_P 32 --compact_R 16 --compact_ste --seed $s
done
echo "SUITE DONE"
