#!/bin/bash
# Frontier 3: separate compaction's GRADIENT wall (STE fixes) from its INFO wall (R vs load).
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/testbed_mqar.py "$@" --steps 6000 --eval_every 1000 --keys_vocab 128 \
     --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "^step|final_tokacc"
}
# control: STE with no-op truncation (R=32=full rank) should recover ~0.98 baseline
run fr3_np64_r1_P32R32_ste --n_pairs 64 --r 1 --compact_P 32 --compact_R 32 --compact_ste
# gradient wall fixed, info wall varying: R ladder under STE
run fr3_np64_r1_P32R16_ste --n_pairs 64 --r 1 --compact_P 32 --compact_R 16 --compact_ste
run fr3_np64_r1_P32R8_ste  --n_pairs 64 --r 1 --compact_P 32 --compact_R 8  --compact_ste
run fr3_np64_r1_P32R4_ste  --n_pairs 64 --r 1 --compact_P 32 --compact_R 4  --compact_ste
# best combined candidate: MIMO + ortho + STE compaction (the full "working GDN" stack)
run fr3_np64_r4_P32R16_ste_ortho --n_pairs 64 --r 4 --compact_P 32 --compact_R 16 --compact_ste --slot_ortho 0.1
echo "SUITE DONE"
