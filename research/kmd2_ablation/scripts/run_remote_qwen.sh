#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"

# Pinned Qwen execution identities; preflight records and verifies these.
export GDN3_FAST_SCAN=1
export GDN3_KMD2_ROUT=4

CONFIG=""
OUT=""
JOB_INDEX=0
NUM_JOBS=1
SUMMARIZE=0
PREFLIGHT_ONLY=0
SKIP_PREFLIGHT=0
RESUME=(--resume)
PASS=()

usage() {
  echo "usage: $0 --config PATH --out PATH [--maximum-control ID]" >&2
  echo "  [--model PATH --tokenizer PATH --native-checkpoint PATH --data PATH --teacher-model PATH]" >&2
  echo "  [--student-device DEV --teacher-device DEV --dtype DTYPE --checkpoint-every N] [--assets-manifest PATH]" >&2
  echo "  [--job-index N --num-jobs N] [--resume|--no-resume] [--preflight-only|--skip-preflight] [--summarize]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --job-index) JOB_INDEX="$2"; shift 2 ;;
    --num-jobs) NUM_JOBS="$2"; shift 2 ;;
    --summarize) SUMMARIZE=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --resume) RESUME=(--resume); shift ;;
    --no-resume) RESUME=(--no-resume); shift ;;
    --maximum-control|--mode|--model|--tokenizer|--native-checkpoint|--checkpoint|--data|--teacher-model|--student-device|--device|--teacher-device|--dtype|--checkpoint-every|--assets-manifest|--model-sha256|--tokenizer-sha256|--checkpoint-sha256|--data-sha256|--teacher-model-sha256|--repo-root)
      PASS+=("$1" "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$CONFIG" && -n "$OUT" ]] || { usage; exit 2; }
if [[ "$PREFLIGHT_ONLY" -eq 1 && "$SKIP_PREFLIGHT" -eq 1 ]]; then
  echo "--preflight-only and --skip-preflight are mutually exclusive" >&2
  exit 2
fi

# Resolve caller-relative paths before changing to the repo root.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
CONFIG="$(abspath "$CONFIG")"
OUT="$(abspath "$OUT")"
resolved=()
i=0
while [[ $i -lt ${#PASS[@]} ]]; do
  flag="${PASS[$i]}"; value="${PASS[$((i+1))]}"
  case "$flag" in
    --model|--tokenizer|--native-checkpoint|--checkpoint|--data|--teacher-model|--assets-manifest|--repo-root)
      value="$(abspath "$value")" ;;
  esac
  resolved+=("$flag" "$value"); i=$((i+2))
done
PASS=("${resolved[@]}")

cd "$ROOT_DIR"

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
  "$PYTHON" -m research.kmd2_ablation.run_ablation preflight \
    --backend qwen --config "$CONFIG" --out "$OUT" \
    --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" "${PASS[@]}"
fi
if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  exit 0
fi

# Live metrics are useful only during an actual run. Starting the watcher
# after the preflight-only exit keeps campaign-wide validation free of its
# shutdown grace period.
WATCHER_PID=""
if [[ -s "$ROOT_DIR/webhook" ]]; then
  export GDNX_LIVE_METRICS=1
  "$PYTHON" research/kmd2_ablation/scripts/metrics_webhook.py \
    --output "$OUT" --webhook-file "$ROOT_DIR/webhook" --poll-seconds 2 &
  WATCHER_PID=$!
  echo "live metrics enabled (watcher pid $WATCHER_PID)" >&2
fi
stop_watcher() {
  if [[ -n "$WATCHER_PID" ]]; then
    sleep 3  # cover one final two-second poll for the completion message
    kill "$WATCHER_PID" 2>/dev/null || true
  fi
}
trap stop_watcher EXIT

"$PYTHON" -m research.kmd2_ablation.run_ablation run \
  --backend qwen --config "$CONFIG" --out "$OUT" \
  --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" "${RESUME[@]}" "${PASS[@]}"

if [[ "$NUM_JOBS" -eq 1 || "$SUMMARIZE" -eq 1 ]]; then
  "$PYTHON" -m research.kmd2_ablation.run_ablation summarize \
    --backend qwen --config "$CONFIG" --out "$OUT" "${PASS[@]}"
fi
