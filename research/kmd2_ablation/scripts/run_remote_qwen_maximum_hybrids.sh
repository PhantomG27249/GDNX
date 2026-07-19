#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
CAMPAIGN="$ROOT_DIR/research/kmd2_ablation/campaigns/qwen08b-maximum-hybrid-v1.json"
SINGLE="${KMD2_QWEN_RUNNER:-$ROOT_DIR/research/kmd2_ablation/scripts/run_remote_qwen.sh}"
SINGLE="${SINGLE//\\//}"
if [[ "$SINGLE" == */* ]] && command -v cygpath >/dev/null 2>&1; then SINGLE="$(cygpath -u "$SINGLE")"; fi
if [[ "$SINGLE" =~ ^([A-Za-z]):/(.*)$ ]]; then
  SINGLE="/${BASH_REMATCH[1],,}/${BASH_REMATCH[2]}"
fi
ASSETS=""
OUT=""
JOB_INDEX=0
NUM_JOBS=1
SUMMARIZE=0
SMOKE=0
PACKAGE="${KMD2_MAXIMUM_PACKAGE:-}"
FORWARD=()
usage() { echo "usage: $0 --assets-manifest PATH --out PATH [--package A|B|both] [--smoke] [--job-index N --num-jobs N --summarize]" >&2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --assets-manifest) ASSETS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --job-index) JOB_INDEX="$2"; shift 2 ;;
    --num-jobs) NUM_JOBS="$2"; shift 2 ;;
    --summarize) SUMMARIZE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --package) PACKAGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --config|--campaign|--maximum-control) echo "$1 is owned by the maximum-hybrid launcher" >&2; exit 2 ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done
if [[ -z "$PACKAGE" && -f "$ROOT_DIR/PACKAGE.json" ]]; then
  PACKAGE="$("$PYTHON" - "$ROOT_DIR/PACKAGE.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["package"])
PY
)"
fi
PACKAGE="${PACKAGE,,}"
[[ "$PACKAGE" == "a" || "$PACKAGE" == "b" || "$PACKAGE" == "both" ]] || { echo "--package A, B, or both is required outside a packaged archive" >&2; exit 2; }
[[ -n "$ASSETS" && -n "$OUT" ]] || { usage; exit 2; }
GENERATED="$OUT/.generated/maximum-hybrid"
PATHS_FILE="$OUT/.generated/maximum-assets.paths"
mkdir -p "$GENERATED"
cd "$ROOT_DIR"
"$PYTHON" - "$CAMPAIGN" "$ASSETS" "$GENERATED" "$PATHS_FILE" "$SMOKE" "$PACKAGE" <<'PY'
import json, sys
from pathlib import Path
from research.kmd2_ablation.runner import materialize_maximum_hybrid_campaign
campaign, manifest, destination, paths_file = map(Path, sys.argv[1:5])
raw = json.loads(manifest.read_text(encoding="utf-8"))
assets = raw.get("assets", {})
names = ("model", "tokenizer", "checkpoint", "data", "teacher_model")
paths = []
for name in names:
    item = assets.get(name)
    if not isinstance(item, dict): raise SystemExit(f"missing asset {name}")
    value = item.get("tree_sha256") or item.get("sha256")
    path = Path(item.get("path", ""))
    if not path.exists() or not (path.is_file() or path.is_dir()): raise SystemExit(f"asset {name} is not a regular path")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise SystemExit(f"asset {name} has invalid SHA-256")
    if "\n" in str(path) or "\r" in str(path): raise SystemExit(f"asset {name} path contains newline")
    paths.append(str(path.resolve()))
materialize_maximum_hybrid_campaign(campaign, raw, destination, smoke=sys.argv[5] == "1")
selected = sys.argv[6]
if selected != "both":
    from research.kmd2_ablation.runner import select_maximum_package_controls
    controls = [item["control_id"] for item in json.loads(campaign.read_text(encoding="utf-8"))["controls"]]
    allowed = set(select_maximum_package_controls(controls, selected))
    for config in destination.glob("*.json"):
        control = config.stem
        control = control.split("-", 1)[1] if control[:2].isdigit() and "-" in control else control
        if control not in allowed:
            config.unlink()
paths_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
PY
mapfile -t ASSET_PATHS < "$PATHS_FILE"
MODEL="${ASSET_PATHS[0]}"; TOKENIZER="${ASSET_PATHS[1]}"; CHECKPOINT="${ASSET_PATHS[2]}"
DATA="${ASSET_PATHS[3]}"; TEACHER="${ASSET_PATHS[4]}"
asset_args=(--model "$MODEL" --tokenizer "$TOKENIZER" --native-checkpoint "$CHECKPOINT" --data "$DATA" --teacher-model "$TEACHER" --assets-manifest "$ASSETS")
invoke_single() {
  if [[ "$SINGLE" == */* ]]; then bash "$SINGLE" "$@"; else "$SINGLE" "$@"; fi
}

run_id() { basename "$1" .json | sed -E 's/^[0-9]+-//'; }
output_for() {
  local name="$1" group="controls"
  [[ "$name" == package-a-* ]] && group="package-a"
  [[ "$name" == package-b-* ]] && group="package-b"
  printf '%s/%s/%s' "$OUT" "$group" "$name"
}

# Real all-controls barrier. No run command is issued until every invocation succeeds.
for CONFIG in "$GENERATED"/*.json; do
  CONTROL="$(run_id "$CONFIG")"
  invoke_single --config "$CONFIG" --out "$(output_for "$CONTROL")" --maximum-control "$CONTROL" --preflight-only "${asset_args[@]}" "${FORWARD[@]}"
done

run_converted_control() {
  local config="$1" control="$2"
  invoke_single --config "$config" --out "$(output_for "$control")" --maximum-control "$control" --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" --resume --skip-preflight "${asset_args[@]}" "${FORWARD[@]}"
}
run_stock_evaluator() {
  local config="$1"
  invoke_single --config "$config" --out "$(output_for stock-qwen)" --maximum-control stock-qwen --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" --resume --skip-preflight "${asset_args[@]}" "${FORWARD[@]}"
}
for CONFIG in "$GENERATED"/*.json; do
  CONTROL="$(run_id "$CONFIG")"
  if [[ "$CONTROL" == stock-qwen ]]; then run_stock_evaluator "$CONFIG"; else run_converted_control "$CONFIG" "$CONTROL"; fi
done

if [[ "$NUM_JOBS" -eq 1 || "$SUMMARIZE" -eq 1 ]]; then
  "$PYTHON" - "$CAMPAIGN" "$OUT" "$PACKAGE" <<'PY'
import json, sys
from pathlib import Path
from research.kmd2_ablation.summarize import aggregate_hybrid_promotion
campaign = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); root = Path(sys.argv[2])
for control in campaign["controls"]:
    arm = control["control_id"]
    selected = sys.argv[3]
    if selected == "a" and arm.startswith("package-b-"): continue
    if selected == "b" and arm.startswith("package-a-"): continue
    rows = []
    for path in root.rglob("*.json"):
        try: record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        observation = record.get("hybrid_promotion_observation") if isinstance(record, dict) else None
        if isinstance(observation, dict) and record.get("arm_id", arm) == arm: rows.append(observation)
    decision = aggregate_hybrid_promotion(rows)
    target = root / ("package-a" if arm.startswith("package-a-") else "package-b" if arm.startswith("package-b-") else "controls") / arm / "campaign-promotion.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema_version":"1.0.0","control_id":arm,**decision}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
PY
fi
