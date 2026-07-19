#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$ROOT_DIR/research/kmd2_ablation/remote_package_b.py" --config "$ROOT_DIR/config.toml" "$@"
