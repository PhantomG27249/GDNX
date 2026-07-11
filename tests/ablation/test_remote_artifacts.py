from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from research.kmd2_ablation.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "research" / "kmd2_ablation"
CONFIGS = SUITE / "configs"
REQUIRED_CONFIGS = {
    "causal_lookahead_screening.json",
    "corrected_momentum_screening.json",
    "smoke.json",
    "screening.json",
    "promotion.json",
    "qwen_exact_cache.json",
    "trapezoid_screening.json",
    "gdn2_decoupled_screening.json",
}


def _documents() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CONFIGS.glob("*.json"))
    }


def test_committed_configs_are_complete_portable_and_cover_required_matrices():
    from research.kmd2_ablation.runner import validate_raw_scientific_config

    documents = _documents()
    assert REQUIRED_CONFIGS <= set(documents)
    assert set(documents) == REQUIRED_CONFIGS
    configs = {
        name: ExperimentConfig.from_dict(document)
        for name, document in documents.items()
    }
    for name, document in documents.items():
        mode = document["qwen"]["run_mode"] if document["backend"] == "qwen" else None
        assert validate_raw_scientific_config(
            document, backend=document["backend"], mode=mode
        ) == [], name
    assert configs["smoke.json"].backend == "tiny"
    assert configs["smoke.json"].seeds == (11,)
    assert configs["smoke.json"].device_preferences == ("cpu",)
    assert configs["smoke.json"].dtype_preferences == ("float32",)
    assert len(configs["screening.json"].seeds) == 3
    assert len(configs["promotion.json"].seeds) == 5
    assert configs["trapezoid_screening.json"].mechanism == "trapezoid"
    assert configs["corrected_momentum_screening.json"].mechanism == "corrected_momentum"
    assert configs["causal_lookahead_screening.json"].mechanism == "causal_lookahead"
    gdn2 = configs["gdn2_decoupled_screening.json"]
    assert gdn2.mechanism == "gdn2_decoupled"
    assert gdn2.variant == "channelwise_erase_write"
    assert len(gdn2.seeds) == 3
    qwen = configs["qwen_exact_cache.json"]
    assert qwen.backend == "qwen"
    assert qwen.required_stage == "qwen_heal"
    assert qwen.qwen.run_mode == "heal"
    assert len(qwen.seeds) == 3
    assert qwen.task.name == "ruler"
    assert qwen.task.params["ruler_long_cells"] == (
        "16k_4q",
        "16k_8q",
        "32k_4q",
        "32k_8q",
    )
    assert qwen.task.params["episodes_per_cell"] >= 64
    assert qwen.task.params["native_r_out"] == 4
    assert qwen.task.params["score_scan"] == (
        "gdn3.kmd2_fast_scan.scan_with_update_norm"
    )
    assert len(qwen.task.params["training_window_token_counts"]) == 64
    assert sum(qwen.task.params["training_window_token_counts"]) == qwen.budget.tokens
    assert len(qwen.task.params["example_ids"]) == 64
    assert sum(qwen.task.params["training_window_example_counts"]) == 64
    from research.kmd2_ablation.resource_probes import _native_addition_count

    assert _native_addition_count(qwen, r_out=4) == 37_952_928
    serialized = json.dumps(documents, sort_keys=True)
    assert not re.search(r"[A-Za-z]:[\\/]", serialized)
    assert "/home/" not in serialized
    assert "--model" not in serialized


def test_requirement_sets_are_minimal_and_backend_complete():
    tiny = (SUITE / "requirements-tiny.txt").read_text(encoding="utf-8").lower()
    qwen = (SUITE / "requirements-qwen.txt").read_text(encoding="utf-8").lower()
    assert "torch" in tiny
    assert "transformers" not in tiny
    assert "triton" not in tiny
    assert "torch" in qwen
    assert "transformers>=5.12.1,<6" in qwen
    assert "safetensors" in qwen
    assert "triton" in qwen


def test_portable_json_schema_declares_complete_top_level_contract():
    schema = json.loads((SUITE / "config.schema.json").read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "suite_version",
        "backend",
        "qwen",
        "baseline",
        "mechanism",
        "variant",
        "task",
        "seeds",
        "budget",
        "optimizer",
        "schedule",
        "model",
        "lengths",
        "evaluation",
        "thresholds",
        "promotion",
        "protected_metrics",
        "device_preferences",
        "dtype_preferences",
        "required_stage",
        "cache",
        "runtime",
    }


@pytest.mark.parametrize(
    ("name", "required_flags"),
    [
        (
            "run_remote_tiny.sh",
            {"--out", "--device", "--job-index", "--num-jobs", "--summarize"},
        ),
        (
            "run_remote_qwen.sh",
            {
                "--model",
                "--native-checkpoint",
                "--data",
                "--out",
                "--student-device",
                "--teacher-device",
                "--job-index",
                "--num-jobs",
                "--summarize",
            },
        ),
    ],
)
def test_remote_scripts_are_strict_relative_preflighted_and_resumable(
    name, required_flags
):
    path = SUITE / "scripts" / name
    source = path.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert required_flags <= set(re.findall(r"--[a-z][a-z-]*", source))
    assert 'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.."' in source
    assert source.index(" preflight ") < source.index(" run ")
    assert source.index(" run ") < source.index(" summarize ")
    assert "--resume" in source
    assert '[[ "$NUM_JOBS" -eq 1 || "$SUMMARIZE" -eq 1 ]]' in source
    assert not re.search(r"[A-Za-z]:[\\/]", source)
    assert "/home/dev" not in source
    git_bash = Path("C:/Program Files/Git/usr/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is not None:
        result = subprocess.run(
            [bash, "-n", path.as_posix()],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_operator_docs_cover_verification_assets_shards_resume_and_promotion():
    suite_readme = (SUITE / "README.md").read_text(encoding="utf-8")
    top_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "verify_bundle.py",
        "requirements-tiny.txt",
        "requirements-qwen.txt",
        "preflight",
        "Slurm",
        "--job-index",
        "--num-jobs",
        "--resume",
        "summarize",
        "manifest.json",
        "jobs.json",
        "quarantine",
        "promotion",
        "MODEL_PATH",
        "NATIVE_CHECKPOINT",
        "DATA_PATH",
        "GDN3_FAST_SCAN=1",
        "GDN3_KMD2_ROUT=4",
        "post-array coordinator",
        "adjacent verifier",
    )
    for term in required:
        assert term in suite_readme
    assert "research/kmd2_ablation/README.md" in top_readme
    assert "train/train_gdn3_distill.py" in top_readme
    assert "results show" not in suite_readme.lower()
