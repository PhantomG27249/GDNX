from __future__ import annotations

import importlib
import hashlib
import io
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from tests.ablation.test_config import minimal_config_dict


def _model_clean_qwen_process(monkeypatch: pytest.MonkeyPatch, *, r_out: int = 4) -> None:
    """Model the launcher's pre-import environment inside the shared test process."""

    monkeypatch.setenv("GDN3_FAST_SCAN", "1")
    monkeypatch.setenv("GDN3_KMD2_ROUT", str(r_out))
    # Other tests legitimately exercise the reference path and may already have
    # imported this module with its import-time gate disabled.  A production
    # launcher starts a fresh process, so remove only that cached module here;
    # the dedicated late-import tests still prove the fail-closed behavior.
    monkeypatch.delitem(sys.modules, "gdn3.kmd2_native", raising=False)


def _cli_module():
    return importlib.import_module("research.kmd2_ablation.run_ablation")


def _invoke(argv: list[str], handlers):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = _cli_module().main(
        argv,
        handlers=handlers,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _success_handler(calls: list):
    def handle(options):
        calls.append(options)
        return {
            "ok": True,
            "schema_version": "1.0.0",
            "codes": [],
            "warnings": [],
        }

    return handle


@pytest.mark.parametrize("command", ["preflight", "run", "summarize", "bundle"])
def test_cli_dispatches_all_subcommands_as_canonical_json(command):
    calls: list = []
    handlers = {name: _success_handler(calls) for name in (
        "preflight",
        "run",
        "summarize",
        "bundle",
    )}

    code, stdout, stderr = _invoke(
        [
            command,
            "--backend",
            "tiny",
            "--config",
            "experiment.json",
            "--out",
            "results",
            "--job-index",
            "1",
            "--num-jobs",
            "3",
            "--resume",
        ],
        handlers,
    )

    assert code == 0
    assert stderr == ""
    assert len(calls) == 1
    options = calls[0]
    assert options.command == command
    assert options.backend == "tiny"
    assert options.config == Path("experiment.json")
    assert options.out == Path("results")
    assert options.job_index == 1
    assert options.num_jobs == 3
    assert options.resume is True
    assert json.loads(stdout) == {
        "codes": [],
        "ok": True,
        "schema_version": "1.0.0",
        "warnings": [],
    }
    assert stdout == (
        '{"codes":[],"ok":true,"schema_version":"1.0.0","warnings":[]}\n'
    )


def test_cli_accepts_qwen_mode_assets_devices_checksums_and_no_resume():
    calls: list = []
    handler = _success_handler(calls)

    code, _, _ = _invoke(
        [
            "preflight",
            "--backend",
            "qwen",
            "--config",
            "experiment.json",
            "--out",
            "results",
            "--mode",
            "heal",
            "--model",
            "model",
            "--tokenizer",
            "tokenizer",
            "--checkpoint",
            "checkpoint.pt",
            "--data",
            "data.jsonl",
            "--teacher-model",
            "teacher",
            "--student-device",
            "cuda:1",
            "--teacher-device",
            "cuda:0",
            "--model-sha256",
            "a" * 64,
            "--assets-manifest",
            "assets.json",
            "--no-resume",
            "--dry-run",
        ],
        {"preflight": handler},
    )

    assert code == 0
    options = calls[0]
    assert options.mode == "heal"
    assert options.model == Path("model")
    assert options.tokenizer == Path("tokenizer")
    assert options.checkpoint == Path("checkpoint.pt")
    assert options.data == Path("data.jsonl")
    assert options.teacher_model == Path("teacher")
    assert options.student_device == "cuda:1"
    assert options.teacher_device == "cuda:0"
    assert options.model_sha256 == "a" * 64
    assert options.assets_manifest == Path("assets.json")
    assert options.resume is False
    assert options.dry_run is True


@pytest.mark.parametrize(
    ("command", "expected"),
    [("preflight", 3), ("run", 4), ("summarize", 5), ("bundle", 6)],
)
def test_cli_stable_failure_exit_codes(command, expected):
    def failed(_options):
        return {"ok": False, "codes": [f"{command}_failed"], "warnings": []}

    code, stdout, _ = _invoke(
        [
            command,
            "--backend",
            "tiny",
            "--config",
            "experiment.json",
            "--out",
            "results",
        ],
        {command: failed},
    )

    assert code == expected
    assert json.loads(stdout)["ok"] is False


def test_cli_usage_errors_return_two_without_calling_handlers():
    called = False

    def handler(_options):
        nonlocal called
        called = True
        return {"ok": True}

    code, stdout, stderr = _invoke(
        ["run", "--backend", "invalid"],
        {"run": handler},
    )

    assert code == 2
    assert called is False
    assert stdout == ""
    assert "invalid choice" in stderr


def test_cli_import_is_lazy_for_torch_transformers_and_production_handlers():
    sys.modules.pop("research.kmd2_ablation.run_ablation", None)
    before = set(sys.modules)

    module = _cli_module()

    imported = set(sys.modules) - before
    assert "torch" not in imported
    assert "transformers" not in imported
    assert "research.kmd2_ablation.runner" not in imported
    assert "research.kmd2_ablation.summarize" not in imported
    assert callable(module.main)


class _FakeCuda:
    def __init__(self, *, available=True, count=2, bf16=True):
        self._available = available
        self._count = count
        self._bf16 = bf16

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def get_device_name(self, ordinal):
        return f"Fake GPU {ordinal}"

    def is_bf16_supported(self):
        return self._bf16


class _FakeTorch:
    __version__ = "2.7.1"

    class version:
        cuda = "12.8"

    def __init__(self, cuda):
        self.cuda = cuda


def test_environment_probe_reports_versions_resources_and_optional_dependencies():
    runner = importlib.import_module("research.kmd2_ablation.runner")
    torch_module = _FakeTorch(_FakeCuda())

    report = runner.probe_environment(
        backend="qwen",
        device_preferences=("cuda", "cpu"),
        dtype_preferences=("bfloat16", "float32"),
        student_device="cuda:1",
        teacher_device="cuda:0",
        requested_dtype="bfloat16",
        python_version=(3, 13, 3),
        python_implementation="CPython",
        torch_module=torch_module,
        dependency_versions={"transformers": "5.12.1", "triton": "3.3.0"},
        dependency_capabilities={"transformers_qwen3_5": True},
    )

    assert report["ok"] is True
    assert report["codes"] == []
    assert report["environment"] == {
        "python": "CPython 3.13.3",
        "pytorch": "2.7.1",
        "cuda": "12.8",
        "gpu": ["Fake GPU 0", "Fake GPU 1"],
        "dependencies": {"transformers": "5.12.1", "triton": "3.3.0"},
        "capabilities": {"transformers_qwen3_5": True},
    }
    assert report["resources"]["student_device"] == "cuda:1"
    assert report["resources"]["teacher_device"] == "cuda:0"
    assert report["resources"]["dtype"] == "bfloat16"
    assert report["warnings"] == []


def test_qwen_environment_requires_transformers_qwen35_import_capability():
    runner = importlib.import_module("research.kmd2_ablation.runner")

    report = runner.probe_environment(
        backend="qwen",
        device_preferences=("cpu",),
        dtype_preferences=("float32",),
        torch_module=_FakeTorch(_FakeCuda(available=False, count=0)),
        dependency_versions={"transformers": "4.57.6", "triton": "3.3.0"},
        dependency_capabilities={"transformers_qwen3_5": False},
    )

    assert report["ok"] is False
    assert report["codes"] == ["transformers_qwen3_5_unavailable"]
    assert report["environment"]["capabilities"] == {
        "transformers_qwen3_5": False
    }
    assert report["warnings"] == []


def test_qwen_score_returning_fast_scan_requires_triton_distribution():
    runner = importlib.import_module("research.kmd2_ablation.runner")

    report = runner.probe_environment(
        backend="qwen",
        device_preferences=("cpu",),
        dtype_preferences=("float32",),
        torch_module=_FakeTorch(_FakeCuda(available=False, count=0)),
        dependency_versions={"transformers": "5.12.1", "triton": None},
        dependency_capabilities={"transformers_qwen3_5": True},
    )

    assert report["ok"] is False
    assert report["codes"] == ["dependency_missing:triton"]
    assert report["warnings"] == []


def test_qwen_dependency_capability_uses_clean_import_without_parent_pollution():
    runner = importlib.import_module("research.kmd2_ablation.runner")
    transformers_before = sys.modules.get("transformers")

    capabilities = runner._dependency_capabilities("qwen")

    assert capabilities == {"transformers_qwen3_5": True}
    assert sys.modules.get("transformers") is transformers_before


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"python_version": (3, 9, 18)}, "python_version_unsupported"),
        ({"dependency_versions": {"transformers": None}}, "dependency_missing:transformers"),
        (
            {
                "student_device": "cuda:2",
                "torch_module": _FakeTorch(_FakeCuda(count=1)),
            },
            "cuda_device_unavailable:cuda:2",
        ),
        (
            {
                "requested_dtype": "bfloat16",
                "torch_module": _FakeTorch(_FakeCuda(bf16=False)),
            },
            "dtype_unsupported:bfloat16",
        ),
    ],
)
def test_environment_probe_returns_stable_actionable_failure_codes(overrides, code):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    arguments = {
        "backend": "qwen",
        "device_preferences": ("cuda", "cpu"),
        "dtype_preferences": ("bfloat16", "float32"),
        "student_device": "cuda:0",
        "teacher_device": None,
        "requested_dtype": "float32",
        "python_version": (3, 13, 3),
        "python_implementation": "CPython",
        "torch_module": _FakeTorch(_FakeCuda()),
        "dependency_versions": {"transformers": "4.53.0"},
    }
    arguments.update(overrides)

    report = runner.probe_environment(**arguments)

    assert report["ok"] is False
    assert code in report["codes"]


def test_output_writability_probe_is_clean_and_actionable(tmp_path, monkeypatch):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    output = tmp_path / "nested" / "results"

    resolved = runner.validate_output_writable(output)

    assert resolved == output.resolve()
    assert output.is_dir()
    assert list(output.iterdir()) == []

    def deny_open(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", deny_open)
    with pytest.raises(runner.PreflightCheckError) as caught:
        runner.validate_output_writable(tmp_path / "denied")
    assert caught.value.code == "output_not_writable"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_asset_identity_validates_files_directory_tree_and_expected_manifest(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    transformers_before = sys.modules.get("transformers")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    model = tmp_path / "model"
    (model / "nested").mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "nested" / "weights.bin").write_bytes(b"weights")

    first = runner.inspect_external_assets(
        {"checkpoint": checkpoint, "model": model}
    )
    expected = {
        name: {
            "kind": record["kind"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
            "files": {
                item["path"]: {
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
                for item in record["tree_manifest"]
            },
        }
        for name, record in first.items()
    }

    repeated = runner.inspect_external_assets(
        {"model": model, "checkpoint": checkpoint}, expected=expected
    )

    assert repeated == first
    assert list(first) == ["checkpoint", "model"]
    assert first["checkpoint"]["sha256"] == _file_digest(checkpoint)
    assert [item["path"] for item in first["model"]["tree_manifest"]] == [
        "config.json",
        "nested/weights.bin",
    ]
    assert sys.modules.get("transformers") is transformers_before


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing", "asset_missing"),
        ("hash", "asset_hash_mismatch"),
        ("tree", "asset_tree_mismatch"),
    ],
)
def test_asset_identity_rejects_missing_hash_and_tree_mismatch(tmp_path, mutation, code):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    asset = tmp_path / "asset"
    asset.mkdir()
    child = asset / "weights.bin"
    child.write_bytes(b"weights")
    actual = runner.inspect_external_assets({"model": asset})["model"]
    path = asset
    expected = {
        "model": {
            "sha256": actual["sha256"],
            "files": {
                "weights.bin": {
                    "size_bytes": child.stat().st_size,
                    "sha256": _file_digest(child),
                }
            },
        }
    }
    if mutation == "missing":
        path = tmp_path / "not-there"
    elif mutation == "hash":
        expected["model"]["sha256"] = "0" * 64
    else:
        expected["model"]["files"]["extra.bin"] = {
            "size_bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }

    with pytest.raises(runner.PreflightCheckError) as caught:
        runner.inspect_external_assets({"model": path}, expected=expected)

    assert caught.value.code == code


def _valid_tiny_raw() -> dict:
    raw = minimal_config_dict()
    raw["required_stage"] = "selector_replay"
    return raw


def _write_config(tmp_path: Path, raw: dict | None = None) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(_valid_tiny_raw() if raw is None else raw), encoding="utf-8")
    return path


def _options(config: Path, output: Path, *, command="preflight", backend="tiny"):
    values = {
        "command": command,
        "backend": backend,
        "config": config,
        "out": output,
        "job_index": 0,
        "num_jobs": 1,
        "resume": True,
        "mode": None,
        "model": None,
        "tokenizer": None,
        "checkpoint": None,
        "data": None,
        "teacher_model": None,
        "student_device": None,
        "teacher_device": None,
        "dtype": None,
        "model_sha256": None,
        "tokenizer_sha256": None,
        "checkpoint_sha256": None,
        "data_sha256": None,
        "teacher_model_sha256": None,
        "assets_manifest": None,
        "repo_root": None,
    }
    if command == "preflight":
        values["dry_run"] = True
    return Namespace(**values)


def _environment_ok(**_kwargs):
    return {
        "ok": True,
        "codes": [],
        "warnings": [],
        "environment": {
            "python": "CPython 3.13.3",
            "pytorch": "2.7.1",
            "cuda": None,
            "gpu": [],
            "dependencies": {},
        },
        "resources": {
            "student_device": "cpu",
            "teacher_device": None,
            "dtype": "float32",
            "cuda_available": False,
            "cuda_device_count": 0,
        },
    }


def _inventory_ok(_root):
    return {
        "inventory_version": "1.0.0",
        "source_files": {"gdn3/kmd2_native.py": "1" * 64},
        "structural_findings": {},
        "compatibility": {},
        "external_assets": {},
    }


def test_scientific_preflight_calls_canonical_variant_validator_exactly_once():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())
    calls = []

    def validate(arm_id, *, backend, task, stage, experiment_kind):
        calls.append((arm_id, backend, task, stage, experiment_kind))
        return variants.validate_variant_compatibility(
            arm_id,
            backend=backend,
            task=task,
            stage=stage,
            experiment_kind=experiment_kind,
        )

    report = runner.validate_scientific_preflight(
        config,
        compatibility_validator=validate,
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
    )

    assert report["ok"] is True
    assert calls == [
        (
            "exact_cache.selector.exact_outer",
            "tiny",
            "mqar",
            "selector_replay",
            "native_warm_start",
        )
    ]
    assert report["arm_id"] == "exact_cache.selector.exact_outer"


def test_scientific_preflight_rejects_compatibility_identity_frozen_and_no_effect():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _valid_tiny_raw()
    raw["backend"] = "qwen"
    raw["mechanism"] = "true_mimo"
    raw["variant"] = "true_mimo_sweep"
    raw["required_stage"] = "mechanism_screen"
    config = config_module.ExperimentConfig.from_dict(raw)

    report = runner.validate_scientific_preflight(
        config,
        gate_evaluator=lambda _config, _spec: {
            "available": True,
            "identity_passed": False,
            "active_effect_passed": False,
            "missing_parameters": [],
            "disconnected_parameters": [],
            "frozen_zero_gates": ["cache_amplitude"],
        },
    )

    assert report["ok"] is False
    assert "variant_incompatible:backend" in report["codes"]
    assert "identity_gate_failed" not in report["codes"]
    assert "active_effect_missing" in report["codes"]
    assert "frozen_zero_gate:cache_amplitude" in report["codes"]


def _gate_evidence(**overrides):
    evidence = {
        "available": True,
        "identity_passed": True,
        "active_effect_passed": True,
        "missing_parameters": [],
        "disconnected_parameters": [],
        "frozen_zero_gates": [],
        "native_feature_present": False,
    }
    evidence.update(overrides)
    return evidence


def test_addition_preflight_refuses_registry_metadata_without_gate_evaluator():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())

    report = runner.validate_scientific_preflight(
        config,
        scientific_evidence=_gate_evidence(),
    )

    assert report["ok"] is False
    assert report["codes"] == ["gate_evaluator_unavailable"]


@pytest.mark.parametrize(
    ("evidence", "code"),
    [
        (
            _gate_evidence(identity_passed=False),
            "identity_gate_failed",
        ),
        (
            _gate_evidence(active_effect_passed=False),
            "active_effect_missing",
        ),
        (
            _gate_evidence(missing_parameters=["cache_amplitude"]),
            "gate_parameter_missing:cache_amplitude",
        ),
        (
            _gate_evidence(disconnected_parameters=["cache_amplitude"]),
            "gate_parameter_disconnected:cache_amplitude",
        ),
        (
            _gate_evidence(frozen_zero_gates=["cache_amplitude"]),
            "frozen_zero_gate:cache_amplitude",
        ),
        (
            {
                "available": True,
                "active_effect_passed": True,
                "missing_parameters": [],
                "disconnected_parameters": [],
                "frozen_zero_gates": [],
            },
            "identity_evidence_missing",
        ),
        (
            {
                "available": True,
                "identity_passed": True,
                "missing_parameters": [],
                "disconnected_parameters": [],
                "frozen_zero_gates": [],
            },
            "active_effect_evidence_missing",
        ),
    ],
)
def test_gate_evaluator_rejects_identity_active_missing_disconnected_and_frozen(
    evidence, code
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())
    calls = []

    def evaluate(received_config, received_spec):
        calls.append((received_config, received_spec.arm_id))
        return evidence

    report = runner.validate_scientific_preflight(
        config,
        gate_evaluator=evaluate,
    )

    assert calls == [(config, "exact_cache.selector.exact_outer")]
    assert report["ok"] is False
    assert code in report["codes"]


def test_gate_evaluator_explicit_identity_and_active_evidence_passes():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())

    report = runner.validate_scientific_preflight(
        config,
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
    )

    assert report["ok"] is True


def test_cache_gate_evidence_is_consumed_by_task9_compatibility_api(monkeypatch):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())
    original = variants.validate_cache_compatibility
    calls = []

    def validate(arm_id, **kwargs):
        calls.append((arm_id, kwargs))
        return original(arm_id, **kwargs)

    monkeypatch.setattr(variants, "validate_cache_compatibility", validate)

    report = runner.validate_scientific_preflight(
        config,
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
    )

    assert report["ok"] is True
    assert calls == [
        (
            "exact_cache.selector.exact_outer",
            {
                "width": config.cache.width,
                "block_size": config.cache.block_size,
                "max_sequence_length": max(config.lengths.curriculum),
                "claimed_evidence_kind": "addition",
                "disabled_identity": True,
                "active_output_changed": True,
                "native_feature_present": False,
            },
        )
    ]


def test_preflight_forwards_gate_evaluator_before_manifest_initialization(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    options = _options(_write_config(tmp_path), tmp_path / "results")
    calls = []

    def evaluate(config, spec):
        calls.append((config.experiment_id, spec.arm_id))
        return _gate_evidence()

    report = runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda *_args: None,
        backend_probe=lambda backend: {"ok": True, "codes": [], "backend": backend},
        gate_evaluator=evaluate,
        resource_evaluator=_passing_resource_evaluator,
    )

    assert report["ok"] is True
    assert calls == [
        (report["jobs"][0]["experiment_id"], "exact_cache.selector.exact_outer")
    ]


def _exact_resource_evidence(**overrides):
    evidence = {
        "available": True,
        "exact": True,
        "trainable_parameters": 12_000,
        "total_parameters": 12_000,
        "recurrent_state_elements": 64,
        "recurrent_state_bytes": 256,
        "cache_persistent_bytes": 512,
        "cache_block_bytes": 1024,
        "cache_storage_dtype": "bf16",
        "cache_compute_dtype": "fp32",
        "ffn_match": {
            "matched": True,
            "target_parameters": 12_000,
            "matched_parameters": 12_000,
            "selected_d_ff": 768,
            "residual_mismatch": 0,
            "tolerance": 1024.0,
        },
    }
    evidence.update(overrides)
    return evidence


def _passing_resource_evaluator(_config, _spec):
    return _exact_resource_evidence()


def test_non_tiny_resources_refuse_success_without_exact_evaluator():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    raw = _valid_tiny_raw()
    raw["backend"] = "qwen"
    config = config_module.ExperimentConfig.from_dict(raw)
    spec = variants.get_variant("exact_cache.selector.exact_outer")

    report = runner.evaluate_exact_resources(
        config,
        spec,
        resource_evaluator=None,
    )

    assert report["ok"] is False
    assert report["codes"] == ["resource_evaluator_unavailable"]


@pytest.mark.parametrize(
    ("evidence", "code"),
    [
        (
            _exact_resource_evidence(exact=False),
            "resource_accounting_not_exact",
        ),
        (
            _exact_resource_evidence(
                ffn_match={
                    "matched": False,
                    "target_parameters": 12_000,
                    "matched_parameters": 10_000,
                    "selected_d_ff": 704,
                    "residual_mismatch": -2_000,
                    "tolerance": 1024.0,
                }
            ),
            "ffn_match_failed",
        ),
        (
            _exact_resource_evidence(
                ffn_match={
                    "matched": True,
                    "target_parameters": 12_000,
                    "matched_parameters": 13_025,
                    "selected_d_ff": 832,
                    "residual_mismatch": 1_025,
                    "tolerance": 1024.0,
                }
            ),
            "ffn_match_failed",
        ),
    ],
)
def test_exact_resource_evaluator_rejects_heuristics_no_match_and_tolerance(
    evidence, code
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())
    spec = variants.get_variant("exact_cache.selector.exact_outer")

    report = runner.evaluate_exact_resources(
        config,
        spec,
        resource_evaluator=lambda _config, _spec: evidence,
    )

    assert report["ok"] is False
    assert code in report["codes"]


def test_exact_resource_evaluator_consumes_task9_parameter_match_accounting():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    tiny_backend = importlib.import_module("research.kmd2_ablation.tiny_backend")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())
    spec = variants.get_variant("exact_cache.selector.exact_outer")
    target = tiny_backend.TinyKMD2Config(
        d_model=8,
        heads=1,
        dk=2,
        dv=2,
        layers=1,
        vocab_size=11,
        d_ff=16,
        rotation_mode="none",
    )
    arm = tiny_backend.TinyKMD2Config(
        d_model=8,
        heads=1,
        dk=4,
        dv=2,
        layers=1,
        vocab_size=11,
        d_ff=16,
        rotation_mode="none",
    )
    match = variants.match_tiny_parameter_count(
        target,
        arm,
        comparison="state_size",
        d_ff_match_min=8,
        d_ff_match_max=128,
    )

    report = runner.evaluate_exact_resources(
        config,
        spec,
        resource_evaluator=lambda _config, _spec: {
            "available": True,
            "exact": True,
            "parameter_match": match,
            "total_parameters": match.matched.trainable_parameters,
            "cache_persistent_bytes": 512,
            "cache_block_bytes": 1024,
            "cache_storage_dtype": "bf16",
            "cache_compute_dtype": "fp32",
        },
    )

    assert report["ok"] is True
    assert report["resources"]["trainable_parameters"] == (
        match.matched.trainable_parameters
    )
    assert report["resources"]["recurrent_state_elements"] == (
        match.matched.recurrent_state_elements
    )
    assert report["resources"]["recurrent_state_bytes"] == (
        match.matched.recurrent_state_bytes
    )
    assert report["resources"]["ffn_match"] == {
        "matched": True,
        "target_parameters": match.target.trainable_parameters,
        "matched_parameters": match.matched.trainable_parameters,
        "selected_d_ff": match.matched.d_ff,
        "residual_mismatch": match.residual_mismatch,
        "tolerance": match.tolerance,
    }


def test_default_tiny_resource_evaluator_uses_instantiated_task9_accounting(
    monkeypatch,
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    config = config_module.ExperimentConfig.from_dict(_valid_tiny_raw())
    spec = variants.get_variant("exact_cache.selector.exact_outer")
    original = variants.construct_equal_state_byte_control
    calls = []

    def account(base_config, *, cache_width, storage_dtype):
        calls.append((base_config, cache_width, storage_dtype))
        return original(
            base_config,
            cache_width=cache_width,
            storage_dtype=storage_dtype,
        )

    monkeypatch.setattr(variants, "construct_equal_state_byte_control", account)

    report = runner.evaluate_exact_resources(
        config,
        spec,
        resource_evaluator=None,
    )

    assert report["ok"] is True
    assert len(calls) == 1
    base_config, width, storage_dtype = calls[0]
    assert base_config.d_model == config.model.hidden_size
    assert base_config.heads == config.model.num_heads
    assert base_config.dk == config.model.state_key_dim
    assert base_config.dv == config.model.state_value_dim
    assert base_config.layers == config.model.num_layers
    assert base_config.d_ff == config.model.ffn_dim
    assert width == config.cache.width
    assert storage_dtype == config.cache.storage_dtype
    assert report["resources"]["cache_persistent_bytes"] > 0
    assert report["resources"]["total_parameters"] >= report["resources"][
        "trainable_parameters"
    ]
    assert report["resources"]["ffn_match"]["matched"] is True


@pytest.mark.parametrize(
    ("config_name", "expected_state_elements"),
    [
        ("trapezoid_screening.json", 2 * 2 * (4 * 4 + 4 + 4)),
        ("corrected_momentum_screening.json", 2 * 2 * (2 * 4 * 4)),
        ("causal_lookahead_screening.json", 2 * 2 * (4 * 4 + 4)),
    ],
)
def test_default_tiny_resource_evaluator_counts_warm_addition_dynamic_state(
    config_name, expected_state_elements
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "kmd2_ablation"
        / "configs"
        / config_name
    )
    config = config_module.ExperimentConfig.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    spec = next(
        item
        for item in variants.all_variants()
        if (item.mechanism, item.variant) == (config.mechanism, config.variant)
    )

    report = runner.evaluate_exact_resources(
        config,
        spec,
        resource_evaluator=None,
    )

    assert report["ok"] is True, report["codes"]
    assert report["resources"]["recurrent_state_elements"] == expected_state_elements
    assert report["resources"]["cache_persistent_bytes"] == 0
    assert report["resources"]["cache_block_bytes"] == 0


def test_default_tiny_matched_arm_uses_task9_matcher_and_reports_no_legal_match(
    monkeypatch,
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    raw = _valid_tiny_raw()
    raw.update(
        mechanism="state_size",
        variant="state_size_sweep",
        required_stage="mechanism_screen",
    )
    raw["task"]["params"]["parameter_match_target"] = {
        "state_key_dim": 32,
        "state_value_dim": 64,
        "mimo_rank": 1,
    }
    config = config_module.ExperimentConfig.from_dict(raw)
    spec = variants.get_variant("state_size.sweep")
    calls = []

    def no_match(target, arm, **kwargs):
        calls.append((target, arm, kwargs))
        raise ValueError("no legal parameter match within tolerance")

    monkeypatch.setattr(variants, "match_tiny_parameter_count", no_match)

    report = runner.evaluate_exact_resources(
        config,
        spec,
        resource_evaluator=None,
    )

    assert len(calls) == 1
    target, arm, kwargs = calls[0]
    assert (target.dk, target.dv, target.mimo_rank) == (32, 64, 1)
    assert (arm.dk, arm.dv, arm.mimo_rank) == (64, 64, 1)
    assert kwargs == {
        "comparison": "state_size",
        "d_ff_match_min": config.model.ffn_match_lower,
        "d_ff_match_max": config.model.ffn_match_upper,
    }
    assert report["ok"] is False
    assert report["codes"] == ["ffn_match_failed"]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda raw: raw["cache"].update(width=0), "cache_width_invalid"),
        (
            lambda raw: raw["lengths"].update(curriculum=[64]),
            "cache_requires_two_blocks",
        ),
        (lambda raw: raw["cache"].update(width=256), "cache_eviction_impossible"),
        (
            lambda raw: raw["model"].update(ffn_dim=840),
            "ffn_match_invalid",
        ),
        (
            lambda raw: raw.update(variant="cache_rotation_factorial"),
            "four_cell_incomplete",
        ),
    ],
)
def test_raw_scientific_preflight_rejects_geometry_factorial_and_ffn(mutate, code):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _valid_tiny_raw()
    mutate(raw)

    codes = runner.validate_raw_scientific_config(raw, backend="tiny", mode=None)

    assert code in codes


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("streaming", True, "qwen_streaming_unsupported"),
        ("decode", True, "qwen_decode_unsupported"),
        ("packing", True, "qwen_packing_unsupported"),
        ("padding", "pad_to_longest", "qwen_padding_unsupported"),
        ("attention_mask", "packed_segments", "qwen_attention_mask_unsupported"),
    ],
)
def test_raw_scientific_preflight_rejects_qwen_option3_inputs(field, value, code):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _valid_tiny_raw()
    raw["backend"] = "qwen"
    raw["qwen"][field] = value

    codes = runner.validate_raw_scientific_config(
        raw, backend="qwen", mode="initial_exact_cache"
    )

    assert code in codes


@pytest.mark.parametrize(
    ("option3_inputs", "code"),
    [
        (
            {"surprise": {}, "unexpected": {}},
            "option3_inputs_incomplete",
        ),
        (
            {"surprise": {}, "recency": {}, "unexpected": {}},
            "option3_inputs_unknown",
        ),
    ],
)
def test_raw_scientific_preflight_rejects_incomplete_or_extra_option3_keys(
    option3_inputs, code
):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _valid_tiny_raw()
    raw["task"]["params"]["option3_inputs"] = option3_inputs

    codes = runner.validate_raw_scientific_config(raw, backend="tiny", mode=None)

    assert code in codes


def test_reproduction_commands_include_every_asset_identity_and_repo_root(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    options = _options(
        tmp_path / "experiment.json",
        tmp_path / "results",
        backend="qwen",
    )
    identities = {
        "mode": "heal",
        "model": tmp_path / "model",
        "tokenizer": tmp_path / "tokenizer",
        "checkpoint": tmp_path / "checkpoint.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher",
        "student_device": "cuda:1",
        "teacher_device": "cuda:0",
        "dtype": "bfloat16",
        "model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "data_sha256": "4" * 64,
        "teacher_model_sha256": "5" * 64,
        "assets_manifest": tmp_path / "assets.json",
        "repo_root": tmp_path / "repo",
    }
    for field, value in identities.items():
        setattr(options, field, value)

    commands = runner.build_reproduction_commands(options)

    expected_flags = {
        "--model": identities["model"],
        "--tokenizer": identities["tokenizer"],
        "--checkpoint": identities["checkpoint"],
        "--data": identities["data"],
        "--teacher-model": identities["teacher_model"],
        "--model-sha256": identities["model_sha256"],
        "--tokenizer-sha256": identities["tokenizer_sha256"],
        "--checkpoint-sha256": identities["checkpoint_sha256"],
        "--data-sha256": identities["data_sha256"],
        "--teacher-model-sha256": identities["teacher_model_sha256"],
        "--assets-manifest": identities["assets_manifest"],
        "--repo-root": identities["repo_root"],
    }
    assert set(commands) == {"preflight", "run", "summarize", "bundle"}
    for command in commands.values():
        for flag, value in expected_flags.items():
            index = command.index(flag)
            assert command[index + 1] == str(value)


def test_preflight_success_reports_contract_and_initializes_immutable_manifests(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config_path = _write_config(tmp_path)
    output = tmp_path / "results"
    options = _options(config_path, output)
    verifier_calls = []

    report = runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda inventory, root: verifier_calls.append(
            (inventory, root)
        ),
        backend_probe=lambda backend: {"ok": True, "codes": [], "backend": backend},
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
        resource_evaluator=_passing_resource_evaluator,
    )

    assert set(report) == {
        "ok",
        "schema_version",
        "codes",
        "warnings",
        "inventory",
        "resources",
        "assets",
        "jobs",
        "commands",
        "manifest_path",
    }
    assert report["ok"] is True
    assert len(report["jobs"]) == 4
    assert len({job["pairing_id"] for job in report["jobs"]}) == 2
    for seed in {job["seed"] for job in report["jobs"]}:
        paired = [job for job in report["jobs"] if job["seed"] == seed]
        assert {job["arm_id"] for job in paired} == {
            "native",
            "exact_cache.selector.exact_outer",
        }
        assert len({job["pairing_id"] for job in paired}) == 1
    assert verifier_calls and verifier_calls[0][1].is_dir()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    jobs = json.loads((output / "jobs.json").read_text(encoding="utf-8"))
    assert jobs["jobs"] == report["jobs"]
    assert manifest["command"] == [
        "python",
        "-m",
        "research.kmd2_ablation.run_ablation",
        "run",
        "--backend",
        "tiny",
    ]
    assert manifest["command"] != report["commands"]["run"]
    assert Path(report["manifest_path"]) == (output / "manifest.json").resolve()
    assert report["resources"]["recurrent_state_bytes"] > 0
    assert report["resources"]["cache_persistent_bytes"] > 0
    assert report["resources"]["cache_block_bytes"] > 0
    assert runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda *_args: None,
        backend_probe=lambda backend: {"ok": True, "codes": [], "backend": backend},
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
        resource_evaluator=_passing_resource_evaluator,
    )["ok"] is True


def test_production_tiny_preflight_uses_measured_gate_probe_and_matched_baseline(
    tmp_path,
):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _valid_tiny_raw()
    raw["device_preferences"] = ["cpu"]
    raw["dtype_preferences"] = ["float32"]
    options = _options(_write_config(tmp_path, raw), tmp_path / "results")
    options.student_device = "cpu"
    options.dtype = "float32"

    report = runner.preflight_command(options)

    assert report["ok"] is True, report["codes"]
    assert len(report["jobs"]) == 4
    assert {job["arm_id"] for job in report["jobs"]} == {
        "native",
        "exact_cache.selector.exact_outer",
    }


@pytest.mark.parametrize(
    ("arm_id", "task"),
    [
        ("trapezoid", "irregular_integration"),
        ("bc_bias", "affine_associative_regression"),
        ("corrected_momentum", "drift_reversal"),
        ("causal_lookahead", "trajectory"),
        ("state_size.sweep", "mqar"),
        ("true_mimo.sweep", "mqar"),
        ("gdn2_decoupled.channelwise", "mqar"),
        ("exact_cache.selector.exact_outer", "mqar"),
    ],
)
def test_default_gate_probe_measures_every_primary_addition_family(arm_id, task):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    probes = importlib.import_module("research.kmd2_ablation.gate_probes")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    spec = variants.get_variant(arm_id)
    raw = _valid_tiny_raw()
    raw.update(
        mechanism=spec.mechanism,
        variant=spec.variant,
        required_stage=spec.required_stage,
    )
    raw["task"] = {"name": task, "params": {}}
    config = config_module.ExperimentConfig.from_dict(raw)

    evidence = probes.measure_scientific_gates(config, spec)

    assert evidence["available"] is True
    assert evidence["active_effect_passed"] is True
    assert evidence["identity_passed"] is spec.native_warm_start
    assert evidence["missing_parameters"] == []
    assert evidence["disconnected_parameters"] == []
    assert evidence["frozen_zero_gates"] == []


def test_cold_redesign_preflight_requires_active_effect_but_not_warm_identity():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    spec = variants.get_variant("state_size.sweep")
    raw = _valid_tiny_raw()
    raw.update(
        mechanism=spec.mechanism,
        variant=spec.variant,
        required_stage=spec.required_stage,
    )
    raw["task"] = {"name": "mqar", "params": {}}
    config = config_module.ExperimentConfig.from_dict(raw)

    report = runner.validate_scientific_preflight(
        config,
        gate_evaluator=lambda _config, _spec: {
            "available": True,
            "identity_passed": False,
            "active_effect_passed": True,
            "missing_parameters": [],
            "disconnected_parameters": [],
            "frozen_zero_gates": [],
        },
    )

    assert report["ok"] is True, report["codes"]


def _write_safetensors_metadata(path: Path, tensors: dict[str, list[int]]) -> None:
    import struct

    offset = 0
    header = {}
    for name in sorted(tensors):
        elements = 1
        for dimension in tensors[name]:
            elements *= dimension
        size = 4 * elements
        header[name] = {
            "dtype": "F32",
            "shape": tensors[name],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def test_production_qwen_dry_run_uses_metadata_only_resources_and_emits_nine_jobs(
    tmp_path, monkeypatch,
):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _valid_tiny_raw()
    raw["backend"] = "qwen"
    raw["qwen"]["run_mode"] = "heal"
    raw["required_stage"] = "qwen_heal"
    raw["seeds"] = [11, 29, 47]
    raw["device_preferences"] = ["cpu"]
    raw["dtype_preferences"] = ["float32"]
    raw["model"].update(
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        state_key_dim=2,
        state_value_dim=2,
        ffn_dim=16,
        ffn_match_lower=8,
        ffn_match_upper=24,
    )
    memory_names = [
        f"model.layers.{index}.linear_attn.in_proj_b.weight"
        for index in range(2)
    ]
    cache_names = [
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in range(2)
        for suffix in (
            "cache_gamma_q",
            "cache_gamma_k",
            "cache_sink_logit",
            "cache_amplitude",
        )
    ]
    raw["task"] = {
        "name": "ruler",
        "params": {
            "example_ids": ["ruler-0", "ruler-1"],
            "objective": "synthetic_only",
            "native_r_out": 4,
            "score_scan": "gdn3.kmd2_fast_scan.scan_with_update_norm",
            "training_window_example_counts": [1, 1],
            "training_window_token_counts": [32, 32],
            "accumulation_steps": 1,
            "memory_parameter_names": memory_names,
            "cache_parameter_names": cache_names,
        },
    }
    raw["budget"] = {"tokens": 64, "updates": 2}
    model = tmp_path / "model.safetensors"
    _write_safetensors_metadata(
        model,
        {
            "model.embed_tokens.weight": [8, 8],
            **{name: [2, 8] for name in memory_names},
        },
    )
    checkpoint = tmp_path / "native-checkpoint.pt"
    checkpoint.write_bytes(b"measured-native-checkpoint")
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"ruler-0"}\n{"id":"ruler-1"}\n', encoding="utf-8")
    options = _options(
        _write_config(tmp_path, raw),
        tmp_path / "results",
        backend="qwen",
    )
    options.mode = "heal"
    options.model = model
    options.checkpoint = checkpoint
    options.data = data
    options.student_device = "cpu"
    options.dtype = "float32"
    _model_clean_qwen_process(monkeypatch)

    report = runner.preflight_command(options)

    assert report["ok"] is True, report["codes"]
    assert len(report["jobs"]) == 9
    assert {job["arm_id"] for job in report["jobs"]} == {
        "native",
        "recency",
        "surprise",
    }
    assert report["resources"]["parameter_metadata_kind"] == "safetensors_header"
    # Two tiny KMD2 layers add 48 parameters each at pinned r_out=4.
    assert report["resources"]["native_addition_parameters"] == 96
    assert report["resources"]["native_r_out"] == 4
    assert report["resources"]["qwen_execution"]["score_scan"] == (
        "gdn3.kmd2_fast_scan.scan_with_update_norm"
    )
    manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["environment"]["qwen_execution"] == report["resources"][
        "qwen_execution"
    ]


def _qwen_window_contract_raw() -> dict:
    raw = _valid_tiny_raw()
    raw["backend"] = "qwen"
    raw["qwen"]["run_mode"] = "heal"
    raw["required_stage"] = "qwen_heal"
    raw["seeds"] = [11, 29, 47]
    raw["task"] = {
        "name": "ruler",
        "params": {
            "objective": "synthetic_only",
            "accumulation_steps": 2,
            "example_ids": ["window-0", "window-1", "window-2", "window-3"],
            "training_window_example_counts": [1, 1, 1, 1],
            "training_window_token_counts": [16, 16, 16, 16],
            "native_r_out": 4,
            "score_scan": "gdn3.kmd2_fast_scan.scan_with_update_norm",
        },
    }
    raw["budget"] = {"tokens": 64, "updates": 2}
    return raw


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["task"]["params"].pop("training_window_token_counts"),
        lambda raw: raw["task"]["params"].update(
            training_window_token_counts=[16, 16, 32]
        ),
        lambda raw: raw["task"]["params"].update(
            training_window_token_counts=[16, 16, 16, 15]
        ),
        lambda raw: raw["task"]["params"].update(
            training_window_example_counts=[1, 1, 1, 2]
        ),
    ],
)
def test_qwen_dry_run_rejects_unrepresentable_window_contracts(mutate):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _qwen_window_contract_raw()
    mutate(raw)

    assert "qwen_data_window_contract_invalid" in runner.validate_raw_scientific_config(
        raw, backend="qwen", mode="heal"
    )


def test_qwen_dry_run_accepts_exact_update_accumulation_window_contract():
    runner = importlib.import_module("research.kmd2_ablation.runner")

    assert runner.validate_raw_scientific_config(
        _qwen_window_contract_raw(), backend="qwen", mode="heal"
    ) == []


def test_qwen_metadata_resource_probe_rejects_shape_offset_lie(tmp_path):
    import struct

    config_module = importlib.import_module("research.kmd2_ablation.config")
    probes = importlib.import_module("research.kmd2_ablation.resource_probes")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    raw = _valid_tiny_raw()
    raw["backend"] = "qwen"
    raw["task"]["params"].update(
        native_r_out=4,
        score_scan="gdn3.kmd2_fast_scan.scan_with_update_norm",
        memory_parameter_names=["model.layers.0.linear_attn.in_proj_b.weight"],
    )
    raw["model"]["num_layers"] = 1
    config = config_module.ExperimentConfig.from_dict(raw)
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    model = tmp_path / "lying.safetensors"
    model.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)

    with pytest.raises(runner.PreflightCheckError) as caught:
        probes.measure_qwen_resources(
            config,
            variants.get_variant("exact_cache.selector.exact_outer"),
            assets={"model": {"path": str(model)}},
            environ={"GDN3_FAST_SCAN": "1", "GDN3_KMD2_ROUT": "4"},
            loaded_modules={},
        )

    assert caught.value.code == "parameter_metadata_invalid"


def test_qwen_resource_probe_counts_pinned_native_additions_and_cache_exactly(
    tmp_path,
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    probes = importlib.import_module("research.kmd2_ablation.resource_probes")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    raw = _qwen_window_contract_raw()
    raw["model"].update(
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        state_key_dim=2,
        state_value_dim=2,
        ffn_dim=16,
        ffn_match_lower=8,
        ffn_match_upper=24,
    )
    memory_names = [f"model.layers.{index}.linear_attn.in_proj_b.weight" for index in range(2)]
    cache_names = [
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in range(2)
        for suffix in (
            "cache_gamma_q",
            "cache_gamma_k",
            "cache_sink_logit",
            "cache_amplitude",
        )
    ]
    raw["task"]["params"].update(
        memory_parameter_names=memory_names,
        cache_parameter_names=cache_names,
    )
    config = config_module.ExperimentConfig.from_dict(raw)
    model = tmp_path / "model.safetensors"
    _write_safetensors_metadata(
        model,
        {
            "model.embed_tokens.weight": [8, 8],
            **{name: [2, 8] for name in memory_names},
        },
    )

    evidence = probes.measure_qwen_resources(
        config,
        variants.get_variant("exact_cache.selector.exact_outer"),
        assets={"model": {"path": str(model)}},
        environ={"GDN3_FAST_SCAN": "1", "GDN3_KMD2_ROUT": "4"},
        loaded_modules={},
    )

    assert evidence["native_addition_parameters"] == 96
    assert evidence["cache_parameter_count"] == 16
    assert evidence["total_base_parameters"] == 96
    assert evidence["total_parameters"] == 208
    assert evidence["arm_total_parameters"] == {
        "native": 192,
        "recency": 208,
        "surprise": 208,
    }
    assert evidence["parameter_scope"] == (
        "full_model_plus_installed_kmd2_native_plus_cache"
    )
    assert evidence["qwen_execution"] == {
        "activation_proof": "preimport_environment_and_source_contract",
        "fast_scan": True,
        "native_r_out": 4,
        "native_scan": "gdn3.kmd2_fast_scan.scan",
        "score_scan": "gdn3.kmd2_fast_scan.scan_with_update_norm",
    }


@pytest.mark.parametrize(
    ("environ", "loaded_modules", "code"),
    [
        ({"GDN3_KMD2_ROUT": "4"}, {}, "qwen_fast_scan_inactive"),
        (
            {"GDN3_FAST_SCAN": "1", "GDN3_KMD2_ROUT": "1"},
            {},
            "qwen_r_out_mismatch",
        ),
        (
            {"GDN3_FAST_SCAN": "1", "GDN3_KMD2_ROUT": "4"},
            {"gdn3.kmd2_native": type("LoadedNative", (), {"_FAST_SCAN": False})()},
            "qwen_fast_scan_import_order_invalid",
        ),
    ],
)
def test_qwen_execution_contract_rejects_inactive_or_late_fast_scan(
    environ, loaded_modules, code
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    probes = importlib.import_module("research.kmd2_ablation.resource_probes")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config = config_module.ExperimentConfig.from_dict(_qwen_window_contract_raw())

    with pytest.raises(runner.PreflightCheckError) as caught:
        probes.verify_qwen_execution_contract(
            config, environ=environ, loaded_modules=loaded_modules
        )

    assert caught.value.code == code


def test_qwen_execution_contract_accepts_real_manifest_semantic_mapping():
    config_module = importlib.import_module("research.kmd2_ablation.config")
    probes = importlib.import_module("research.kmd2_ablation.resource_probes")
    canonical = config_module.ExperimentConfig.from_dict(
        _qwen_window_contract_raw()
    ).semantic_dict()
    assert "runtime" not in canonical

    contract = probes.verify_qwen_execution_contract(
        canonical,
        environ={"GDN3_FAST_SCAN": "1", "GDN3_KMD2_ROUT": "4"},
        loaded_modules={},
    )

    assert contract["fast_scan"] is True
    assert contract["native_r_out"] == 4
    assert contract["score_scan"] == (
        "gdn3.kmd2_fast_scan.scan_with_update_norm"
    )


def test_qwen_resource_probe_rejects_memory_layout_that_cannot_install_native(
    tmp_path,
):
    config_module = importlib.import_module("research.kmd2_ablation.config")
    probes = importlib.import_module("research.kmd2_ablation.resource_probes")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    variants = importlib.import_module("research.kmd2_ablation.variants")
    raw = _qwen_window_contract_raw()
    raw["model"].update(
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        state_key_dim=2,
        state_value_dim=2,
        ffn_dim=16,
        ffn_match_lower=8,
        ffn_match_upper=24,
    )
    memory_name = "model.layers.0.linear_attn.in_proj_b.weight"
    raw["task"]["params"].update(
        memory_parameter_names=[memory_name],
        cache_parameter_names=[
            f"model.layers.0.linear_attn.{suffix}"
            for suffix in (
                "cache_gamma_q",
                "cache_gamma_k",
                "cache_sink_logit",
                "cache_amplitude",
            )
        ],
    )
    config = config_module.ExperimentConfig.from_dict(raw)
    model = tmp_path / "bad-layout.safetensors"
    _write_safetensors_metadata(model, {memory_name: [8, 2]})

    with pytest.raises(runner.PreflightCheckError) as caught:
        probes.measure_qwen_resources(
            config,
            variants.get_variant("exact_cache.selector.exact_outer"),
            assets={"model": {"path": str(model)}},
            environ={"GDN3_FAST_SCAN": "1", "GDN3_KMD2_ROUT": "4"},
            loaded_modules={},
        )

    assert caught.value.code == "parameter_metadata_mismatch"


def test_qwen_native_addition_accounting_rejects_integer_overflow():
    from dataclasses import replace

    config_module = importlib.import_module("research.kmd2_ablation.config")
    probes = importlib.import_module("research.kmd2_ablation.resource_probes")
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config = config_module.ExperimentConfig.from_dict(_qwen_window_contract_raw())
    oversized = replace(
        config,
        model=replace(config.model, hidden_size=1 << 62),
    )

    with pytest.raises(runner.PreflightCheckError) as caught:
        probes._native_addition_count(oversized, r_out=4)

    assert caught.value.code == "parameter_accounting_overflow"


def test_preflight_source_failure_and_manifest_conflict_are_machine_readable(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    options = _options(_write_config(tmp_path), tmp_path / "results")

    stale = runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda *_args: (_ for _ in ()).throw(
            ValueError("source mismatch")
        ),
        backend_probe=lambda backend: {"ok": True, "codes": [], "backend": backend},
    )
    assert stale["ok"] is False
    assert stale["codes"] == ["source_hash_stale"]

    clean = runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda *_args: None,
        backend_probe=lambda backend: {"ok": True, "codes": [], "backend": backend},
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
        resource_evaluator=_passing_resource_evaluator,
    )
    assert clean["ok"] is True
    manifest_path = Path(clean["manifest_path"])
    manifest_path.write_text("{}\n", encoding="utf-8")
    conflict = runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda *_args: None,
        backend_probe=lambda backend: {"ok": True, "codes": [], "backend": backend},
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
        resource_evaluator=_passing_resource_evaluator,
    )
    assert conflict["ok"] is False
    assert "immutable_manifest_conflict" in conflict["codes"]


def test_run_reads_preflight_documents_and_uses_result_store_execute_jobs(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    options = _options(_write_config(tmp_path), tmp_path / "results")
    ready = lambda backend: {"ok": True, "codes": [], "backend": backend}
    assert runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda *_args: None,
        backend_probe=ready,
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
        resource_evaluator=_passing_resource_evaluator,
    )["ok"]
    run_options = Namespace(**{**vars(options), "command": "run"})
    captured = {}

    def execute(jobs, *, store, command, dispatchers, resume):
        captured.update(
            jobs=jobs,
            store=store,
            command=command,
            dispatchers=dispatchers,
            resume=resume,
        )
        return [{"job_id": jobs[0]["job_id"], "status": "skipped"}]

    report = runner.run(
        run_options,
        execute_fn=execute,
        dispatchers={"tiny": lambda job: job},
    )

    assert report["ok"] is True
    assert isinstance(captured["store"], runner.ResultStore)
    assert captured["jobs"] == json.loads(
        (options.out / "jobs.json").read_text(encoding="utf-8")
    )["jobs"]
    assert captured["store"].job_index == 0
    assert captured["store"].num_jobs == 1
    assert captured["resume"] is True
    assert captured["command"][3] == "run"


def test_runtime_dispatcher_builder_receives_qwen_runtime_outside_jobs(
    tmp_path, monkeypatch
):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config_module = importlib.import_module("research.kmd2_ablation.config")
    options = _options(
        tmp_path / "experiment.json",
        tmp_path / "results",
        command="run",
        backend="qwen",
    )
    options.mode = "heal"
    options.model = tmp_path / "model"
    options.tokenizer = tmp_path / "tokenizer"
    options.checkpoint = tmp_path / "checkpoint.pt"
    options.data = tmp_path / "data.jsonl"
    options.teacher_model = tmp_path / "teacher"
    options.student_device = "cuda:1"
    options.teacher_device = "cuda:0"
    options.dtype = "bfloat16"
    asset_hashes = {
        "model": "1" * 64,
        "checkpoint": "2" * 64,
        "data": "3" * 64,
        "teacher_model": "4" * 64,
    }
    captured = {}
    dispatcher = lambda job: {"job_id": job["job_id"]}

    class FakeQwenTraining:
        @staticmethod
        def build_job_dispatcher(runtime, dependencies=None):
            captured["runtime"] = runtime
            captured["dependencies"] = dependencies
            return dispatcher

    def load(module_name):
        captured["module_name"] = module_name
        return FakeQwenTraining

    _model_clean_qwen_process(monkeypatch)
    execution = {
        "activation_proof": "preimport_environment_and_source_contract",
        "fast_scan": True,
        "native_r_out": 4,
        "native_scan": "gdn3.kmd2_fast_scan.scan",
        "score_scan": "gdn3.kmd2_fast_scan.scan_with_update_norm",
    }

    dispatchers = runner.build_runtime_dispatchers(
        options,
        manifest={
            "asset_hashes": asset_hashes,
            "canonical_config": config_module.ExperimentConfig.from_dict(
                _qwen_window_contract_raw()
            ).semantic_dict(),
            "environment": {"qwen_execution": execution},
        },
        module_loader=load,
    )

    assert dispatchers == {"qwen": dispatcher}
    assert captured["module_name"] == "research.kmd2_ablation.qwen_training"
    assert captured["dependencies"] is None
    assert captured["runtime"] == {
        "model": options.model,
        "tokenizer": options.tokenizer,
        "checkpoint": options.checkpoint,
        "data": options.data,
        "teacher_model": options.teacher_model,
        "output": options.out,
        "student_device": "cuda:1",
        "teacher_device": "cuda:0",
        "dtype": "bfloat16",
        "asset_hashes": asset_hashes,
        "resume": True,
    }
    assert not {"job", "jobs", "canonical_config"} & set(captured["runtime"])


def test_real_qwen_runtime_dispatcher_binds_from_manifest_semantic_config(
    tmp_path, monkeypatch
):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config_module = importlib.import_module("research.kmd2_ablation.config")
    options = _options(
        tmp_path / "experiment.json",
        tmp_path / "results",
        command="run",
        backend="qwen",
    )
    options.model = tmp_path / "model"
    options.checkpoint = tmp_path / "checkpoint.pt"
    options.data = tmp_path / "data.jsonl"
    options.student_device = "cuda:0"
    options.dtype = "bfloat16"
    _model_clean_qwen_process(monkeypatch)
    contract = {
        "activation_proof": "preimport_environment_and_source_contract",
        "fast_scan": True,
        "native_r_out": 4,
        "native_scan": "gdn3.kmd2_fast_scan.scan",
        "score_scan": "gdn3.kmd2_fast_scan.scan_with_update_norm",
    }

    dispatchers = runner.build_runtime_dispatchers(
        options,
        manifest={
            "asset_hashes": {
                "model": "1" * 64,
                "checkpoint": "2" * 64,
                "data": "3" * 64,
            },
            "canonical_config": config_module.ExperimentConfig.from_dict(
                _qwen_window_contract_raw()
            ).semantic_dict(),
            "environment": {"qwen_execution": contract},
        },
    )

    assert set(dispatchers) == {"qwen"}
    assert callable(dispatchers["qwen"])


def test_qwen_runtime_binding_rejects_scan_environment_drift_before_backend_import(
    tmp_path, monkeypatch
):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    config_module = importlib.import_module("research.kmd2_ablation.config")
    options = _options(
        tmp_path / "experiment.json",
        tmp_path / "results",
        command="run",
        backend="qwen",
    )
    imported = []
    monkeypatch.delenv("GDN3_FAST_SCAN", raising=False)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")

    with pytest.raises(runner.BackendUnavailable, match="fast scan"):
        runner.build_runtime_dispatchers(
            options,
            manifest={
                "asset_hashes": {},
                "canonical_config": config_module.ExperimentConfig.from_dict(
                    _qwen_window_contract_raw()
                ).semantic_dict(),
                "environment": {
                    "qwen_execution": {
                        "activation_proof": "preimport_environment_and_source_contract",
                        "fast_scan": True,
                        "native_r_out": 4,
                        "native_scan": "gdn3.kmd2_fast_scan.scan",
                        "score_scan": "gdn3.kmd2_fast_scan.scan_with_update_norm",
                    }
                },
            },
            module_loader=lambda name: imported.append(name),
        )

    assert imported == []


def test_tiny_runtime_binding_consumes_cli_device_without_forwarding_qwen_keys(
    tmp_path,
):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    options = _options(
        tmp_path / "experiment.json",
        tmp_path / "results",
        command="run",
        backend="tiny",
    )
    options.student_device = "cpu"
    options.teacher_device = "cuda:0"
    options.dtype = "float32"
    captured = {}
    dispatcher = lambda job: job

    class FakeTinyTraining:
        @staticmethod
        def build_job_dispatcher(runtime):
            captured["runtime"] = runtime
            return dispatcher

    result = runner.build_runtime_dispatchers(
        options,
        manifest={
            "asset_hashes": {},
            "canonical_config": {"dtype_preferences": ["float32"]},
        },
        module_loader=lambda _name: FakeTinyTraining,
    )

    assert result == {"tiny": dispatcher}
    assert captured["runtime"] == {
        "output": options.out,
        "dtype": "float32",
        "asset_hashes": {},
        "resume": True,
    }


def test_run_builds_bound_dispatcher_when_not_explicitly_injected(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    options = _options(_write_config(tmp_path), tmp_path / "results")
    ready = lambda backend: {"ok": True, "codes": [], "backend": backend}
    assert runner.preflight(
        options,
        environment_probe=_environment_ok,
        inventory_builder=_inventory_ok,
        inventory_verifier=lambda *_args: None,
        backend_probe=ready,
        gate_evaluator=lambda _config, _spec: _gate_evidence(),
        resource_evaluator=_passing_resource_evaluator,
    )["ok"]
    run_options = Namespace(**{**vars(options), "command": "run"})
    captured = {}
    dispatcher = lambda job: job

    def bind(received_options, *, manifest):
        captured["bound_options"] = received_options
        captured["manifest"] = manifest
        return {"tiny": dispatcher}

    def execute(jobs, *, store, command, dispatchers, resume):
        captured["dispatchers"] = dispatchers
        return [{"job_id": jobs[0]["job_id"], "status": "skipped"}]

    report = runner.run(
        run_options,
        execute_fn=execute,
        dispatcher_builder=bind,
    )

    assert report["ok"] is True
    assert captured["bound_options"] is run_options
    assert captured["dispatchers"] == {"tiny": dispatcher}
    assert captured["manifest"]["asset_hashes"] == {}


def test_runtime_cli_changes_do_not_change_jobs_or_canonical_manifest(tmp_path):
    runner = importlib.import_module("research.kmd2_ablation.runner")
    raw = _valid_tiny_raw()
    raw["backend"] = "qwen"
    config_path = _write_config(tmp_path, raw)
    first = _options(config_path, tmp_path / "first", backend="qwen")
    second = _options(config_path, tmp_path / "second", backend="qwen")
    first.model = tmp_path / "model-a"
    second.model = tmp_path / "model-b"
    first.checkpoint = tmp_path / "checkpoint-a.pt"
    second.checkpoint = tmp_path / "checkpoint-b.pt"
    first.data = tmp_path / "data-a.jsonl"
    second.data = tmp_path / "data-b.jsonl"
    first.student_device = "cuda:0"
    second.student_device = "cuda:7"
    first.model_sha256 = "1" * 64
    second.model_sha256 = "1" * 64
    first.checkpoint_sha256 = second.checkpoint_sha256 = "2" * 64
    first.data_sha256 = second.data_sha256 = "3" * 64

    def inspect(paths, *, expected):
        return {
            name: {
                "path": str(path),
                "kind": "file",
                "size_bytes": 1,
                "sha256": expected[name]["sha256"],
                "tree_manifest": [],
            }
            for name, path in paths.items()
        }

    common = {
        "environment_probe": _environment_ok,
        "inventory_builder": _inventory_ok,
        "inventory_verifier": lambda *_args: None,
        "asset_inspector": inspect,
        "backend_probe": lambda backend: {
            "ok": True,
            "codes": [],
            "backend": backend,
        },
        "gate_evaluator": lambda _config, _spec: _gate_evidence(),
        "resource_evaluator": _passing_resource_evaluator,
    }

    first_report = runner.preflight(first, **common)
    second_report = runner.preflight(second, **common)

    assert first_report["ok"] is second_report["ok"] is True
    assert first_report["jobs"] == second_report["jobs"]
    assert first_report["commands"] != second_report["commands"]
    first_manifest = json.loads((first.out / "manifest.json").read_text("utf-8"))
    second_manifest = json.loads((second.out / "manifest.json").read_text("utf-8"))
    assert first_manifest == second_manifest
    serialized_jobs = json.dumps(first_report["jobs"], sort_keys=True)
    for runtime_value in (
        first.model,
        second.model,
        first.checkpoint,
        second.checkpoint,
        first.data,
        second.data,
        first.student_device,
        second.student_device,
        first.model_sha256,
    ):
        assert str(runtime_value) not in serialized_jobs


@pytest.mark.parametrize(
    ("source", "ok"),
    [
        ("def run_job(job):\n    return job\n", True),
        ("def build_job_dispatcher(runtime):\n    return lambda job: job\n", True),
        ("def helper(job):\n    return job\n", False),
    ],
)
def test_backend_probe_reports_entrypoint_from_injected_source_without_importing(
    source, ok
):
    runner = importlib.import_module("research.kmd2_ablation.runner")

    report = runner.probe_backend_dispatch(
        "tiny", source_loader=lambda module_name: source
    )

    assert report["ok"] is ok
    assert report["backend"] == "tiny"
    assert report["codes"] == (
        [] if ok else ["backend_dispatch_unavailable:tiny"]
    )


def test_qwen_backend_readiness_probe_never_imports_transformers():
    script = """
import sys
from importlib.abc import MetaPathFinder

class RejectTransformers(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] == 'transformers':
            raise AssertionError('preflight imported Transformers')
        return None

sys.meta_path.insert(0, RejectTransformers())
from research.kmd2_ablation.runner import probe_backend_dispatch
report = probe_backend_dispatch('qwen')
assert report['backend'] == 'qwen'
assert report['codes'] == (
    [] if report['ok'] else ['backend_dispatch_unavailable:qwen']
)
assert 'transformers' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
