from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "research/kmd2_ablation/campaigns/qwen08b-maximum-hybrid-v1.json"


def test_campaign_has_exact_eleven_controls() -> None:
    document = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    assert [item["control_id"] for item in document["controls"]] == [
        "gdn2-r1", "gdn2-mimo-r2", "gdn2-mimo-r4",
        "package-a-native-decay", "package-a-braid-no-cache",
        "package-a-recency-w64", "package-a-hola-w64",
        "package-b-recency-w64", "package-b-hola-w64",
        "shared-query-widening", "stock-qwen",
    ]
    assert all(item["features"]["convolution"] is True for item in document["controls"])
    assert document["matched"]["seeds"] == [11, 29, 47]
    assert document["promotion"]["requires_all_seeds"] is True
    assert document["promotion"]["silent_promotion"] is False


def test_materialization_requires_full_asset_identities(tmp_path: Path) -> None:
    from research.kmd2_ablation.runner import materialize_maximum_hybrid_campaign

    assets = {"schema_version": "1.0.0", "assets": {}}
    with pytest.raises(ValueError, match="asset identities"):
        materialize_maximum_hybrid_campaign(CAMPAIGN, assets, tmp_path)


def test_materialized_controls_are_real_experiment_configs(tmp_path: Path) -> None:
    from research.kmd2_ablation.config import ExperimentConfig
    from research.kmd2_ablation.gate_probes import measure_scientific_gates
    from research.kmd2_ablation.qwen_variants import validate_maximum_control_config
    from research.kmd2_ablation.runner import (
        _resolve_variant,
        materialize_maximum_hybrid_campaign,
        validate_scientific_preflight,
    )

    assets = {"schema_version": "1.0.0", "assets": {
        name: {"path": str(tmp_path / name), ("tree_sha256" if name in {"model", "tokenizer"} else "sha256"): "a" * 64}
        for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model")
    }}
    paths = materialize_maximum_hybrid_campaign(CAMPAIGN, assets, tmp_path / "configs", smoke=True)
    parsed = [ExperimentConfig.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    assert len(parsed) == 11
    assert all(config.task.params["maximum_control"] for config in parsed)
    assert all(config.task.params["lambda_spec"] == 0.001 for config in parsed)
    assert all(config.task.params["lambda_gate"] == 0.001 for config in parsed)
    assert all(config.task.params["specialization_updates"] == 8 for config in parsed)
    assert parsed[-1].qwen.run_mode == "reliance"
    assert all(config.task.params["objective"] == "synthetic_only" for config in parsed)
    assert all(
        validate_maximum_control_config(
            json.loads(path.read_text(encoding="utf-8"))
        ) is not None
        for path in paths
    )
    assert all(_resolve_variant(config) is not None for config in parsed)
    for config in (parsed[6], parsed[8]):
        evidence = measure_scientific_gates(config, _resolve_variant(config))
        assert evidence["available"] is True
        # 2026-07-14 "Option B": the HOLA warm gate sigmoid(-4) intentionally
        # breaks exact source parity, so native parity is no longer asserted.
        # Package B's identity contract is conversion determinism, which the
        # gate probe already encodes (identity_passed for four_state).
        assert evidence["probe"]["neutral_repeat_deterministic"] is True
        assert evidence["active_effect_passed"] is True
        assert evidence["probe"]["kind"] == "real_converted_kmd2_hybrid_module"
        assert evidence["probe"]["native_base_class"] == "KMD2NativeAttn"
        assert evidence["probe"]["forward_executed"] is True
        assert evidence["probe"]["scan_executed"] is True
        assert evidence["probe"]["cache_schema"] in {"state", "states"}
        assert evidence["probe"]["hola_admissions"] > 0
        assert evidence["probe"]["finite_connected_gradients"] is True
        assert evidence["probe"]["lane_specialization_gradients"] is True
        assert evidence["probe"]["braid_staging_passed"] is True
        assert "maximum_control_contract_fixture" not in repr(evidence)
        rejected = validate_scientific_preflight(
            config,
            gate_evaluator=lambda *_: {
                "available": True, "identity_passed": True,
                "active_effect_passed": True, "missing_parameters": (),
                "disconnected_parameters": (), "frozen_zero_gates": (),
                "probe": {"kind": "maximum_control_contract_fixture"},
            },
        )
        assert "real_module_gate_required" in rejected["codes"]
    # Package B's identity contract (conversion determinism) must still hold.
    assert measure_scientific_gates(parsed[8], _resolve_variant(parsed[8]))["identity_passed"] is True


def test_maximum_runner_jobs_use_canonical_architecture_and_runtime_pairing(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.config import ExperimentConfig
    from research.kmd2_ablation.qwen_training import (
        _architecture_dispatch_contract,
        derive_three_arm_pairing,
    )
    from research.kmd2_ablation.runner import (
        _expand_jobs,
        _resolve_variant,
        materialize_maximum_hybrid_campaign,
    )

    assets = {"schema_version": "1.0.0", "assets": {
        name: {
            "path": str(tmp_path / name),
            ("tree_sha256" if name in {"model", "tokenizer"} else "sha256"): "a" * 64,
        }
        for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model")
    }}
    paths = materialize_maximum_hybrid_campaign(
        CAMPAIGN, assets, tmp_path / "configs", smoke=True
    )
    expected_architectures = {
        "gdn2-r1": "gdn2-channel-r1",
        "gdn2-mimo-r2": "mimo-r2",
        "gdn2-mimo-r4": "mimo-r4",
        "package-a-native-decay": "gdn2-mimo-r4-braid-shared-hola-w64",
        "package-a-braid-no-cache": "gdn2-mimo-r4-braid-shared-hola-w64",
        "package-a-recency-w64": "gdn2-mimo-r4-braid-shared-hola-w64",
        "package-a-hola-w64": "gdn2-mimo-r4-braid-shared-hola-w64",
        "package-b-recency-w64": "gdn2-mimo-r4-braid-four-state-hola-w64",
        "package-b-hola-w64": "gdn2-mimo-r4-braid-four-state-hola-w64",
        "shared-query-widening": "rout-4",
    }

    all_jobs: list[dict[str, object]] = []
    for path in paths:
        config = ExperimentConfig.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        control_id = config.task.params["maximum_control"]
        jobs = _expand_jobs(
            config,
            _resolve_variant(config).arm_id,
            asset_hashes={"checkpoint": "a" * 64, "data": "a" * 64},
        )
        all_jobs.extend(jobs)
        assert len(jobs) == 3
        assert {job["seed"] for job in jobs} == {11, 29, 47}

        if control_id == "stock-qwen":
            assert {job["arm_id"] for job in jobs} == {"native"}
            assert all("architecture_registry_sha256" not in job for job in jobs)
        else:
            expected_arm_id = expected_architectures[control_id]
            assert {job["arm_id"] for job in jobs} == {expected_arm_id}
            assert all("architecture_registry_sha256" in job for job in jobs)
        for job in jobs:
            _architecture_dispatch_contract(job, job["canonical_config"])
            pairing = derive_three_arm_pairing(
                job,
                example_ids=tuple(config.task.params["example_ids"]),
                pre_replacement_checkpoint_sha256="a" * 64,
                data_sha256="a" * 64,
            )
            assert job["pairing_id"] == pairing.pairing_id

    assert len(all_jobs) == 33


def test_all_eleven_preflight_before_first_run(tmp_path: Path) -> None:
    from research.kmd2_ablation.runner import run_campaign

    events: list[tuple[str, str]] = []
    controls = [{"control_id": str(index)} for index in range(11)]
    run_campaign(
        {"controls": controls, "promotion": {"requires_all_seeds": True}},
        preflight=lambda item: events.append(("preflight", item["control_id"])) or {"ok": True},
        execute=lambda item: events.append(("run", item["control_id"])) or {"ok": True},
    )
    assert [kind for kind, _ in events[:11]] == ["preflight"] * 11
    assert events[11][0] == "run"


def test_package_b_identity_is_conversion_determinism_not_native_output_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.config import ExperimentConfig
    from research.kmd2_ablation.gate_probes import measure_scientific_gates
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.runner import (
        _resolve_variant, materialize_maximum_hybrid_campaign,
        validate_scientific_preflight,
    )

    assets = {"schema_version": "1.0.0", "assets": {
        name: {"path": str(tmp_path / name), ("tree_sha256" if name in {"model", "tokenizer"} else "sha256"): "a" * 64}
        for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model")
    }}
    paths = materialize_maximum_hybrid_campaign(CAMPAIGN, assets, tmp_path / "configs", smoke=True)
    config = ExperimentConfig.from_dict(json.loads(paths[8].read_text(encoding="utf-8")))
    original = QwenFourStateHybrid.from_native.__func__
    def perturbed(cls, native):
        converted = original(cls, native)
        with __import__("torch").no_grad():
            converted.components.out_proj.weight[0, 0].add_(0.25)
        return converted
    monkeypatch.setattr(QwenFourStateHybrid, "from_native", classmethod(perturbed))
    evidence = measure_scientific_gates(config, _resolve_variant(config))
    assert evidence["probe"]["native_parity_passed"] is False
    assert evidence["probe"]["native_parity_expected"] is False
    assert evidence["probe"]["neutral_repeat_deterministic"] is True
    assert evidence["identity_passed"] is True
    # 2026-07-14 "Option B": Package B's identity contract is conversion
    # determinism, not source-output parity, so a parity delta alone must NOT
    # fail the real-module preflight gate (braided horizons, CMS clocks, the
    # 0.5-initialized trapezoid, and the warm HOLA gate all move the output).
    report = validate_scientific_preflight(
        config, gate_evaluator=lambda *_: evidence
    )
    assert "real_module_gate_required" not in report["codes"]
    # Parity remains load-bearing exactly where the probe declares it expected
    # (Package A): a parity failure there must still fail closed.
    doctored = json.loads(json.dumps(evidence))
    doctored["probe"]["native_parity_expected"] = True
    rejected = validate_scientific_preflight(
        config, gate_evaluator=lambda *_: doctored
    )
    assert "real_module_gate_required" in rejected["codes"]


def test_three_seed_promotion_never_silent() -> None:
    from research.kmd2_ablation.runner import promotion_decision

    result = promotion_decision([{"seed": 11, "passed": True}, {"seed": 29, "passed": True}])
    assert result == {"promoted": False, "reason": "missing_required_seeds", "missing_seeds": [47]}
