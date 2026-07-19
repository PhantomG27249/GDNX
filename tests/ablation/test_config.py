from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def minimal_config_dict() -> dict:
    """A complete, runnable scientific configuration used by every test."""
    return {
        "schema_version": "1.0.0",
        "suite_version": "1.0.0",
        "backend": "tiny",
        "qwen": {
            "model_asset": "qwen_model",
            "tokenizer_asset": "qwen_tokenizer",
            "run_mode": "initial_exact_cache",
            "streaming": False,
            "decode": False,
            "packing": False,
            "padding": "none",
            "attention_mask": "all_ones",
        },
        "baseline": "gdn2_native",
        "mechanism": "exact_cache",
        "variant": "top_surprise",
        "task": {
            "name": "mqar",
            "params": {
                "num_pairs": 8,
                "vocab_size": 256,
                "distractor_lengths": [32, 64],
            },
        },
        "seeds": [11, 29],
        "budget": {"tokens": 65_536, "updates": 100},
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1.0e-3,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
        "schedule": {"name": "cosine", "warmup_updates": 10},
        "model": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "state_key_dim": 64,
            "state_value_dim": 64,
            "ffn_dim": 768,
            "ffn_match_lower": 704,
            "ffn_match_upper": 832,
        },
        "lengths": {
            "curriculum": [128, 256],
            "extrapolation": [512, 1024],
        },
        "evaluation": {
            "primary_metric": "exact_match",
            "direction": "maximize",
        },
        "thresholds": {
            "min_useful_addition": 0.02,
            "min_reliance": 0.10,
            "equivalence_tolerance": 0.01,
            "harm_threshold": 0.03,
            "synergy_threshold": 0.02,
        },
        "promotion": {
            "min_gate_mean": 0.005,
            "min_gate_max": 0.02,
            "min_persistent_hit_rate": 0.25,
            "min_conditional_read_accuracy": 0.50,
            "min_shuffled_cache_dependence": 0.05,
            "min_adjacent_capacity_lcb": 0.05,
        },
        "protected_metrics": [
            {"name": "validation_loss", "max_regression": 0.01},
            {"name": "perplexity", "max_regression": 0.02},
        ],
        "device_preferences": ["cuda", "cpu"],
        "dtype_preferences": ["bfloat16", "float32"],
        "required_stage": "mechanism_screen",
        "cache": {
            "width": 32,
            "block_size": 64,
            "score": "exact_outer",
            "read": "unit_l2",
            "read_init": "gamma_one_sink_zero_amplitude_zero",
            "eps_cache": 1.0e-6,
            "coordinate_frame": "rotated_recurrence",
            "pre_rotation_diagnostic": False,
            "storage_dtype": "bf16",
            "compute_dtype": "fp32",
            "inclusive": True,
            "tie_policy": "score_desc_position_desc",
            "lr_cache": 2.0e-3,
            "weight_decay_cache": 0.0,
        },
        "runtime": {
            "output_path": "runs/kmd2-ablation/minimal",
            "device_ordinal": 0,
        },
    }


def test_every_committed_ablation_config_parses_as_complete_schema() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "research" / "kmd2_ablation" / "configs"
    paths = sorted(config_dir.glob("*.json"))
    assert paths, "no committed ablation configs found"
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = _config_module().ExperimentConfig.from_dict(raw)
        assert config.semantic_dict()


def _config_module():
    try:
        module = importlib.import_module("research.kmd2_ablation.config")
    except ModuleNotFoundError:
        pytest.fail("ExperimentConfig is missing")
    if not hasattr(module, "ExperimentConfig"):
        pytest.fail("ExperimentConfig is missing")
    return module


def _build(raw: dict | None = None):
    return _config_module().ExperimentConfig.from_dict(
        minimal_config_dict() if raw is None else raw
    )


def test_maximum_hybrid_configs_fail_closed():
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256

    for arm_id, topology, phase_topology, read_topology, read_paths, variant in (
        ("gdn2-mimo-r4-braid-shared-hola-w64", "shared", "shared", "shared_state_outputs", 4, "braid_shared_hola_w64"),
        ("gdn2-mimo-r4-braid-four-state-hola-w64", "mimo_rank_contributions", "per_rank", "full_cross_state", 16, "braid_four_state_hola_w64"),
    ):
        raw = minimal_config_dict()
        raw.update(mechanism="maximum_hybrid", variant=variant)
        raw["model"].update(num_heads=16, state_key_dim=128, state_value_dim=128)
        raw["seeds"] = [11, 29, 47]
        raw["budget"] = {"updates": 1536, "tokens": 6291456}
        raw["lengths"] = {"curriculum": [512, 2048, 4096, 8192], "extrapolation": [16384, 32768]}
        raw["task"]["params"].update(
            target_layers=list(TARGET_LAYERS), mimo_rank=4, output_width=4,
            gate_mode="channelwise", timescales=([1, 16, 64, 256]
                                                  if topology == "mimo_rank_contributions"
                                                  else [64, 512, 4096, 32768]),
            state_topology=topology, cache_scope="per_head_shared_across_ranks",
            update_paths=4, convolution_on=True,
            rotation_mode="mamba3_complex_input_dependent_cumulative",
            phase_topology=phase_topology,
            input_rank=4, output_rank=4, read_topology=read_topology,
            read_paths=read_paths, state_input_mode="trapezoid", lookahead=(topology == "shared"),
            qk_contract=("affine_diagonal_plus_additive_identity_init" if topology == "shared" else "gdn_unit_directional_affine_postrenorm_v1"),
        )
        raw["cache"].update(
            width=64, block_size=256, score="exact_outer", read="rmsnorm",
            storage_dtype="bf16", read_init="hola_gate_logit_v2_minus4",
        )
        raw["architecture"] = {
            "arm_id": arm_id, "registry_sha256": registry_sha256(),
            "model_tree_sha256": "a" * 64, "ordered_examples_sha256": "b" * 64,
            "pretraining_checkpoint_sha256": "c" * 64,
        }
        assert _build(raw).architecture.arm_id == arm_id
        for path, value in (
            (("task", "params", "convolution_on"), False),
            (("task", "params", "mimo_rank"), 2),
            (("task", "params", "timescales"), [64, 512]),
            (("task", "params", "rotation_mode"), "current"),
            (("task", "params", "phase_topology"), "per_rank" if phase_topology == "shared" else "shared"),
            (("task", "params", "input_rank"), 2),
            (("task", "params", "output_rank"), 2),
            (("task", "params", "read_topology"), "full_cross_state" if read_topology == "shared_state_outputs" else "shared_state_outputs"),
            (("task", "params", "read_paths"), 16 if read_paths == 4 else 4),
            (("task", "params", "state_input_mode"), "native"),
            (("task", "params", "lookahead"), topology != "shared"),
            (("task", "params", "qk_contract"), "none"),
            (("task", "params", "cache_scope"), "per_state"),
            (("task", "params", "update_paths"), 16),
        ):
            drifted = copy.deepcopy(raw)
            target = drifted
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with pytest.raises(ValueError, match="architecture arm does not match"):
                _build(drifted)


def test_optional_architecture_identity_is_exact_and_rejects_drift():
    from research.kmd2_ablation.architecture import registry_sha256
    raw = minimal_config_dict()
    raw["architecture"] = {
        "arm_id": "cache-surprise-w64", "registry_sha256": registry_sha256(),
        "model_tree_sha256": "a" * 64, "ordered_examples_sha256": "b" * 64,
        "pretraining_checkpoint_sha256": "c" * 64,
    }
    raw["cache"]["width"] = 64
    raw["cache"]["block_size"] = 256
    raw["seeds"] = [11, 29, 47]
    raw["budget"] = {"updates": 1536, "tokens": 6291456}
    raw["lengths"] = {"curriculum": [512, 2048, 4096, 8192], "extrapolation": [16384, 32768]}
    raw["model"]["num_heads"] = 16
    raw["model"]["state_key_dim"] = 128
    raw["model"]["state_value_dim"] = 128
    config = _build(raw)
    assert config.architecture.arm_id == "cache-surprise-w64"
    raw["architecture"]["registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry_sha256"):
        _build(raw)


def test_architecture_identity_rejects_legacy_mechanism_variant_mismatch():
    from research.kmd2_ablation.architecture import registry_sha256
    raw = minimal_config_dict()
    raw["architecture"] = {
        "arm_id": "mimo-r2", "registry_sha256": registry_sha256(),
        "model_tree_sha256": "a" * 64, "ordered_examples_sha256": "b" * 64,
        "pretraining_checkpoint_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="active legacy cache"):
        _build(raw)


def test_architecture_identity_checks_rank_gate_output_and_model_values():
    from research.kmd2_ablation.architecture import registry_sha256
    raw = minimal_config_dict()
    raw.update(mechanism="true_mimo", variant="true_mimo_sweep")
    raw["model"].update(num_heads=16, state_key_dim=128, state_value_dim=128)
    raw["cache"].update(width=64, block_size=256)
    raw["seeds"] = [11, 29, 47]
    raw["budget"] = {"updates": 1536, "tokens": 6291456}
    raw["lengths"] = {"curriculum": [512, 2048, 4096, 8192], "extrapolation": [16384, 32768]}
    raw["task"]["params"].update(mimo_rank=2, output_width=1, gate_mode="scalar")
    raw["architecture"] = {
        "arm_id": "mimo-r2", "registry_sha256": registry_sha256(),
        "model_tree_sha256": "a" * 64, "ordered_examples_sha256": "b" * 64,
        "pretraining_checkpoint_sha256": "c" * 64,
    }
    assert _build(raw).architecture.arm_id == "mimo-r2"
    for container, field, value in (
        ("task", "mimo_rank", 4), ("task", "output_width", 4),
        ("task", "gate_mode", "channelwise"), ("model", "state_key_dim", 64),
    ):
        drifted = copy.deepcopy(raw)
        target = drifted["task"]["params"] if container == "task" else drifted["model"]
        target[field] = value
        with pytest.raises(ValueError, match="architecture arm does not match"):
            _build(drifted)


@pytest.mark.parametrize("path,value,code", [
    (("seeds",), [11, 29], "architecture.seeds_mismatch"),
    (("budget", "updates"), 1535, "architecture.updates_mismatch"),
    (("budget", "tokens"), 6291455, "architecture.tokens_mismatch"),
    (("lengths", "curriculum"), [512], "architecture.curriculum_lengths_mismatch"),
    (("lengths", "extrapolation"), [16384], "architecture.extrapolation_lengths_mismatch"),
    (("task", "params", "target_layers"), [0], "architecture.target_layers_mismatch"),
])
def test_architecture_identity_rejects_frozen_campaign_drift(path, value, code):
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    raw = minimal_config_dict()
    raw.update(mechanism="true_mimo", variant="true_mimo_sweep")
    raw["model"].update(num_heads=16, state_key_dim=128, state_value_dim=128)
    raw["cache"].update(width=64, block_size=256)
    raw["seeds"] = [11, 29, 47]
    raw["budget"] = {"updates": 1536, "tokens": 6291456}
    raw["lengths"] = {"curriculum": [512, 2048, 4096, 8192], "extrapolation": [16384, 32768]}
    raw["task"]["params"].update(mimo_rank=2, output_width=1, gate_mode="scalar", target_layers=list(TARGET_LAYERS))
    raw["architecture"] = {"arm_id": "mimo-r2", "registry_sha256": registry_sha256(), "model_tree_sha256": "a"*64, "ordered_examples_sha256": "b"*64, "pretraining_checkpoint_sha256": "c"*64}
    target = raw
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=code):
        _build(raw)


@pytest.mark.parametrize("arm_id,mechanism,variant", [
    ("mimo-r2", "exact_cache", "top_surprise"),
    ("gdn2-channel-r1", "exact_cache", "top_surprise"),
])
def test_noncache_architecture_rejects_active_legacy_cache(arm_id, mechanism, variant):
    from research.kmd2_ablation.architecture import registry_sha256
    raw = minimal_config_dict()
    raw.update(mechanism=mechanism, variant=variant)
    raw["model"].update(num_heads=16, state_key_dim=128, state_value_dim=128)
    raw["architecture"] = {
        "arm_id": arm_id, "registry_sha256": registry_sha256(),
        "model_tree_sha256": "a" * 64, "ordered_examples_sha256": "b" * 64,
        "pretraining_checkpoint_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="active legacy cache"):
        _build(raw)


def _changed(path: str, value) -> dict:
    raw = copy.deepcopy(minimal_config_dict())
    target = raw
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    return raw


def test_complete_schema_builds_frozen_nested_configuration():
    raw = minimal_config_dict()
    config = _build(raw)
    protected_metric = _config_module().ProtectedMetric

    assert config.schema_version == "1.0.0"
    assert config.suite_version == "1.0.0"
    assert config.backend == "tiny"
    assert config.qwen.run_mode == "initial_exact_cache"
    assert config.baseline == "gdn2_native"
    assert (config.mechanism, config.variant) == ("exact_cache", "top_surprise")
    assert config.task.name == "mqar"
    assert config.task.params["distractor_lengths"] == (32, 64)
    assert config.seeds == (11, 29)
    assert (config.budget.tokens, config.budget.updates) == (65_536, 100)
    assert config.optimizer.betas == (0.9, 0.95)
    assert config.schedule.warmup_updates == 10
    assert config.model.state_key_dim == 64
    assert config.model.state_value_dim == 64
    assert config.lengths.curriculum == (128, 256)
    assert config.lengths.extrapolation == (512, 1024)
    assert config.evaluation.primary_metric == "exact_match"
    assert config.evaluation.direction == "maximize"
    assert config.protected_metrics == (
        protected_metric(name="validation_loss", max_regression=0.01),
        protected_metric(name="perplexity", max_regression=0.02),
    )
    assert config.device_preferences == ("cuda", "cpu")
    assert config.dtype_preferences == ("bfloat16", "float32")
    assert config.required_stage == "mechanism_screen"
    assert config.runtime.output_path == "runs/kmd2-ablation/minimal"
    assert config.runtime.device_ordinal == 0

    raw["task"]["params"]["distractor_lengths"].append(999)
    assert config.task.params["distractor_lengths"] == (32, 64)
    with pytest.raises(TypeError):
        config.task.params["num_pairs"] = 9
    with pytest.raises(FrozenInstanceError):
        config.runtime.device_ordinal = 1


def test_task_params_accept_and_freeze_recursive_json_values():
    raw = minimal_config_dict()
    raw["task"]["params"] = {
        "null": None,
        "boolean": True,
        "integer": 7,
        "finite_float": 1.25,
        "string": "value",
        "list": [1, {"nested": (False, None)}],
        "tuple": ("item", 2),
    }

    config = _build(raw)

    assert config.task.params["list"] == (1, {"nested": (False, None)})
    assert config.task.params["tuple"] == ("item", 2)
    assert json.loads(config.canonical_json)["task"]["params"] == {
        "boolean": True,
        "finite_float": 1.25,
        "integer": 7,
        "list": [1, {"nested": [False, None]}],
        "null": None,
        "string": "value",
        "tuple": ["item", 2],
    }


@pytest.mark.parametrize(
    "params",
    [
        {1: "non-string key"},
        {"nested": {2: "non-string nested key"}},
        {"unsupported": {1, 2}},
        {"unsupported": b"bytes"},
        {"unsupported": object()},
        {"non_finite": math.nan},
        {"nested": [math.inf]},
    ],
    ids=[
        "non-string-key",
        "nested-non-string-key",
        "set",
        "bytes",
        "object",
        "nan",
        "infinity",
    ],
)
def test_task_params_reject_non_json_values_during_construction(params):
    raw = minimal_config_dict()
    raw["task"]["params"] = params

    with pytest.raises((TypeError, ValueError), match=r"task\.params"):
        _build(raw)


@pytest.mark.parametrize("container_type", ["mapping", "list"])
def test_task_config_direct_construction_rejects_json_cycles(container_type):
    if container_type == "mapping":
        params = {}
        params["self"] = params
    else:
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        params = {"value": cyclic_list}

    with pytest.raises(ValueError, match=r"task\.params.*cycle"):
        _config_module().TaskConfig(name="mqar", params=params)


def test_task_config_direct_construction_enforces_json_depth_limit():
    def nested_params(depth):
        value = "leaf"
        for _ in range(depth):
            value = [value]
        return {"value": value}

    assert _config_module().TaskConfig(
        name="mqar", params=nested_params(64)
    ).params
    with pytest.raises(ValueError, match=r"task\.params.*depth"):
        _config_module().TaskConfig(name="mqar", params=nested_params(65))


def test_task_config_direct_construction_detaches_nested_json_values():
    module = _config_module()
    source = [1]
    task = module.TaskConfig(name="mqar", params={"x": source})
    config = replace(_build(), task=task)
    experiment_id = config.experiment_id

    source.append(2)

    assert config.task.params["x"] == (1,)
    assert config.experiment_id == experiment_id
    with pytest.raises(TypeError):
        config.task.params["x"] = (9,)


def test_experiment_config_direct_construction_detaches_tuple_inputs():
    seeds = [11, 29]
    device_preferences = ["cuda", "cpu"]
    dtype_preferences = ["bfloat16", "float32"]
    config = replace(
        _build(),
        seeds=seeds,
        device_preferences=device_preferences,
        dtype_preferences=dtype_preferences,
    )
    experiment_id = config.experiment_id

    seeds.append(31)
    device_preferences.reverse()
    dtype_preferences.clear()

    assert config.seeds == (11, 29)
    assert config.device_preferences == ("cuda", "cpu")
    assert config.dtype_preferences == ("bfloat16", "float32")
    assert config.experiment_id == experiment_id


@pytest.mark.parametrize(
    ("field", "value", "record_name"),
    [
        ("task", {"name": "mqar", "params": {}}, "TaskConfig"),
        (
            "protected_metrics",
            ({"name": "validation_loss", "max_regression": 0.01},),
            "ProtectedMetric",
        ),
    ],
)
def test_experiment_config_direct_construction_rejects_non_record_values(
    field, value, record_name
):
    with pytest.raises(TypeError, match=rf"{field}.*{record_name}"):
        replace(_build(), **{field: value})


@pytest.mark.parametrize(
    ("field", "integer_value", "float_value"),
    [
        ("optimizer.learning_rate", 1, 1.0),
        ("optimizer.eps", 1, 1.0),
        ("optimizer.weight_decay", 0, 0.0),
        ("thresholds.min_useful_addition", 1, 1.0),
        ("thresholds.min_reliance", 1, 1.0),
        ("thresholds.equivalence_tolerance", 0, 0.0),
        ("thresholds.harm_threshold", 1, 1.0),
        ("thresholds.synergy_threshold", 0, 0.0),
        ("promotion.min_gate_mean", 0, 0.0),
        ("promotion.min_gate_max", 0, 0.0),
        ("promotion.min_persistent_hit_rate", 0, 0.0),
        ("promotion.min_conditional_read_accuracy", 0, 0.0),
        ("promotion.min_shuffled_cache_dependence", 0, 0.0),
        ("promotion.min_adjacent_capacity_lcb", 0, 0.0),
        ("cache.eps_cache", 1, 1.0),
        ("cache.lr_cache", 1, 1.0),
        ("cache.weight_decay_cache", 0, 0.0),
    ],
)
def test_declared_real_fields_canonicalize_integer_and_float_spellings(
    field, integer_value, float_value
):
    integer_config = _build(_changed(field, integer_value))
    float_config = _build(_changed(field, float_value))
    stored_value = integer_config
    for field_name in field.split("."):
        stored_value = getattr(stored_value, field_name)

    assert type(stored_value) is float
    assert integer_config.canonical_json == float_config.canonical_json
    assert integer_config.experiment_id == float_config.experiment_id


def test_optimizer_betas_canonicalize_each_integer_and_float_spelling():
    integer = minimal_config_dict()
    integer["optimizer"]["betas"] = [0, 0]
    floating = copy.deepcopy(integer)
    floating["optimizer"]["betas"] = [0.0, 0.0]

    integer_config = _build(integer)
    float_config = _build(floating)

    assert all(type(beta) is float for beta in integer_config.optimizer.betas)
    assert integer_config.canonical_json == float_config.canonical_json
    assert integer_config.experiment_id == float_config.experiment_id


def test_protected_metric_real_field_canonicalizes_numeric_spellings():
    integer = minimal_config_dict()
    integer["protected_metrics"][0]["max_regression"] = 0
    floating = copy.deepcopy(integer)
    floating["protected_metrics"][0]["max_regression"] = 0.0

    integer_config = _build(integer)
    float_config = _build(floating)

    assert type(integer_config.protected_metrics[0].max_regression) is float
    assert integer_config.canonical_json == float_config.canonical_json
    assert integer_config.experiment_id == float_config.experiment_id


@pytest.mark.parametrize(
    "field",
    [
        "optimizer.weight_decay",
        "thresholds.equivalence_tolerance",
        "thresholds.synergy_threshold",
        "promotion.min_gate_mean",
        "promotion.min_gate_max",
        "promotion.min_persistent_hit_rate",
        "promotion.min_conditional_read_accuracy",
        "promotion.min_shuffled_cache_dependence",
        "promotion.min_adjacent_capacity_lcb",
        "cache.weight_decay_cache",
    ],
)
def test_declared_real_fields_canonicalize_signed_zero(field):
    negative_config = _build(_changed(field, -0.0))
    positive_config = _build(_changed(field, 0.0))
    stored_value = negative_config
    for field_name in field.split("."):
        stored_value = getattr(stored_value, field_name)

    assert math.copysign(1.0, stored_value) == 1.0
    assert negative_config.canonical_json == positive_config.canonical_json
    assert negative_config.experiment_id == positive_config.experiment_id


def test_optimizer_betas_and_protected_metrics_canonicalize_signed_zero():
    negative = minimal_config_dict()
    negative["optimizer"]["betas"][0] = -0.0
    negative["protected_metrics"][0]["max_regression"] = -0.0
    positive = copy.deepcopy(negative)
    positive["optimizer"]["betas"][0] = 0.0
    positive["protected_metrics"][0]["max_regression"] = 0.0

    negative_config = _build(negative)
    positive_config = _build(positive)

    assert math.copysign(1.0, negative_config.optimizer.betas[0]) == 1.0
    assert (
        math.copysign(
            1.0, negative_config.protected_metrics[0].max_regression
        )
        == 1.0
    )
    assert negative_config.canonical_json == positive_config.canonical_json
    assert negative_config.experiment_id == positive_config.experiment_id


def test_canonical_json_and_experiment_id_are_stable_sha256():
    config = _build()
    canonical = json.dumps(
        config.semantic_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    assert config.canonical_json == canonical
    assert config.experiment_id == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert len(config.experiment_id) == 64


def test_json_key_order_and_explicit_runtime_fields_do_not_change_identity():
    raw = minimal_config_dict()
    reordered = dict(reversed(list(raw.items())))
    reordered["task"] = {
        "params": dict(reversed(list(raw["task"]["params"].items()))),
        "name": raw["task"]["name"],
    }
    reordered["runtime"] = {
        "output_path": "D:/different/operator/path",
        "device_ordinal": 7,
    }

    assert _build(raw).experiment_id == _build(reordered).experiment_id
    semantic = _build(reordered).semantic_dict()
    assert "runtime" not in semantic


@pytest.mark.parametrize("backend", ["tiny", "qwen"])
def test_public_backend_names_are_accepted_and_preserved(backend):
    config = _build(_changed("backend", backend))

    assert config.backend == backend
    assert config.semantic_dict()["backend"] == backend
    assert json.loads(config.canonical_json)["backend"] == backend


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("torch_reference", "tiny"), ("qwen_native", "qwen")],
)
def test_backend_aliases_share_canonical_identity(alias, canonical):
    alias_config = _build(_changed("backend", alias))
    canonical_config = _build(_changed("backend", canonical))

    assert alias_config.backend == canonical
    assert alias_config.semantic_dict()["backend"] == canonical
    assert alias_config.canonical_json == canonical_config.canonical_json
    assert alias_config.experiment_id == canonical_config.experiment_id


@pytest.mark.parametrize("run_mode", ["reliance", "heal", "initial_exact_cache"])
def test_qwen_run_modes_are_accepted(run_mode):
    raw = _changed("qwen.run_mode", run_mode)
    raw["backend"] = "qwen"
    config = _build(raw)

    assert config.qwen.run_mode == run_mode
    assert config.semantic_dict()["qwen"]["run_mode"] == run_mode


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", [11, 30]),
        ("budget.tokens", 65_537),
        ("cache.width", 31),
        ("thresholds.synergy_threshold", 0.025),
        ("task.params", {"num_pairs": 9, "vocab_size": 256, "distractor_lengths": [32, 64]}),
    ],
)
def test_scientific_fields_change_experiment_identity(field, value):
    assert _build().experiment_id != _build(_changed(field, value)).experiment_id


def test_promotion_thresholds_have_six_separate_defaults_and_fields():
    cls = _config_module().PromotionThresholds
    defaults = cls()

    assert defaults.min_gate_mean == 0.005
    assert defaults.min_gate_max == 0.02
    assert defaults.min_persistent_hit_rate == 0.25
    assert defaults.min_conditional_read_accuracy == 0.50
    assert defaults.min_shuffled_cache_dependence == 0.05
    assert defaults.min_adjacent_capacity_lcb == 0.05
    assert len(defaults.__dataclass_fields__) == 6

    raw = minimal_config_dict()
    raw.pop("promotion")
    assert _build(raw).promotion == defaults


def test_promotion_canonical_json_uses_only_approved_public_field_names():
    config = _build()
    promotion = config.semantic_dict()["promotion"]

    assert set(promotion) == {
        "min_gate_mean",
        "min_gate_max",
        "min_persistent_hit_rate",
        "min_conditional_read_accuracy",
        "min_shuffled_cache_dependence",
        "min_adjacent_capacity_lcb",
    }
    assert "min_conditional_read_rate" not in promotion
    assert "min_shuffled_dependence" not in promotion
    assert config.promotion.min_conditional_read_rate == 0.50
    assert config.promotion.min_shuffled_dependence == 0.05

    legacy = minimal_config_dict()
    legacy["promotion"]["min_conditional_read_rate"] = legacy["promotion"].pop(
        "min_conditional_read_accuracy"
    )
    with pytest.raises(ValueError, match="min_conditional_read_rate"):
        _build(legacy)


def test_cache_config_exposes_exact_cache_contract():
    cache = _build().cache

    assert cache.width == 32
    assert cache.block_size == 64
    assert cache.score == "exact_outer"
    assert cache.read == "unit_l2"
    assert cache.read_init == "gamma_one_sink_zero_amplitude_zero"
    assert cache.eps_cache == 1.0e-6
    assert cache.coordinate_frame == "rotated_recurrence"
    assert cache.pre_rotation_diagnostic is False
    assert cache.storage_dtype == "bf16"
    assert cache.compute_dtype == "fp32"
    assert cache.inclusive is True
    assert cache.tie_policy == "score_desc_position_desc"
    assert cache.lr_cache == 2.0e-3
    assert cache.weight_decay_cache == 0.0


def test_cache_config_defaults_use_canonical_policy_names():
    cache = _config_module().CacheConfig()

    assert cache.score == "exact_outer"
    assert cache.read == "unit_l2"
    assert cache.read_init == "gamma_one_sink_zero_amplitude_zero"
    assert cache.storage_dtype == "bf16"
    assert cache.compute_dtype == "fp32"


@pytest.mark.parametrize(
    "score",
    [
        "exact_outer",
        "coupled_paper",
        "residual_only",
        "write_value",
        "recency",
        "reservoir",
        "future_query_oracle",
    ],
)
def test_cache_accepts_every_approved_score_policy(score):
    assert _build(_changed("cache.score", score)).cache.score == score


@pytest.mark.parametrize("read", ["unit_l2", "fixed_temperature", "rmsnorm"])
def test_cache_accepts_every_approved_read_policy(read):
    assert _build(_changed("cache.read", read)).cache.read == read


def test_cache_accepts_canonical_read_initialization():
    read_init = "gamma_one_sink_zero_amplitude_zero"

    assert _build(_changed("cache.read_init", read_init)).cache.read_init == read_init


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cache.storage_dtype", "fp32"),
        ("cache.storage_dtype", "bf16"),
        ("cache.compute_dtype", "fp32"),
    ],
)
def test_cache_accepts_canonical_precision_names(field, value):
    path = field.removeprefix("cache.")

    assert getattr(_build(_changed(field, value)).cache, path) == value


@pytest.mark.parametrize(
    ("field", "legacy", "canonical"),
    [
        ("cache.score", "surprise_l2", "exact_outer"),
        ("cache.read", "softmax", "fixed_temperature"),
        ("cache.read_init", "zero", "gamma_one_sink_zero_amplitude_zero"),
        ("cache.storage_dtype", "float32", "fp32"),
        ("cache.storage_dtype", "bfloat16", "bf16"),
        ("cache.compute_dtype", "float32", "fp32"),
    ],
)
def test_cache_legacy_aliases_normalize_to_canonical_semantics(
    field, legacy, canonical
):
    config = _build(_changed(field, legacy))
    cache_field = field.removeprefix("cache.")

    assert getattr(config.cache, cache_field) == canonical
    assert config.semantic_dict()["cache"][cache_field] == canonical


@pytest.mark.parametrize(
    "field",
    [
        "cache.score",
        "cache.read",
        "cache.read_init",
        "cache.storage_dtype",
        "cache.compute_dtype",
    ],
)
def test_cache_policy_type_errors_retain_field_context(field):
    with pytest.raises(TypeError, match=field):
        _build(_changed(field, ["not", "a", "string"]))


@pytest.mark.parametrize("storage_dtype", ["fp32", "bf16"])
def test_cache_storage_dtype_allows_only_supported_precisions(storage_dtype):
    assert _build(_changed("cache.storage_dtype", storage_dtype)).cache.storage_dtype == storage_dtype


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cache.storage_dtype", "float16", "storage_dtype"),
        ("cache.compute_dtype", "bfloat16", "compute_dtype"),
        ("cache.inclusive", False, "inclusive"),
        ("cache.tie_policy", "position_asc", "tie_policy"),
        ("cache.weight_decay_cache", 0.01, "weight_decay_cache"),
        ("cache.score", "not_a_score", "score"),
        ("cache.read", "not_a_read", "read"),
        ("cache.read_init", "not_an_init", "read_init"),
        ("cache.eps_cache", 0.0, "eps_cache"),
        ("cache.lr_cache", 0.0, "lr_cache"),
    ],
)
def test_cache_rejects_unsupported_precision_policy_and_optimizer_values(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        _build(_changed(field, value))


def test_pre_rotation_coordinate_is_diagnostic_only_and_explicitly_gated():
    raw = _changed("cache.coordinate_frame", "pre_rotation")
    with pytest.raises(ValueError, match="pre_rotation_diagnostic"):
        _build(raw)

    raw["cache"]["pre_rotation_diagnostic"] = True
    assert _build(raw).cache.coordinate_frame == "pre_rotation"


@pytest.mark.parametrize(
    ("mechanism", "variant", "width", "valid"),
    [
        ("current_block_only", "chunk_only", 0, True),
        ("exact_cache", "top_surprise", 0, False),
        ("current_block_only", "chunk_only", 1, False),
        ("exact_cache", "chunk_only", 0, False),
    ],
)
def test_zero_width_is_reserved_for_current_block_chunk_only_control(
    mechanism, variant, width, valid
):
    raw = minimal_config_dict()
    raw.update(mechanism=mechanism, variant=variant)
    raw["cache"]["width"] = width
    if valid:
        assert _build(raw).cache.width == 0
    else:
        with pytest.raises(ValueError, match="width|chunk_only|current_block_only"):
            _build(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lengths.curriculum", [64], "two processing blocks"),
        ("cache.width", 256, "eviction"),
    ],
)
def test_top_surprise_requires_two_blocks_and_enough_candidates_for_eviction(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        _build(_changed(field, value))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("thresholds.min_useful_addition", 0.0, "min_useful_addition"),
        ("thresholds.min_reliance", 0.01, "min_reliance"),
        ("thresholds.equivalence_tolerance", -0.01, "equivalence_tolerance"),
        ("thresholds.harm_threshold", 0.01, "harm_threshold"),
        ("thresholds.synergy_threshold", -0.01, "synergy_threshold"),
        ("promotion.min_gate_mean", 1.1, "min_gate_mean"),
        ("promotion.min_persistent_hit_rate", -0.1, "min_persistent_hit_rate"),
    ],
)
def test_threshold_relationships_probabilities_and_nonnegative_values(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        _build(_changed(field, value))


def test_protected_metrics_have_frozen_per_metric_limits_in_canonical_semantics():
    module = _config_module()
    raw = minimal_config_dict()
    raw["protected_metrics"] = [
        {"name": "validation_loss", "max_regression": 0.01},
        {"name": "perplexity", "max_regression": 0.02},
    ]

    config = _build(raw)

    assert config.protected_metrics == (
        module.ProtectedMetric(name="validation_loss", max_regression=0.01),
        module.ProtectedMetric(name="perplexity", max_regression=0.02),
    )
    assert config.semantic_dict()["protected_metrics"] == [
        {"name": "validation_loss", "max_regression": 0.01},
        {"name": "perplexity", "max_regression": 0.02},
    ]
    raw["protected_metrics"][0]["max_regression"] = 0.99
    assert config.protected_metrics[0].max_regression == 0.01
    with pytest.raises(FrozenInstanceError):
        config.protected_metrics[0].max_regression = 0.99


@pytest.mark.parametrize(
    ("protected_metrics", "message"),
    [
        ([], "must not be empty"),
        (["validation_loss"], "must be a mapping"),
        ([{"name": "validation_loss"}], "max_regression"),
        ([{"max_regression": 0.01}], "name"),
        ([{"name": "", "max_regression": 0.01}], "name"),
        ([{"name": "validation_loss", "max_regression": -0.01}], "max_regression"),
        ([{"name": "validation_loss", "max_regression": math.nan}], "finite"),
        ([{"name": "validation_loss", "max_regression": True}], "finite number"),
        (
            [{"name": "validation_loss", "max_regression": 0.01, "global": 0.1}],
            "global",
        ),
        (
            [
                {"name": "validation_loss", "max_regression": 0.01},
                {"name": "validation_loss", "max_regression": 0.02},
            ],
            "duplicate",
        ),
    ],
)
def test_protected_metrics_reject_ambiguous_or_invalid_limits(
    protected_metrics, message
):
    raw = minimal_config_dict()
    raw["protected_metrics"] = protected_metrics

    with pytest.raises((TypeError, ValueError), match=message):
        _build(raw)


def test_global_protected_regression_threshold_is_rejected_as_ambiguous():
    raw = minimal_config_dict()
    raw["thresholds"]["max_protected_regression"] = 0.01

    with pytest.raises(ValueError, match="max_protected_regression"):
        _build(raw)


def test_per_metric_regression_limit_changes_experiment_identity():
    changed = minimal_config_dict()
    changed["protected_metrics"][0]["max_regression"] = 0.011

    assert _build().experiment_id != _build(changed).experiment_id


def test_every_declared_scientific_field_is_behaviorally_immutable_and_canonicalized():
    raw = minimal_config_dict()
    config = _build(raw)
    expected_semantics = copy.deepcopy(raw)
    expected_semantics.pop("runtime")

    assert config.semantic_dict() == expected_semantics
    experiment_id = config.experiment_id
    exported = config.semantic_dict()
    exported["task"]["params"]["distractor_lengths"].append(999)

    assert config.task.params["distractor_lengths"] == (32, 64)
    assert config.experiment_id == experiment_id
    with pytest.raises(TypeError):
        config.task.params["num_pairs"] = 9
    with pytest.raises(TypeError):
        config.seeds[0] = 7
    with pytest.raises(FrozenInstanceError):
        config.optimizer.learning_rate = 0.5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model.ffn_dim", 770, "ffn_dim"),
        ("model.ffn_match_lower", 706, "ffn_match_lower"),
        ("model.ffn_match_upper", 830, "ffn_match_upper"),
        ("model.ffn_match_lower", 800, "ffn_dim"),
        ("model.ffn_match_upper", 736, "ffn_dim"),
        ("optimizer.learning_rate", math.inf, "learning_rate"),
        ("cache.eps_cache", math.nan, "eps_cache"),
    ],
)
def test_dimensions_and_numeric_settings_are_finite_and_valid(field, value, message):
    with pytest.raises(ValueError, match=message):
        _build(_changed(field, value))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seeds", ["11"], "seeds"),
        ("budget.tokens", "65536", "tokens"),
        ("model.hidden_size", 256.0, "hidden_size"),
        ("cache.width", "32", "width"),
        ("cache.inclusive", 1, "inclusive"),
    ],
)
def test_invalid_scientific_types_are_rejected_without_coercion(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _build(_changed(field, value))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0.0", "schema_version"),
        ("suite_version", "9.0.0", "suite_version"),
        ("backend", "triton", "backend"),
        ("baseline", "unknown", "baseline"),
        ("mechanism", "unknown", "mechanism"),
        ("variant", "unknown", "variant"),
        ("task.name", "unknown", "task.name"),
        ("optimizer.name", "sgd", "optimizer.name"),
        ("schedule.name", "unknown", "schedule.name"),
        ("evaluation.direction", "sideways", "direction"),
        ("required_stage", "unknown", "required_stage"),
        ("device_preferences", ["tpu"], "device_preferences"),
        ("dtype_preferences", ["float16"], "dtype_preferences"),
    ],
)
def test_schema_versions_enums_and_execution_preferences_are_validated(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        _build(_changed(field, value))


@pytest.mark.parametrize("attention_mask", ["none", "all_ones"])
def test_initial_exact_cache_accepts_only_approved_attention_masks(attention_mask):
    config = _build(_changed("qwen.attention_mask", attention_mask))

    assert config.qwen.attention_mask == attention_mask


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qwen.run_mode", "streaming"),
        ("qwen.streaming", True),
        ("qwen.decode", True),
        ("qwen.packing", True),
        ("qwen.padding", "pad_to_longest"),
        ("qwen.attention_mask", "causal_full_sequence"),
        ("qwen.attention_mask", "packed_segments"),
    ],
)
def test_initial_exact_cache_rejects_streaming_decode_packing_padding_and_masks(
    field, value
):
    with pytest.raises(ValueError, match="initial exact-cache|qwen"):
        _build(_changed(field, value))


def test_unknown_or_missing_schema_keys_are_rejected():
    extra = minimal_config_dict()
    extra["typo_field"] = 123
    with pytest.raises(ValueError, match="typo_field"):
        _build(extra)

    missing = minimal_config_dict()
    del missing["model"]["state_key_dim"]
    with pytest.raises(ValueError, match="state_key_dim"):
        _build(missing)


def test_package_imports_config_and_inventory_without_qwen_dependencies():
    import_script = textwrap.dedent(
        """
        import importlib
        import sys
        from importlib.abc import MetaPathFinder

        blocked_dependency_roots = {"transformers", "triton"}

        class RejectOptionalDependencies(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in blocked_dependency_roots:
                    raise AssertionError(
                        f"unexpected optional dependency import: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, RejectOptionalDependencies())

        def import_transformers():
            import transformers

        def import_triton():
            import triton

        def assert_import_is_blocked(importer, dependency, mechanism):
            try:
                importer()
            except AssertionError as exc:
                expected = f"unexpected optional dependency import: {dependency}"
                assert str(exc) == expected
            else:
                raise AssertionError(
                    f"{mechanism} did not block optional dependency: {dependency}"
                )

        assert_import_is_blocked(import_transformers, "transformers", "ordinary import")
        assert_import_is_blocked(import_triton, "triton", "ordinary import")
        assert_import_is_blocked(
            lambda: importlib.import_module("transformers"),
            "transformers",
            "importlib.import_module",
        )
        assert_import_is_blocked(
            lambda: importlib.import_module("triton"),
            "triton",
            "importlib.import_module",
        )

        import research.kmd2_ablation as suite
        public_names = {name for name in vars(suite) if not name.startswith("_")}
        assert public_names == {"SUITE_VERSION"}

        config_module = importlib.import_module("research.kmd2_ablation.config")
        inventory_module = importlib.import_module("research.kmd2_ablation.inventory")

        assert suite.SUITE_VERSION == "1.0.0"
        assert config_module.ExperimentConfig
        assert config_module.CacheConfig
        assert inventory_module.build_inventory
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
