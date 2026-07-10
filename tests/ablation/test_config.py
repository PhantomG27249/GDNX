from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, fields, is_dataclass
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
            "attention_mask": "causal_full_sequence",
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


@pytest.mark.parametrize(
    "backend", ["tiny", "qwen", "torch_reference", "qwen_native"]
)
def test_public_and_legacy_backend_names_are_accepted_and_preserved(backend):
    config = _build(_changed("backend", backend))

    assert config.backend == backend
    assert config.semantic_dict()["backend"] == backend
    assert json.loads(config.canonical_json)["backend"] == backend


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


def test_every_declared_scientific_field_is_frozen_and_canonicalized():
    raw = minimal_config_dict()
    config = _build(raw)
    expected_semantics = copy.deepcopy(raw)
    expected_semantics.pop("runtime")

    assert config.semantic_dict() == expected_semantics

    def assert_frozen_dataclass(instance):
        assert is_dataclass(instance)
        assert instance.__dataclass_params__.frozen is True
        for dataclass_field in fields(instance):
            value = getattr(instance, dataclass_field.name)
            if is_dataclass(value) and not isinstance(value, type):
                assert_frozen_dataclass(value)
            elif isinstance(value, tuple):
                for item in value:
                    if is_dataclass(item) and not isinstance(item, type):
                        assert_frozen_dataclass(item)

    assert_frozen_dataclass(config)
    with pytest.raises(TypeError):
        config.task.params["num_pairs"] = 9
    with pytest.raises(TypeError):
        config.seeds[0] = 7


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qwen.run_mode", "streaming"),
        ("qwen.streaming", True),
        ("qwen.decode", True),
        ("qwen.packing", True),
        ("qwen.padding", "pad_to_longest"),
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
