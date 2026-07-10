from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import random

import pytest

from research.kmd2_ablation.metrics import (
    METRIC_REGISTRY,
    BootstrapInterval,
    FactorialBootstrapResult,
    MetricContribution,
    MetricSample,
    MetricSpec,
    MetricValue,
    NamedInterval,
    OPTION3_REJECTION_CODES,
    Option3BranchResult,
    Option3Decision,
    Option3Evidence,
    Option3Thresholds,
    ProtectedEffect,
    accumulate_metrics,
    affine_contributions,
    binding_mqar_contributions,
    cache_diagnostic_contributions,
    classify_addition,
    classify_reliance,
    classification_contributions,
    drift_contributions,
    freshness_contributions,
    integration_contributions,
    metric_direction,
    paired_bootstrap,
    paired_factorial_bootstrap,
    decide_option3,
    evaluate_option3_branch,
    selector_auprc_contribution,
    structured_contributions,
    trajectory_contributions,
)


def _values(contributions: tuple[MetricContribution, ...]):
    return {
        (value.metric, value.strata): value
        for value in accumulate_metrics(contributions)
    }


def test_metric_accumulators_core_records_registry_and_exact_classification() -> None:
    required_directions = {
        "token_accuracy": 1,
        "episode_exact": 1,
        "chance_adjusted_accuracy": 1,
        "integration_mse": -1,
        "cache_persistent_bytes": -1,
        "cache_block_bytes": -1,
        "latency_ms": -1,
        "throughput_tokens_per_second": 1,
        "peak_vram_bytes": -1,
    }
    assert {name: metric_direction(name) for name in required_directions} == required_directions
    assert set(required_directions).issubset(METRIC_REGISTRY)
    for diagnostic in (
        "cache_attention_entropy",
        "cache_effective_support",
        "cache_sink_mass",
        "cache_output_norm",
        "state_output_norm",
        "cache_retention_rate",
        "cache_eviction_rate",
        "cache_selection_score_mean",
    ):
        assert metric_direction(diagnostic) is None
    with pytest.raises(ValueError, match="direction"):
        MetricSpec("bad", True)  # type: ignore[arg-type]

    contribution = MetricContribution("token_accuracy", 1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        contribution.numerator = 2.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        MetricContribution("token_accuracy", float("nan"), 1.0)
    with pytest.raises(ValueError, match="zero denominator"):
        MetricContribution("token_accuracy", 1.0, 0.0)

    contributions = classification_contributions(
        predictions=[1, 2, 0, 4],
        targets=[1, 0, 0, 4],
        eligible=[True, True, True, True],
        episode_ids=["a", "a", "b", "b"],
        chance_probabilities=[0.25, 0.25, 0.25, 0.25],
    )
    values = _values(contributions)
    assert values[("token_accuracy", ())].numerator == 3.0
    assert values[("token_accuracy", ())].denominator == 4.0
    assert values[("token_accuracy", ())].value == 0.75
    assert values[("episode_exact", ())].value == 0.5
    assert values[("chance_adjusted_accuracy", ())].value == pytest.approx(2.0 / 3.0)
    assert all(value.available for value in values.values())

    doubled = _values(contributions + contributions)
    assert doubled[("token_accuracy", ())].numerator == 6.0
    assert doubled[("token_accuracy", ())].denominator == 8.0
    assert doubled[("token_accuracy", ())].value == 0.75


def test_episode_exact_composes_across_classification_and_mqar_partitions() -> None:
    first = classification_contributions(
        predictions=[1],
        targets=[1],
        eligible=[True],
        episode_ids=["shared"],
        chance_probabilities=[0.25],
    )
    second = classification_contributions(
        predictions=[0],
        targets=[1],
        eligible=[True],
        episode_ids=["shared"],
        chance_probabilities=[0.25],
    )
    classification = _values(first + second)[("episode_exact", ())]
    assert classification.numerator == 0.0
    assert classification.denominator == 1.0
    assert classification.value == 0.0

    common = {
        "task": "mqar",
        "eligible": [True],
        "episode_ids": ["shared"],
        "distance_bins": ["near"],
        "load_bins": ["low"],
        "declared_distance_bins": ("near",),
        "declared_load_bins": ("low",),
    }
    first_binding = binding_mqar_contributions(
        predictions=[1], targets=[1], **common
    )
    second_binding = binding_mqar_contributions(
        predictions=[0], targets=[1], **common
    )
    mqar = _values(first_binding + second_binding)[("mqar_episode_exact", ())]
    assert mqar.numerator == 0.0
    assert mqar.denominator == 1.0
    assert mqar.value == 0.0


def test_metric_accumulators_integration_strata_and_empty_unavailable() -> None:
    contributions = integration_contributions(
        predictions=[1.0, 3.0, 2.0],
        targets=[0.0, 1.0, 4.0],
        eligible=[True, True, True],
        gap_bins=["small", "large", "large"],
        curvature_bins=["low", "low", "high"],
        declared_gap_bins=("small", "large", "empty"),
        declared_curvature_bins=("low", "high", "empty"),
    )
    values = _values(contributions)
    assert values[("integration_mse", ())].value == 3.0
    assert values[("integration_mse", (("gap", "small"),))].value == 1.0
    assert values[("integration_mse", (("gap", "large"),))].value == 4.0
    assert values[("integration_mse", (("curvature", "low"),))].value == 2.5
    empty = values[("integration_mse", (("gap", "empty"),))]
    assert empty.numerator == empty.denominator == 0.0
    assert empty.value is None and empty.available is False
    with pytest.raises(ValueError, match="finite"):
        integration_contributions(
            predictions=[float("inf")],
            targets=[0.0],
            eligible=[True],
            gap_bins=["small"],
            curvature_bins=["low"],
            declared_gap_bins=("small",),
            declared_curvature_bins=("low",),
        )


def test_metric_accumulators_temporal_affine_binding_and_structured_exact() -> None:
    temporal = _values(
        drift_contributions(
            steady_state_errors=[1.0, 3.0],
            adaptation_lags=[2.0, 4.0],
            peak_overshoots=[0.5, 1.5],
            recovery_times=[5.0, 7.0],
        )
        + trajectory_contributions(
            smooth_errors=[1.0, -1.0],
            phase_lags=[0.25, 0.75],
            change_point_errors=[2.0, -2.0],
            change_point_overshoots=[1.0, 3.0],
            recovery_times=[4.0, 8.0],
        )
    )
    expected = {
        "drift_steady_state_error": 2.0,
        "drift_adaptation_lag": 3.0,
        "drift_peak_overshoot": 1.0,
        "drift_recovery_time": 6.0,
        "trajectory_smooth_mse": 1.0,
        "trajectory_phase_lag": 0.5,
        "trajectory_change_point_mse": 4.0,
        "trajectory_change_point_overshoot": 2.0,
        "trajectory_recovery_time": 6.0,
    }
    for metric, value in expected.items():
        assert temporal[(metric, ())].value == value

    affine = _values(
        affine_contributions(
            query_errors=[1.0, -1.0],
            intercept_errors=[2.0, 0.0],
            slope_errors=[1.0, 3.0],
            zero_intercept=[True, False],
        )
    )
    assert affine[("affine_query_mse", ())].value == 1.0
    assert affine[("affine_intercept_mse", ())].value == 2.0
    assert affine[("affine_slope_mse", ())].value == 5.0
    assert affine[("affine_query_mse", (("intercept", "zero"),))].value == 1.0

    binding = _values(
        binding_mqar_contributions(
            task="mqar",
            predictions=[1, 2, 0, 4],
            targets=[1, 0, 0, 4],
            eligible=[True, True, True, True],
            episode_ids=["a", "a", "b", "b"],
            distance_bins=["near", "far", "far", "near"],
            load_bins=["low", "low", "high", "high"],
            declared_distance_bins=("near", "far", "empty"),
            declared_load_bins=("low", "high", "empty"),
        )
    )
    assert binding[("mqar_token_accuracy", ())].value == 0.75
    assert binding[("mqar_episode_exact", ())].value == 0.5
    assert binding[("mqar_token_accuracy", (("distance", "near"),))].value == 1.0
    assert not binding[("mqar_token_accuracy", (("load", "empty"),))].available

    structured = _values(
        structured_contributions(
            correct=[True, False, True, False],
            is_exception=[False, False, True, True],
        )
    )
    assert structured[("structured_rule_accuracy", ())].value == 0.5
    assert structured[("structured_exception_accuracy", ())].value == 0.5


def test_metric_accumulators_freshness_cache_bytes_and_performance_exact() -> None:
    freshness = _values(
        freshness_contributions(
            latest_correct=[True, False, True],
            stale_old_predicted=[False, True, False],
            update_latencies=[1.0, 3.0, 2.0],
            duplicate_occupancies=[2.0, 4.0, 3.0],
            old_attention_masses=[0.2, 0.4, 0.3],
            new_attention_masses=[0.8, 0.6, 0.7],
        )
    )
    assert freshness[("freshness_latest_accuracy", ())].value == pytest.approx(2 / 3)
    assert freshness[("freshness_stale_old_rate", ())].value == pytest.approx(1 / 3)
    assert freshness[("freshness_update_latency", ())].value == 2.0
    assert freshness[("freshness_duplicate_occupancy", ())].value == 3.0
    assert freshness[("freshness_old_attention_mass", ())].value == pytest.approx(0.3)
    assert freshness[("freshness_new_attention_mass", ())].value == pytest.approx(0.7)

    cache = _values(
        cache_diagnostic_contributions(
            span_hits=[True, False, True],
            top1_key_correct=[True, False, True],
            top1_attention_masses=[0.7, 0.2, 0.5],
            gold_attention_masses=[0.8, 0.0, 0.4],
            cache_value_correct=[True, False, True],
            wrong_key=[False, True, True],
            non_sink_read=[True, True, False],
            selector_labels=[True, False, True],
            selector_scores=[0.9, 0.8, 0.1],
            selector_positions=[10, 20, 30],
            survived=[True, False, True],
            ages=["young", "old", "old"],
            declared_ages=("young", "old", "empty"),
            attention_entropies=[1.0, 2.0, 3.0],
            effective_supports=[2.0, 3.0, 4.0],
            sink_masses=[0.1, 0.2, 0.3],
            cache_output_norms=[2.0, 3.0, 4.0],
            state_output_norms=[1.0, 2.0, 3.0],
            retained_counts=[3.0, 0.0, 1.0],
            evicted_counts=[1.0, 0.0, 1.0],
            selection_scores=[1.0, 2.0, 3.0],
            persistent_bytes=[10.0, 12.0, 14.0],
            block_bytes=[20.0, 22.0, 24.0],
            latencies_ms=[2.0, 3.0, 4.0],
            token_counts=[100.0, 200.0, 150.0],
            example_counts=[10.0, 20.0, 15.0],
            wall_seconds=[1.0, 1.0, 1.0],
            peak_vram_bytes=[1000.0, 2000.0, 3000.0],
        )
    )
    expected = {
        "cache_span_hit_rate": 2 / 3,
        "cache_top1_key_accuracy": 2 / 3,
        "cache_top1_attention_mass": 0.4666666666666667,
        "cache_gold_attention_mass": 0.4,
        "cache_conditional_read_accuracy": 1.0,
        "cache_value_exact_match": 1.0,
        "cache_wrong_key_rate": 0.5,
        "cache_selector_auprc": 5 / 6,
        "cache_survival_rate": 2 / 3,
        "cache_attention_entropy": 2.0,
        "cache_effective_support": 3.0,
        "cache_sink_mass": 0.2,
        "cache_output_norm": 3.0,
        "state_output_norm": 2.0,
        "cache_retention_rate": 2 / 3,
        "cache_eviction_rate": 1 / 3,
        "cache_selection_score_mean": 2.0,
        "cache_selection_score_min": 1.0,
        "cache_selection_score_max": 3.0,
        "cache_persistent_bytes": 14.0,
        "cache_block_bytes": 24.0,
        "latency_ms": 3.0,
        "throughput_tokens_per_second": 150.0,
        "throughput_examples_per_second": 15.0,
        "peak_vram_bytes": 3000.0,
    }
    for metric, value in expected.items():
        assert cache[(metric, ())].value == pytest.approx(value)
    assert cache[("cache_survival_rate", (("age", "young"),))].value == 1.0
    assert not cache[("cache_survival_rate", (("age", "empty"),))].available


def test_metric_accumulators_selector_ties_domains_alignment_and_empty_conditionals() -> None:
    tied = selector_auprc_contribution(
        labels=[True, False, True],
        scores=[0.5, 0.5, 0.1],
        positions=[1, 2, 3],
    )
    permuted = selector_auprc_contribution(
        labels=[True, True, False],
        scores=[0.1, 0.5, 0.5],
        positions=[3, 1, 2],
    )
    assert tied == permuted
    assert _values((tied,))[("cache_selector_auprc", ())].value == pytest.approx(7 / 12)

    with pytest.raises(ValueError, match="equal lengths"):
        freshness_contributions(
            latest_correct=[True],
            stale_old_predicted=[False, True],
            update_latencies=[1.0],
            duplicate_occupancies=[1.0],
            old_attention_masses=[0.1],
            new_attention_masses=[0.9],
        )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        freshness_contributions(
            latest_correct=[True],
            stale_old_predicted=[False],
            update_latencies=[1.0],
            duplicate_occupancies=[1.0],
            old_attention_masses=[1.1],
            new_attention_masses=[0.9],
        )
    with pytest.raises(TypeError, match="episode_ids"):
        binding_mqar_contributions(
            task="mqar",
            predictions=[0],
            targets=[0],
            eligible=[False],
            episode_ids=[None],  # type: ignore[list-item]
            distance_bins=["near"],
            load_bins=["low"],
            declared_distance_bins=("near",),
            declared_load_bins=("low",),
        )

    empty = _values(
        cache_diagnostic_contributions(
            span_hits=[False],
            top1_key_correct=[False],
            top1_attention_masses=[0.0],
            gold_attention_masses=[0.0],
            cache_value_correct=[False],
            wrong_key=[False],
            non_sink_read=[False],
            selector_labels=[False],
            selector_scores=[0.0],
            selector_positions=[0],
            survived=[False],
            ages=["old"],
            declared_ages=("old",),
            attention_entropies=[0.0],
            effective_supports=[1.0],
            sink_masses=[1.0],
            cache_output_norms=[0.0],
            state_output_norms=[0.0],
            retained_counts=[0.0],
            evicted_counts=[0.0],
            selection_scores=[0.0],
            persistent_bytes=[0.0],
            block_bytes=[0.0],
            latencies_ms=[0.0],
            token_counts=[0.0],
            example_counts=[0.0],
            wall_seconds=[1.0],
            peak_vram_bytes=[0.0],
        )
    )
    for metric in (
        "cache_conditional_read_accuracy",
        "cache_value_exact_match",
        "cache_wrong_key_rate",
        "cache_selector_auprc",
        "cache_retention_rate",
        "cache_eviction_rate",
    ):
        assert not empty[(metric, ())].available


def test_selector_auprc_composes_raw_ranked_state_across_partitions() -> None:
    first = selector_auprc_contribution(
        labels=[True], scores=[0.9], positions=[0]
    )
    second = selector_auprc_contribution(
        labels=[False, True], scores=[0.8, 0.1], positions=[0, 1]
    )
    result = _values((first, second))[("cache_selector_auprc", ())]
    assert result.numerator == pytest.approx(5 / 3)
    assert result.denominator == 2.0
    assert result.value == pytest.approx(5 / 6)


def test_metric_accumulators_cache_semantic_groups_allow_different_populations() -> None:
    values = _values(
        cache_diagnostic_contributions(
            span_hits=[True],
            top1_key_correct=[True],
            top1_attention_masses=[0.8],
            gold_attention_masses=[0.7],
            cache_value_correct=[True],
            wrong_key=[False],
            non_sink_read=[True],
            selector_labels=[True, False],
            selector_scores=[0.9, 0.2],
            selector_positions=[4, 9],
            survived=[True, False, True],
            ages=["young", "old", "old"],
            declared_ages=("young", "old"),
            attention_entropies=[0.5],
            effective_supports=[1.5],
            sink_masses=[0.1],
            cache_output_norms=[2.0],
            state_output_norms=[1.0],
            retained_counts=[2.0, 1.0],
            evicted_counts=[1.0, 0.0],
            selection_scores=[1.0, 2.0, 3.0, 4.0],
            persistent_bytes=[10.0, 12.0],
            block_bytes=[20.0, 22.0],
            latencies_ms=[2.0, 4.0],
            token_counts=[100.0, 200.0],
            example_counts=[10.0, 20.0],
            wall_seconds=[1.0, 1.0],
            peak_vram_bytes=[1000.0, 2000.0],
        )
    )
    assert values[("cache_selector_auprc", ())].value == 1.0
    assert values[("cache_survival_rate", ())].value == pytest.approx(2 / 3)
    assert values[("cache_selection_score_mean", ())].value == 2.5
    assert values[("throughput_tokens_per_second", ())].value == 150.0


def _sample(
    seed: int, example: str, numerator: float, denominator: float = 1.0
) -> MetricSample:
    return MetricSample(seed, example, "budget-a", numerator, denominator)


def _independent_bootstrap_values(
    variant: tuple[MetricSample, ...],
    baseline: tuple[MetricSample, ...],
    *,
    random_seed: int,
    resamples: int,
) -> list[float]:
    variant_map = {(item.seed, item.example_id): item for item in variant}
    baseline_map = {(item.seed, item.example_id): item for item in baseline}
    seeds = sorted({seed for seed, _ in variant_map})
    examples = {
        seed: sorted(example for item_seed, example in variant_map if item_seed == seed)
        for seed in seeds
    }
    rng = random.Random(random_seed)
    values: list[float] = []
    for _ in range(resamples):
        seed_effects: list[float] = []
        for seed in (rng.choice(seeds) for _ in seeds):
            selected = [rng.choice(examples[seed]) for _ in examples[seed]]
            variant_n = math.fsum(variant_map[(seed, item)].numerator for item in selected)
            variant_d = math.fsum(variant_map[(seed, item)].denominator for item in selected)
            baseline_n = math.fsum(baseline_map[(seed, item)].numerator for item in selected)
            baseline_d = math.fsum(baseline_map[(seed, item)].denominator for item in selected)
            seed_effects.append(variant_n / variant_d - baseline_n / baseline_d)
        values.append(math.fsum(seed_effects) / len(seed_effects))
    return values


def test_paired_bootstrap_hierarchical_exact_deterministic_and_order_invariant() -> None:
    variant = (
        _sample(1, "a", 9.0, 10.0),
        _sample(2, "a", 1.0),
        _sample(2, "b", 1.0),
        _sample(2, "c", 1.0),
    )
    baseline = (
        _sample(1, "a", 5.0, 10.0),
        _sample(2, "a", 0.0),
        _sample(2, "b", 0.0),
        _sample(2, "c", 0.0),
    )
    result = paired_bootstrap(
        variant,
        baseline,
        direction=1,
        random_seed=71,
        resamples=101,
    )
    assert result.point == pytest.approx(0.7)
    independent = sorted(
        _independent_bootstrap_values(
            variant, baseline, random_seed=71, resamples=101
        )
    )
    assert result.lower == independent[2]
    assert result.upper == independent[98]
    assert result.seed_count == 2 and result.example_count == 4
    repeated = paired_bootstrap(
        tuple(reversed(variant)),
        tuple(reversed(baseline)),
        direction=1,
        random_seed=71,
        resamples=101,
    )
    assert repeated == result
    assert repeated.canonical_bytes == result.canonical_bytes
    reversed_direction = paired_bootstrap(
        variant, baseline, direction=-1, random_seed=71, resamples=101
    )
    assert reversed_direction.point == pytest.approx(-0.7)
    assert reversed_direction.lower == pytest.approx(-result.upper)
    assert reversed_direction.upper == pytest.approx(-result.lower)


def test_paired_bootstrap_singleton_signed_zero_and_strict_pairing() -> None:
    singleton = paired_bootstrap(
        (_sample(1, "a", 2.0, 4.0),),
        (_sample(1, "a", 1.0, 4.0),),
        direction=1,
        random_seed=3,
        resamples=9,
    )
    assert singleton.point == singleton.lower == singleton.upper == 0.25
    zero = paired_bootstrap(
        (_sample(1, "a", 0.0),),
        (_sample(1, "a", -0.0),),
        direction=1,
        random_seed=3,
        resamples=9,
    )
    assert zero.point == 0.0 and math.copysign(1.0, zero.point) == 1.0
    assert b"-0.0" not in zero.canonical_bytes
    with pytest.raises(ValueError, match="must not be empty"):
        paired_bootstrap((), (), direction=1, random_seed=1, resamples=9)
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap(
            (_sample(1, "a", 1.0), _sample(1, "a", 1.0)),
            (_sample(1, "a", 0.0),),
            direction=1,
            random_seed=1,
            resamples=9,
        )
    with pytest.raises(ValueError, match="matched seed/example"):
        paired_bootstrap(
            (_sample(1, "a", 1.0),),
            (_sample(1, "b", 0.0),),
            direction=1,
            random_seed=1,
            resamples=9,
        )
    with pytest.raises(ValueError, match="budget"):
        paired_bootstrap(
            (_sample(1, "a", 1.0),),
            (MetricSample(1, "a", "budget-b", 0.0, 1.0),),
            direction=1,
            random_seed=1,
            resamples=9,
        )
    with pytest.raises(ValueError, match="finite"):
        MetricSample(1, "a", "budget-a", float("nan"), 1.0)


def test_paired_bootstrap_factorial_joint_complete_cells_and_contrasts() -> None:
    cells = {
        "M00": (_sample(1, "a", 0.0),),
        "M10": (_sample(1, "a", 2.0),),
        "M01": (_sample(1, "a", 1.0),),
        "M11": (_sample(1, "a", 4.0),),
    }
    result = paired_factorial_bootstrap(
        cells, direction=1, random_seed=11, resamples=17
    )
    assert result.interaction.point == result.interaction.lower == result.interaction.upper == 1.0
    assert result.current_effect.point == 3.0
    assert result.feature_off_effect.point == 2.0
    assert result.canonical_bytes == paired_factorial_bootstrap(
        dict(reversed(tuple(cells.items()))),
        direction=1,
        random_seed=11,
        resamples=17,
    ).canonical_bytes
    with pytest.raises(ValueError, match="four factorial cells"):
        paired_factorial_bootstrap(
            {key: value for key, value in cells.items() if key != "M11"},
            direction=1,
            random_seed=11,
            resamples=17,
        )
    bad_budget = dict(cells)
    bad_budget["M11"] = (MetricSample(1, "a", "other", 4.0, 1.0),)
    with pytest.raises(ValueError, match="budget"):
        paired_factorial_bootstrap(
            bad_budget, direction=1, random_seed=11, resamples=17
        )


@pytest.mark.parametrize("bad_direction", [True, False, 1.0, -1.0])
def test_bootstrap_directions_require_exact_signed_ints(bad_direction) -> None:
    with pytest.raises(ValueError, match="direction"):
        BootstrapInterval(0.0, 0.0, 0.0, bad_direction, 1, 1, 1)
    with pytest.raises(ValueError, match="direction"):
        paired_bootstrap(
            (_sample(1, "a", 1.0),),
            (_sample(1, "a", 0.0),),
            direction=bad_direction,
            random_seed=1,
            resamples=1,
        )
    with pytest.raises(ValueError, match="direction"):
        paired_factorial_bootstrap(
            {
                "M00": (_sample(1, "a", 0.0),),
                "M10": (_sample(1, "a", 1.0),),
                "M01": (_sample(1, "a", 1.0),),
                "M11": (_sample(1, "a", 2.0),),
            },
            direction=bad_direction,
            random_seed=1,
            resamples=1,
        )


def _interval(
    lower: float, upper: float, point: float | None = None, direction: int = 1
):
    return BootstrapInterval(
        (lower + upper) / 2 if point is None else point,
        lower,
        upper,
        direction,
        3,
        12,
        101,
    )


def test_metric_value_and_factorial_result_exported_invariants() -> None:
    assert MetricValue("token_accuracy", 1.0, 2.0, 0.5, 1, True).available
    with pytest.raises(ValueError, match="unknown metric"):
        MetricValue("missing", 0.0, 0.0, None, None, False)
    with pytest.raises(ValueError, match="direction"):
        MetricValue("token_accuracy", 1.0, 2.0, 0.5, -1, True)
    with pytest.raises(ValueError, match="available"):
        MetricValue("token_accuracy", 0.0, 0.0, 0.0, 1, True)
    with pytest.raises(ValueError, match="unavailable"):
        MetricValue("token_accuracy", 1.0, 2.0, None, 1, False)
    with pytest.raises(ValueError, match="value"):
        MetricValue("token_accuracy", 1.0, 2.0, 0.4, 1, True)

    interval = _interval(0.1, 0.2)
    assert FactorialBootstrapResult(interval, interval, interval).interaction == interval
    with pytest.raises(TypeError, match="BootstrapInterval"):
        FactorialBootstrapResult(interval, interval, "bad")  # type: ignore[arg-type]
    opposite = _interval(0.1, 0.2, direction=-1)
    with pytest.raises(ValueError, match="direction"):
        FactorialBootstrapResult(interval, interval, opposite)
    different_counts = BootstrapInterval(0.15, 0.1, 0.2, 1, 4, 12, 101)
    with pytest.raises(ValueError, match="counts"):
        FactorialBootstrapResult(interval, interval, different_counts)


@pytest.mark.parametrize(
    ("expected", "kwargs"),
    [
        ("failed/invalid", {"primary": None}),
        ("failed/invalid", {"valid": False}),
        ("harmful", {"primary": _interval(-0.3, -0.1)}),
        (
            "harmful",
            {
                "primary": _interval(-0.01, 0.01),
                "protected": (
                    ProtectedEffect(
                        "latency_ms", _interval(-0.2, -0.11, direction=-1), 0.1
                    ),
                ),
                "interaction": _interval(0.2, 0.3),
            },
        ),
        (
            "synergistic",
            {
                "primary": _interval(-0.01, 0.01),
                "interaction": _interval(0.05, 0.1),
            },
        ),
        ("incremental", {"primary": _interval(0.1, 0.2)}),
        (
            "replacement-only",
            {
                "primary": _interval(-0.01, 0.09),
                "existing_feature_off": _interval(0.1, 0.2),
            },
        ),
        ("redundant", {"primary": _interval(-0.09, 0.099)}),
        ("inconclusive", {"primary": _interval(-0.1, 0.1)}),
    ],
)
def test_addition_classification_all_ordered_labels(expected: str, kwargs: dict) -> None:
    defaults = {
        "metric": "token_accuracy",
        "primary": _interval(-0.01, 0.01),
        "protected": (),
        "valid": True,
        "min_useful": 0.1,
        "harm_threshold": 0.1,
        "min_synergy": 0.05,
        "interaction": None,
        "existing_feature_off": None,
    }
    defaults.update(kwargs)
    assert classify_addition(**defaults) == expected


def test_addition_classification_exact_protection_boundaries_and_diagnostic_reject() -> None:
    safe = ProtectedEffect(
        "latency_ms", _interval(-0.1, 0.2, direction=-1), 0.1
    )
    assert classify_addition(
        metric="token_accuracy",
        primary=_interval(0.1, 0.2),
        protected=(safe,),
        valid=True,
        min_useful=0.1,
        harm_threshold=0.1,
        min_synergy=0.05,
    ) == "incremental"
    with pytest.raises(ValueError, match="diagnostic.*direction"):
        classify_addition(
            metric="cache_attention_entropy",
            primary=_interval(0.1, 0.2),
            protected=(),
            valid=True,
            min_useful=0.1,
            harm_threshold=0.1,
            min_synergy=0.05,
        )

    harmful = ProtectedEffect(
        "latency_ms", _interval(-0.2, -0.11, direction=-1), 0.1
    )
    assert classify_addition(
        metric="token_accuracy",
        primary=_interval(0.1, 0.2),
        protected=(item for item in (harmful,)),
        valid=True,
        min_useful=0.1,
        harm_threshold=0.1,
        min_synergy=0.05,
    ) == "harmful"


@pytest.mark.parametrize(
    ("expected", "interval", "valid"),
    [
        ("failed/invalid", None, True),
        ("failed/invalid", _interval(0.2, 0.3), False),
        ("harmful-current", _interval(-0.3, -0.1), True),
        ("relied-on", _interval(0.05, 0.2), True),
        ("dispensable", _interval(-0.02, 0.02), True),
        ("inconclusive-reliance", _interval(-0.03, 0.049), True),
    ],
)
def test_reliance_classification_all_ordered_labels(
    expected: str, interval, valid: bool
) -> None:
    assert classify_reliance(
        metric="token_accuracy",
        effect=interval,
        valid=valid,
        min_reliance=0.05,
        equivalence=0.02,
        harm_threshold=0.1,
    ) == expected


EXPECTED_OPTION3_CODES = (
    "invalid_evidence",
    "mechanism_changed",
    "missing_required_cells",
    "missing_required_evidence",
    "long_macro_lcb",
    "long_cell_lcb_count",
    "surprise_vs_recency_lcb",
    "short_macro_lcb",
    "short_cell_lcb",
    "eight_k_lcb",
    "episode_exact_lcb",
    "freshness_latest_native_lcb",
    "freshness_latest_recency_lcb",
    "freshness_stale_native_lcb",
    "freshness_stale_recency_lcb",
    "nonfinite_or_skipped",
    "ce_ucb",
    "kl_ucb",
    "capacity_w64_lcb",
    "capacity_adjacent_lcb",
    "capacity_above_128",
    "gate_mean",
    "gate_max",
    "persistent_hit_lcb",
    "conditional_read_lcb",
    "shuffle_drop_lcb",
    "persistent_memory_limit",
    "decode_throughput",
    "prefill_throughput",
    "dynamic_memory_not_flat",
)


def _named(name: str, lower: float, upper: float | None = None) -> NamedInterval:
    return NamedInterval(name, _interval(lower, lower if upper is None else upper))


def _passing_option3(branch: str) -> Option3Evidence:
    return Option3Evidence(
        branch=branch,
        valid=True,
        mechanism_unchanged=True,
        long_macro=_interval(0.10, 0.20),
        long_cells=(
            _named("16k_4q", 0.10, 0.20),
            _named("16k_8q", 0.10, 0.20),
            _named("32k_4q", 0.09, 0.20),
            _named("32k_8q", 0.09, 0.20),
        ),
        surprise_vs_recency=(
            _interval(0.05, 0.10) if branch == "surprise" else None
        ),
        short_macro=_interval(-0.02, 0.02),
        short_cells=(
            _named("512", -0.03, 0.01),
            _named("1k", -0.03, 0.01),
            _named("2k", -0.03, 0.01),
            _named("4k", -0.03, 0.01),
        ),
        eight_k=_interval(-0.03, 0.01),
        episode_exact=_interval(-0.05, 0.01),
        freshness_latest_native=_interval(-0.02, 0.02),
        freshness_latest_recency=_interval(-0.02, 0.02),
        freshness_stale_native=_interval(-0.02, 0.02, direction=-1),
        freshness_stale_recency=_interval(-0.02, 0.02, direction=-1),
        ce_delta=_interval(-0.01, 0.02),
        kl_delta=_interval(-0.001, 0.005),
        mean_kl_native=0.1,
        nonfinite_count=0,
        skipped_steps=0,
        capacity_w64=_interval(0.10, 0.20),
        capacity_w32=_interval(0.05, 0.10),
        capacity_w128=_interval(0.04, 0.10),
        requires_width_above_128=False,
        amplitudes=(0.005, 0.02),
        persistent_hit=_interval(0.25, 0.40),
        conditional_read=_interval(0.50, 0.70),
        shuffle_drop=_interval(0.05, 0.10),
        persistent_bytes=10 * 1024 * 1024,
        persistent_bytes_limit=10 * 1024 * 1024,
        decode_throughput_ratio=0.80,
        prefill_throughput_ratio=0.75,
        dynamic_memory_flat=True,
    )


@pytest.mark.parametrize("min_long_cells", [0, 5, True])
def test_option3_threshold_long_cell_domain(min_long_cells) -> None:
    with pytest.raises(ValueError, match="min_long_cells"):
        Option3Thresholds(min_long_cells=min_long_cells)


@pytest.mark.parametrize(
    "field",
    [
        "long_macro_lcb",
        "long_cell_lcb",
        "surprise_recency_lcb",
        "short_macro_lcb",
        "short_cell_lcb",
        "eight_k_lcb",
        "episode_exact_lcb",
        "freshness_lcb",
        "capacity_w64_lcb",
        "capacity_adjacent_lcb",
    ],
)
def test_option3_effect_thresholds_stay_in_feasible_domain(field: str) -> None:
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        replace(Option3Thresholds(), **{field: 1.01})
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        replace(Option3Thresholds(), **{field: -1.01})


@pytest.mark.parametrize(
    "field",
    [
        "ce_ucb",
        "kl_absolute_ucb",
        "kl_native_fraction",
        "min_decode_throughput",
        "min_prefill_throughput",
    ],
)
def test_option3_loss_and_throughput_thresholds_are_nonnegative(field: str) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        replace(Option3Thresholds(), **{field: -0.01})


@pytest.mark.parametrize(
    "field",
    [
        "min_gate_mean",
        "min_gate_max",
        "min_persistent_hit",
        "min_conditional_read",
        "min_shuffle_drop",
    ],
)
def test_option3_gate_and_retrieval_thresholds_are_probabilities(field: str) -> None:
    for value in (-0.01, 1.01):
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            replace(Option3Thresholds(), **{field: value})
    with pytest.raises(ValueError, match="mean.*max"):
        Option3Thresholds(min_gate_mean=0.5, min_gate_max=0.4)


def test_option3_evidence_requires_typed_field_specific_interval_directions() -> None:
    evidence = _passing_option3("surprise")
    with pytest.raises(TypeError, match="long_macro.*BootstrapInterval"):
        replace(evidence, long_macro="bad")  # type: ignore[arg-type]

    positive_direction_fields = (
        "long_macro",
        "surprise_vs_recency",
        "short_macro",
        "eight_k",
        "episode_exact",
        "freshness_latest_native",
        "freshness_latest_recency",
        "ce_delta",
        "kl_delta",
        "capacity_w64",
        "capacity_w32",
        "capacity_w128",
        "persistent_hit",
        "conditional_read",
        "shuffle_drop",
    )
    for field in positive_direction_fields:
        with pytest.raises(ValueError, match=f"{field}.*direction"):
            replace(evidence, **{field: _interval(0.0, 0.1, direction=-1)})
    for field in ("freshness_stale_native", "freshness_stale_recency"):
        with pytest.raises(ValueError, match=f"{field}.*direction"):
            replace(evidence, **{field: _interval(0.0, 0.1, direction=1)})

    bad_long = tuple(
        NamedInterval(item.name, _interval(0.1, 0.2, direction=-1))
        for item in evidence.long_cells
    )
    with pytest.raises(ValueError, match="long_cells.*direction"):
        replace(evidence, long_cells=bad_long)
    bad_short = tuple(
        NamedInterval(item.name, _interval(0.1, 0.2, direction=-1))
        for item in evidence.short_cells
    )
    with pytest.raises(ValueError, match="short_cells.*direction"):
        replace(evidence, short_cells=bad_short)
    with pytest.raises(TypeError, match="long_cells.*NamedInterval"):
        replace(evidence, long_cells=("bad",))  # type: ignore[arg-type]


def test_option3_result_and_decision_exported_invariants() -> None:
    passed_surprise = Option3BranchResult("surprise", True, ())
    passed_recency = Option3BranchResult("recency", True, ())
    failed_surprise = Option3BranchResult(
        "surprise", False, ("long_macro_lcb",)
    )
    with pytest.raises(ValueError, match="branch"):
        Option3BranchResult("other", True, ())
    with pytest.raises(TypeError, match="passed"):
        Option3BranchResult("surprise", 1, ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rejection"):
        Option3BranchResult("surprise", False, ("not_a_code",))
    with pytest.raises(ValueError, match="unique"):
        Option3BranchResult(
            "surprise", False, ("long_macro_lcb", "long_macro_lcb")
        )
    with pytest.raises(ValueError, match="canonical"):
        Option3BranchResult(
            "surprise", False, ("long_macro_lcb", "invalid_evidence")
        )
    with pytest.raises(ValueError, match="passed"):
        Option3BranchResult("surprise", True, ("long_macro_lcb",))

    assert Option3Decision("surprise", passed_surprise, passed_recency).selected_branch == "surprise"
    assert Option3Decision("recency", failed_surprise, passed_recency).selected_branch == "recency"
    with pytest.raises(ValueError, match="selected"):
        Option3Decision("other", passed_surprise, passed_recency)
    with pytest.raises(ValueError, match="branches"):
        Option3Decision("surprise", passed_recency, passed_surprise)
    with pytest.raises(ValueError, match="selection"):
        Option3Decision("recency", passed_surprise, passed_recency)


@pytest.mark.parametrize("code", EXPECTED_OPTION3_CODES)
def test_option3_every_rejection_code_is_individually_enforced(code: str) -> None:
    evidence = _passing_option3("surprise")
    if code == "invalid_evidence":
        evidence = replace(evidence, valid=False)
    elif code == "mechanism_changed":
        evidence = replace(evidence, mechanism_unchanged=False)
    elif code == "missing_required_cells":
        evidence = replace(evidence, long_cells=evidence.long_cells[:-1])
    elif code == "missing_required_evidence":
        evidence = replace(evidence, long_macro=None)
    elif code == "long_macro_lcb":
        evidence = replace(evidence, long_macro=_interval(0.099, 0.2))
    elif code == "long_cell_lcb_count":
        evidence = replace(
            evidence,
            long_cells=tuple(_named(item.name, 0.099, 0.2) for item in evidence.long_cells),
        )
    elif code == "surprise_vs_recency_lcb":
        evidence = replace(evidence, surprise_vs_recency=_interval(0.049, 0.1))
    elif code == "short_macro_lcb":
        evidence = replace(evidence, short_macro=_interval(-0.021, 0.01))
    elif code == "short_cell_lcb":
        evidence = replace(
            evidence,
            short_cells=(_named("512", -0.031, 0.01), *evidence.short_cells[1:]),
        )
    elif code == "eight_k_lcb":
        evidence = replace(evidence, eight_k=_interval(-0.031, 0.01))
    elif code == "episode_exact_lcb":
        evidence = replace(evidence, episode_exact=_interval(-0.051, 0.01))
    elif code == "freshness_latest_native_lcb":
        evidence = replace(evidence, freshness_latest_native=_interval(-0.021, 0.01))
    elif code == "freshness_latest_recency_lcb":
        evidence = replace(evidence, freshness_latest_recency=_interval(-0.021, 0.01))
    elif code == "freshness_stale_native_lcb":
        evidence = replace(
            evidence,
            freshness_stale_native=_interval(-0.021, 0.01, direction=-1),
        )
    elif code == "freshness_stale_recency_lcb":
        evidence = replace(
            evidence,
            freshness_stale_recency=_interval(-0.021, 0.01, direction=-1),
        )
    elif code == "nonfinite_or_skipped":
        evidence = replace(evidence, nonfinite_count=1)
    elif code == "ce_ucb":
        evidence = replace(evidence, ce_delta=_interval(-0.01, 0.021))
    elif code == "kl_ucb":
        evidence = replace(evidence, kl_delta=_interval(-0.001, 0.0051))
    elif code == "capacity_w64_lcb":
        evidence = replace(evidence, capacity_w64=_interval(0.099, 0.2))
    elif code == "capacity_adjacent_lcb":
        evidence = replace(
            evidence,
            capacity_w32=_interval(0.049, 0.1),
            capacity_w128=_interval(0.049, 0.1),
        )
    elif code == "capacity_above_128":
        evidence = replace(evidence, requires_width_above_128=True)
    elif code == "gate_mean":
        evidence = replace(evidence, amplitudes=(0.0, 0.0, 0.0, 0.0, 0.02))
    elif code == "gate_max":
        evidence = replace(evidence, amplitudes=(0.005, 0.019))
    elif code == "persistent_hit_lcb":
        evidence = replace(evidence, persistent_hit=_interval(0.249, 0.4))
    elif code == "conditional_read_lcb":
        evidence = replace(evidence, conditional_read=_interval(0.499, 0.7))
    elif code == "shuffle_drop_lcb":
        evidence = replace(evidence, shuffle_drop=_interval(0.049, 0.1))
    elif code == "persistent_memory_limit":
        evidence = replace(evidence, persistent_bytes=evidence.persistent_bytes_limit + 1)
    elif code == "decode_throughput":
        evidence = replace(evidence, decode_throughput_ratio=0.799)
    elif code == "prefill_throughput":
        evidence = replace(evidence, prefill_throughput_ratio=0.749)
    elif code == "dynamic_memory_not_flat":
        evidence = replace(evidence, dynamic_memory_flat=False)
    result = evaluate_option3_branch(evidence, Option3Thresholds())
    assert result.passed is False
    assert result.rejection_codes == (code,)


def test_option3_exact_boundaries_branch_independence_and_ordering() -> None:
    assert OPTION3_REJECTION_CODES == EXPECTED_OPTION3_CODES
    surprise = _passing_option3("surprise")
    recency = _passing_option3("recency")
    assert evaluate_option3_branch(surprise, Option3Thresholds()).passed
    assert evaluate_option3_branch(recency, Option3Thresholds()).passed
    decision = decide_option3(surprise, recency, Option3Thresholds())
    assert decision.selected_branch == "surprise"
    assert decision.surprise.passed and decision.recency.passed

    failed_surprise = replace(surprise, long_macro=_interval(0.099, 0.2))
    recency_decision = decide_option3(failed_surprise, recency, Option3Thresholds())
    assert recency_decision.selected_branch == "recency"
    assert recency_decision.surprise.rejection_codes == ("long_macro_lcb",)
    both_failed = decide_option3(
        failed_surprise,
        replace(recency, long_macro=_interval(0.099, 0.2)),
        Option3Thresholds(),
    )
    assert both_failed.selected_branch == "no_promote"

    higher_native_kl = replace(
        surprise,
        mean_kl_native=0.2,
        kl_delta=_interval(-0.001, 0.01),
    )
    assert evaluate_option3_branch(higher_native_kl, Option3Thresholds()).passed
