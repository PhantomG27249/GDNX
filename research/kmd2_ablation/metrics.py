"""Dependency-light metrics and statistical decisions for KMD-2 ablations."""

from __future__ import annotations

import math
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: int | None
    reducer: str = "ratio"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("metric name must be a non-empty string")
        if self.direction is not None and (
            type(self.direction) is not int or self.direction not in (-1, 1)
        ):
            raise ValueError("metric direction must be +1, -1, or None")
        if self.reducer not in {"ratio", "min", "max", "episode_exact", "auprc"}:
            raise ValueError(
                "metric reducer must be ratio, min, max, episode_exact, or auprc"
            )


_DIRECTIONS = {
    "token_accuracy": 1,
    "episode_exact": 1,
    "chance_adjusted_accuracy": 1,
    "integration_mse": -1,
    "drift_steady_state_error": -1,
    "drift_adaptation_lag": -1,
    "drift_peak_overshoot": -1,
    "drift_recovery_time": -1,
    "trajectory_smooth_mse": -1,
    "trajectory_phase_lag": -1,
    "trajectory_change_point_mse": -1,
    "trajectory_change_point_overshoot": -1,
    "trajectory_recovery_time": -1,
    "affine_query_mse": -1,
    "affine_intercept_mse": -1,
    "affine_slope_mse": -1,
    "binding_token_accuracy": 1,
    "binding_episode_exact": 1,
    "mqar_token_accuracy": 1,
    "mqar_episode_exact": 1,
    "structured_rule_accuracy": 1,
    "structured_exception_accuracy": 1,
    "freshness_latest_accuracy": 1,
    "freshness_stale_old_rate": -1,
    "freshness_update_latency": -1,
    "freshness_duplicate_occupancy": -1,
    "freshness_old_attention_mass": -1,
    "freshness_new_attention_mass": 1,
    "cache_span_hit_rate": 1,
    "cache_top1_key_accuracy": 1,
    "cache_top1_attention_mass": None,
    "cache_gold_attention_mass": 1,
    "cache_conditional_read_accuracy": 1,
    "cache_value_exact_match": 1,
    "cache_wrong_key_rate": -1,
    "cache_selector_auprc": 1,
    "cache_survival_rate": 1,
    "cache_attention_entropy": None,
    "cache_effective_support": None,
    "cache_sink_mass": None,
    "cache_output_norm": None,
    "state_output_norm": None,
    "cache_retention_rate": None,
    "cache_eviction_rate": None,
    "cache_selection_score_mean": None,
    "cache_selection_score_min": None,
    "cache_selection_score_max": None,
    "cache_persistent_bytes": -1,
    "cache_block_bytes": -1,
    "latency_ms": -1,
    "throughput_tokens_per_second": 1,
    "throughput_examples_per_second": 1,
    "peak_vram_bytes": -1,
}

_MAX_METRICS = {
    "cache_persistent_bytes",
    "cache_block_bytes",
    "cache_selection_score_max",
    "peak_vram_bytes",
}
_MIN_METRICS = {"cache_selection_score_min"}
_EPISODE_EXACT_METRICS = {
    "episode_exact",
    "binding_episode_exact",
    "mqar_episode_exact",
}
METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType(
    {
        name: MetricSpec(
            name,
            direction,
            "max"
            if name in _MAX_METRICS
            else "min"
            if name in _MIN_METRICS
            else "episode_exact"
            if name in _EPISODE_EXACT_METRICS
            else "auprc"
            if name == "cache_selector_auprc"
            else "ratio",
        )
        for name, direction in _DIRECTIONS.items()
    }
)


def metric_direction(metric: str) -> int | None:
    try:
        return METRIC_REGISTRY[metric].direction
    except KeyError as error:
        raise ValueError(f"unknown metric {metric!r}") from error


def _canonical_strata(
    strata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
) -> tuple[tuple[str, str], ...]:
    items = tuple(strata.items()) if isinstance(strata, Mapping) else tuple(strata)
    if any(
        type(key) is not str
        or not key
        or type(value) is not str
        or not value
        for key, value in items
    ):
        raise TypeError("metric strata keys and values must be non-empty strings")
    if len({key for key, _ in items}) != len(items):
        raise ValueError("metric strata keys must be unique")
    return tuple(sorted(items))


def _finite_number(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class MetricContribution:
    metric: str
    numerator: float
    denominator: float
    strata: tuple[tuple[str, str], ...] = ()
    state: tuple[tuple[object, ...], ...] = ()

    def __post_init__(self) -> None:
        metric_direction(self.metric)
        reducer = METRIC_REGISTRY[self.metric].reducer
        numerator = _finite_number("metric numerator", self.numerator)
        denominator = _finite_number("metric denominator", self.denominator)
        if denominator < 0:
            raise ValueError("metric denominator must be nonnegative")
        if denominator == 0 and numerator != 0:
            raise ValueError("zero denominator requires a zero numerator")
        state = tuple(tuple(entry) for entry in self.state)
        if reducer in {"episode_exact", "auprc"}:
            if numerator != 0 or denominator != 0:
                raise ValueError(
                    f"{reducer} contributions must retain raw state with zero totals"
                )
        elif state:
            raise ValueError(f"{reducer} contributions cannot contain raw state")
        if reducer == "episode_exact":
            normalized_episode_state: list[tuple[object, ...]] = []
            for entry in state:
                if len(entry) != 2:
                    raise ValueError("episode_exact state entries must have two fields")
                episode_id, correct = entry
                if type(episode_id) is not str or not episode_id:
                    raise TypeError(
                        "episode_exact state ids must be non-empty strings"
                    )
                if type(correct) is not bool:
                    raise TypeError("episode_exact state outcomes must be bool")
                normalized_episode_state.append((episode_id, correct))
            state = tuple(sorted(normalized_episode_state, key=lambda item: item[0]))
        elif reducer == "auprc":
            normalized_ranked_state: list[tuple[object, ...]] = []
            for entry in state:
                if len(entry) != 3:
                    raise ValueError("auprc state entries must have three fields")
                score, label, position = entry
                score = _finite_number("selector score", score)
                if score < 0:
                    raise ValueError("selector score values must be nonnegative")
                if type(label) is not bool:
                    raise TypeError("selector labels must be bool")
                if type(position) is not int or position < 0:
                    raise ValueError("selector positions must be nonnegative ints")
                normalized_ranked_state.append((score, label, position))
            state = tuple(
                sorted(
                    normalized_ranked_state,
                    key=lambda item: (item[2], -item[0], not item[1]),
                )
            )
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "strata", _canonical_strata(self.strata))
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class MetricValue:
    metric: str
    numerator: float
    denominator: float
    value: float | None
    direction: int | None
    available: bool
    strata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        expected_direction = metric_direction(self.metric)
        if self.direction != expected_direction or (
            expected_direction is not None and type(self.direction) is not int
        ):
            raise ValueError("metric value direction does not match registry")
        if type(self.available) is not bool:
            raise TypeError("metric value available must be bool")
        numerator = _finite_number("metric value numerator", self.numerator)
        denominator = _finite_number("metric value denominator", self.denominator)
        if denominator < 0:
            raise ValueError("metric value denominator must be nonnegative")
        if self.available and denominator == 0:
            raise ValueError("available metric value requires a positive denominator")
        if not self.available and denominator != 0:
            raise ValueError("unavailable metric value requires a zero denominator")
        if not self.available:
            if self.value is not None or numerator != 0:
                raise ValueError(
                    "unavailable metric value requires zero numerator and None value"
                )
            value = None
        else:
            value = _finite_number("metric value", self.value)
            reducer = METRIC_REGISTRY[self.metric].reducer
            expected_value = (
                numerator / denominator
                if reducer in {"ratio", "episode_exact", "auprc"}
                else numerator
            )
            if not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("metric value does not match its reducer totals")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "strata", _canonical_strata(self.strata))


def accumulate_metrics(
    contributions: Iterable[MetricContribution],
) -> tuple[MetricValue, ...]:
    grouped: dict[
        tuple[str, tuple[tuple[str, str], ...]], list[MetricContribution]
    ] = defaultdict(list)
    for contribution in contributions:
        if not isinstance(contribution, MetricContribution):
            raise TypeError("contributions must contain MetricContribution records")
        grouped[(contribution.metric, contribution.strata)].append(contribution)
    values: list[MetricValue] = []
    for (metric, strata), items in sorted(grouped.items()):
        reducer = METRIC_REGISTRY[metric].reducer
        if reducer == "ratio":
            denominator = math.fsum(item.denominator for item in items)
            available = denominator > 0
            numerator = math.fsum(item.numerator for item in items)
            value = numerator / denominator if available else None
        elif reducer in {"min", "max"}:
            denominator = math.fsum(item.denominator for item in items)
            available_items = [item.numerator for item in items if item.denominator > 0]
            available = bool(available_items)
            value = (
                (min(available_items) if reducer == "min" else max(available_items))
                if available
                else None
            )
            numerator = value if value is not None else 0.0
        elif reducer == "episode_exact":
            outcomes: dict[str, bool] = {}
            for item in items:
                for episode_id, correct in item.state:
                    assert isinstance(episode_id, str) and isinstance(correct, bool)
                    outcomes[episode_id] = outcomes.get(episode_id, True) and correct
            numerator = float(sum(outcomes.values()))
            denominator = float(len(outcomes))
            available = denominator > 0
            value = numerator / denominator if available else None
        else:
            ranked = [entry for item in items for entry in item.state]
            positive_count = sum(bool(entry[1]) for entry in ranked)
            if positive_count == 0:
                numerator = 0.0
                denominator = 0.0
                available = False
                value = None
            else:
                score_groups: dict[float, list[bool]] = defaultdict(list)
                for score, label, _position in ranked:
                    score_groups[float(score)].append(bool(label))
                true_positives = 0
                seen = 0
                precision_sum = 0.0
                for score in sorted(score_groups, reverse=True):
                    labels = score_groups[score]
                    group_positives = sum(labels)
                    true_positives += group_positives
                    seen += len(labels)
                    precision_sum += group_positives * (true_positives / seen)
                numerator = precision_sum
                denominator = float(positive_count)
                available = True
                value = numerator / denominator
        if value == 0:
            value = 0.0
        values.append(
            MetricValue(
                metric=metric,
                numerator=numerator,
                denominator=denominator,
                value=value,
                direction=metric_direction(metric),
                available=available,
                strata=strata,
            )
        )
    return tuple(values)


def _same_length(name: str, sequences: Sequence[Sequence[object]]) -> int:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"{name} inputs must have equal lengths")
    return next(iter(lengths))


def classification_contributions(
    *,
    predictions: Sequence[int],
    targets: Sequence[int],
    eligible: Sequence[bool],
    episode_ids: Sequence[str],
    chance_probabilities: Sequence[float],
) -> tuple[MetricContribution, ...]:
    count = _same_length(
        "classification",
        (predictions, targets, eligible, episode_ids, chance_probabilities),
    )
    correct: list[bool] = []
    episode_state: list[tuple[object, ...]] = []
    chance_numerator: list[float] = []
    chance_denominator: list[float] = []
    for index in range(count):
        if type(eligible[index]) is not bool:
            raise TypeError("eligible entries must be bool")
        if type(predictions[index]) is not int or type(targets[index]) is not int:
            raise TypeError("classification predictions and targets must be ints")
        episode_id = episode_ids[index]
        if type(episode_id) is not str or not episode_id:
            raise TypeError("episode_ids entries must be non-empty strings")
        chance = _finite_number("chance probability", chance_probabilities[index])
        if not 0 <= chance < 1:
            raise ValueError("chance probabilities must be in [0,1)")
        if not eligible[index]:
            continue
        is_correct = predictions[index] == targets[index]
        correct.append(is_correct)
        episode_state.append((episode_id, is_correct))
        chance_numerator.append(float(is_correct) - chance)
        chance_denominator.append(1.0 - chance)
    return (
        MetricContribution("token_accuracy", float(sum(correct)), float(len(correct))),
        MetricContribution(
            "episode_exact", 0.0, 0.0, state=tuple(episode_state)
        ),
        MetricContribution(
            "chance_adjusted_accuracy",
            math.fsum(chance_numerator),
            math.fsum(chance_denominator),
        ),
    )


def integration_contributions(
    *,
    predictions: Sequence[float],
    targets: Sequence[float],
    eligible: Sequence[bool],
    gap_bins: Sequence[str],
    curvature_bins: Sequence[str],
    declared_gap_bins: Sequence[str],
    declared_curvature_bins: Sequence[str],
) -> tuple[MetricContribution, ...]:
    count = _same_length(
        "integration", (predictions, targets, eligible, gap_bins, curvature_bins)
    )
    gap_levels = tuple(declared_gap_bins)
    curvature_levels = tuple(declared_curvature_bins)
    if len(set(gap_levels)) != len(gap_levels) or len(set(curvature_levels)) != len(
        curvature_levels
    ):
        raise ValueError("declared integration strata must be unique")
    squared_errors: list[float] = []
    by_gap: dict[str, list[float]] = {level: [] for level in gap_levels}
    by_curvature: dict[str, list[float]] = {
        level: [] for level in curvature_levels
    }
    for index in range(count):
        prediction = _finite_number("integration prediction", predictions[index])
        target = _finite_number("integration target", targets[index])
        if type(eligible[index]) is not bool:
            raise TypeError("eligible entries must be bool")
        gap = gap_bins[index]
        curvature = curvature_bins[index]
        if gap not in by_gap or curvature not in by_curvature:
            raise ValueError("integration stratum is not declared")
        if not eligible[index]:
            continue
        squared = (prediction - target) ** 2
        squared_errors.append(squared)
        by_gap[gap].append(squared)
        by_curvature[curvature].append(squared)
    contributions = [
        MetricContribution(
            "integration_mse", math.fsum(squared_errors), float(len(squared_errors))
        )
    ]
    contributions.extend(
        MetricContribution(
            "integration_mse",
            math.fsum(values),
            float(len(values)),
            (("gap", level),),
        )
        for level, values in by_gap.items()
    )
    contributions.extend(
        MetricContribution(
            "integration_mse",
            math.fsum(values),
            float(len(values)),
            (("curvature", level),),
        )
        for level, values in by_curvature.items()
    )
    return tuple(contributions)


def _mean_contribution(
    metric: str,
    values: Sequence[float],
    strata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
) -> MetricContribution:
    finite = [_finite_number(f"{metric} value", value) for value in values]
    return MetricContribution(metric, math.fsum(finite), float(len(finite)), strata)


def _nonnegative_values(name: str, values: Sequence[float]) -> list[float]:
    finite = [_finite_number(name, value) for value in values]
    if any(value < 0 for value in finite):
        raise ValueError(f"{name} values must be nonnegative")
    return finite


def _probability_values(name: str, values: Sequence[float]) -> list[float]:
    finite = [_finite_number(name, value) for value in values]
    if any(not 0 <= value <= 1 for value in finite):
        raise ValueError(f"{name} values must be in [0,1]")
    return finite


def _positive_values(name: str, values: Sequence[float]) -> list[float]:
    finite = _nonnegative_values(name, values)
    if any(value <= 0 for value in finite):
        raise ValueError(f"{name} values must be positive")
    return finite


def _support_values(values: Sequence[float]) -> list[float]:
    finite = _positive_values("effective support", values)
    if any(value < 1 for value in finite):
        raise ValueError("effective support values must be at least 1")
    return finite


def _extreme_contribution(metric: str, values: Sequence[float]) -> MetricContribution:
    finite = _nonnegative_values(f"{metric} value", values)
    if not finite:
        return MetricContribution(metric, 0.0, 0.0)
    reducer = METRIC_REGISTRY[metric].reducer
    if reducer not in {"min", "max"}:
        raise ValueError(f"{metric} is not an extrema metric")
    extreme = min(finite) if reducer == "min" else max(finite)
    return MetricContribution(metric, extreme, float(len(finite)))


def _rate_contribution(
    metric: str,
    values: Sequence[bool],
    strata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
) -> MetricContribution:
    if any(type(value) is not bool for value in values):
        raise TypeError(f"{metric} values must be bool")
    return MetricContribution(metric, float(sum(values)), float(len(values)), strata)


def drift_contributions(
    *,
    steady_state_errors: Sequence[float],
    adaptation_lags: Sequence[float],
    peak_overshoots: Sequence[float],
    recovery_times: Sequence[float],
) -> tuple[MetricContribution, ...]:
    return (
        _mean_contribution("drift_steady_state_error", _nonnegative_values("steady-state error", steady_state_errors)),
        _mean_contribution("drift_adaptation_lag", _nonnegative_values("adaptation lag", adaptation_lags)),
        _mean_contribution("drift_peak_overshoot", _nonnegative_values("peak overshoot", peak_overshoots)),
        _mean_contribution("drift_recovery_time", _nonnegative_values("recovery time", recovery_times)),
    )


def trajectory_contributions(
    *,
    smooth_errors: Sequence[float],
    phase_lags: Sequence[float],
    change_point_errors: Sequence[float],
    change_point_overshoots: Sequence[float],
    recovery_times: Sequence[float],
) -> tuple[MetricContribution, ...]:
    smooth_squared = [
        _finite_number("trajectory smooth error", value) ** 2
        for value in smooth_errors
    ]
    change_squared = [
        _finite_number("trajectory change-point error", value) ** 2
        for value in change_point_errors
    ]
    return (
        _mean_contribution("trajectory_smooth_mse", smooth_squared),
        _mean_contribution("trajectory_phase_lag", _nonnegative_values("phase lag", phase_lags)),
        _mean_contribution("trajectory_change_point_mse", change_squared),
        _mean_contribution(
            "trajectory_change_point_overshoot", _nonnegative_values("change-point overshoot", change_point_overshoots)
        ),
        _mean_contribution("trajectory_recovery_time", _nonnegative_values("trajectory recovery time", recovery_times)),
    )


def affine_contributions(
    *,
    query_errors: Sequence[float],
    intercept_errors: Sequence[float],
    slope_errors: Sequence[float],
    zero_intercept: Sequence[bool],
) -> tuple[MetricContribution, ...]:
    if len(query_errors) != len(zero_intercept):
        raise ValueError("affine query_errors and zero_intercept must align")
    if any(type(value) is not bool for value in zero_intercept):
        raise TypeError("zero_intercept entries must be bool")
    query_squared = [
        _finite_number("affine query error", value) ** 2 for value in query_errors
    ]
    intercept_squared = [
        _finite_number("affine intercept error", value) ** 2
        for value in intercept_errors
    ]
    slope_squared = [
        _finite_number("affine slope error", value) ** 2 for value in slope_errors
    ]
    contributions = [
        _mean_contribution("affine_query_mse", query_squared),
        _mean_contribution("affine_intercept_mse", intercept_squared),
        _mean_contribution("affine_slope_mse", slope_squared),
    ]
    for is_zero, label in ((True, "zero"), (False, "nonzero")):
        values = [
            error
            for error, flag in zip(query_squared, zero_intercept, strict=True)
            if flag is is_zero
        ]
        contributions.append(
            _mean_contribution(
                "affine_query_mse", values, (("intercept", label),)
            )
        )
    return tuple(contributions)


def binding_mqar_contributions(
    *,
    task: str,
    predictions: Sequence[int],
    targets: Sequence[int],
    eligible: Sequence[bool],
    episode_ids: Sequence[str],
    distance_bins: Sequence[str],
    load_bins: Sequence[str],
    declared_distance_bins: Sequence[str],
    declared_load_bins: Sequence[str],
) -> tuple[MetricContribution, ...]:
    if task not in {"binding", "mqar"}:
        raise ValueError("task must be binding or mqar")
    count = _same_length(
        task,
        (predictions, targets, eligible, episode_ids, distance_bins, load_bins),
    )
    distance_levels = tuple(declared_distance_bins)
    load_levels = tuple(declared_load_bins)
    if len(set(distance_levels)) != len(distance_levels) or len(set(load_levels)) != len(
        load_levels
    ):
        raise ValueError("binding/MQAR declared strata must be unique")
    by_distance: dict[str, list[bool]] = {level: [] for level in distance_levels}
    by_load: dict[str, list[bool]] = {level: [] for level in load_levels}
    correct: list[bool] = []
    episode_state: list[tuple[object, ...]] = []
    for index in range(count):
        if type(eligible[index]) is not bool:
            raise TypeError("eligible entries must be bool")
        if type(predictions[index]) is not int or type(targets[index]) is not int:
            raise TypeError("binding/MQAR predictions and targets must be ints")
        if distance_bins[index] not in by_distance or load_bins[index] not in by_load:
            raise ValueError("binding/MQAR stratum is not declared")
        episode_id = episode_ids[index]
        if type(episode_id) is not str or not episode_id:
            raise TypeError("episode_ids entries must be non-empty strings")
        if not eligible[index]:
            continue
        item = predictions[index] == targets[index]
        correct.append(item)
        by_distance[distance_bins[index]].append(item)
        by_load[load_bins[index]].append(item)
        episode_state.append((episode_id, item))
    accuracy_metric = f"{task}_token_accuracy"
    episode_metric = f"{task}_episode_exact"
    contributions = [
        _rate_contribution(accuracy_metric, correct),
        MetricContribution(episode_metric, 0.0, 0.0, state=tuple(episode_state)),
    ]
    contributions.extend(
        _rate_contribution(accuracy_metric, values, (("distance", level),))
        for level, values in by_distance.items()
    )
    contributions.extend(
        _rate_contribution(accuracy_metric, values, (("load", level),))
        for level, values in by_load.items()
    )
    return tuple(contributions)


def structured_contributions(
    *, correct: Sequence[bool], is_exception: Sequence[bool]
) -> tuple[MetricContribution, ...]:
    _same_length("structured", (correct, is_exception))
    if any(type(value) is not bool for value in (*correct, *is_exception)):
        raise TypeError("structured masks and outcomes must be bool")
    rule = [item for item, exception in zip(correct, is_exception, strict=True) if not exception]
    exception = [
        item for item, exception in zip(correct, is_exception, strict=True) if exception
    ]
    return (
        _rate_contribution("structured_rule_accuracy", rule),
        _rate_contribution("structured_exception_accuracy", exception),
    )


def freshness_contributions(
    *,
    latest_correct: Sequence[bool],
    stale_old_predicted: Sequence[bool],
    update_latencies: Sequence[float],
    duplicate_occupancies: Sequence[float],
    old_attention_masses: Sequence[float],
    new_attention_masses: Sequence[float],
) -> tuple[MetricContribution, ...]:
    _same_length(
        "freshness",
        (
            latest_correct,
            stale_old_predicted,
            update_latencies,
            duplicate_occupancies,
            old_attention_masses,
            new_attention_masses,
        ),
    )
    return (
        _rate_contribution("freshness_latest_accuracy", latest_correct),
        _rate_contribution("freshness_stale_old_rate", stale_old_predicted),
        _mean_contribution(
            "freshness_update_latency",
            _nonnegative_values("freshness update latency", update_latencies),
        ),
        _mean_contribution(
            "freshness_duplicate_occupancy",
            _nonnegative_values(
                "freshness duplicate occupancy", duplicate_occupancies
            ),
        ),
        _mean_contribution(
            "freshness_old_attention_mass",
            _probability_values("old attention mass", old_attention_masses),
        ),
        _mean_contribution(
            "freshness_new_attention_mass",
            _probability_values("new attention mass", new_attention_masses),
        ),
    )


def selector_auprc_contribution(
    *,
    labels: Sequence[bool],
    scores: Sequence[float],
    positions: Sequence[int],
) -> MetricContribution:
    _same_length("selector", (labels, scores, positions))
    if any(type(label) is not bool for label in labels):
        raise TypeError("selector_labels entries must be bool")
    finite_scores = _nonnegative_values("selector score", scores)
    if any(type(position) is not int or position < 0 for position in positions):
        raise ValueError("selector positions must be nonnegative ints")
    return MetricContribution(
        "cache_selector_auprc",
        0.0,
        0.0,
        state=tuple(
            (score, label, position)
            for label, score, position in zip(
                labels, finite_scores, positions, strict=True
            )
        ),
    )


def cache_diagnostic_contributions(
    *,
    span_hits: Sequence[bool],
    top1_key_correct: Sequence[bool],
    top1_attention_masses: Sequence[float],
    gold_attention_masses: Sequence[float],
    cache_value_correct: Sequence[bool],
    wrong_key: Sequence[bool],
    non_sink_read: Sequence[bool],
    selector_labels: Sequence[bool],
    selector_scores: Sequence[float],
    selector_positions: Sequence[int],
    survived: Sequence[bool],
    ages: Sequence[str],
    declared_ages: Sequence[str],
    attention_entropies: Sequence[float],
    effective_supports: Sequence[float],
    sink_masses: Sequence[float],
    cache_output_norms: Sequence[float],
    state_output_norms: Sequence[float],
    retained_counts: Sequence[float],
    evicted_counts: Sequence[float],
    selection_scores: Sequence[float],
    persistent_bytes: Sequence[float],
    block_bytes: Sequence[float],
    latencies_ms: Sequence[float],
    token_counts: Sequence[float],
    example_counts: Sequence[float],
    wall_seconds: Sequence[float],
    peak_vram_bytes: Sequence[float],
) -> tuple[MetricContribution, ...]:
    query_count = _same_length(
        "cache query diagnostics",
        (
            span_hits,
            top1_key_correct,
            top1_attention_masses,
            gold_attention_masses,
            cache_value_correct,
            wrong_key,
            non_sink_read,
            attention_entropies,
            effective_supports,
            sink_masses,
            cache_output_norms,
            state_output_norms,
        ),
    )
    _same_length(
        "cache selector diagnostics",
        (selector_labels, selector_scores, selector_positions),
    )
    _same_length("cache survival diagnostics", (survived, ages))
    _same_length("cache retention diagnostics", (retained_counts, evicted_counts))
    _same_length(
        "cache performance diagnostics",
        (
            persistent_bytes,
            block_bytes,
            latencies_ms,
            token_counts,
            example_counts,
            wall_seconds,
            peak_vram_bytes,
        ),
    )
    if any(
        type(value) is not bool
        for sequence in (
            span_hits,
            top1_key_correct,
            cache_value_correct,
            wrong_key,
            non_sink_read,
        )
        for value in sequence
    ):
        raise TypeError("cache query diagnostic outcomes must be bool")
    hit_indices = [index for index in range(query_count) if span_hits[index]]
    hit_conditional = [top1_key_correct[index] for index in hit_indices]
    hit_value = [cache_value_correct[index] for index in hit_indices]
    non_sink_wrong = [
        wrong_key[index] for index in range(query_count) if non_sink_read[index]
    ]

    if any(type(value) is not bool for value in survived):
        raise TypeError("survived entries must be bool")
    age_levels = tuple(declared_ages)
    if len(set(age_levels)) != len(age_levels):
        raise ValueError("declared ages must be unique")
    by_age: dict[str, list[bool]] = {level: [] for level in age_levels}
    for survived_item, age in zip(survived, ages, strict=True):
        if age not in by_age:
            raise ValueError("cache survival age is not declared")
        by_age[age].append(survived_item)

    retained = _nonnegative_values("retained count", retained_counts)
    evicted = _nonnegative_values("evicted count", evicted_counts)
    retained_total = math.fsum(retained)
    evicted_total = math.fsum(evicted)
    candidate_total = retained_total + evicted_total
    measured_wall_seconds = _positive_values("wall seconds", wall_seconds)
    contributions = [
        _rate_contribution("cache_span_hit_rate", span_hits),
        _rate_contribution("cache_top1_key_accuracy", top1_key_correct),
        _mean_contribution(
            "cache_top1_attention_mass",
            _probability_values("top1 attention mass", top1_attention_masses),
        ),
        _mean_contribution(
            "cache_gold_attention_mass",
            _probability_values("gold attention mass", gold_attention_masses),
        ),
        _rate_contribution("cache_conditional_read_accuracy", hit_conditional),
        _rate_contribution("cache_value_exact_match", hit_value),
        _rate_contribution("cache_wrong_key_rate", non_sink_wrong),
        selector_auprc_contribution(
            labels=selector_labels,
            scores=selector_scores,
            positions=selector_positions,
        ),
        _rate_contribution("cache_survival_rate", survived),
        _mean_contribution(
            "cache_attention_entropy",
            _nonnegative_values("attention entropy", attention_entropies),
        ),
        _mean_contribution(
            "cache_effective_support",
            _support_values(effective_supports),
        ),
        _mean_contribution(
            "cache_sink_mass", _probability_values("sink mass", sink_masses)
        ),
        _mean_contribution(
            "cache_output_norm",
            _nonnegative_values("cache output norm", cache_output_norms),
        ),
        _mean_contribution(
            "state_output_norm",
            _nonnegative_values("state output norm", state_output_norms),
        ),
        MetricContribution(
            "cache_retention_rate", retained_total, candidate_total
        ),
        MetricContribution("cache_eviction_rate", evicted_total, candidate_total),
        _mean_contribution(
            "cache_selection_score_mean",
            _nonnegative_values("selection score", selection_scores),
        ),
        _extreme_contribution("cache_selection_score_min", selection_scores),
        _extreme_contribution("cache_selection_score_max", selection_scores),
        _extreme_contribution("cache_persistent_bytes", persistent_bytes),
        _extreme_contribution("cache_block_bytes", block_bytes),
        _mean_contribution(
            "latency_ms", _nonnegative_values("latency", latencies_ms)
        ),
        MetricContribution(
            "throughput_tokens_per_second",
            math.fsum(_nonnegative_values("token count", token_counts)),
            math.fsum(measured_wall_seconds),
        ),
        MetricContribution(
            "throughput_examples_per_second",
            math.fsum(_nonnegative_values("example count", example_counts)),
            math.fsum(measured_wall_seconds),
        ),
        _extreme_contribution("peak_vram_bytes", peak_vram_bytes),
    ]
    contributions.extend(
        _rate_contribution("cache_survival_rate", values, (("age", age),))
        for age, values in by_age.items()
    )
    return tuple(contributions)


@dataclass(frozen=True)
class MetricSample:
    seed: int
    example_id: str
    budget_id: str
    numerator: float
    denominator: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("sample seed must be an int")
        for name in ("example_id", "budget_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"sample {name} must be a non-empty string")
        numerator = _finite_number("sample numerator", self.numerator)
        denominator = _finite_number("sample denominator", self.denominator)
        if denominator <= 0:
            raise ValueError("sample denominator must be positive")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    direction: int
    seed_count: int
    example_count: int
    resamples: int

    def __post_init__(self) -> None:
        for name in ("point", "lower", "upper"):
            object.__setattr__(
                self, name, _canonical_zero(_finite_number(name, getattr(self, name)))
            )
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("bootstrap direction must be +1 or -1")
        if not self.lower <= self.upper:
            raise ValueError("bootstrap lower must not exceed upper")
        for name in ("seed_count", "example_count", "resamples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"bootstrap {name} must be a positive int")

    def _canonical_dict(self) -> dict[str, float | int]:
        return {
            "direction": self.direction,
            "example_count": self.example_count,
            "lower": self.lower,
            "point": self.point,
            "resamples": self.resamples,
            "seed_count": self.seed_count,
            "upper": self.upper,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self._canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


@dataclass(frozen=True)
class FactorialBootstrapResult:
    interaction: BootstrapInterval
    current_effect: BootstrapInterval
    feature_off_effect: BootstrapInterval

    def __post_init__(self) -> None:
        intervals = (self.interaction, self.current_effect, self.feature_off_effect)
        if any(not isinstance(interval, BootstrapInterval) for interval in intervals):
            raise TypeError("factorial results must contain BootstrapInterval records")
        directions = {interval.direction for interval in intervals}
        if len(directions) != 1:
            raise ValueError("factorial interval directions must match")
        counts = {
            (interval.seed_count, interval.example_count, interval.resamples)
            for interval in intervals
        }
        if len(counts) != 1:
            raise ValueError("factorial interval counts and resamples must match")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "current_effect": self.current_effect._canonical_dict(),
                "feature_off_effect": self.feature_off_effect._canonical_dict(),
                "interaction": self.interaction._canonical_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def _index_samples(
    name: str, samples: Sequence[MetricSample]
) -> dict[tuple[int, str], MetricSample]:
    if not samples:
        raise ValueError(f"{name} samples must not be empty")
    indexed: dict[tuple[int, str], MetricSample] = {}
    for sample in samples:
        if not isinstance(sample, MetricSample):
            raise TypeError(f"{name} samples must be MetricSample records")
        identity = (sample.seed, sample.example_id)
        if identity in indexed:
            raise ValueError(f"{name} contains duplicate seed/example identity")
        indexed[identity] = sample
    return indexed


def _validate_matched_arms(
    arms: Mapping[str, Sequence[MetricSample]],
) -> tuple[
    dict[str, dict[tuple[int, str], MetricSample]],
    tuple[int, ...],
    dict[int, tuple[str, ...]],
]:
    indexed = {name: _index_samples(name, samples) for name, samples in arms.items()}
    names = tuple(indexed)
    reference = indexed[names[0]]
    identities = set(reference)
    for name in names[1:]:
        if set(indexed[name]) != identities:
            raise ValueError("arms must have exactly matched seed/example identities")
    for identity in identities:
        budgets = {indexed[name][identity].budget_id for name in names}
        if len(budgets) != 1:
            raise ValueError("matched observations must have identical budget IDs")
    seeds = tuple(sorted({seed for seed, _ in identities}))
    examples = {
        seed: tuple(sorted(example for item_seed, example in identities if item_seed == seed))
        for seed in seeds
    }
    return indexed, seeds, examples


def _ratio(
    arm: Mapping[tuple[int, str], MetricSample],
    seed: int,
    examples: Sequence[str],
) -> float:
    numerator = math.fsum(arm[(seed, example)].numerator for example in examples)
    denominator = math.fsum(arm[(seed, example)].denominator for example in examples)
    if denominator <= 0:
        raise ValueError("resampled metric denominator must be positive")
    return numerator / denominator


def _fixed_interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    span = len(ordered) - 1
    lower_index = math.floor(0.025 * span)
    upper_index = math.ceil(0.975 * span)
    return ordered[lower_index], ordered[upper_index]


def _validate_bootstrap_args(direction: int, random_seed: int, resamples: int) -> None:
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("bootstrap direction must be +1 or -1")
    if type(random_seed) is not int:
        raise TypeError("bootstrap random_seed must be an int")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("bootstrap resamples must be a positive int")


def paired_bootstrap(
    variant: Sequence[MetricSample],
    baseline: Sequence[MetricSample],
    *,
    direction: int,
    random_seed: int,
    resamples: int,
) -> BootstrapInterval:
    _validate_bootstrap_args(direction, random_seed, resamples)
    indexed, seeds, examples = _validate_matched_arms(
        {"baseline": baseline, "variant": variant}
    )
    per_seed = [
        direction
        * (
            _ratio(indexed["variant"], seed, examples[seed])
            - _ratio(indexed["baseline"], seed, examples[seed])
        )
        for seed in seeds
    ]
    point = _canonical_zero(math.fsum(per_seed) / len(per_seed))
    rng = random.Random(random_seed)
    draws: list[float] = []
    for _ in range(resamples):
        effects: list[float] = []
        for sampled_seed in (rng.choice(seeds) for _ in seeds):
            seed_examples = examples[sampled_seed]
            sampled_examples = tuple(
                rng.choice(seed_examples) for _ in seed_examples
            )
            effects.append(
                direction
                * (
                    _ratio(indexed["variant"], sampled_seed, sampled_examples)
                    - _ratio(indexed["baseline"], sampled_seed, sampled_examples)
                )
            )
        draws.append(_canonical_zero(math.fsum(effects) / len(effects)))
    lower, upper = _fixed_interval(draws)
    return BootstrapInterval(
        point,
        lower,
        upper,
        direction,
        len(seeds),
        sum(len(items) for items in examples.values()),
        resamples,
    )


def paired_factorial_bootstrap(
    cells: Mapping[str, Sequence[MetricSample]],
    *,
    direction: int,
    random_seed: int,
    resamples: int,
) -> FactorialBootstrapResult:
    _validate_bootstrap_args(direction, random_seed, resamples)
    required = {"M00", "M10", "M01", "M11"}
    if set(cells) != required:
        raise ValueError("factorial bootstrap requires all four factorial cells")
    ordered_cells = {name: cells[name] for name in sorted(required)}
    indexed, seeds, examples = _validate_matched_arms(ordered_cells)

    def contrasts(seed: int, selected: Sequence[str]) -> tuple[float, float, float]:
        values = {
            name: _ratio(indexed[name], seed, selected) for name in required
        }
        interaction = direction * (
            values["M11"] - values["M10"] - values["M01"] + values["M00"]
        )
        current = direction * (values["M11"] - values["M01"])
        feature_off = direction * (values["M10"] - values["M00"])
        return interaction, current, feature_off

    points_by_seed = [contrasts(seed, examples[seed]) for seed in seeds]
    points = tuple(
        _canonical_zero(math.fsum(items[index] for items in points_by_seed) / len(seeds))
        for index in range(3)
    )
    rng = random.Random(random_seed)
    draws: tuple[list[float], list[float], list[float]] = ([], [], [])
    for _ in range(resamples):
        sampled_contrasts: list[tuple[float, float, float]] = []
        for sampled_seed in (rng.choice(seeds) for _ in seeds):
            seed_examples = examples[sampled_seed]
            selected = tuple(rng.choice(seed_examples) for _ in seed_examples)
            sampled_contrasts.append(contrasts(sampled_seed, selected))
        for index in range(3):
            draws[index].append(
                _canonical_zero(
                    math.fsum(item[index] for item in sampled_contrasts) / len(seeds)
                )
            )
    count = sum(len(items) for items in examples.values())
    intervals = []
    for index, point in enumerate(points):
        lower, upper = _fixed_interval(draws[index])
        intervals.append(
            BootstrapInterval(
                point, lower, upper, direction, len(seeds), count, resamples
            )
        )
    return FactorialBootstrapResult(*intervals)


@dataclass(frozen=True)
class ProtectedEffect:
    metric: str
    interval: BootstrapInterval
    max_regression: float

    def __post_init__(self) -> None:
        direction = metric_direction(self.metric)
        if direction is None:
            raise ValueError("diagnostic metric has no normalized decision direction")
        if not isinstance(self.interval, BootstrapInterval):
            raise TypeError("protected interval must be a BootstrapInterval")
        if self.interval.direction != direction:
            raise ValueError("protected interval direction does not match metric")
        maximum = _finite_number("max_regression", self.max_regression)
        if maximum < 0:
            raise ValueError("max_regression must be nonnegative")
        object.__setattr__(self, "max_regression", maximum)


def _decision_interval(
    metric: str, interval: BootstrapInterval | None
) -> int:
    direction = metric_direction(metric)
    if direction is None:
        raise ValueError("diagnostic metric has no normalized decision direction")
    if interval is not None:
        if not isinstance(interval, BootstrapInterval):
            raise TypeError("decision effect must be a BootstrapInterval or None")
        if interval.direction != direction:
            raise ValueError("decision interval direction does not match metric")
    return direction


def classify_addition(
    *,
    metric: str,
    primary: BootstrapInterval | None,
    protected: Iterable[ProtectedEffect],
    valid: bool,
    min_useful: float,
    harm_threshold: float,
    min_synergy: float,
    interaction: BootstrapInterval | None = None,
    existing_feature_off: BootstrapInterval | None = None,
) -> str:
    _decision_interval(metric, primary)
    _decision_interval(metric, interaction)
    _decision_interval(metric, existing_feature_off)
    if type(valid) is not bool:
        raise TypeError("valid must be bool")
    useful = _finite_number("min_useful", min_useful)
    harm = _finite_number("harm_threshold", harm_threshold)
    synergy = _finite_number("min_synergy", min_synergy)
    if useful <= 0 or harm <= 0 or synergy <= 0:
        raise ValueError("addition thresholds must be positive")
    protected_effects = tuple(protected)
    if any(not isinstance(item, ProtectedEffect) for item in protected_effects):
        raise TypeError("protected effects must be ProtectedEffect records")
    if not valid or primary is None:
        return "failed/invalid"
    protected_harm = any(
        item.interval.upper < -item.max_regression for item in protected_effects
    )
    if primary.upper <= -harm or protected_harm:
        return "harmful"
    protected_safe = all(
        item.interval.lower >= -item.max_regression for item in protected_effects
    )
    if interaction is not None and protected_safe and interaction.lower >= synergy:
        return "synergistic"
    if primary.lower >= useful and protected_safe:
        return "incremental"
    if (
        existing_feature_off is not None
        and existing_feature_off.lower >= useful
        and protected_safe
    ):
        return "replacement-only"
    if primary.upper < useful and primary.lower > -harm:
        return "redundant"
    return "inconclusive"


def classify_reliance(
    *,
    metric: str,
    effect: BootstrapInterval | None,
    valid: bool,
    min_reliance: float,
    equivalence: float,
    harm_threshold: float,
) -> str:
    _decision_interval(metric, effect)
    if type(valid) is not bool:
        raise TypeError("valid must be bool")
    minimum = _finite_number("min_reliance", min_reliance)
    band = _finite_number("equivalence", equivalence)
    harm = _finite_number("harm_threshold", harm_threshold)
    if band < 0 or minimum <= band or harm <= band:
        raise ValueError(
            "reliance thresholds require min_reliance and harm_threshold "
            "greater than nonnegative equivalence"
        )
    if not valid or effect is None:
        return "failed/invalid"
    if effect.upper <= -harm:
        return "harmful-current"
    if effect.lower >= minimum:
        return "relied-on"
    if effect.lower >= -band and effect.upper <= band:
        return "dispensable"
    return "inconclusive-reliance"


OPTION3_REJECTION_CODES = (
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

_OPTION3_LONG_CELLS = {"16k_4q", "16k_8q", "32k_4q", "32k_8q"}
_OPTION3_SHORT_CELLS = {"512", "1k", "2k", "4k"}


@dataclass(frozen=True)
class NamedInterval:
    name: str
    interval: BootstrapInterval

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("named interval name must be a non-empty string")
        if not isinstance(self.interval, BootstrapInterval):
            raise TypeError("named interval must contain a BootstrapInterval")


@dataclass(frozen=True)
class Option3Thresholds:
    long_macro_lcb: float = 0.10
    long_cell_lcb: float = 0.10
    min_long_cells: int = 2
    surprise_recency_lcb: float = 0.05
    short_macro_lcb: float = -0.02
    short_cell_lcb: float = -0.03
    eight_k_lcb: float = -0.03
    episode_exact_lcb: float = -0.05
    freshness_lcb: float = -0.02
    ce_ucb: float = 0.02
    kl_absolute_ucb: float = 0.005
    kl_native_fraction: float = 0.05
    capacity_w64_lcb: float = 0.10
    capacity_adjacent_lcb: float = 0.05
    min_gate_mean: float = 0.005
    min_gate_max: float = 0.02
    min_persistent_hit: float = 0.25
    min_conditional_read: float = 0.50
    min_shuffle_drop: float = 0.05
    min_decode_throughput: float = 0.80
    min_prefill_throughput: float = 0.75

    def __post_init__(self) -> None:
        if (
            type(self.min_long_cells) is not int
            or not 1 <= self.min_long_cells <= len(_OPTION3_LONG_CELLS)
        ):
            raise ValueError("min_long_cells must be an int from 1 through 4")
        effect_fields = (
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
        )
        nonnegative_fields = (
            "ce_ucb",
            "kl_absolute_ucb",
            "kl_native_fraction",
            "min_decode_throughput",
            "min_prefill_throughput",
        )
        probability_fields = (
            "min_gate_mean",
            "min_gate_max",
            "min_persistent_hit",
            "min_conditional_read",
            "min_shuffle_drop",
        )
        for name in (*effect_fields, *nonnegative_fields, *probability_fields):
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))
        if any(not -1 <= getattr(self, name) <= 1 for name in effect_fields):
            raise ValueError("effect thresholds must be in [-1,1]")
        if any(getattr(self, name) < 0 for name in nonnegative_fields):
            raise ValueError("loss and throughput thresholds must be nonnegative")
        if any(not 0 <= getattr(self, name) <= 1 for name in probability_fields):
            raise ValueError("gate and retrieval thresholds must be in [0,1]")
        if self.min_gate_mean > self.min_gate_max:
            raise ValueError("minimum gate mean must not exceed minimum gate max")


@dataclass(frozen=True)
class Option3Evidence:
    branch: str
    valid: bool
    mechanism_unchanged: bool
    long_macro: BootstrapInterval | None
    long_cells: tuple[NamedInterval, ...]
    surprise_vs_recency: BootstrapInterval | None
    short_macro: BootstrapInterval | None
    short_cells: tuple[NamedInterval, ...]
    eight_k: BootstrapInterval | None
    episode_exact: BootstrapInterval | None
    freshness_latest_native: BootstrapInterval | None
    freshness_latest_recency: BootstrapInterval | None
    freshness_stale_native: BootstrapInterval | None
    freshness_stale_recency: BootstrapInterval | None
    ce_delta: BootstrapInterval | None
    kl_delta: BootstrapInterval | None
    mean_kl_native: float
    nonfinite_count: int
    skipped_steps: int
    capacity_w64: BootstrapInterval | None
    capacity_w32: BootstrapInterval | None
    capacity_w128: BootstrapInterval | None
    requires_width_above_128: bool
    amplitudes: tuple[float, ...]
    persistent_hit: BootstrapInterval | None
    conditional_read: BootstrapInterval | None
    shuffle_drop: BootstrapInterval | None
    persistent_bytes: int
    persistent_bytes_limit: int
    decode_throughput_ratio: float
    prefill_throughput_ratio: float
    dynamic_memory_flat: bool

    def __post_init__(self) -> None:
        if self.branch not in {"surprise", "recency"}:
            raise ValueError("Option3 branch must be surprise or recency")
        for name in (
            "valid",
            "mechanism_unchanged",
            "requires_width_above_128",
            "dynamic_memory_flat",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in ("nonfinite_count", "skipped_steps", "persistent_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative int")
        if type(self.persistent_bytes_limit) is not int or self.persistent_bytes_limit < 1:
            raise ValueError("persistent_bytes_limit must be a positive int")
        object.__setattr__(
            self, "mean_kl_native", _finite_number("mean_kl_native", self.mean_kl_native)
        )
        if self.mean_kl_native < 0:
            raise ValueError("mean_kl_native must be nonnegative")
        for name in ("decode_throughput_ratio", "prefill_throughput_ratio"):
            value = _finite_number(name, getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        amplitudes = tuple(
            _finite_number("cache amplitude", value) for value in self.amplitudes
        )
        if any(not 0 <= value <= 1 for value in amplitudes):
            raise ValueError("cache amplitudes must be in [0,1]")
        object.__setattr__(self, "amplitudes", amplitudes)
        interval_directions = {
            "long_macro": 1,
            "surprise_vs_recency": 1,
            "short_macro": 1,
            "eight_k": 1,
            "episode_exact": 1,
            "freshness_latest_native": 1,
            "freshness_latest_recency": 1,
            "freshness_stale_native": -1,
            "freshness_stale_recency": -1,
            "ce_delta": 1,
            "kl_delta": 1,
            "capacity_w64": 1,
            "capacity_w32": 1,
            "capacity_w128": 1,
            "persistent_hit": 1,
            "conditional_read": 1,
            "shuffle_drop": 1,
        }
        for name, expected_direction in interval_directions.items():
            interval = getattr(self, name)
            if interval is None:
                continue
            if not isinstance(interval, BootstrapInterval):
                raise TypeError(f"{name} must be a BootstrapInterval or None")
            if interval.direction != expected_direction:
                raise ValueError(
                    f"{name} interval direction must be {expected_direction:+d}"
                )
        for field_name in ("long_cells", "short_cells"):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, NamedInterval) for value in values):
                raise TypeError(f"{field_name} must contain NamedInterval records")
            if any(value.interval.direction != 1 for value in values):
                raise ValueError(f"{field_name} interval direction must be +1")
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True)
class Option3BranchResult:
    branch: str
    passed: bool
    rejection_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.branch not in {"surprise", "recency"}:
            raise ValueError("Option3 result branch must be surprise or recency")
        if type(self.passed) is not bool:
            raise TypeError("Option3 result passed must be bool")
        if type(self.rejection_codes) is not tuple:
            raise TypeError("Option3 rejection codes must be a tuple")
        if any(code not in OPTION3_REJECTION_CODES for code in self.rejection_codes):
            raise ValueError("Option3 rejection codes must use the fixed registry")
        if len(set(self.rejection_codes)) != len(self.rejection_codes):
            raise ValueError("Option3 rejection codes must be unique")
        code_order = {code: index for index, code in enumerate(OPTION3_REJECTION_CODES)}
        if self.rejection_codes != tuple(
            sorted(self.rejection_codes, key=code_order.__getitem__)
        ):
            raise ValueError("Option3 rejection codes must use canonical order")
        if self.passed != (not self.rejection_codes):
            raise ValueError("Option3 passed must be equivalent to having no rejection codes")


@dataclass(frozen=True)
class Option3Decision:
    selected_branch: str
    surprise: Option3BranchResult
    recency: Option3BranchResult

    def __post_init__(self) -> None:
        if self.selected_branch not in {"surprise", "recency", "no_promote"}:
            raise ValueError("Option3 selected branch is invalid")
        if not isinstance(self.surprise, Option3BranchResult) or not isinstance(
            self.recency, Option3BranchResult
        ):
            raise TypeError("Option3 decisions require branch result records")
        if self.surprise.branch != "surprise" or self.recency.branch != "recency":
            raise ValueError("Option3 decision result branches are inconsistent")
        expected = (
            "surprise"
            if self.surprise.passed
            else "recency"
            if self.recency.passed
            else "no_promote"
        )
        if self.selected_branch != expected:
            raise ValueError("Option3 selection does not follow branch precedence")


def _named_interval_map(
    values: Sequence[NamedInterval], expected: set[str]
) -> dict[str, BootstrapInterval] | None:
    if any(not isinstance(value, NamedInterval) for value in values):
        return None
    names = [value.name for value in values]
    if len(set(names)) != len(names) or set(names) != expected:
        return None
    return {value.name: value.interval for value in values}


def evaluate_option3_branch(
    evidence: Option3Evidence, thresholds: Option3Thresholds
) -> Option3BranchResult:
    if not isinstance(evidence, Option3Evidence):
        raise TypeError("evidence must be Option3Evidence")
    if not isinstance(thresholds, Option3Thresholds):
        raise TypeError("thresholds must be Option3Thresholds")
    codes: list[str] = []
    if not evidence.valid:
        codes.append("invalid_evidence")
    if not evidence.mechanism_unchanged:
        codes.append("mechanism_changed")
    long_cells = _named_interval_map(evidence.long_cells, _OPTION3_LONG_CELLS)
    short_cells = _named_interval_map(evidence.short_cells, _OPTION3_SHORT_CELLS)
    if long_cells is None or short_cells is None:
        codes.append("missing_required_cells")
        return Option3BranchResult(evidence.branch, False, tuple(codes))
    required_intervals = (
        evidence.long_macro,
        evidence.short_macro,
        evidence.eight_k,
        evidence.episode_exact,
        evidence.freshness_latest_native,
        evidence.freshness_latest_recency,
        evidence.freshness_stale_native,
        evidence.freshness_stale_recency,
        evidence.ce_delta,
        evidence.kl_delta,
        evidence.capacity_w64,
        evidence.capacity_w32,
        evidence.capacity_w128,
        evidence.persistent_hit,
        evidence.conditional_read,
        evidence.shuffle_drop,
    )
    if (
        any(value is None for value in required_intervals)
        or not evidence.amplitudes
        or (evidence.branch == "surprise" and evidence.surprise_vs_recency is None)
    ):
        codes.append("missing_required_evidence")
        return Option3BranchResult(evidence.branch, False, tuple(codes))
    assert evidence.long_macro is not None
    assert evidence.short_macro is not None
    assert evidence.eight_k is not None
    assert evidence.episode_exact is not None
    assert evidence.freshness_latest_native is not None
    assert evidence.freshness_latest_recency is not None
    assert evidence.freshness_stale_native is not None
    assert evidence.freshness_stale_recency is not None
    assert evidence.ce_delta is not None
    assert evidence.kl_delta is not None
    assert evidence.capacity_w64 is not None
    assert evidence.capacity_w32 is not None
    assert evidence.capacity_w128 is not None
    assert evidence.persistent_hit is not None
    assert evidence.conditional_read is not None
    assert evidence.shuffle_drop is not None

    if evidence.long_macro.lower < thresholds.long_macro_lcb:
        codes.append("long_macro_lcb")
    if (
        sum(
            interval.lower >= thresholds.long_cell_lcb
            for interval in long_cells.values()
        )
        < thresholds.min_long_cells
    ):
        codes.append("long_cell_lcb_count")
    if evidence.branch == "surprise":
        assert evidence.surprise_vs_recency is not None
        if evidence.surprise_vs_recency.lower < thresholds.surprise_recency_lcb:
            codes.append("surprise_vs_recency_lcb")
    if evidence.short_macro.lower < thresholds.short_macro_lcb:
        codes.append("short_macro_lcb")
    if any(
        interval.lower < thresholds.short_cell_lcb
        for interval in short_cells.values()
    ):
        codes.append("short_cell_lcb")
    if evidence.eight_k.lower < thresholds.eight_k_lcb:
        codes.append("eight_k_lcb")
    if evidence.episode_exact.lower < thresholds.episode_exact_lcb:
        codes.append("episode_exact_lcb")
    for field_name, code in (
        ("freshness_latest_native", "freshness_latest_native_lcb"),
        ("freshness_latest_recency", "freshness_latest_recency_lcb"),
        ("freshness_stale_native", "freshness_stale_native_lcb"),
        ("freshness_stale_recency", "freshness_stale_recency_lcb"),
    ):
        interval = getattr(evidence, field_name)
        assert isinstance(interval, BootstrapInterval)
        if interval.lower < thresholds.freshness_lcb:
            codes.append(code)
    if evidence.nonfinite_count or evidence.skipped_steps:
        codes.append("nonfinite_or_skipped")
    if evidence.ce_delta.upper > thresholds.ce_ucb:
        codes.append("ce_ucb")
    kl_limit = max(
        thresholds.kl_absolute_ucb,
        thresholds.kl_native_fraction * evidence.mean_kl_native,
    )
    if evidence.kl_delta.upper > kl_limit:
        codes.append("kl_ucb")
    if evidence.capacity_w64.lower < thresholds.capacity_w64_lcb:
        codes.append("capacity_w64_lcb")
    if max(evidence.capacity_w32.lower, evidence.capacity_w128.lower) < thresholds.capacity_adjacent_lcb:
        codes.append("capacity_adjacent_lcb")
    if evidence.requires_width_above_128:
        codes.append("capacity_above_128")
    gate_mean = math.fsum(evidence.amplitudes) / len(evidence.amplitudes)
    if gate_mean < thresholds.min_gate_mean:
        codes.append("gate_mean")
    if max(evidence.amplitudes) < thresholds.min_gate_max:
        codes.append("gate_max")
    if evidence.persistent_hit.lower < thresholds.min_persistent_hit:
        codes.append("persistent_hit_lcb")
    if evidence.conditional_read.lower < thresholds.min_conditional_read:
        codes.append("conditional_read_lcb")
    if evidence.shuffle_drop.lower < thresholds.min_shuffle_drop:
        codes.append("shuffle_drop_lcb")
    if evidence.persistent_bytes > evidence.persistent_bytes_limit:
        codes.append("persistent_memory_limit")
    if evidence.decode_throughput_ratio < thresholds.min_decode_throughput:
        codes.append("decode_throughput")
    if evidence.prefill_throughput_ratio < thresholds.min_prefill_throughput:
        codes.append("prefill_throughput")
    if not evidence.dynamic_memory_flat:
        codes.append("dynamic_memory_not_flat")
    return Option3BranchResult(evidence.branch, not codes, tuple(codes))


def decide_option3(
    surprise: Option3Evidence,
    recency: Option3Evidence,
    thresholds: Option3Thresholds,
) -> Option3Decision:
    if surprise.branch != "surprise" or recency.branch != "recency":
        raise ValueError("Option3 decision requires surprise then recency evidence")
    surprise_result = evaluate_option3_branch(surprise, thresholds)
    recency_result = evaluate_option3_branch(recency, thresholds)
    selected = (
        "surprise"
        if surprise_result.passed
        else "recency"
        if recency_result.passed
        else "no_promote"
    )
    return Option3Decision(selected, surprise_result, recency_result)


__all__ = [
    "METRIC_REGISTRY",
    "BootstrapInterval",
    "FactorialBootstrapResult",
    "MetricContribution",
    "MetricSample",
    "MetricSpec",
    "MetricValue",
    "NamedInterval",
    "OPTION3_REJECTION_CODES",
    "Option3BranchResult",
    "Option3Decision",
    "Option3Evidence",
    "Option3Thresholds",
    "ProtectedEffect",
    "accumulate_metrics",
    "affine_contributions",
    "binding_mqar_contributions",
    "cache_diagnostic_contributions",
    "classify_addition",
    "classify_reliance",
    "classification_contributions",
    "drift_contributions",
    "decide_option3",
    "evaluate_option3_branch",
    "freshness_contributions",
    "integration_contributions",
    "metric_direction",
    "paired_bootstrap",
    "paired_factorial_bootstrap",
    "selector_auprc_contribution",
    "structured_contributions",
    "trajectory_contributions",
]
