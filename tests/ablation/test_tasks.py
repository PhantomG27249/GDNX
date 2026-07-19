from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from types import MappingProxyType

import pytest
import torch

from research.kmd2_ablation.tasks import (
    TASK_NAMES,
    TASK_SCHEMA_VERSION,
    EpisodeBatch,
    generate_task,
)
from research.kmd2_ablation.tasks.state_tracking import (
    MODULAR_TOKENS,
    PARITY_TOKENS,
    STATE_TRACKING_SCHEMA_VERSION,
    STATE_TRACKING_TOKENS_PER_QUERY,
    TOGGLE_TOKENS,
)
from research.kmd2_ablation.tasks.integration import (
    INTEGRATION_DECAY_RANGE,
    INTEGRATION_DELTA_VALUES,
    INTEGRATION_FORCING_RANGE,
    INTEGRATION_RK4_STEPS,
    INTEGRATION_SCHEMA_VERSION,
    rk4_piecewise_linear_oracle,
)
from research.kmd2_ablation.tasks.dynamics import (
    DRIFT_SLOPE_RANGE,
    DYNAMICS_SCHEMA_VERSION,
    DYNAMICS_TOKENS_PER_STEP,
    DYNAMICS_WARMUP_OBSERVATIONS,
    TRAJECTORY_FREQUENCY_RANGE,
)
from research.kmd2_ablation.tasks.local_binding import (
    LOCAL_BINDING_DEFAULT_WIDTH,
    LOCAL_BINDING_MODES,
    LOCAL_BINDING_SCHEMA_VERSION,
    LOCAL_BINDING_TOKENS,
    LOCAL_BINDING_VOCAB_VERSION,
)
from research.kmd2_ablation.tasks.mqar import (
    MQAR_DEFAULT_WIDTH,
    MQAR_LOAD_FACTORS,
    MQAR_OVERWRITE_FRACTION,
    MQAR_SCHEMA_VERSION,
    MQAR_TOKENS,
)
from research.kmd2_ablation.tasks.structured import (
    STRUCTURED_EXCEPTION_FRACTION,
    STRUCTURED_RULE_MODULUS,
    STRUCTURED_SCHEMA_VERSION,
    STRUCTURED_TOKENS,
)
from research.kmd2_ablation.tasks.far_surprise import (
    FAR_SURPRISE_DEFAULT_WIDTH,
    FAR_SURPRISE_DISTRACTOR_MARGIN,
    FAR_SURPRISE_SCHEMA_VERSION,
    FAR_SURPRISE_TOKENS,
)
from research.kmd2_ablation.tasks.freshness import (
    FRESHNESS_DEFAULT_WIDTH,
    FRESHNESS_SCHEMA_VERSION,
    FRESHNESS_STALE_SCORE,
    FRESHNESS_TOKENS,
)
from research.kmd2_ablation.tasks.affine import (
    AFFINE_DEFAULT_INPUT_DIM,
    AFFINE_DEFAULT_OUTPUT_DIM,
    AFFINE_SCHEMA_VERSION,
)


EXPECTED_TASK_NAMES = frozenset(
    {
        "affine_associative_regression",
        "drift_reversal",
        "far_surprise",
        "freshness",
        "irregular_integration",
        "local_binding",
        "modular_counter",
        "mqar",
        "parity",
        "state_tracking",
        "structured_exceptions",
        "toggle_fsm",
        "trajectory",
    }
)

EXPECTED_FOCUSED_TEST_MARKERS = {
    "affine_associative_regression": "test_affine_",
    "drift_reversal": "test_drift_reversal_",
    "far_surprise": "test_far_surprise_",
    "freshness": "test_freshness_",
    "irregular_integration": "test_irregular_integration_",
    "local_binding": "test_local_binding_",
    "modular_counter": "test_state_tracking_exact_ops_and_ood_modular_counter",
    "mqar": "test_mqar_",
    "parity": "test_state_tracking_exact_ops_and_ood_parity",
    "state_tracking": "test_state_tracking_dispatcher_",
    "structured_exceptions": "test_structured_exceptions_",
    "toggle_fsm": "test_state_tracking_exact_ops_and_ood_toggle_fsm",
    "trajectory": "test_trajectory_",
}


def _minimal_episode(**overrides: object) -> EpisodeBatch:
    values: dict[str, object] = {
        "task": "parity",
        "split": "train",
        "seed": 7,
        "example_ids": ("example-0", "example-1"),
        "input_ids": torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64),
        "continuous_inputs": None,
        "direct_factors": None,
        "targets": torch.tensor([[-100, -100, 0], [-100, -100, 1]]),
        "valid": torch.ones(2, 3, dtype=torch.bool),
        "positions": torch.arange(3, dtype=torch.int64).repeat(2, 1),
        "loss_mask": torch.tensor([[False, False, True], [False, False, True]]),
        "query_mask": torch.tensor([[False, False, True], [False, False, True]]),
        "boundaries": torch.tensor([[True, False, False], [True, False, False]]),
        "source_spans": torch.tensor(
            [[[-1, -1], [-1, -1], [0, 1]], [[-1, -1], [-1, -1], [0, 1]]],
            dtype=torch.int64,
        ),
        "strata": {"kind": torch.zeros(2, 3, dtype=torch.int64)},
        "metadata": (
            {"logical_length": 3, "actual_length": 3},
            {"logical_length": 3, "actual_length": 3},
        ),
    }
    values.update(overrides)
    return EpisodeBatch(**values)  # type: ignore[arg-type]


def test_episode_contract_and_registry() -> None:
    assert TASK_SCHEMA_VERSION == "1.0.0"
    assert TASK_NAMES == EXPECTED_TASK_NAMES

    original = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
    episode = _minimal_episode(input_ids=original)
    original.zero_()

    assert episode.input_ids is not None
    assert episode.input_ids.tolist() == [[1, 2, 3], [3, 2, 1]]
    assert isinstance(episode.strata, MappingProxyType)
    assert isinstance(episode.metadata[0], MappingProxyType)
    json.dumps([dict(item) for item in episode.metadata])
    with pytest.raises(Exception):
        episode.seed = 8  # type: ignore[misc]

    with pytest.raises(ValueError, match="unknown task.*not-a-task"):
        generate_task("not-a-task", 1, 4, 0, "train", {})
    with pytest.raises(ValueError, match="RULER.*Qwen"):
        generate_task("ruler", 1, 4, 0, "train", {})

    with pytest.raises(ValueError, match="task.*registered"):
        _minimal_episode(task="not-a-registered-task")


def test_all_named_task_families_have_collected_focused_tests() -> None:
    collected_test_names = {
        name for name, value in globals().items() if name.startswith("test_") and callable(value)
    }
    assert EXPECTED_FOCUSED_TEST_MARKERS.keys() == EXPECTED_TASK_NAMES
    for task_name, marker in EXPECTED_FOCUSED_TEST_MARKERS.items():
        assert any(name.startswith(marker) for name in collected_test_names), task_name


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("continuous_inputs", torch.zeros(2, 3, 1), "exactly one input modality"),
        ("targets", torch.zeros(2, 2, dtype=torch.int64), "targets"),
        ("loss_mask", torch.tensor([[False, True, False]] * 2), "loss_mask"),
        ("query_mask", torch.tensor([[False, False, False]] * 2), "loss_mask"),
        ("valid", torch.tensor([[True, True, False]] * 2), "query_mask"),
        ("positions", torch.tensor([[0, 2, 1]] * 2), "positions"),
        (
            "source_spans",
            torch.tensor([[[0, 1], [-1, -1], [3, 2]]] * 2),
            "source_spans",
        ),
    ],
)
def test_episode_contract_rejects_invalid_records(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _minimal_episode(**{field: value})


def test_episode_contract_supports_continuous_and_direct_factor_modalities() -> None:
    continuous = _minimal_episode(
        input_ids=None,
        continuous_inputs=torch.zeros(2, 3, 4, dtype=torch.float64),
        targets=torch.zeros(2, 3, 2, dtype=torch.float64),
    )
    assert continuous.continuous_inputs is not None
    assert continuous.continuous_inputs.dtype == torch.float64

    factors = {
        "q": torch.zeros(2, 3, 1, 1, 2),
        "k": torch.zeros(2, 3, 1, 1, 2),
    }
    direct = _minimal_episode(input_ids=None, direct_factors=factors)
    assert isinstance(direct.direct_factors, MappingProxyType)
    assert direct.direct_factors is not None
    assert direct.direct_factors["q"].data_ptr() != factors["q"].data_ptr()


def test_episode_contract_rejects_non_json_metadata_and_tensor_aliases() -> None:
    with pytest.raises(TypeError, match="metadata.*JSON"):
        _minimal_episode(metadata=({"bad": object()}, {"ok": True}))

    shared = torch.zeros(2, 3, dtype=torch.int64)
    with pytest.raises(ValueError, match="direct_factors.*alias"):
        _minimal_episode(
            input_ids=None,
            direct_factors={"q": shared, "k": shared},
        )


def test_episode_metadata_is_deeply_immutable_and_json_serializable() -> None:
    original = {
        "nested": {"items": [1, {"value": 2}]},
        "logical_length": 3,
        "actual_length": 3,
    }
    episode = _minimal_episode(metadata=(original, original))
    assert isinstance(episode.metadata[0], MappingProxyType)
    encoded = json.dumps([dict(item) for item in episode.metadata], sort_keys=True)
    assert '"value": 2' in encoded

    with pytest.raises(TypeError):
        episode.metadata[0]["nested"]["new"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        episode.metadata[0]["nested"]["items"][1]["value"] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        episode.metadata[0]["nested"]["items"][0] = 9  # type: ignore[index]
    original["nested"]["items"][1]["value"] = 99  # type: ignore[index]
    assert episode.metadata[0]["nested"]["items"][1]["value"] == 2

    nested = episode.metadata[0]["nested"]
    with pytest.raises(TypeError, match="immutable"):
        nested |= {"new": 3}
    assert "new" not in nested

    with pytest.raises(TypeError, match="immutable"):
        nested.__init__({"replacement": True})
    assert "replacement" not in nested
    assert nested["items"][1]["value"] == 2


def test_example_identity_includes_canonical_validated_params_and_task_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mqar_a = generate_task(
        "mqar",
        2,
        4,
        113,
        "train",
        {"width": 8, "overwrite_fraction": 0.25},
    )
    mqar_reordered = generate_task(
        "mqar",
        2,
        4,
        113,
        "train",
        {"overwrite_fraction": 0.25, "width": 8},
    )
    assert mqar_a.example_ids == mqar_reordered.example_ids
    assert mqar_a.example_ids != generate_task(
        "mqar", 2, 4, 113, "train", {"width": 16, "overwrite_fraction": 0.25}
    ).example_ids

    assert generate_task(
        "modular_counter", 2, 5, 127, "train", {"modulus": 5}
    ).example_ids != generate_task(
        "modular_counter", 2, 5, 127, "train", {"modulus": 7}
    ).example_ids
    assert generate_task(
        "affine_associative_regression",
        2,
        4,
        131,
        "train",
        {"input_dim": 3, "output_dim": 2},
    ).example_ids != generate_task(
        "affine_associative_regression",
        2,
        4,
        131,
        "train",
        {"input_dim": 4, "output_dim": 2},
    ).example_ids

    from research.kmd2_ablation.tasks import state_tracking

    before = generate_task("parity", 2, 5, 137, "train", {}).example_ids
    monkeypatch.setattr(state_tracking, "STATE_TRACKING_SCHEMA_VERSION", "1.0.1")
    after = generate_task("parity", 2, 5, 137, "train", {}).example_ids
    assert before != after


def _assert_batch_equal(left: EpisodeBatch, right: EpisodeBatch) -> None:
    assert left.task == right.task
    assert left.split == right.split
    assert left.seed == right.seed
    assert left.example_ids == right.example_ids
    for name in (
        "input_ids",
        "continuous_inputs",
        "targets",
        "valid",
        "positions",
        "loss_mask",
        "query_mask",
        "boundaries",
        "source_spans",
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None:
            assert right_value is None
        else:
            assert torch.equal(left_value, right_value), name
    if left.direct_factors is None:
        assert right.direct_factors is None
    else:
        assert right.direct_factors is not None
        assert left.direct_factors.keys() == right.direct_factors.keys()
        for name in left.direct_factors:
            assert torch.equal(left.direct_factors[name], right.direct_factors[name]), name
    assert left.strata.keys() == right.strata.keys()
    for name in left.strata:
        assert torch.equal(left.strata[name], right.strata[name]), name
    assert tuple(dict(item) for item in left.metadata) == tuple(
        dict(item) for item in right.metadata
    )


def _query_labels(batch: EpisodeBatch) -> list[int]:
    return batch.targets[batch.query_mask].tolist()


def test_state_tracking_exact_ops_and_ood_parity() -> None:
    assert STATE_TRACKING_SCHEMA_VERSION == "1.0.0"
    assert STATE_TRACKING_TOKENS_PER_QUERY == 3
    assert dict(PARITY_TOKENS) == {
        "PAD": 0,
        "HOLD": 1,
        "ONE": 2,
        "QUERY": 3,
    }

    batch = generate_task("parity", 3, 8, 19, "train", {})
    repeated = generate_task("parity", 3, 8, 19, "train", {})
    _assert_batch_equal(batch, repeated)
    assert batch.input_ids is not None

    for example in range(3):
        state = 0
        segment_start = 0
        for token_index, token in enumerate(batch.input_ids[example].tolist()):
            if batch.boundaries[example, token_index]:
                state = 0
                segment_start = token_index
            if token == PARITY_TOKENS["ONE"]:
                state ^= 1
            elif token not in {
                PARITY_TOKENS["HOLD"],
                PARITY_TOKENS["QUERY"],
            }:
                raise AssertionError(f"unexpected parity token {token}")
            if batch.query_mask[example, token_index]:
                assert token == PARITY_TOKENS["QUERY"]
                assert batch.targets[example, token_index].item() == state
                assert batch.source_spans[example, token_index].tolist() == [
                    segment_start,
                    token_index,
                ]

    labels = _query_labels(batch)
    assert abs(labels.count(0) - labels.count(1)) <= 1
    assert PARITY_TOKENS["HOLD"] in batch.input_ids
    assert PARITY_TOKENS["QUERY"] in batch.input_ids

    two_x = generate_task("parity", 3, 8, 19, "ood_2x", {})
    four_x = generate_task("parity", 3, 8, 19, "ood_4x", {})
    assert two_x.input_ids is not None and four_x.input_ids is not None
    assert two_x.input_ids.shape[1] == 2 * batch.input_ids.shape[1]
    assert four_x.input_ids.shape[1] == 4 * batch.input_ids.shape[1]
    assert set(batch.example_ids).isdisjoint(two_x.example_ids)
    assert set(two_x.example_ids).isdisjoint(four_x.example_ids)
    assert batch.metadata[0]["logical_length"] == 8
    assert batch.metadata[0]["actual_length"] == 24
    assert two_x.metadata[0]["operation_count"] == 16


def test_state_tracking_exact_ops_and_ood_modular_counter() -> None:
    assert dict(MODULAR_TOKENS) == {
        "PAD": 0,
        "HOLD": 1,
        "QUERY": 2,
        "RESET": 3,
        "ADD_BASE": 16,
    }
    modulus = 5
    batch = generate_task(
        "modular_counter", 2, 10, 23, "id", {"modulus": modulus}
    )
    assert batch.input_ids is not None
    for example in range(2):
        state = 0
        for token_index, token in enumerate(batch.input_ids[example].tolist()):
            if batch.boundaries[example, token_index]:
                state = 0
            if token == MODULAR_TOKENS["RESET"]:
                state = 0
            elif token >= MODULAR_TOKENS["ADD_BASE"]:
                state = (state + token - MODULAR_TOKENS["ADD_BASE"]) % modulus
            elif token not in {
                MODULAR_TOKENS["HOLD"],
                MODULAR_TOKENS["QUERY"],
            }:
                raise AssertionError(f"unexpected modular token {token}")
            if batch.query_mask[example, token_index]:
                assert batch.targets[example, token_index].item() == state
    counts = torch.bincount(batch.targets[batch.query_mask], minlength=modulus)
    assert int(counts.max() - counts.min()) <= 1
    assert MODULAR_TOKENS["HOLD"] in batch.input_ids
    assert MODULAR_TOKENS["QUERY"] in batch.input_ids


def test_state_tracking_exact_ops_and_ood_toggle_fsm() -> None:
    assert dict(TOGGLE_TOKENS) == {
        "PAD": 0,
        "SET0": 1,
        "SET1": 2,
        "TOGGLE": 3,
        "NOOP": 4,
        "QUERY": 5,
    }
    batch = generate_task("toggle_fsm", 2, 12, 29, "train", {})
    assert batch.input_ids is not None
    for example in range(2):
        state = 0
        for token_index, token in enumerate(batch.input_ids[example].tolist()):
            if batch.boundaries[example, token_index]:
                state = 0
            if token == TOGGLE_TOKENS["SET0"]:
                state = 0
            elif token == TOGGLE_TOKENS["SET1"]:
                state = 1
            elif token == TOGGLE_TOKENS["TOGGLE"]:
                state ^= 1
            elif token not in {TOGGLE_TOKENS["NOOP"], TOGGLE_TOKENS["QUERY"]}:
                raise AssertionError(f"unexpected toggle token {token}")
            if batch.query_mask[example, token_index]:
                assert batch.targets[example, token_index].item() == state
    assert set(TOGGLE_TOKENS.values()) - {TOGGLE_TOKENS["PAD"]} <= set(
        batch.input_ids.flatten().tolist()
    )
    labels = _query_labels(batch)
    assert abs(labels.count(0) - labels.count(1)) <= 1


def test_state_tracking_dispatcher_determinism_and_rng_isolation() -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    via_dispatch = generate_task(
        "state_tracking", 2, 7, 31, "train", {"kind": "parity"}
    )
    after = torch.random.get_rng_state()
    direct = generate_task("parity", 2, 7, 31, "train", {})
    _assert_batch_equal(via_dispatch, direct)
    assert torch.equal(before, after)

    larger = generate_task("parity", 4, 7, 31, "train", {})
    assert via_dispatch.example_ids == larger.example_ids[:2]
    assert torch.equal(via_dispatch.input_ids, larger.input_ids[:2])
    assert generate_task("parity", 2, 7, 32, "train", {}).example_ids != direct.example_ids
    assert generate_task("parity", 2, 7, 31, "id", {}).example_ids != direct.example_ids

    with pytest.raises(ValueError, match="state_tracking.*kind"):
        generate_task("state_tracking", 1, 4, 0, "train", {})
    with pytest.raises(ValueError, match="kind.*parity"):
        generate_task(
            "state_tracking", 1, 4, 0, "train", {"kind": "unknown"}
        )


@pytest.mark.parametrize(
    ("task", "params", "classes", "batch_size", "length"),
    [
        ("parity", {}, 2, 2, 3),
        ("toggle_fsm", {}, 2, 3, 1),
        ("modular_counter", {"modulus": 5}, 5, 3, 3),
    ],
)
def test_state_tracking_counterbalances_odd_batches_and_preserves_prefixes(
    task: str,
    params: dict[str, int],
    classes: int,
    batch_size: int,
    length: int,
) -> None:
    batch = generate_task(task, batch_size, length, 0, "train", params)
    labels = batch.targets[batch.query_mask]
    counts = torch.bincount(labels, minlength=classes)
    assert int(counts.max() - counts.min()) <= 1

    larger = generate_task(task, batch_size + 2, length, 0, "train", params)
    assert batch.example_ids == larger.example_ids[:batch_size]
    assert torch.equal(batch.input_ids, larger.input_ids[:batch_size])
    assert torch.equal(batch.targets, larger.targets[:batch_size])
    assert torch.equal(batch.source_spans, larger.source_spans[:batch_size])


def _independent_rk4_step(
    h0: torch.Tensor,
    u0: torch.Tensor,
    u1: torch.Tensor,
    delta: float,
    decay: torch.Tensor,
    *,
    steps: int = 4096,
) -> torch.Tensor:
    dt = delta / steps
    state = h0.clone()

    def derivative(value: torch.Tensor, elapsed: float) -> torch.Tensor:
        forcing = u0 + (u1 - u0) * (elapsed / delta)
        return -decay * value + forcing

    elapsed = 0.0
    for _ in range(steps):
        k1 = derivative(state, elapsed)
        k2 = derivative(state + 0.5 * dt * k1, elapsed + 0.5 * dt)
        k3 = derivative(state + 0.5 * dt * k2, elapsed + 0.5 * dt)
        k4 = derivative(state + dt * k3, elapsed + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        elapsed += dt
    return state


def test_irregular_integration_matches_rk4() -> None:
    assert INTEGRATION_SCHEMA_VERSION == "1.0.0"
    assert INTEGRATION_DECAY_RANGE == (0.05, 2.0)
    assert INTEGRATION_FORCING_RANGE == (-1.5, 1.5)
    assert INTEGRATION_DELTA_VALUES == (1e-07, 0.0001, 0.05, 0.5, 2.5)
    assert INTEGRATION_RK4_STEPS == 4096

    batch = generate_task(
        "irregular_integration", 2, 10, 37, "train", {"components": 2}
    )
    assert batch.continuous_inputs is not None
    assert batch.continuous_inputs.dtype == torch.float64
    assert batch.targets.dtype == torch.float64
    assert batch.continuous_inputs.shape == (2, 10, 5)
    assert batch.targets.shape == (2, 10, 2)
    assert not torch.equal(batch.continuous_inputs[..., :2], batch.targets)

    deltas = batch.continuous_inputs[..., 2]
    assert torch.any(deltas == INTEGRATION_DELTA_VALUES[0])
    assert torch.any(deltas == INTEGRATION_DELTA_VALUES[-1])
    assert batch.strata["delta_bin"].unique().numel() >= 3
    assert batch.strata["curvature_bin"][batch.query_mask].unique().numel() >= 2

    for example in range(2):
        decay = batch.continuous_inputs[example, 0, 3:]
        state = torch.zeros(2, dtype=torch.float64)
        previous_u = torch.zeros(2, dtype=torch.float64)
        for token_index in range(batch.continuous_inputs.shape[1]):
            current_u = batch.continuous_inputs[example, token_index, :2]
            delta = float(batch.continuous_inputs[example, token_index, 2])
            if batch.boundaries[example, token_index]:
                state.zero_()
                previous_u = current_u
                assert not batch.query_mask[example, token_index]
                continue
            state = _independent_rk4_step(
                state, previous_u, current_u, delta, decay, steps=4096
            )
            assert torch.allclose(
                batch.targets[example, token_index], state, rtol=2e-8, atol=2e-9
            )
            previous_u = current_u

    analytic = batch.targets[0, 1]
    helper = rk4_piecewise_linear_oracle(
        torch.zeros(2, dtype=torch.float64),
        batch.continuous_inputs[0, 0, :2],
        batch.continuous_inputs[0, 1, :2],
        float(batch.continuous_inputs[0, 1, 2]),
        batch.continuous_inputs[0, 1, 3:],
    )
    assert torch.allclose(analytic, helper, rtol=2e-8, atol=2e-9)


def test_irregular_integration_determinism_boundaries_and_ood() -> None:
    baseline = generate_task(
        "irregular_integration", 2, 6, 41, "train", {"components": 1}
    )
    _assert_batch_equal(
        baseline,
        generate_task(
            "irregular_integration", 2, 6, 41, "train", {"components": 1}
        ),
    )
    assert baseline.boundaries[:, 0].all()
    assert baseline.boundaries[:, 3].all()
    assert not baseline.loss_mask[baseline.boundaries].any()
    assert torch.all(baseline.targets[~baseline.loss_mask] == 0)
    assert torch.all(
        baseline.source_spans[baseline.query_mask, 1]
        <= torch.nonzero(baseline.query_mask, as_tuple=False)[:, 1]
    )

    two_x = generate_task(
        "irregular_integration", 2, 6, 41, "ood_2x", {"components": 1}
    )
    four_x = generate_task(
        "irregular_integration", 2, 6, 41, "ood_4x", {"components": 1}
    )
    assert two_x.continuous_inputs is not None and four_x.continuous_inputs is not None
    assert two_x.continuous_inputs.shape[1] == 12
    assert four_x.continuous_inputs.shape[1] == 24
    assert set(baseline.example_ids).isdisjoint(two_x.example_ids)
    assert set(two_x.example_ids).isdisjoint(four_x.example_ids)

    with pytest.raises(ValueError, match="components"):
        generate_task(
            "irregular_integration", 1, 4, 0, "train", {"components": 0}
        )
    with pytest.raises(ValueError, match="at least two.*timepoints"):
        generate_task(
            "irregular_integration", 1, 1, 0, "train", {"components": 1}
        )


def test_drift_reversal_queries_precede_targets_and_strata() -> None:
    assert DYNAMICS_SCHEMA_VERSION == "1.0.0"
    assert DYNAMICS_TOKENS_PER_STEP == 2
    assert DYNAMICS_WARMUP_OBSERVATIONS == 3
    assert DRIFT_SLOPE_RANGE == (0.02, 0.12)
    batch = generate_task("drift_reversal", 3, 9, 43, "train", {})
    assert batch.continuous_inputs is not None
    assert batch.continuous_inputs.shape == (3, 21, 3)

    for example in range(3):
        metadata = batch.metadata[example]
        base = float(metadata["base"])
        slope = float(metadata["slope"])
        reversal_step = int(metadata["reversal_step"])
        first_changed_query_step = int(metadata["first_changed_query_step"])
        reversal_value = base + slope * reversal_step
        expected_values = []
        for step in range(DYNAMICS_WARMUP_OBSERVATIONS + 9):
            if step <= reversal_step:
                expected_values.append(base + slope * step)
            else:
                expected_values.append(
                    reversal_value - slope * (step - reversal_step)
                )
        for warmup_position in range(DYNAMICS_WARMUP_OBSERVATIONS):
            assert not batch.query_mask[example, warmup_position]
            assert not batch.loss_mask[example, warmup_position]
            assert batch.continuous_inputs[
                example, warmup_position, 0
            ].item() == pytest.approx(expected_values[warmup_position])
        for step in range(9):
            query_position = DYNAMICS_WARMUP_OBSERVATIONS + 2 * step
            observation_position = query_position + 1
            target_step = DYNAMICS_WARMUP_OBSERVATIONS + step
            assert batch.query_mask[example, query_position]
            assert batch.continuous_inputs[example, query_position, 0] == 0
            assert batch.targets[example, query_position, 0].item() == pytest.approx(
                expected_values[target_step]
            )
            assert batch.continuous_inputs[
                example, observation_position, 0
            ].item() == pytest.approx(expected_values[target_step])
            assert batch.source_spans[example, query_position].tolist() == [
                0,
                query_position,
            ]
            lag = int(batch.strata["causal_lag"][example, query_position])
            expected_lag = (
                -1 if step < first_changed_query_step else step - first_changed_query_step
            )
            assert lag == expected_lag
        zero_lag_positions = torch.nonzero(
            (batch.strata["causal_lag"][example] == 0)
            & batch.query_mask[example],
            as_tuple=False,
        ).flatten()
        assert zero_lag_positions.tolist() == [
            DYNAMICS_WARMUP_OBSERVATIONS + 2 * first_changed_query_step
        ]

    phases = batch.strata["phase"][batch.query_mask]
    counts = torch.bincount(phases, minlength=3)
    assert torch.equal(counts, torch.tensor([9, 9, 9]))
    assert torch.any(batch.strata["causal_lag"][batch.query_mask] == 0)
    assert torch.any(batch.strata["overshoot"][batch.query_mask] == 1)


def test_drift_reversal_determinism_and_ood_horizons() -> None:
    baseline = generate_task("drift_reversal", 2, 6, 47, "train", {})
    _assert_batch_equal(
        baseline, generate_task("drift_reversal", 2, 6, 47, "train", {})
    )
    two_x = generate_task("drift_reversal", 2, 6, 47, "ood_2x", {})
    four_x = generate_task("drift_reversal", 2, 6, 47, "ood_4x", {})
    assert two_x.continuous_inputs is not None and four_x.continuous_inputs is not None
    assert two_x.continuous_inputs.shape[1] == 27
    assert four_x.continuous_inputs.shape[1] == 51
    assert int(two_x.query_mask.sum()) == 24
    assert int(four_x.query_mask.sum()) == 48
    assert set(baseline.example_ids).isdisjoint(two_x.example_ids)
    assert set(two_x.example_ids).isdisjoint(four_x.example_ids)


def test_trajectory_balances_modes_changes_and_causal_targets() -> None:
    assert TRAJECTORY_FREQUENCY_RANGE == (0.15, 0.45)
    batch = generate_task("trajectory", 4, 9, 53, "train", {})
    assert batch.continuous_inputs is not None
    modes = [str(item["mode"]) for item in batch.metadata]
    cases = [bool(item["has_change_point"]) for item in batch.metadata]
    assert modes.count("linear") == modes.count("sinusoidal") == 2
    assert cases.count(False) == cases.count(True) == 2

    for example in range(4):
        assert not batch.query_mask[
            example, :DYNAMICS_WARMUP_OBSERVATIONS
        ].any()
        for step in range(9):
            query_position = DYNAMICS_WARMUP_OBSERVATIONS + 2 * step
            observation_position = query_position + 1
            assert batch.query_mask[example, query_position]
            assert not batch.query_mask[example, observation_position]
            assert batch.continuous_inputs[example, query_position, 0] == 0
            assert torch.equal(
                batch.targets[example, query_position],
                batch.continuous_inputs[example, observation_position, :1],
            )
            assert batch.source_spans[example, query_position, 1] <= query_position
            assert batch.source_spans[example, query_position].tolist() == [
                0,
                query_position,
            ]
        if cases[example]:
            first_changed = int(batch.metadata[example]["first_changed_query_step"])
            changed_position = DYNAMICS_WARMUP_OBSERVATIONS + 2 * first_changed
            assert batch.strata["causal_lag"][example, changed_position] == 0
        else:
            assert not torch.any(
                batch.strata["causal_lag"][example, batch.query_mask[example]] >= 0
            )
    phases = batch.strata["phase"][batch.query_mask]
    assert torch.equal(torch.bincount(phases, minlength=3), torch.tensor([12, 12, 12]))
    assert set(batch.strata["trajectory_type"][batch.query_mask].tolist()) == {0, 1}
    assert set(batch.strata["change_case"][batch.query_mask].tolist()) == {0, 1}


def test_trajectory_seed_split_and_length_variation() -> None:
    baseline = generate_task("trajectory", 4, 6, 59, "train", {})
    _assert_batch_equal(
        baseline, generate_task("trajectory", 4, 6, 59, "train", {})
    )
    assert generate_task("trajectory", 4, 6, 60, "train", {}).example_ids != baseline.example_ids
    assert generate_task("trajectory", 4, 6, 59, "id", {}).example_ids != baseline.example_ids
    assert generate_task(
        "trajectory", 4, 6, 59, "ood_2x", {}
    ).continuous_inputs.shape[1] == 27
    assert generate_task(
        "trajectory", 4, 6, 59, "ood_4x", {}
    ).continuous_inputs.shape[1] == 51


def test_local_binding_modes_overwrite_spans_and_width_cells() -> None:
    assert LOCAL_BINDING_SCHEMA_VERSION == "1.0.0"
    assert LOCAL_BINDING_VOCAB_VERSION == "1.0.0"
    assert LOCAL_BINDING_DEFAULT_WIDTH == 8
    assert LOCAL_BINDING_MODES == (
        "adjacent",
        "separated",
        "motif",
        "delayed_copy",
    )
    assert dict(LOCAL_BINDING_TOKENS) == {
        "PAD": 0,
        "QUERY": 1,
        "COPY": 2,
        "FILLER": 3,
        "KEY_BASE": 100,
        "VALUE_BASE": 1000,
        "MOTIF_BASE": 2000,
    }
    batch = generate_task("local_binding", 3, 12, 61, "train", {"width": 8})
    assert batch.input_ids is not None
    query_modes = batch.strata["mode"][batch.query_mask]
    assert set(query_modes.tolist()) == {0, 1, 2, 3}
    assert set(batch.strata["distance_bin"][batch.query_mask].tolist()) == {0, 1, 2}
    assert set(batch.strata["load_bin"][batch.query_mask].tolist()) == {0, 1, 2}
    assert set(batch.strata["overwrite"][batch.query_mask].tolist()) == {0, 1}

    for example in range(3):
        ids = batch.input_ids[example]
        for query_position in torch.nonzero(
            batch.query_mask[example], as_tuple=False
        ).flatten().tolist():
            source_start, source_end = batch.source_spans[
                example, query_position
            ].tolist()
            assert source_end == source_start + 1 <= query_position
            assert batch.targets[example, query_position] == ids[source_start]
            assert batch.strata["source_distance"][example, query_position] == (
                query_position - source_start
            )
            mode = int(batch.strata["mode"][example, query_position])
            if mode in (0, 1):
                query_key = int(ids[query_position - 1])
                prior_key_positions = torch.nonzero(
                    ids[: query_position - 1] == query_key, as_tuple=False
                ).flatten()
                assert prior_key_positions.numel() >= 1
                latest_key = int(prior_key_positions[-1])
                scan = latest_key + 1
                while ids[scan] == LOCAL_BINDING_TOKENS["FILLER"]:
                    scan += 1
                assert scan == source_start
            elif mode == 3:
                assert ids[query_position] == LOCAL_BINDING_TOKENS["COPY"]


def test_local_binding_determinism_and_ood_query_counts() -> None:
    baseline = generate_task("local_binding", 2, 4, 67, "train", {"width": 8})
    _assert_batch_equal(
        baseline,
        generate_task("local_binding", 2, 4, 67, "train", {"width": 8}),
    )
    assert int(baseline.query_mask.sum()) == 8
    assert int(
        generate_task("local_binding", 2, 4, 67, "ood_2x", {"width": 8})
        .query_mask.sum()
    ) == 16
    assert int(
        generate_task("local_binding", 2, 4, 67, "ood_4x", {"width": 8})
        .query_mask.sum()
    ) == 32


def test_local_binding_distance_cells_are_computed_from_actual_spans() -> None:
    batch = generate_task("local_binding", 2, 3, 69, "train", {"width": 4})
    for example, query_position in torch.nonzero(batch.query_mask, as_tuple=False):
        distance = int(batch.strata["source_distance"][example, query_position])
        expected_bin = 0 if distance < 4 else 1 if distance <= 8 else 2
        assert int(batch.strata["distance_bin"][example, query_position]) == expected_bin


def test_mqar_latest_overwrite_exact_spans_and_capacity_cells() -> None:
    assert MQAR_SCHEMA_VERSION == "1.0.0"
    assert MQAR_DEFAULT_WIDTH == 8
    assert MQAR_LOAD_FACTORS == (0.5, 1.0, 1.5)
    assert MQAR_OVERWRITE_FRACTION == 0.25
    assert dict(MQAR_TOKENS) == {
        "PAD": 0,
        "QUERY": 1,
        "FILLER": 2,
        "KEY_BASE": 100,
        "VALUE_BASE": 1000,
    }
    batch = generate_task("mqar", 6, 9, 71, "train", {"width": 8})
    assert batch.input_ids is not None
    assert set(batch.strata["load_bin"][batch.query_mask].tolist()) == {0, 1, 2}
    assert set(batch.strata["overwrite"][batch.query_mask].tolist()) == {0, 1}
    assert set(batch.strata["distance_bin"][batch.query_mask].tolist()) == {0, 1, 2}

    for example in range(6):
        latest: dict[int, tuple[int, int]] = {}
        ids = batch.input_ids[example]
        valid_length = int(batch.valid[example].sum())
        position = 0
        while position < valid_length:
            token = int(ids[position])
            next_token = int(ids[position + 1]) if position + 1 < valid_length else 0
            if token >= MQAR_TOKENS["KEY_BASE"] and token < MQAR_TOKENS["VALUE_BASE"]:
                if next_token == MQAR_TOKENS["QUERY"]:
                    query_position = position + 1
                    expected_value, source_position = latest[token]
                    assert batch.targets[example, query_position] == expected_value
                    assert batch.source_spans[example, query_position].tolist() == [
                        source_position,
                        source_position + 1,
                    ]
                    position += 2
                    continue
                assert next_token >= MQAR_TOKENS["VALUE_BASE"]
                latest[token] = (next_token, position + 1)
                position += 2
                continue
            position += 1


def test_mqar_determinism_seed_split_and_ood() -> None:
    baseline = generate_task("mqar", 3, 5, 73, "train", {"width": 8})
    _assert_batch_equal(
        baseline, generate_task("mqar", 3, 5, 73, "train", {"width": 8})
    )
    assert generate_task("mqar", 3, 5, 74, "train", {"width": 8}).example_ids != baseline.example_ids
    assert generate_task("mqar", 3, 5, 73, "id", {"width": 8}).example_ids != baseline.example_ids
    assert int(
        generate_task("mqar", 3, 5, 73, "ood_2x", {"width": 8}).query_mask.sum()
    ) == 30
    assert int(
        generate_task("mqar", 3, 5, 73, "ood_4x", {"width": 8}).query_mask.sum()
    ) == 60


def test_structured_exceptions_rule_and_exception_oracles() -> None:
    assert STRUCTURED_SCHEMA_VERSION == "1.0.0"
    assert STRUCTURED_RULE_MODULUS == 97
    assert STRUCTURED_EXCEPTION_FRACTION == 0.25
    assert dict(STRUCTURED_TOKENS) == {
        "PAD": 0,
        "QUERY": 1,
        "KEY_BASE": 100,
        "VALUE_BASE": 1000,
    }
    batch = generate_task("structured_exceptions", 2, 10, 79, "train", {})
    assert batch.input_ids is not None
    strata = batch.strata["item_type"][batch.query_mask]
    assert set(strata.tolist()) == {0, 1}
    assert abs(int((strata == 0).sum()) - int((strata == 1).sum())) <= 1
    for example in range(2):
        factor = int(batch.metadata[example]["rule_factor"])
        offset = int(batch.metadata[example]["rule_offset"])
        for query_position in torch.nonzero(
            batch.query_mask[example], as_tuple=False
        ).flatten().tolist():
            key = int(batch.input_ids[example, query_position - 1])
            source = int(batch.source_spans[example, query_position, 0])
            observed = int(batch.input_ids[example, source])
            expected_rule = STRUCTURED_TOKENS["VALUE_BASE"] + (
                factor * (key - STRUCTURED_TOKENS["KEY_BASE"]) + offset
            ) % STRUCTURED_RULE_MODULUS
            assert int(batch.targets[example, query_position]) == observed
            if batch.strata["item_type"][example, query_position] == 0:
                assert observed == expected_rule
            else:
                assert observed != expected_rule


def test_structured_exceptions_determinism_and_ood() -> None:
    baseline = generate_task("structured_exceptions", 2, 6, 83, "train", {})
    _assert_batch_equal(
        baseline,
        generate_task("structured_exceptions", 2, 6, 83, "train", {}),
    )
    assert generate_task("structured_exceptions", 2, 6, 84, "train", {}).example_ids != baseline.example_ids
    assert generate_task("structured_exceptions", 2, 6, 83, "id", {}).example_ids != baseline.example_ids
    assert int(generate_task("structured_exceptions", 2, 6, 83, "ood_2x", {}).query_mask.sum()) == 24
    assert int(generate_task("structured_exceptions", 2, 6, 83, "ood_4x", {}).query_mask.sum()) == 48


def test_structured_exception_query_order_never_derives_from_a_set() -> None:
    from research.kmd2_ablation.tasks import structured

    tree = ast.parse(inspect.getsource(structured.generate_structured_exceptions))
    set_variables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "set":
            continue
        set_variables.update(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
    list_from_set = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in set_variables
    ]
    assert not list_from_set


def test_structured_exceptions_are_cross_pythonhashseed_deterministic() -> None:
    script = """
import hashlib
import json
from research.kmd2_ablation.tasks import generate_task

batch = generate_task('structured_exceptions', 3, 9, 149, 'ood_2x', {})
payload = {
    'ids': batch.example_ids,
    'input_ids': batch.input_ids.tolist(),
    'targets': batch.targets.tolist(),
    'source_spans': batch.source_spans.tolist(),
    'item_type': batch.strata['item_type'].tolist(),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
"""
    digests = []
    for hash_seed in ("1", "8675309"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        digests.append(completed.stdout.strip())
    assert digests[0] == digests[1]


def test_far_surprise_early_sources_lose_native_score_to_distractors() -> None:
    assert FAR_SURPRISE_SCHEMA_VERSION == "1.0.0"
    assert FAR_SURPRISE_DEFAULT_WIDTH == 8
    assert FAR_SURPRISE_DISTRACTOR_MARGIN == 8
    assert dict(FAR_SURPRISE_TOKENS) == {
        "PAD": 0,
        "QUERY": 1,
        "KEY_BASE": 100,
        "VALUE_BASE": 1000,
    }
    batch = generate_task("far_surprise", 2, 5, 89, "train", {"width": 8})
    assert batch.input_ids is not None
    assert all(int(item["distractor_count"]) > 8 for item in batch.metadata)
    for example in range(2):
        distractor_scores = batch.strata["native_score"][
            example, batch.strata["distractor"][example].bool()
        ]
        assert distractor_scores.numel() > 8
        for query_position in torch.nonzero(
            batch.query_mask[example], as_tuple=False
        ).flatten().tolist():
            source = int(batch.source_spans[example, query_position, 0])
            assert source < int(batch.metadata[example]["distractor_start"])
            assert batch.targets[example, query_position] == batch.input_ids[example, source]
            assert distractor_scores.min() > batch.strata["native_score"][example, source]
            assert batch.strata["load_over_width"][example, query_position] == 1
    queried_keys = set(
        batch.input_ids[:, :-1][batch.query_mask[:, 1:]].flatten().tolist()
    )
    distractor_keys = set(
        batch.input_ids[batch.strata["distractor_key"].bool()].flatten().tolist()
    )
    assert queried_keys.isdisjoint(distractor_keys)


def test_far_surprise_determinism_seed_split_and_length() -> None:
    baseline = generate_task("far_surprise", 2, 4, 97, "train", {"width": 8})
    _assert_batch_equal(
        baseline,
        generate_task("far_surprise", 2, 4, 97, "train", {"width": 8}),
    )
    assert int(generate_task("far_surprise", 2, 4, 97, "ood_2x", {"width": 8}).query_mask.sum()) == 16
    assert int(generate_task("far_surprise", 2, 4, 97, "ood_4x", {"width": 8}).query_mask.sum()) == 32


def test_far_surprise_long_ood_distractor_keys_remain_disjoint() -> None:
    batch = generate_task(
        "far_surprise", 1, 128, 151, "ood_4x", {"width": 8}
    )
    assert batch.input_ids is not None
    query_positions = torch.nonzero(batch.query_mask[0], as_tuple=False).flatten()
    query_keys = set(batch.input_ids[0, query_positions - 1].tolist())
    distractor_keys = set(
        batch.input_ids[0, batch.strata["distractor_key"][0].bool()].tolist()
    )
    assert len(query_positions) == 512
    assert query_keys.isdisjoint(distractor_keys)
    for query_position in query_positions.tolist():
        source = int(batch.source_spans[0, query_position, 0])
        assert source < int(batch.metadata[0]["distractor_start"])
        assert batch.targets[0, query_position] == batch.input_ids[0, source]


def test_freshness_latest_and_historical_queries_respect_versions() -> None:
    assert FRESHNESS_SCHEMA_VERSION == "1.0.0"
    assert FRESHNESS_DEFAULT_WIDTH == 8
    assert FRESHNESS_STALE_SCORE == 10.0
    assert dict(FRESHNESS_TOKENS) == {
        "PAD": 0,
        "QUERY": 1,
        "FILLER": 2,
        "LATEST": 3,
        "HISTORY_BASE": 10,
        "KEY_BASE": 100,
        "VALUE_BASE": 1000,
    }
    batch = generate_task("freshness", 2, 4, 101, "train", {"width": 8})
    assert batch.input_ids is not None
    query_types = batch.strata["query_type"][batch.query_mask]
    assert int((query_types == 0).sum()) == int((query_types == 1).sum())
    for example in range(2):
        versions: dict[int, list[tuple[int, int]]] = {}
        ids = batch.input_ids[example]
        valid_length = int(batch.valid[example].sum())
        position = 0
        while position < valid_length:
            token = int(ids[position])
            if FRESHNESS_TOKENS["KEY_BASE"] <= token < FRESHNESS_TOKENS["VALUE_BASE"]:
                next_token = int(ids[position + 1])
                if next_token >= FRESHNESS_TOKENS["VALUE_BASE"]:
                    versions.setdefault(token, []).append((next_token, position + 1))
                    position += 2
                    continue
                request = next_token
                query_position = position + 2
                assert ids[query_position] == FRESHNESS_TOKENS["QUERY"]
                if request == FRESHNESS_TOKENS["LATEST"]:
                    expected_value, expected_source = versions[token][-1]
                    assert batch.strata["query_type"][example, query_position] == 0
                    old_source = versions[token][0][1]
                    assert batch.strata["admission_score"][example, old_source] > batch.strata["admission_score"][example, expected_source]
                else:
                    version_index = request - FRESHNESS_TOKENS["HISTORY_BASE"]
                    expected_value, expected_source = versions[token][version_index]
                    assert batch.strata["query_type"][example, query_position] == 1
                assert batch.targets[example, query_position] == expected_value
                assert batch.source_spans[example, query_position].tolist() == [
                    expected_source,
                    expected_source + 1,
                ]
                position += 3
                continue
            position += 1


def test_freshness_determinism_and_ood_groups() -> None:
    baseline = generate_task("freshness", 2, 3, 103, "train", {"width": 8})
    _assert_batch_equal(
        baseline,
        generate_task("freshness", 2, 3, 103, "train", {"width": 8}),
    )
    assert int(generate_task("freshness", 2, 3, 103, "ood_2x", {"width": 8}).query_mask.sum()) == 24
    assert int(generate_task("freshness", 2, 3, 103, "ood_4x", {"width": 8}).query_mask.sum()) == 48


def test_affine_symmetric_direct_factors_recover_slope_and_intercept() -> None:
    assert AFFINE_SCHEMA_VERSION == "1.0.0"
    assert AFFINE_DEFAULT_INPUT_DIM == 3
    assert AFFINE_DEFAULT_OUTPUT_DIM == 2
    batch = generate_task(
        "affine_associative_regression",
        4,
        4,
        107,
        "train",
        {"input_dim": 3, "output_dim": 2},
    )
    assert batch.direct_factors is not None
    factors = batch.direct_factors
    assert factors["q"].shape == (4, 9, 1, 1, 3)
    assert factors["k"].shape == (4, 9, 1, 1, 3)
    assert factors["v"].shape == (4, 9, 1, 1, 2)
    assert factors["decay"].shape == (4, 9, 1, 3)
    assert not torch.any(factors["k"] == 1.0, dim=(0, 1, 2, 3)).all()
    controls = batch.strata["intercept_control"][batch.query_mask]
    assert int((controls == 1).sum()) == int((controls == 0).sum()) == 2

    for example in range(4):
        write_positions = torch.nonzero(
            factors["write_mask"][example], as_tuple=False
        ).flatten()
        query_position = int(torch.nonzero(batch.query_mask[example])[0])
        x = factors["k"][example, write_positions, 0, 0]
        y = factors["v"][example, write_positions, 0, 0]
        assert torch.equal(x[0::2], -x[1::2])
        assert torch.allclose(x.sum(dim=0), torch.zeros(3, dtype=x.dtype), atol=0, rtol=0)
        design = torch.cat(
            [x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1
        )
        solution = torch.linalg.lstsq(design, y).solution
        query_x = factors["q"][example, query_position, 0, 0]
        predicted = query_x @ solution[:-1] + solution[-1]
        assert torch.allclose(
            predicted,
            batch.targets[example, query_position],
            rtol=1e-8,
            atol=1e-8,
        )
        inferred_intercept = 0.5 * (y[0] + y[1])
        if batch.strata["intercept_control"][example, query_position] == 1:
            assert torch.equal(inferred_intercept, torch.zeros_like(inferred_intercept))
        else:
            assert torch.linalg.vector_norm(inferred_intercept) > 0.1
        assert batch.source_spans[example, query_position].tolist() == [
            0,
            query_position,
        ]


def test_affine_determinism_no_constant_path_and_ood_writes() -> None:
    params = {"input_dim": 3, "output_dim": 2}
    baseline = generate_task(
        "affine_associative_regression", 4, 4, 109, "train", params
    )
    _assert_batch_equal(
        baseline,
        generate_task(
            "affine_associative_regression", 4, 4, 109, "train", params
        ),
    )
    assert baseline.direct_factors is not None
    assert "constant_coordinate" not in baseline.direct_factors
    assert baseline.metadata[0]["qk_bias"] is False
    two_x = generate_task(
        "affine_associative_regression", 4, 4, 109, "ood_2x", params
    )
    four_x = generate_task(
        "affine_associative_regression", 4, 4, 109, "ood_4x", params
    )
    assert two_x.direct_factors is not None and four_x.direct_factors is not None
    assert two_x.direct_factors["q"].shape[1] == 17
    assert four_x.direct_factors["q"].shape[1] == 33
def test_ruler_parity_and_modular_state_tracking_and_lm_loss() -> None:
    from research.kmd2_ablation.tasks.ruler import score_state_tracking

    parity = score_state_tracking("parity", [1, 0, 1, 1], [1, 1, 0, 1], lm_loss=0.25)
    modular = score_state_tracking("modular", [4, 5, 7], [4, 0, 2], modulus=5, lm_loss=0.5)
    assert parity.correct == (True, False, False, True)
    assert modular.correct == (True, True, True)
    assert parity.lm_loss == 0.25 and modular.lm_loss == 0.5
    with pytest.raises(ValueError, match="modulus"):
        score_state_tracking("modular", [1], [1], lm_loss=0.1)
