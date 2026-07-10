from __future__ import annotations

import pytest
import torch

from research.kmd2_ablation import exact_cache as exact_cache_module
from research.kmd2_ablation.config import CacheConfig
from research.kmd2_ablation.exact_cache import (
    ExactCacheState,
    merge_persistent_cache,
)


def test_merge_selects_candidates_independently_for_each_head() -> None:
    block_k = torch.tensor(
        [[[[10.0], [100.0]], [[20.0], [200.0]], [[30.0], [300.0]]]]
    )
    block_v = block_k + 0.5
    block_scores = torch.tensor([[[9.0, 1.0], [1.0, 8.0], [7.0, 6.0]]])
    block_positions = torch.tensor([[10, 20, 30]])
    block_valid = torch.ones(1, 3, dtype=torch.bool)

    state = merge_persistent_cache(
        None,
        block_k,
        block_v,
        block_scores,
        block_positions,
        block_valid,
        width=2,
        storage_dtype=torch.float32,
    )

    torch.testing.assert_close(
        state.keys,
        torch.tensor([[[[10.0], [30.0]], [[200.0], [300.0]]]]),
    )
    torch.testing.assert_close(
        state.values,
        torch.tensor([[[[10.5], [30.5]], [[200.5], [300.5]]]]),
    )
    torch.testing.assert_close(state.scores, torch.tensor([[[9.0, 7.0], [8.0, 6.0]]]))
    torch.testing.assert_close(state.positions, torch.tensor([[[10, 30], [20, 30]]]))
    assert state.valid.all()


def test_merge_breaks_exact_score_ties_by_newer_position() -> None:
    block_k = torch.tensor([[[[10.0]], [[30.0]], [[20.0]]]])
    block_v = block_k.clone()
    block_scores = torch.full((1, 3, 1), 5.0)
    block_positions = torch.tensor([[10, 30, 20]])
    block_valid = torch.ones(1, 3, dtype=torch.bool)

    state = merge_persistent_cache(
        None,
        block_k,
        block_v,
        block_scores,
        block_positions,
        block_valid,
        width=2,
        storage_dtype=torch.float32,
    )

    torch.testing.assert_close(state.positions, torch.tensor([[[30, 20]]]))
    torch.testing.assert_close(state.keys[..., 0], torch.tensor([[[30.0, 20.0]]]))


def test_merge_keeps_old_survivor_admits_new_candidate_and_evicts_losers() -> None:
    first = merge_persistent_cache(
        None,
        torch.tensor([[[[1.0]], [[2.0]]]]),
        torch.tensor([[[[101.0]], [[102.0]]]]),
        torch.tensor([[[10.0], [1.0]]]),
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        width=2,
        storage_dtype=torch.float32,
    )

    second = merge_persistent_cache(
        first,
        torch.tensor([[[[3.0]], [[4.0]]]]),
        torch.tensor([[[[103.0]], [[104.0]]]]),
        torch.tensor([[[5.0], [0.0]]]),
        torch.tensor([[3, 4]]),
        torch.ones(1, 2, dtype=torch.bool),
        width=2,
        storage_dtype=torch.float32,
    )

    torch.testing.assert_close(second.positions, torch.tensor([[[1, 3]]]))
    torch.testing.assert_close(second.keys[..., 0], torch.tensor([[[1.0, 3.0]]]))
    torch.testing.assert_close(second.values[..., 0], torch.tensor([[[101.0, 103.0]]]))
    assert second.keys.shape[2] == 2
    assert set(second.positions.flatten().tolist()) == {1, 3}


def test_merge_returns_only_width_candidates_and_clears_block_workspace() -> None:
    block_k = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1)
    block_v = block_k + 100.0
    state = merge_persistent_cache(
        None,
        block_k,
        block_v,
        torch.arange(6, dtype=torch.float32).reshape(1, 6, 1),
        torch.arange(6).reshape(1, 6),
        torch.ones(1, 6, dtype=torch.bool),
        width=2,
        storage_dtype=torch.float32,
    )

    assert state.keys.shape == (1, 1, 2, 1)
    assert state.values.shape == (1, 1, 2, 1)
    assert state.scores.shape == state.positions.shape == state.valid.shape == (1, 1, 2)
    torch.testing.assert_close(state.positions, torch.tensor([[[5, 4]]]))


def test_merge_from_none_pads_insufficient_and_all_invalid_candidates() -> None:
    state = merge_persistent_cache(
        None,
        torch.tensor([[[[7.0], [70.0]], [[8.0], [80.0]]]]),
        torch.tensor([[[[17.0], [170.0]], [[18.0], [180.0]]]]),
        torch.tensor([[[2.0, 200.0], [1.0, 100.0]]]),
        torch.tensor([[7, 8]]),
        torch.tensor([[True, False]]),
        width=3,
        storage_dtype=torch.float32,
    )

    torch.testing.assert_close(
        state.keys,
        torch.tensor([[[[7.0], [0.0], [0.0]], [[70.0], [0.0], [0.0]]]]),
    )
    torch.testing.assert_close(state.scores, torch.tensor([[[2.0, 0.0, 0.0], [200.0, 0.0, 0.0]]]))
    torch.testing.assert_close(state.positions, torch.tensor([[[7, -1, -1], [7, -1, -1]]]))
    torch.testing.assert_close(
        state.valid,
        torch.tensor([[[True, False, False], [True, False, False]]]),
    )

    all_invalid = merge_persistent_cache(
        None,
        torch.ones(1, 2, 1, 1),
        torch.ones(1, 2, 1, 1),
        torch.tensor([[[float("nan")], [float("inf")]]]),
        torch.tensor([[1, 2]]),
        torch.zeros(1, 2, dtype=torch.bool),
        width=2,
        storage_dtype=torch.float32,
    )
    assert not all_invalid.valid.any()
    assert not all_invalid.keys.any()
    assert not all_invalid.values.any()
    assert not all_invalid.scores.any()
    assert (all_invalid.positions == -1).all()


def test_merge_width_zero_returns_canonical_empty_state() -> None:
    state = merge_persistent_cache(
        None,
        torch.ones(2, 3, 4, 5),
        torch.ones(2, 3, 4, 6),
        torch.ones(2, 3, 4),
        torch.arange(3).expand(2, 3),
        torch.ones(2, 3, dtype=torch.bool),
        width=0,
        storage_dtype=torch.bfloat16,
    )

    assert state.keys.shape == (2, 4, 0, 5)
    assert state.values.shape == (2, 4, 0, 6)
    assert state.scores.shape == state.positions.shape == state.valid.shape == (2, 4, 0)
    assert state.keys.dtype == state.values.dtype == torch.bfloat16
    assert state.scores.dtype == torch.float32
    assert state.positions.dtype == torch.int64
    assert state.valid.dtype == torch.bool


def test_merge_storage_cast_roundtrips_between_bfloat16_and_float32() -> None:
    source_k = torch.tensor([[[[1.25, -2.5]], [[3.75, 4.5]]]])
    source_v = torch.tensor([[[[5.25]], [[-6.5]]]])
    bf16_state = merge_persistent_cache(
        None,
        source_k,
        source_v,
        torch.tensor([[[2.0], [1.0]]]),
        torch.tensor([[2, 1]]),
        torch.ones(1, 2, dtype=torch.bool),
        width=2,
        storage_dtype=torch.bfloat16,
    )
    fp32_state = merge_persistent_cache(
        bf16_state,
        torch.empty(1, 0, 1, 2),
        torch.empty(1, 0, 1, 1),
        torch.empty(1, 0, 1),
        torch.empty(1, 0, dtype=torch.int64),
        torch.empty(1, 0, dtype=torch.bool),
        width=2,
        storage_dtype=torch.float32,
    )

    assert bf16_state.keys.dtype == bf16_state.values.dtype == torch.bfloat16
    assert fp32_state.keys.dtype == fp32_state.values.dtype == torch.float32
    torch.testing.assert_close(fp32_state.keys, source_k.permute(0, 2, 1, 3))
    torch.testing.assert_close(fp32_state.values, source_v.permute(0, 2, 1, 3))


def test_merge_stores_raw_block_v_not_a_different_recurrence_update() -> None:
    raw_v = torch.tensor([[[[11.0, 12.0]], [[21.0, 22.0]]]])
    deliberately_different_u = torch.tensor([[[[-91.0, -92.0]], [[-81.0, -82.0]]]])

    state = merge_persistent_cache(
        None,
        torch.tensor([[[[1.0]], [[2.0]]]]),
        raw_v,
        torch.tensor([[[1.0], [2.0]]]),
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        width=2,
        storage_dtype=torch.float32,
    )

    torch.testing.assert_close(state.values, raw_v.permute(0, 2, 1, 3).flip(2))
    assert not torch.equal(state.values, deliberately_different_u.permute(0, 2, 1, 3))


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_merge_preserves_key_value_gradients_but_detaches_admission_metadata(
    storage_dtype: torch.dtype,
) -> None:
    block_k = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]], requires_grad=True)
    block_v = torch.tensor([[[[11.0]], [[12.0]], [[13.0]]]], requires_grad=True)
    block_scores = torch.tensor([[[1.0], [3.0], [2.0]]], requires_grad=True)

    state = merge_persistent_cache(
        None,
        block_k,
        block_v,
        block_scores,
        torch.tensor([[1, 2, 3]]),
        torch.ones(1, 3, dtype=torch.bool),
        width=2,
        storage_dtype=storage_dtype,
    )

    assert state.keys.requires_grad and state.values.requires_grad
    assert not state.scores.requires_grad and state.scores.grad_fn is None
    assert not state.positions.requires_grad and state.positions.grad_fn is None
    assert not state.valid.requires_grad and state.valid.grad_fn is None
    (state.keys.float().sum() + state.values.float().sum()).backward()
    torch.testing.assert_close(block_k.grad, torch.tensor([[[[0.0]], [[1.0]], [[1.0]]]]))
    torch.testing.assert_close(block_v.grad, torch.tensor([[[[0.0]], [[1.0]], [[1.0]]]]))
    assert block_scores.grad is None


def test_exact_cache_state_nbytes_sums_actual_tensor_storage() -> None:
    state = ExactCacheState(
        keys=torch.zeros(2, 3, 4, 5, dtype=torch.bfloat16),
        values=torch.zeros(2, 3, 4, 7, dtype=torch.bfloat16),
        scores=torch.zeros(2, 3, 4, dtype=torch.float32),
        positions=torch.zeros(2, 3, 4, dtype=torch.int64),
        valid=torch.zeros(2, 3, 4, dtype=torch.bool),
    )
    tensors = (state.keys, state.values, state.scores, state.positions, state.valid)
    expected = sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    assert state.nbytes == expected
    with pytest.raises((AttributeError, TypeError)):
        state.nbytes = 0  # type: ignore[misc]


def _valid_state_tensors() -> dict[str, torch.Tensor]:
    return {
        "keys": torch.zeros(2, 3, 4, 5),
        "values": torch.zeros(2, 3, 4, 7),
        "scores": torch.zeros(2, 3, 4, dtype=torch.float32),
        "positions": torch.zeros(2, 3, 4, dtype=torch.int64),
        "valid": torch.ones(2, 3, 4, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("field", "replacement", "error", "match"),
    [
        ("keys", torch.zeros(2, 3, 5), ValueError, "keys.*4"),
        ("values", torch.zeros(2, 3, 5, 7), ValueError, "values.*shape"),
        ("scores", torch.zeros(2, 3, 5), ValueError, "scores.*shape"),
        ("positions", torch.zeros(2, 2, 4, dtype=torch.int64), ValueError, "positions.*shape"),
        ("valid", torch.ones(2, 3, 5, dtype=torch.bool), ValueError, "valid.*shape"),
        ("keys", torch.zeros(2, 3, 4, 5, dtype=torch.float16), TypeError, "keys.*float32.*bfloat16"),
        ("values", torch.zeros(2, 3, 4, 7, dtype=torch.float64), TypeError, "values.*float32.*bfloat16"),
        ("scores", torch.zeros(2, 3, 4, dtype=torch.bfloat16), TypeError, "scores.*float32"),
        ("positions", torch.zeros(2, 3, 4, dtype=torch.int32), TypeError, "positions.*int64"),
        ("valid", torch.ones(2, 3, 4, dtype=torch.uint8), TypeError, "valid.*bool"),
    ],
)
def test_exact_cache_state_rejects_noncanonical_shapes_and_dtypes(
    field: str,
    replacement: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    tensors = _valid_state_tensors()
    tensors[field] = replacement

    with pytest.raises(error, match=match):
        ExactCacheState(**tensors)


def test_exact_cache_state_requires_matching_key_value_storage_dtype() -> None:
    tensors = _valid_state_tensors()
    tensors["keys"] = tensors["keys"].to(torch.bfloat16)

    with pytest.raises(TypeError, match="keys.*values.*same dtype"):
        ExactCacheState(**tensors)


def test_exact_cache_state_rejects_non_tensor_and_cross_device_fields() -> None:
    tensors: dict[str, object] = _valid_state_tensors()
    tensors["keys"] = []
    with pytest.raises(TypeError, match="keys.*Tensor"):
        ExactCacheState(**tensors)  # type: ignore[arg-type]

    tensors = _valid_state_tensors()
    tensors["values"] = torch.zeros(2, 3, 4, 7, device="meta")
    with pytest.raises(ValueError, match="device"):
        ExactCacheState(**tensors)


def test_exact_cache_state_requires_detached_scores() -> None:
    tensors = _valid_state_tensors()
    tensors["scores"] = tensors["scores"].requires_grad_(True)

    with pytest.raises(ValueError, match="scores.*detached"):
        ExactCacheState(**tensors)


def test_exact_cache_state_rejects_nonfinite_scores_only_when_valid() -> None:
    tensors = _valid_state_tensors()
    tensors["scores"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="scores.*finite.*valid"):
        ExactCacheState(**tensors)

    tensors["valid"][0, 0, 0] = False
    state = ExactCacheState(**tensors)
    assert torch.isnan(state.scores[0, 0, 0])


@pytest.mark.parametrize("field", ["keys", "values"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_exact_cache_state_rejects_nonfinite_key_values_only_when_valid(
    field: str,
    nonfinite: float,
) -> None:
    tensors = _valid_state_tensors()
    tensors[field][0, 0, 0, 0] = nonfinite

    with pytest.raises(ValueError, match=f"{field}.*finite.*valid"):
        ExactCacheState(**tensors)

    tensors["valid"][0, 0, 0] = False
    state = ExactCacheState(**tensors)
    assert not torch.isfinite(getattr(state, field)[0, 0, 0, 0])


def _valid_block_inputs() -> list[torch.Tensor]:
    return [
        torch.zeros(2, 3, 4, 5),
        torch.zeros(2, 3, 4, 7),
        torch.zeros(2, 3, 4),
        torch.arange(3).expand(2, 3),
        torch.ones(2, 3, dtype=torch.bool),
    ]


@pytest.mark.parametrize(
    ("index", "replacement", "error", "match"),
    [
        (0, torch.zeros(2, 3, 5), ValueError, "block_k.*4"),
        (1, torch.zeros(2, 3, 5, 7), ValueError, "block_v.*shape"),
        (2, torch.zeros(2, 3, 5), ValueError, "block_scores.*shape"),
        (3, torch.zeros(2, 4, dtype=torch.int64), ValueError, "block_positions.*shape"),
        (4, torch.ones(2, 4, dtype=torch.bool), ValueError, "block_valid.*shape"),
        (0, torch.zeros(2, 3, 4, 5, dtype=torch.int64), TypeError, "block_k.*floating"),
        (1, torch.zeros(2, 3, 4, 7, dtype=torch.int64), TypeError, "block_v.*floating"),
        (2, torch.zeros(2, 3, 4, dtype=torch.int64), TypeError, "block_scores.*floating"),
        (3, torch.zeros(2, 3), TypeError, "block_positions.*integer"),
        (3, torch.zeros(2, 3, dtype=torch.bool), TypeError, "block_positions.*integer"),
        (4, torch.ones(2, 3, dtype=torch.uint8), TypeError, "block_valid.*bool"),
    ],
)
def test_merge_rejects_noncanonical_block_shapes_and_dtypes(
    index: int,
    replacement: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    inputs = _valid_block_inputs()
    inputs[index] = replacement

    with pytest.raises(error, match=match):
        merge_persistent_cache(
            None,
            *inputs,
            width=2,
            storage_dtype=torch.float32,
        )


def test_merge_rejects_non_tensor_and_cross_device_block_inputs() -> None:
    inputs: list[object] = _valid_block_inputs()
    inputs[0] = []
    with pytest.raises(TypeError, match="block_k.*Tensor"):
        merge_persistent_cache(  # type: ignore[arg-type]
            None,
            *inputs,
            width=2,
            storage_dtype=torch.float32,
        )

    inputs = _valid_block_inputs()
    inputs[1] = torch.zeros(2, 3, 4, 7, device="meta")
    with pytest.raises(ValueError, match="device"):
        merge_persistent_cache(
            None,
            *inputs,
            width=2,
            storage_dtype=torch.float32,
        )


@pytest.mark.parametrize(
    ("state", "match"),
    [
        (
            ExactCacheState(
                keys=torch.zeros(1, 4, 2, 5),
                values=torch.zeros(1, 4, 2, 7),
                scores=torch.zeros(1, 4, 2),
                positions=torch.zeros(1, 4, 2, dtype=torch.int64),
                valid=torch.ones(1, 4, 2, dtype=torch.bool),
            ),
            "state.*batch",
        ),
        (
            ExactCacheState(
                keys=torch.zeros(2, 4, 2, 6),
                values=torch.zeros(2, 4, 2, 7),
                scores=torch.zeros(2, 4, 2),
                positions=torch.zeros(2, 4, 2, dtype=torch.int64),
                valid=torch.ones(2, 4, 2, dtype=torch.bool),
            ),
            "state.*key",
        ),
        (
            ExactCacheState(
                keys=torch.zeros(2, 4, 2, 5),
                values=torch.zeros(2, 4, 2, 8),
                scores=torch.zeros(2, 4, 2),
                positions=torch.zeros(2, 4, 2, dtype=torch.int64),
                valid=torch.ones(2, 4, 2, dtype=torch.bool),
            ),
            "state.*value",
        ),
    ],
)
def test_merge_rejects_state_shape_mismatches(
    state: ExactCacheState, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        merge_persistent_cache(
            state,
            *_valid_block_inputs(),
            width=2,
            storage_dtype=torch.float32,
        )


def test_merge_rejects_non_state_object() -> None:
    with pytest.raises(TypeError, match="state.*ExactCacheState"):
        merge_persistent_cache(  # type: ignore[arg-type]
            object(),
            *_valid_block_inputs(),
            width=2,
            storage_dtype=torch.float32,
        )


def test_merge_rejects_negative_width() -> None:
    with pytest.raises(ValueError, match="width.*nonnegative"):
        merge_persistent_cache(
            None,
            *_valid_block_inputs(),
            width=-1,
            storage_dtype=torch.float32,
        )


@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.float64, "float32"])
def test_merge_rejects_invalid_storage_dtype(storage_dtype: object) -> None:
    with pytest.raises(TypeError, match="storage_dtype.*float32.*bfloat16"):
        merge_persistent_cache(
            None,
            *_valid_block_inputs(),
            width=2,
            storage_dtype=storage_dtype,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_merge_rejects_nonfinite_scores_at_valid_block_positions(
    nonfinite: float,
) -> None:
    inputs = _valid_block_inputs()
    inputs[2][0, 0, 0] = nonfinite

    with pytest.raises(ValueError, match="scores.*finite.*valid"):
        merge_persistent_cache(
            None,
            *inputs,
            width=2,
            storage_dtype=torch.float32,
        )


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_merge_rejects_nonfinite_key_values_at_valid_block_positions(
    input_index: int,
    nonfinite: float,
) -> None:
    inputs = _valid_block_inputs()
    inputs[input_index][0, 0, 0, 0] = nonfinite
    field = "block_k" if input_index == 0 else "block_v"

    with pytest.raises(ValueError, match=f"{field}.*finite.*valid"):
        merge_persistent_cache(
            None,
            *inputs,
            width=2,
            storage_dtype=torch.float32,
        )


def test_merge_ignores_nonfinite_key_values_at_invalid_block_positions() -> None:
    state = merge_persistent_cache(
        None,
        torch.tensor([[[[float("nan")]], [[7.0]]]]),
        torch.tensor([[[[float("inf")]], [[17.0]]]]),
        torch.tensor([[[float("-inf")], [3.0]]]),
        torch.tensor([[4, 5]]),
        torch.tensor([[False, True]]),
        width=2,
        storage_dtype=torch.float32,
    )

    torch.testing.assert_close(state.keys, torch.tensor([[[[7.0], [0.0]]]]))
    torch.testing.assert_close(state.values, torch.tensor([[[[17.0], [0.0]]]]))
    torch.testing.assert_close(state.positions, torch.tensor([[[5, -1]]]))
    torch.testing.assert_close(state.valid, torch.tensor([[[True, False]]]))
    assert torch.isfinite(state.keys).all()
    assert torch.isfinite(state.values).all()


def test_merge_ignores_nonfinite_key_values_in_invalid_persistent_slots() -> None:
    prior = ExactCacheState(
        keys=torch.tensor([[[[float("nan")]]]]),
        values=torch.tensor([[[[float("-inf")]]]]),
        scores=torch.tensor([[[float("inf")]]]),
        positions=torch.tensor([[[99]]]),
        valid=torch.tensor([[[False]]]),
    )

    state = merge_persistent_cache(
        prior,
        torch.tensor([[[[8.0]]]]),
        torch.tensor([[[[18.0]]]]),
        torch.tensor([[[4.0]]]),
        torch.tensor([[6]]),
        torch.tensor([[True]]),
        width=2,
        storage_dtype=torch.float32,
    )

    torch.testing.assert_close(state.keys, torch.tensor([[[[8.0], [0.0]]]]))
    torch.testing.assert_close(state.values, torch.tensor([[[[18.0], [0.0]]]]))
    torch.testing.assert_close(state.positions, torch.tensor([[[6, -1]]]))
    torch.testing.assert_close(state.valid, torch.tensor([[[True, False]]]))
    assert torch.isfinite(state.keys).all()
    assert torch.isfinite(state.values).all()


def _cache_config(
    *,
    read: str = "unit_l2",
    storage_dtype: str = "fp32",
    width: int = 2,
) -> CacheConfig:
    return CacheConfig(
        width=width,
        block_size=4,
        read=read,
        storage_dtype=storage_dtype,
    )


@pytest.mark.parametrize(
    ("key_dim", "heads", "error", "match"),
    [
        (True, 2, TypeError, "key_dim.*exact.*int"),
        (2.0, 2, TypeError, "key_dim.*exact.*int"),
        (0, 2, ValueError, "key_dim.*positive"),
        (2, False, TypeError, "heads.*exact.*int"),
        (2, 1.0, TypeError, "heads.*exact.*int"),
        (2, -1, ValueError, "heads.*positive"),
    ],
)
def test_initialize_cache_read_parameters_validates_exact_positive_dimensions(
    key_dim: object,
    heads: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        exact_cache_module.initialize_cache_read_parameters(  # type: ignore[arg-type]
            key_dim,
            heads,
        )


def test_initialize_cache_read_parameters_returns_direct_trainable_fp32_values() -> None:
    parameters = exact_cache_module.initialize_cache_read_parameters(3, 2)

    assert isinstance(parameters, exact_cache_module.CacheReadParameters)
    assert parameters.gamma_q.shape == parameters.gamma_k.shape == (3,)
    assert parameters.sink_logit.shape == parameters.amplitude.shape == (2,)
    for parameter in (
        parameters.gamma_q,
        parameters.gamma_k,
        parameters.sink_logit,
        parameters.amplitude,
    ):
        assert isinstance(parameter, torch.nn.Parameter)
        assert parameter.dtype == torch.float32
        assert parameter.requires_grad
    torch.testing.assert_close(parameters.gamma_q, torch.ones(3))
    torch.testing.assert_close(parameters.gamma_k, torch.ones(3))
    torch.testing.assert_close(parameters.sink_logit, torch.zeros(2))
    torch.testing.assert_close(parameters.amplitude, torch.zeros(2))
    assert parameters.gamma_q.numel() == 3
    assert parameters.gamma_k.numel() == 3


@pytest.mark.parametrize("read", ["unit_l2", "fixed_temperature", "rmsnorm"])
def test_cache_read_matches_manual_logits_and_raw_value_output(read: str) -> None:
    eps = 1.0e-6
    q = torch.tensor([[[[3.0, 4.0]], [[3.0, 4.0]]]])
    block_k = torch.tensor([[[[3.0, 4.0]], [[-4.0, 3.0]]]])
    raw_v = torch.tensor([[[[2.0]], [[10.0]]]])
    deliberately_different_u = torch.tensor([[[[-20.0]], [[-100.0]]]])
    sink_logit = torch.tensor([-0.25])

    y_cache, diagnostics = exact_cache_module.cache_read_blocks(
        q,
        torch.tensor([[7, 8]]),
        None,
        block_k,
        raw_v,
        torch.zeros(1, 2, 1),
        torch.tensor([[7, 8]]),
        torch.ones(1, 2, dtype=torch.bool),
        _cache_config(read=read),
        torch.ones(2),
        torch.ones(2),
        sink_logit,
    )

    if read == "unit_l2":
        first_logit = 1.0 / (2.0**0.5)
    elif read == "fixed_temperature":
        first_logit = 2.0**0.5
    else:
        q_rms = (12.5 + eps) ** 0.5
        first_logit = (25.0 / (q_rms * q_rms)) / (2.0**0.5)
    expected_weights = torch.softmax(
        torch.tensor([first_logit, 0.0, -0.25]),
        dim=0,
    )
    expected_y = expected_weights[0] * 2.0 + expected_weights[1] * 10.0

    torch.testing.assert_close(diagnostics.attention_weights[0, 1, 0], expected_weights)
    torch.testing.assert_close(y_cache[0, 1, 0, 0], expected_y)
    assert y_cache.dtype == torch.float32
    assert not torch.equal(y_cache, deliberately_different_u)


def _causal_read(
    block_k: torch.Tensor,
    block_v: torch.Tensor,
) -> tuple[torch.Tensor, object]:
    state = ExactCacheState(
        keys=torch.tensor([[[[1.0], [1.0]]]]),
        values=torch.tensor([[[[10.0], [99.0]]]]),
        scores=torch.tensor([[[2.0, 1.0]]]),
        positions=torch.tensor([[[3, 100]]]),
        valid=torch.ones(1, 1, 2, dtype=torch.bool),
    )
    return exact_cache_module.cache_read_blocks(
        torch.ones(1, 2, 1, 1),
        torch.tensor([[7, 11]]),
        state,
        block_k,
        block_v,
        torch.zeros(1, 2, 1),
        torch.tensor([[7, 11]]),
        torch.ones(1, 2, dtype=torch.bool),
        _cache_config(),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1),
    )


def test_cache_read_is_inclusive_causal_and_orders_persistent_before_block() -> None:
    y_cache, diagnostics = _causal_read(
        torch.ones(1, 2, 1, 1),
        torch.tensor([[[[20.0]], [[30.0]]]]),
    )

    torch.testing.assert_close(
        diagnostics.hit_ready_positions,
        torch.tensor([[[[3, -1, 7, -1]], [[3, -1, 7, 11]]]]),
    )
    torch.testing.assert_close(
        diagnostics.candidate_valid,
        torch.tensor([[[[True, False, True, False]], [[True, False, True, True]]]]),
    )
    first_weights = torch.softmax(torch.tensor([1.0, 1.0, 0.0]), dim=0)
    second_weights = torch.softmax(torch.tensor([1.0, 1.0, 1.0, 0.0]), dim=0)
    torch.testing.assert_close(
        y_cache[0, 0, 0, 0],
        first_weights[0] * 10.0 + first_weights[1] * 20.0,
    )
    torch.testing.assert_close(
        y_cache[0, 1, 0, 0],
        second_weights[0] * 10.0
        + second_weights[1] * 20.0
        + second_weights[2] * 30.0,
    )


def test_cache_read_prior_outputs_are_invariant_to_future_block_changes() -> None:
    baseline, _ = _causal_read(
        torch.ones(1, 2, 1, 1),
        torch.tensor([[[[20.0]], [[30.0]]]]),
    )
    changed_future, _ = _causal_read(
        torch.tensor([[[[1.0]], [[-1000.0]]]]),
        torch.tensor([[[[20.0]], [[-9000.0]]]]),
    )

    torch.testing.assert_close(baseline[:, :1], changed_future[:, :1])
    assert not torch.equal(baseline[:, 1:], changed_future[:, 1:])


def test_cache_read_masks_future_block_indices_even_when_positions_are_visible() -> None:
    _, diagnostics = exact_cache_module.cache_read_blocks(
        torch.ones(1, 2, 1, 1),
        torch.tensor([[100, 100]]),
        None,
        torch.ones(1, 2, 1, 1),
        torch.tensor([[[[10.0]], [[9000.0]]]]),
        torch.zeros(1, 2, 1),
        torch.tensor([[0, 0]]),
        torch.ones(1, 2, dtype=torch.bool),
        _cache_config(),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1),
    )

    torch.testing.assert_close(
        diagnostics.candidate_valid,
        torch.tensor([[[[True, False]], [[True, True]]]]),
    )


def test_cache_read_keeps_batches_and_heads_independent() -> None:
    block_v = torch.tensor([2.0, 20.0, 200.0, 2000.0]).reshape(2, 1, 2, 1)
    y_cache, _ = exact_cache_module.cache_read_blocks(
        torch.zeros(2, 1, 2, 1),
        torch.tensor([[50], [500]]),
        None,
        torch.ones(2, 1, 2, 1),
        block_v,
        torch.zeros(2, 1, 2),
        torch.tensor([[50], [500]]),
        torch.ones(2, 1, dtype=torch.bool),
        _cache_config(),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(2),
    )

    torch.testing.assert_close(y_cache, block_v / 2.0)


def test_cache_read_sink_only_is_exact_zero_and_has_sink_semantics() -> None:
    y_cache, diagnostics = exact_cache_module.cache_read_blocks(
        torch.randn(1, 2, 2, 3),
        torch.tensor([[10, 20]]),
        None,
        torch.empty(1, 0, 2, 3),
        torch.empty(1, 0, 2, 4),
        torch.empty(1, 0, 2),
        torch.empty(1, 0, dtype=torch.int64),
        torch.empty(1, 0, dtype=torch.bool),
        _cache_config(),
        torch.ones(3),
        torch.ones(3),
        torch.tensor([0.25, -0.5]),
    )

    assert torch.equal(y_cache, torch.zeros_like(y_cache))
    assert torch.isfinite(y_cache).all()
    assert torch.equal(diagnostics.sink_mass, torch.ones(1, 2, 2))
    assert torch.equal(diagnostics.top1_positions, torch.full((1, 2, 2), -1))
    assert torch.equal(diagnostics.attention_entropy, torch.zeros(1, 2, 2))


def test_cache_read_learned_sink_competes_with_candidates() -> None:
    y_cache, diagnostics = exact_cache_module.cache_read_blocks(
        torch.ones(1, 1, 1, 1),
        torch.tensor([[4]]),
        None,
        torch.ones(1, 1, 1, 1),
        torch.full((1, 1, 1, 1), 5.0),
        torch.zeros(1, 1, 1),
        torch.tensor([[4]]),
        torch.ones(1, 1, dtype=torch.bool),
        _cache_config(),
        torch.ones(1),
        torch.ones(1),
        torch.tensor([2.0]),
    )
    expected = torch.softmax(torch.tensor([1.0, 2.0]), dim=0)

    torch.testing.assert_close(diagnostics.attention_weights[0, 0, 0], expected)
    torch.testing.assert_close(y_cache[0, 0, 0, 0], expected[0] * 5.0)
    assert diagnostics.top1_positions.item() == -1
    torch.testing.assert_close(diagnostics.sink_mass[0, 0, 0], expected[1])


def test_cache_read_bfloat16_roundtrips_block_storage_before_fp32_read() -> None:
    value = torch.tensor([[[[1.001]]]])
    common = (
        torch.ones(1, 1, 1, 1),
        torch.tensor([[1]]),
        None,
        torch.tensor([[[[1.001]]]]),
        value,
        torch.zeros(1, 1, 1),
        torch.tensor([[1]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    fp32, _ = exact_cache_module.cache_read_blocks(
        *common,
        _cache_config(storage_dtype="fp32"),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1),
    )
    bf16, _ = exact_cache_module.cache_read_blocks(
        *common,
        _cache_config(storage_dtype="bf16"),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1),
    )
    candidate_mass = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[0]

    torch.testing.assert_close(fp32, value * candidate_mass)
    torch.testing.assert_close(bf16, value.to(torch.bfloat16).float() * candidate_mass)
    assert not torch.equal(fp32, bf16)


def test_cache_read_gradients_flow_only_through_differentiable_read_inputs() -> None:
    q = torch.tensor([[[[1.0, 2.0]]]], requires_grad=True)
    block_k = torch.tensor(
        [[[[2.0, -1.0]], [[1.0, 1.0]]]],
        requires_grad=True,
    )
    block_v = torch.tensor(
        [[[[3.0, 4.0]], [[-2.0, 5.0]]]],
        requires_grad=True,
    )
    block_scores = torch.zeros(1, 2, 1, requires_grad=True)
    gamma_q = torch.nn.Parameter(torch.tensor([1.1, 0.9]))
    gamma_k = torch.nn.Parameter(torch.tensor([0.8, 1.2]))
    sink_logit = torch.nn.Parameter(torch.tensor([0.2]))

    y_cache, diagnostics = exact_cache_module.cache_read_blocks(
        q,
        torch.tensor([[2]]),
        None,
        block_k,
        block_v,
        block_scores,
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        _cache_config(read="rmsnorm"),
        gamma_q,
        gamma_k,
        sink_logit,
    )
    (y_cache.square().sum() + diagnostics.attention_entropy.sum()).backward()

    for tensor in (q, block_k, block_v, gamma_q, gamma_k, sink_logit):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert block_scores.grad is None


def test_cache_read_diagnostics_report_exact_shapes_values_and_bytes() -> None:
    state = ExactCacheState(
        keys=torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16),
        values=torch.tensor(
            [[[[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]]],
            dtype=torch.bfloat16,
        ),
        scores=torch.tensor([[[2.0, 1.0]]]),
        positions=torch.tensor([[[5, 6]]]),
        valid=torch.tensor([[[True, False]]]),
    )
    y_cache, diagnostics = exact_cache_module.cache_read_blocks(
        torch.zeros(1, 2, 1, 2),
        torch.tensor([[5, 9]]),
        state,
        torch.zeros(1, 2, 1, 2),
        torch.tensor([[[[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]]),
        torch.zeros(1, 2, 1),
        torch.tensor([[5, 9]]),
        torch.ones(1, 2, dtype=torch.bool),
        _cache_config(storage_dtype="bf16"),
        torch.ones(2),
        torch.ones(2),
        torch.zeros(1),
    )

    assert isinstance(diagnostics, exact_cache_module.CacheReadDiagnostics)
    assert y_cache.shape == (1, 2, 1, 3)
    assert diagnostics.persistent_selected_positions.shape == (1, 1, 2)
    assert diagnostics.hit_ready_positions.shape == (1, 2, 1, 4)
    assert diagnostics.candidate_valid.shape == (1, 2, 1, 4)
    assert diagnostics.attention_weights.shape == (1, 2, 1, 5)
    for field in (
        diagnostics.top1_positions,
        diagnostics.attention_entropy,
        diagnostics.top1_mass,
        diagnostics.sink_mass,
    ):
        assert field.shape == (1, 2, 1)
    torch.testing.assert_close(
        diagnostics.persistent_selected_positions,
        torch.tensor([[[5, -1]]]),
    )
    torch.testing.assert_close(
        diagnostics.hit_ready_positions,
        torch.tensor([[[[5, -1, 5, -1]], [[5, -1, 5, 9]]]]),
    )
    torch.testing.assert_close(
        diagnostics.candidate_valid,
        torch.tensor([[[[True, False, True, False]], [[True, False, True, True]]]]),
    )
    torch.testing.assert_close(
        diagnostics.attention_weights,
        torch.tensor(
            [[[[1 / 3, 0.0, 1 / 3, 0.0, 1 / 3]], [[0.25, 0.0, 0.25, 0.25, 0.25]]]]
        ),
    )
    torch.testing.assert_close(
        diagnostics.top1_positions,
        torch.tensor([[[5], [5]]]),
    )
    torch.testing.assert_close(
        diagnostics.attention_entropy,
        torch.tensor([[[torch.log(torch.tensor(3.0))], [torch.log(torch.tensor(4.0))]]]),
    )
    torch.testing.assert_close(
        diagnostics.top1_mass,
        torch.tensor([[[1 / 3], [0.25]]]),
    )
    torch.testing.assert_close(
        diagnostics.sink_mass,
        torch.tensor([[[1 / 3], [0.25]]]),
    )
    assert diagnostics.persistent_bytes == state.nbytes
    assert diagnostics.block_bytes == 1 * 1 * 2 * (2 * 2 + 3 * 2 + 4 + 8 + 1)


def test_cache_read_width_zero_uses_current_block_only() -> None:
    empty_state = ExactCacheState(
        keys=torch.empty(1, 2, 0, 1),
        values=torch.empty(1, 2, 0, 1),
        scores=torch.empty(1, 2, 0),
        positions=torch.empty(1, 2, 0, dtype=torch.int64),
        valid=torch.empty(1, 2, 0, dtype=torch.bool),
    )
    y_cache, diagnostics = exact_cache_module.cache_read_blocks(
        torch.ones(1, 1, 2, 1),
        torch.tensor([[101]]),
        empty_state,
        torch.ones(1, 2, 2, 1),
        torch.tensor([2.0, 20.0, 4.0, 40.0]).reshape(1, 2, 2, 1),
        torch.zeros(1, 2, 2),
        torch.tensor([[100, 102]]),
        torch.ones(1, 2, dtype=torch.bool),
        _cache_config(width=0),
        torch.ones(1),
        torch.ones(1),
        torch.zeros(2),
    )
    expected_mass = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[0]

    assert diagnostics.persistent_selected_positions.shape == (1, 2, 0)
    torch.testing.assert_close(
        diagnostics.hit_ready_positions,
        torch.tensor([[[[100, -1], [100, -1]]]]),
    )
    torch.testing.assert_close(
        y_cache,
        torch.tensor([[[[2.0], [20.0]]]]) * expected_mass,
    )
    assert diagnostics.persistent_bytes == 0


@pytest.mark.parametrize("read", ["unit_l2", "rmsnorm"])
def test_cache_read_normalization_does_not_mutate_inputs_or_state(read: str) -> None:
    q = torch.tensor([[[[3.0, 4.0]]]])
    block_k = torch.tensor([[[[5.0, 12.0]]]])
    block_v = torch.tensor([[[[7.0]]]])
    state = ExactCacheState(
        keys=torch.tensor([[[[8.0, 15.0], [0.0, 0.0]]]]),
        values=torch.tensor([[[[9.0], [0.0]]]]),
        scores=torch.tensor([[[1.0, 0.0]]]),
        positions=torch.tensor([[[1, -1]]]),
        valid=torch.tensor([[[True, False]]]),
    )
    originals = {
        "q": q.clone(),
        "block_k": block_k.clone(),
        "block_v": block_v.clone(),
        "state_keys": state.keys.clone(),
        "state_values": state.values.clone(),
        "state_positions": state.positions.clone(),
        "state_valid": state.valid.clone(),
    }

    exact_cache_module.cache_read_blocks(
        q,
        torch.tensor([[2]]),
        state,
        block_k,
        block_v,
        torch.zeros(1, 1, 1),
        torch.tensor([[2]]),
        torch.ones(1, 1, dtype=torch.bool),
        _cache_config(read=read),
        torch.ones(2),
        torch.ones(2),
        torch.zeros(1),
    )

    assert torch.equal(q, originals["q"])
    assert torch.equal(block_k, originals["block_k"])
    assert torch.equal(block_v, originals["block_v"])
    assert torch.equal(state.keys, originals["state_keys"])
    assert torch.equal(state.values, originals["state_values"])
    assert torch.equal(state.positions, originals["state_positions"])
    assert torch.equal(state.valid, originals["state_valid"])


def _valid_cache_read_kwargs() -> dict[str, object]:
    return {
        "q_eff": torch.ones(2, 3, 4, 5),
        "query_positions": torch.arange(3, dtype=torch.int64).expand(2, 3),
        "state": None,
        "block_k": torch.ones(2, 3, 4, 5),
        "block_v": torch.ones(2, 3, 4, 7),
        "block_scores": torch.zeros(2, 3, 4),
        "block_positions": torch.arange(3, dtype=torch.int64).expand(2, 3),
        "block_valid": torch.ones(2, 3, dtype=torch.bool),
        "config": _cache_config(),
        "gamma_q": torch.ones(5),
        "gamma_k": torch.ones(5),
        "sink_logit": torch.zeros(4),
    }


@pytest.mark.parametrize(
    ("field", "replacement", "error", "match"),
    [
        ("q_eff", torch.ones(2, 3, 5), ValueError, "q_eff.*4"),
        ("query_positions", torch.zeros(2, 4, dtype=torch.int64), ValueError, "query_positions.*shape"),
        ("block_k", torch.ones(2, 3, 4), ValueError, "block_k.*4"),
        ("block_k", torch.ones(2, 3, 4, 6), ValueError, "block_k.*shape"),
        ("block_v", torch.ones(2, 3, 3, 7), ValueError, "block_v.*shape"),
        ("block_scores", torch.zeros(2, 3, 3), ValueError, "block_scores.*shape"),
        ("block_positions", torch.zeros(2, 4, dtype=torch.int64), ValueError, "block_positions.*shape"),
        ("block_valid", torch.ones(2, 4, dtype=torch.bool), ValueError, "block_valid.*shape"),
        ("gamma_q", torch.ones(4), ValueError, "gamma_q.*shape"),
        ("gamma_k", torch.ones(4), ValueError, "gamma_k.*shape"),
        ("sink_logit", torch.zeros(3), ValueError, "sink_logit.*shape"),
        ("q_eff", torch.ones(2, 3, 4, 5, dtype=torch.float64), TypeError, "q_eff.*float32"),
        ("query_positions", torch.zeros(2, 3, dtype=torch.int32), TypeError, "query_positions.*int64"),
        ("block_k", torch.ones(2, 3, 4, 5, dtype=torch.float64), TypeError, "block_k.*float32"),
        ("block_v", torch.ones(2, 3, 4, 7, dtype=torch.bfloat16), TypeError, "block_v.*float32"),
        ("block_scores", torch.zeros(2, 3, 4, dtype=torch.float64), TypeError, "block_scores.*float32"),
        ("block_positions", torch.zeros(2, 3, dtype=torch.int32), TypeError, "block_positions.*int64"),
        ("block_valid", torch.ones(2, 3, dtype=torch.uint8), TypeError, "block_valid.*bool"),
        ("gamma_q", torch.ones(5, dtype=torch.float64), TypeError, "gamma_q.*float32"),
        ("gamma_k", torch.ones(5, dtype=torch.bfloat16), TypeError, "gamma_k.*float32"),
        ("sink_logit", torch.zeros(4, dtype=torch.float64), TypeError, "sink_logit.*float32"),
    ],
)
def test_cache_read_rejects_noncanonical_shapes_and_dtypes(
    field: str,
    replacement: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    kwargs = _valid_cache_read_kwargs()
    kwargs[field] = replacement

    with pytest.raises(error, match=match):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]


def test_cache_read_rejects_non_tensor_cross_device_and_non_state_inputs() -> None:
    kwargs = _valid_cache_read_kwargs()
    kwargs["q_eff"] = []
    with pytest.raises(TypeError, match="q_eff.*Tensor"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]

    kwargs = _valid_cache_read_kwargs()
    kwargs["sink_logit"] = torch.zeros(4, device="meta")
    with pytest.raises(ValueError, match="device"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]

    kwargs = _valid_cache_read_kwargs()
    kwargs["state"] = object()
    with pytest.raises(TypeError, match="state.*ExactCacheState"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "index", "nonfinite"),
    [
        ("q_eff", (0, 0, 0, 0), float("nan")),
        ("block_k", (0, 0, 0, 0), float("inf")),
        ("block_v", (0, 0, 0, 0), float("-inf")),
        ("block_scores", (0, 0, 0), float("nan")),
        ("gamma_q", (0,), float("inf")),
        ("gamma_k", (0,), float("nan")),
        ("sink_logit", (0,), float("-inf")),
    ],
)
def test_cache_read_rejects_nonfinite_values_used_by_the_read(
    field: str,
    index: tuple[int, ...],
    nonfinite: float,
) -> None:
    kwargs = _valid_cache_read_kwargs()
    tensor = kwargs[field]
    assert isinstance(tensor, torch.Tensor)
    tensor[index] = nonfinite

    with pytest.raises(ValueError, match=f"{field}.*finite"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]


def test_cache_read_ignores_nonfinite_values_at_invalid_candidate_slots() -> None:
    kwargs = _valid_cache_read_kwargs()
    block_k = kwargs["block_k"]
    block_v = kwargs["block_v"]
    block_scores = kwargs["block_scores"]
    block_valid = kwargs["block_valid"]
    assert isinstance(block_k, torch.Tensor)
    assert isinstance(block_v, torch.Tensor)
    assert isinstance(block_scores, torch.Tensor)
    assert isinstance(block_valid, torch.Tensor)
    block_valid[0, 0] = False
    block_k[0, 0] = float("nan")
    block_v[0, 0] = float("inf")
    block_scores[0, 0] = float("-inf")

    y_cache, diagnostics = exact_cache_module.cache_read_blocks(  # type: ignore[arg-type]
        **kwargs
    )

    assert torch.isfinite(y_cache).all()
    assert not diagnostics.candidate_valid[0, :, :, 0].any()


def test_cache_read_rejects_nonfinite_persistent_values_only_when_valid() -> None:
    kwargs = _valid_cache_read_kwargs()
    state = ExactCacheState(
        keys=torch.ones(2, 4, 2, 5),
        values=torch.ones(2, 4, 2, 7),
        scores=torch.zeros(2, 4, 2),
        positions=torch.zeros(2, 4, 2, dtype=torch.int64),
        valid=torch.ones(2, 4, 2, dtype=torch.bool),
    )
    state.keys[0, 0, 0, 0] = float("nan")
    kwargs["state"] = state
    with pytest.raises(ValueError, match="state.keys.*finite.*valid"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]

    state.valid[0, 0, 0] = False
    y_cache, _ = exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]
    assert torch.isfinite(y_cache).all()


def test_cache_read_rejects_nonfinite_computed_logits() -> None:
    kwargs = _valid_cache_read_kwargs()
    kwargs["config"] = _cache_config(read="rmsnorm")
    kwargs["gamma_q"] = torch.full((5,), 3.0e38)
    kwargs["gamma_k"] = torch.full((5,), 3.0e38)

    with pytest.raises(ValueError, match="nonfinite.*logit"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]


def test_cache_read_rejects_unsupported_config_and_read_policy() -> None:
    kwargs = _valid_cache_read_kwargs()
    kwargs["config"] = object()
    with pytest.raises(TypeError, match="config.*CacheConfig"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]

    kwargs = _valid_cache_read_kwargs()
    config = kwargs["config"]
    assert isinstance(config, CacheConfig)
    object.__setattr__(config, "read", "unsupported")
    with pytest.raises(ValueError, match="unsupported.*read"):
        exact_cache_module.cache_read_blocks(**kwargs)  # type: ignore[arg-type]
