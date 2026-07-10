from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from research.kmd2_ablation.exact_cache import (
    deterministic_topw,
    reference_scan_with_scores,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_exact_cache_import_isolated_from_optional_acceleration_dependencies() -> None:
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

        module = importlib.import_module("research.kmd2_ablation.exact_cache")
        expected_symbols = {
            "CacheReadDiagnostics",
            "CacheReadParameters",
            "ExactCacheState",
            "cache_read_blocks",
            "deterministic_topw",
            "initialize_cache_read_parameters",
            "merge_persistent_cache",
            "reference_scan_with_scores",
        }

        assert expected_symbols <= set(vars(module))
        assert blocked_dependency_roots.isdisjoint(sys.modules)
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


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _scalar_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    out_mix: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deliberately scalar recurrence, independent of batched matrix ops."""
    dtype = _compute_dtype(q.dtype)
    q_c, k_c, v_c, decay_c, beta_e_c, beta_w_c = (
        tensor.to(dtype) for tensor in (q, k, v, decay, beta_e, beta_w)
    )
    mix_c = None if out_mix is None else out_mix.to(dtype)
    batch, steps, heads, slots, key_dim = q.shape
    value_dim = v.shape[-1]

    states = [
        [
            [
                [torch.zeros((), dtype=dtype, device=q.device) for _ in range(value_dim)]
                for _ in range(key_dim)
            ]
            for _ in range(heads)
        ]
        for _ in range(batch)
    ]
    token_outputs: list[torch.Tensor] = []
    token_scores: list[torch.Tensor] = []

    for token in range(steps):
        batch_outputs: list[torch.Tensor] = []
        batch_scores: list[torch.Tensor] = []
        for batch_idx in range(batch):
            head_outputs: list[torch.Tensor] = []
            head_scores: list[torch.Tensor] = []
            for head in range(heads):
                state_bar = [
                    [
                        decay_c[batch_idx, token, head, row]
                        * states[batch_idx][head][row][column]
                        for column in range(value_dim)
                    ]
                    for row in range(key_dim)
                ]
                memory = [
                    sum(
                        k_c[batch_idx, token, head, row] * state_bar[row][column]
                        for row in range(key_dim)
                    )
                    for column in range(value_dim)
                ]
                update = [
                    beta_w_c[batch_idx, token, head]
                    * v_c[batch_idx, token, head, column]
                    - beta_e_c[batch_idx, token, head] * memory[column]
                    for column in range(value_dim)
                ]
                state = [
                    [
                        state_bar[row][column]
                        + k_c[batch_idx, token, head, row] * update[column]
                        for column in range(value_dim)
                    ]
                    for row in range(key_dim)
                ]
                states[batch_idx][head] = state

                slot_reads = [
                    [
                        sum(
                            q_c[batch_idx, token, head, slot, row]
                            * state[row][column]
                            for row in range(key_dim)
                        )
                        for column in range(value_dim)
                    ]
                    for slot in range(slots)
                ]
                if slots == 1:
                    mixed_read = slot_reads[0]
                else:
                    assert mix_c is not None
                    mixed_read = [
                        sum(
                            mix_c[head, slot] * slot_reads[slot][column]
                            for slot in range(slots)
                        )
                        for column in range(value_dim)
                    ]
                head_outputs.append(torch.stack(mixed_read))

                key_norm = torch.sqrt(
                    sum(
                        k_c[batch_idx, token, head, row].square()
                        for row in range(key_dim)
                    )
                )
                update_norm = torch.sqrt(sum(component.square() for component in update))
                head_scores.append(key_norm * update_norm)
            batch_outputs.append(torch.stack(head_outputs))
            batch_scores.append(torch.stack(head_scores))
        token_outputs.append(torch.stack(batch_outputs))
        token_scores.append(torch.stack(batch_scores))

    return torch.stack(token_outputs, dim=1), torch.stack(token_scores, dim=1)


def _random_inputs(
    *, dtype: torch.dtype, slots: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2718 + slots)
    batch, steps, heads, key_dim, value_dim = 2, 3, 2, 2, 3
    q = torch.randn(batch, steps, heads, slots, key_dim, generator=generator, dtype=dtype)
    k = torch.randn(batch, steps, heads, key_dim, generator=generator, dtype=dtype)
    v = torch.randn(batch, steps, heads, value_dim, generator=generator, dtype=dtype)
    decay = 0.55 + 0.4 * torch.rand(
        batch, steps, heads, key_dim, generator=generator, dtype=dtype
    )
    beta_e = 0.1 + 0.75 * torch.rand(
        batch, steps, heads, generator=generator, dtype=dtype
    )
    beta_w = 0.1 + 0.75 * torch.rand(
        batch, steps, heads, generator=generator, dtype=dtype
    )
    tensors: list[torch.Tensor] = [q, k, v, decay, beta_e, beta_w]
    if slots > 1:
        tensors.append(
            torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [-0.3, 0.7, 0.15, 0.45]],
                dtype=dtype,
            )
        )
    return tuple(tensor.detach().requires_grad_(True) for tensor in tensors)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float64, 1.0e-10, 1.0e-8),
        (torch.float32, 1.0e-6, 1.0e-5),
    ],
)
@pytest.mark.parametrize("slots", [1, 4])
def test_reference_scan_matches_independent_scalar_oracle_forward_and_gradients(
    dtype: torch.dtype, atol: float, rtol: float, slots: int
) -> None:
    actual_inputs = _random_inputs(dtype=dtype, slots=slots)
    oracle_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in actual_inputs
    )
    actual_mix = actual_inputs[6] if slots > 1 else None
    oracle_mix = oracle_inputs[6] if slots > 1 else None

    actual_y, actual_score = reference_scan_with_scores(
        *actual_inputs[:6], out_mix=actual_mix
    )
    oracle_y, oracle_score = _scalar_oracle(
        *oracle_inputs[:6], out_mix=oracle_mix
    )

    torch.testing.assert_close(actual_y, oracle_y, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_score, oracle_score.detach(), atol=atol, rtol=rtol)
    assert actual_y.dtype == dtype
    assert actual_score.dtype == dtype

    loss_weight = torch.linspace(
        -0.75, 1.25, actual_y.numel(), dtype=dtype
    ).reshape_as(actual_y)
    actual_grads = torch.autograd.grad((actual_y * loss_weight).sum(), actual_inputs)
    oracle_grads = torch.autograd.grad((oracle_y * loss_weight).sum(), oracle_inputs)
    for actual_grad, oracle_grad in zip(actual_grads, oracle_grads, strict=True):
        torch.testing.assert_close(actual_grad, oracle_grad, atol=atol, rtol=rtol)


def test_reference_scan_reads_the_updated_state() -> None:
    q = torch.tensor([[[[[3.0]]]]], dtype=torch.float64)
    k = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[5.0]]]], dtype=torch.float64)
    decay = torch.tensor([[[[0.25]]]], dtype=torch.float64)
    beta_e = torch.tensor([[[0.7]]], dtype=torch.float64)
    beta_w = torch.tensor([[[0.4]]], dtype=torch.float64)

    y, score = reference_scan_with_scores(q, k, v, decay, beta_e, beta_w)

    torch.testing.assert_close(y, torch.tensor([[[[12.0]]]], dtype=torch.float64))
    torch.testing.assert_close(score, torch.tensor([[[4.0]]], dtype=torch.float64))


@pytest.mark.parametrize(
    ("key", "expected_score"),
    [
        ([0.0, 0.0], 0.0),
        ([1.0e-12, 0.0], 2.5e-12),
    ],
)
def test_update_score_keeps_the_exact_zero_or_subepsilon_key_factor(
    key: list[float], expected_score: float
) -> None:
    q = torch.ones(1, 1, 1, 1, 2, dtype=torch.float64)
    k = torch.tensor([[[key]]], dtype=torch.float64)
    v = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float64)
    decay = torch.ones(1, 1, 1, 2, dtype=torch.float64)
    beta_e = torch.ones(1, 1, 1, dtype=torch.float64)
    beta_w = torch.full((1, 1, 1), 0.5, dtype=torch.float64)

    _, score = reference_scan_with_scores(q, k, v, decay, beta_e, beta_w)

    assert score.item() == pytest.approx(expected_score, rel=1.0e-12, abs=0.0)


def test_update_scores_are_detached_while_reads_remain_differentiable() -> None:
    q, k, v, decay, beta_e, beta_w = _random_inputs(dtype=torch.float32, slots=1)

    y, score = reference_scan_with_scores(q, k, v, decay, beta_e, beta_w)

    assert y.requires_grad
    assert y.grad_fn is not None
    assert not score.requires_grad
    assert score.grad_fn is None


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_non_float64_inputs_compute_and_return_float32(
    input_dtype: torch.dtype,
) -> None:
    q = torch.ones(1, 1, 1, 1, 2, dtype=input_dtype)
    k = torch.ones(1, 1, 1, 2, dtype=input_dtype)
    v = torch.ones(1, 1, 1, 3, dtype=input_dtype)
    decay = torch.ones(1, 1, 1, 2, dtype=input_dtype)
    beta_e = torch.zeros(1, 1, 1, dtype=input_dtype)
    beta_w = torch.ones(1, 1, 1, dtype=input_dtype)

    y, score = reference_scan_with_scores(q, k, v, decay, beta_e, beta_w)

    assert y.dtype == torch.float32
    assert score.dtype == torch.float32


@pytest.mark.parametrize(
    ("input_dtype", "expected_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_reference_scan_disables_ambient_cpu_autocast_and_preserves_gradients(
    input_dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    baseline_inputs = _random_inputs(dtype=input_dtype, slots=4)
    autocast_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in baseline_inputs
    )

    baseline_y, baseline_score = reference_scan_with_scores(
        *baseline_inputs[:6], out_mix=baseline_inputs[6]
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_y, autocast_score = reference_scan_with_scores(
            *autocast_inputs[:6], out_mix=autocast_inputs[6]
        )

    assert autocast_y.dtype == expected_dtype
    assert autocast_score.dtype == expected_dtype
    torch.testing.assert_close(autocast_y, baseline_y)
    torch.testing.assert_close(autocast_score, baseline_score)

    baseline_grads = torch.autograd.grad(baseline_y.square().sum(), baseline_inputs)
    autocast_grads = torch.autograd.grad(autocast_y.square().sum(), autocast_inputs)
    for autocast_grad, baseline_grad in zip(
        autocast_grads, baseline_grads, strict=True
    ):
        assert torch.isfinite(autocast_grad).all()
        torch.testing.assert_close(autocast_grad, baseline_grad)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("input_dtype", "expected_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_reference_scan_disables_ambient_cuda_autocast_and_preserves_gradients(
    input_dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    baseline_inputs = tuple(
        tensor.detach().cuda().requires_grad_(True)
        for tensor in _random_inputs(dtype=input_dtype, slots=4)
    )
    autocast_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in baseline_inputs
    )

    baseline_y, baseline_score = reference_scan_with_scores(
        *baseline_inputs[:6], out_mix=baseline_inputs[6]
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        autocast_y, autocast_score = reference_scan_with_scores(
            *autocast_inputs[:6], out_mix=autocast_inputs[6]
        )

    assert autocast_y.dtype == expected_dtype
    assert autocast_score.dtype == expected_dtype
    torch.testing.assert_close(autocast_y, baseline_y)
    torch.testing.assert_close(autocast_score, baseline_score)

    baseline_grads = torch.autograd.grad(baseline_y.square().sum(), baseline_inputs)
    autocast_grads = torch.autograd.grad(autocast_y.square().sum(), autocast_inputs)
    for autocast_grad, baseline_grad in zip(
        autocast_grads, baseline_grads, strict=True
    ):
        assert torch.isfinite(autocast_grad).all()
        torch.testing.assert_close(autocast_grad, baseline_grad)


def _valid_shape_inputs() -> list[torch.Tensor | None]:
    return [
        torch.zeros(1, 2, 2, 4, 3),
        torch.zeros(1, 2, 2, 3),
        torch.zeros(1, 2, 2, 5),
        torch.ones(1, 2, 2, 3),
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, 2),
        torch.full((2, 4), 0.25),
    ]


ShapeMutation = Callable[[list[torch.Tensor | None]], None]


def _replace(index: int, value: torch.Tensor | None) -> ShapeMutation:
    def mutate(inputs: list[torch.Tensor | None]) -> None:
        inputs[index] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_replace(0, torch.zeros(1, 2, 2, 3)), "q.*5"),
        (_replace(1, torch.zeros(1, 3, 2, 3)), "k.*shape"),
        (_replace(2, torch.zeros(1, 2, 3, 5)), "v.*shape"),
        (_replace(3, torch.ones(1, 2, 2, 4)), "decay.*shape"),
        (_replace(4, torch.zeros(1, 2, 3)), "beta_e.*shape"),
        (_replace(5, torch.ones(1, 3, 2)), "beta_w.*shape"),
        (_replace(6, None), "out_mix.*required"),
        (_replace(6, torch.ones(2, 3)), "out_mix.*shape"),
    ],
)
def test_reference_scan_rejects_shape_contract_violations(
    mutate: ShapeMutation, match: str
) -> None:
    inputs = _valid_shape_inputs()
    mutate(inputs)

    with pytest.raises(ValueError, match=match):
        reference_scan_with_scores(*inputs[:6], out_mix=inputs[6])


def test_reference_scan_rejects_out_mix_for_one_query_slot() -> None:
    inputs = _valid_shape_inputs()
    inputs[0] = torch.zeros(1, 2, 2, 1, 3)

    with pytest.raises(ValueError, match="out_mix.*None"):
        reference_scan_with_scores(*inputs[:6], out_mix=torch.ones(2, 1))


@pytest.mark.parametrize("input_index", range(7))
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_reference_scan_rejects_nonfinite_inputs(
    input_index: int, nonfinite: float
) -> None:
    inputs = _valid_shape_inputs()
    tensor = inputs[input_index]
    assert isinstance(tensor, torch.Tensor)
    tensor = tensor.clone()
    tensor.reshape(-1)[0] = nonfinite
    inputs[input_index] = tensor

    with pytest.raises(ValueError, match="finite"):
        reference_scan_with_scores(*inputs[:6], out_mix=inputs[6])


def test_reference_scan_returns_declared_shapes_and_finite_values() -> None:
    q, k, v, decay, beta_e, beta_w, out_mix = _random_inputs(
        dtype=torch.float32, slots=4
    )

    y, score = reference_scan_with_scores(
        q, k, v, decay, beta_e, beta_w, out_mix=out_mix
    )

    assert y.shape == (2, 3, 2, 3)
    assert score.shape == (2, 3, 2)
    assert torch.isfinite(y).all()
    assert torch.isfinite(score).all()


def test_deterministic_topw_selects_each_batch_and_head_independently() -> None:
    scores = torch.tensor(
        [
            [[1.0, 3.0, 3.0, 2.0, 999.0], [5.0, 5.0, 4.0, -1.0, 0.0]],
            [[0.0, 1.0, 1.0, 2.0, 2.0], [7.0, 7.0, 7.0, 7.0, 7.0]],
        ],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [
            [[10, 20, 25, 100, 999], [1, 9, 4, 7, 3]],
            [[50, 0, 8, 4, 3], [0, 4, 2, 9, 3]],
        ],
        dtype=torch.int64,
    )
    valid = torch.tensor(
        [
            [[True, True, True, True, False], [True, True, False, True, True]],
            [[True, True, True, True, True], [False, True, False, True, False]],
        ]
    )

    actual = deterministic_topw(scores, positions, valid, width=7)

    expected = torch.tensor(
        [
            [[2, 1, 3, 0, -1, -1, -1], [1, 0, 4, 3, -1, -1, -1]],
            [[3, 4, 2, 1, 0, -1, -1], [3, 1, -1, -1, -1, -1, -1]],
        ],
        dtype=torch.int64,
    )
    torch.testing.assert_close(actual, expected)


def test_deterministic_topw_returns_empty_int64_indices_for_zero_width() -> None:
    scores = torch.ones(2, 2, 3)
    positions = torch.arange(3).expand(2, 2, 3)
    valid = torch.ones(2, 2, 3, dtype=torch.bool)

    actual = deterministic_topw(scores, positions, valid, width=0)

    assert actual.shape == (2, 2, 0)
    assert actual.dtype == torch.int64
    assert actual.device == scores.device


def test_deterministic_topw_pads_all_invalid_rows() -> None:
    scores = torch.tensor([[[3.0, 2.0, 1.0]]])
    positions = torch.tensor([[[30, 20, 10]]])
    valid = torch.zeros(1, 1, 3, dtype=torch.bool)

    actual = deterministic_topw(scores, positions, valid, width=4)

    torch.testing.assert_close(actual, torch.full((1, 1, 4), -1, dtype=torch.int64))


def test_deterministic_topw_ignores_nonfinite_scores_at_invalid_positions() -> None:
    scores = torch.tensor([[[float("nan"), 4.0, float("inf"), -1.0]]])
    positions = torch.tensor([[[400, 10, 500, 20]]])
    valid = torch.tensor([[[False, True, False, True]]])

    actual = deterministic_topw(scores, positions, valid, width=4)

    torch.testing.assert_close(
        actual, torch.tensor([[[1, 3, -1, -1]]], dtype=torch.int64)
    )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_deterministic_topw_rejects_nonfinite_valid_scores(nonfinite: float) -> None:
    scores = torch.tensor([[[1.0, nonfinite]]])
    positions = torch.tensor([[[0, 1]]])
    valid = torch.tensor([[[False, True]]])

    with pytest.raises(ValueError, match="finite"):
        deterministic_topw(scores, positions, valid, width=1)


@pytest.mark.parametrize(
    ("score_shape", "position_shape", "valid_shape"),
    [
        ((2, 3), (2, 3), (2, 3)),
        ((1, 2, 3), (1, 2, 4), (1, 2, 3)),
        ((1, 2, 3), (1, 2, 3), (1, 1, 3)),
    ],
)
def test_deterministic_topw_rejects_noncanonical_or_mismatched_shapes(
    score_shape: tuple[int, ...],
    position_shape: tuple[int, ...],
    valid_shape: tuple[int, ...],
) -> None:
    scores = torch.zeros(score_shape)
    positions = torch.zeros(position_shape, dtype=torch.int64)
    valid = torch.zeros(valid_shape, dtype=torch.bool)

    with pytest.raises(ValueError, match="shape"):
        deterministic_topw(scores, positions, valid, width=1)


@pytest.mark.parametrize(
    ("scores_dtype", "positions_dtype", "valid_dtype", "match"),
    [
        (torch.int64, torch.int64, torch.bool, "scores.*floating"),
        (torch.float32, torch.float32, torch.bool, "positions.*integer"),
        (torch.float32, torch.bool, torch.bool, "positions.*integer"),
        (torch.float32, torch.int64, torch.uint8, "valid.*bool"),
    ],
)
def test_deterministic_topw_rejects_invalid_input_dtypes(
    scores_dtype: torch.dtype,
    positions_dtype: torch.dtype,
    valid_dtype: torch.dtype,
    match: str,
) -> None:
    scores = torch.zeros(1, 1, 2, dtype=scores_dtype)
    positions = torch.zeros(1, 1, 2, dtype=positions_dtype)
    valid = torch.ones(1, 1, 2, dtype=valid_dtype)

    with pytest.raises(TypeError, match=match):
        deterministic_topw(scores, positions, valid, width=1)


@pytest.mark.parametrize("width", [True, 1.0, torch.tensor(1)])
def test_deterministic_topw_requires_width_to_be_an_exact_int(width: object) -> None:
    scores = torch.ones(1, 1, 1)
    positions = torch.zeros(1, 1, 1, dtype=torch.int64)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)

    with pytest.raises(TypeError, match="width.*int"):
        deterministic_topw(scores, positions, valid, width=width)  # type: ignore[arg-type]


def test_deterministic_topw_rejects_negative_width() -> None:
    scores = torch.ones(1, 1, 1)
    positions = torch.zeros(1, 1, 1, dtype=torch.int64)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="width.*nonnegative"):
        deterministic_topw(scores, positions, valid, width=-1)


def test_deterministic_topw_repeats_exact_tie_order_deterministically() -> None:
    scores = torch.ones(1, 1, 6)
    positions = torch.tensor([[[2, 9, 4, 7, 1, 5]]])
    valid = torch.ones(1, 1, 6, dtype=torch.bool)
    expected = torch.tensor([[[1, 3, 5, 2]]], dtype=torch.int64)

    for _ in range(20):
        torch.testing.assert_close(
            deterministic_topw(scores, positions, valid, width=4), expected
        )


def test_deterministic_topw_detaches_selection_from_differentiable_scores() -> None:
    scores = torch.tensor([[[1.0, 3.0, 2.0]]], requires_grad=True)
    positions = torch.tensor([[[10, 20, 30]]])
    valid = torch.ones(1, 1, 3, dtype=torch.bool)

    actual = deterministic_topw(scores, positions, valid, width=2)

    torch.testing.assert_close(actual, torch.tensor([[[1, 2]]], dtype=torch.int64))
    assert not actual.requires_grad
    assert actual.grad_fn is None
    assert scores.grad is None
