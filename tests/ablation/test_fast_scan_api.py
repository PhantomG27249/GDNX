from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
FAST_SCAN_PATH = REPO_ROOT / "gdn3" / "kmd2_fast_scan.py"
WRAPPER_ARGUMENTS = ("q", "k", "v", "g", "beta_e", "beta_w", "out_mix")


def _source_tree() -> ast.Module:
    return ast.parse(FAST_SCAN_PATH.read_text(encoding="utf-8"))


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one top-level {name} function"
    return matches[0]


def _assert_exact_wrapper_signature(function: ast.FunctionDef) -> None:
    arguments = function.args
    assert tuple(argument.arg for argument in arguments.args) == WRAPPER_ARGUMENTS
    assert arguments.posonlyargs == []
    assert arguments.vararg is None
    assert arguments.kwonlyargs == []
    assert arguments.kwarg is None
    assert len(arguments.defaults) == 1
    assert isinstance(arguments.defaults[0], ast.Constant)
    assert arguments.defaults[0].value is None


def _assert_core_call(call: ast.Call, return_scores: bool) -> None:
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "_scan_core"
    assert [
        argument.id if isinstance(argument, ast.Name) else None
        for argument in call.args
    ] == list(WRAPPER_ARGUMENTS)
    assert len(call.keywords) == 1
    keyword = call.keywords[0]
    assert keyword.arg == "return_scores"
    assert isinstance(keyword.value, ast.Constant)
    assert keyword.value.value is return_scores


def _top_level_assignment(tree: ast.Module, name: str) -> ast.AST:
    matches = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(node.value)
    assert len(matches) == 1, f"expected one top-level assignment to {name}"
    return matches[0]


def _assert_compiled_assignment(
    tree: ast.Module, export_name: str, implementation_name: str
) -> None:
    value = _top_level_assignment(tree, export_name)
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Attribute)
    assert isinstance(value.func.value, ast.Name)
    assert (value.func.value.id, value.func.attr) == ("torch", "compile")
    assert len(value.args) == 1
    assert isinstance(value.args[0], ast.Name)
    assert value.args[0].id == implementation_name
    assert value.keywords == []


def test_fast_scan_has_exact_legacy_and_score_wrapper_contracts_without_import() -> None:
    tree = _source_tree()
    _top_level_function(tree, "_scan_core")

    legacy = _top_level_function(tree, "_scan_impl")
    _assert_exact_wrapper_signature(legacy)
    assert len(legacy.body) == 2
    assignment, return_statement = legacy.body
    assert isinstance(assignment, ast.Assign)
    assert len(assignment.targets) == 1
    target = assignment.targets[0]
    assert isinstance(target, ast.Tuple)
    assert [element.id for element in target.elts if isinstance(element, ast.Name)] == [
        "y",
        "_",
    ]
    _assert_core_call(assignment.value, return_scores=False)
    assert isinstance(return_statement, ast.Return)
    assert isinstance(return_statement.value, ast.Name)
    assert return_statement.value.id == "y"

    scored = _top_level_function(tree, "_scan_with_update_norm_impl")
    _assert_exact_wrapper_signature(scored)
    assert len(scored.body) == 1
    assert isinstance(scored.body[0], ast.Return)
    _assert_core_call(scored.body[0].value, return_scores=True)


def test_fast_scan_exports_exact_compiled_public_assignments_without_import() -> None:
    tree = _source_tree()
    _assert_compiled_assignment(tree, "scan", "_scan_impl")
    _assert_compiled_assignment(
        tree, "scan_with_update_norm", "_scan_with_update_norm_impl"
    )


def test_trsm_compiler_fallback_is_limited_to_exact_incompatible_stack() -> None:
    tree = _source_tree()
    fallback = _top_level_assignment(tree, "_NEEDS_TRSM_COMPILER_FALLBACK")
    assert ast.unparse(fallback) == (
        "os.name == 'nt' and sys.version_info[:2] == (3, 13) and "
        "(torch.__version__ == '2.10.0+cu128') and "
        "(triton.__version__ == '3.7.0')"
    )

    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_NEEDS_TRSM_COMPILER_FALLBACK"
    ]
    assert len(guards) == 1
    assert guards[0].orelse == []
    assert [ast.unparse(statement) for statement in guards[0].body] == [
        "_trsm_triton = torch.compiler.disable(_trsm_triton)",
        "_trsm_upper_triton = torch.compiler.disable(_trsm_upper_triton)",
    ]


def _cuda_fast_scan_module():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton fast-scan executable tests")
    return importlib.import_module("gdn3.kmd2_fast_scan")


def _leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().requires_grad_(True)


def _make_inputs(
    *, steps: int, r_out: int, seed: int
) -> tuple[torch.Tensor | None, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch, heads, key_dim, value_dim = 1, 1, 8, 64
    q = _leaf(
        torch.randn(
            batch,
            steps,
            heads,
            r_out,
            key_dim,
            generator=generator,
            device="cuda",
        )
        * 0.20
    )
    raw_k = torch.randn(
        batch,
        steps,
        heads,
        key_dim,
        generator=generator,
        device="cuda",
    )
    k = _leaf(torch.nn.functional.normalize(raw_k, dim=-1) * 0.45)
    v = _leaf(
        torch.randn(
            batch,
            steps,
            heads,
            value_dim,
            generator=generator,
            device="cuda",
        )
        * 0.20
    )
    g = _leaf(
        0.97
        + 0.02
        * torch.rand(
            batch,
            steps,
            heads,
            key_dim,
            generator=generator,
            device="cuda",
        )
    )
    beta_e = _leaf(
        0.10
        + 0.30
        * torch.rand(
            batch,
            steps,
            heads,
            generator=generator,
            device="cuda",
        )
    )
    beta_w = _leaf(
        0.20
        + 0.30
        * torch.rand(
            batch,
            steps,
            heads,
            generator=generator,
            device="cuda",
        )
    )
    if r_out == 1:
        return q, k, v, g, beta_e, beta_w, None

    out_mix = _leaf(
        torch.tensor([[0.10, 0.20, 0.30, 0.40]], device="cuda")
    )
    return q, k, v, g, beta_e, beta_w, out_mix


def _clone_inputs(
    inputs: tuple[torch.Tensor | None, ...],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if tensor is None else _leaf(tensor) for tensor in inputs)


def _relative_mse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.mean((actual.float() - expected.float()) ** 2)
    denominator = torch.mean(expected.float() ** 2).clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())


def _differentiable(
    inputs: tuple[torch.Tensor | None, ...],
) -> list[torch.Tensor]:
    return [tensor for tensor in inputs if tensor is not None]


@pytest.mark.cuda
@pytest.mark.parametrize("r_out", [1, 4])
@pytest.mark.parametrize("extra_step", [0, 1])
def test_fast_scan_y_scores_and_gradients_match_independent_reference(
    r_out: int, extra_step: int
) -> None:
    module = _cuda_fast_scan_module()
    from research.kmd2_ablation.exact_cache import reference_scan_with_scores

    steps = module.C + extra_step
    initial = _make_inputs(
        steps=steps,
        r_out=r_out,
        seed=7100 + 10 * r_out + extra_step,
    )

    old_inputs = _clone_inputs(initial)
    score_inputs = _clone_inputs(initial)
    old_y = module._scan_impl(*old_inputs)
    fast_y, fast_scores = module._scan_with_update_norm_impl(*score_inputs)

    torch.testing.assert_close(fast_y, old_y, rtol=0.0, atol=0.0)
    assert fast_scores.shape == (1, steps, 1)
    assert fast_scores.dtype == torch.float32
    assert fast_scores.requires_grad is False
    assert fast_scores.grad_fn is None

    reference_inputs = _clone_inputs(initial)
    reference_y, reference_scores = reference_scan_with_scores(
        *reference_inputs[:6], out_mix=reference_inputs[6]
    )
    assert _relative_mse(fast_y, reference_y) < 2e-3
    assert _relative_mse(fast_scores, reference_scores) < 2e-3

    probe_generator = torch.Generator(device="cuda").manual_seed(
        7300 + 10 * r_out + extra_step
    )
    probe = torch.randn(
        fast_y.shape,
        dtype=fast_y.dtype,
        device=fast_y.device,
        generator=probe_generator,
    )
    fast_gradients = torch.autograd.grad(
        fast_y,
        _differentiable(score_inputs),
        grad_outputs=probe,
    )
    old_gradients = torch.autograd.grad(
        old_y,
        _differentiable(old_inputs),
        grad_outputs=probe,
    )
    reference_gradients = torch.autograd.grad(
        reference_y,
        _differentiable(reference_inputs),
        grad_outputs=probe,
    )
    gradient_names = ["q", "k", "v", "g", "beta_e", "beta_w"]
    if r_out > 1:
        gradient_names.append("out_mix")
    assert len(fast_gradients) == len(reference_gradients) == len(gradient_names)
    assert len(old_gradients) == len(gradient_names)
    for name, old_gradient, fast_gradient, reference_gradient in zip(
        gradient_names, old_gradients, fast_gradients, reference_gradients
    ):
        torch.testing.assert_close(fast_gradient, old_gradient, rtol=0.0, atol=0.0)
        assert _relative_mse(fast_gradient, reference_gradient) < 1e-2, name


@pytest.mark.cuda
def test_compiled_public_scan_and_scored_scan_have_identical_y() -> None:
    module = _cuda_fast_scan_module()
    initial = _make_inputs(steps=module.C, r_out=1, seed=7200)
    legacy_inputs = _clone_inputs(initial)
    scored_inputs = _clone_inputs(initial)
    eager_inputs = _clone_inputs(initial)

    legacy_y = module.scan(*legacy_inputs)
    scored_y, scores = module.scan_with_update_norm(*scored_inputs)
    eager_y, eager_scores = module._scan_with_update_norm_impl(*eager_inputs)

    for tensor in (legacy_y, scored_y, scores):
        assert bool(torch.isfinite(tensor).all())
    assert _relative_mse(scored_y, legacy_y) < 2e-3
    assert _relative_mse(legacy_y, eager_y) < 2e-3
    assert _relative_mse(scored_y, eager_y) < 2e-3
    assert _relative_mse(scores, eager_scores) < 2e-3
    assert scores.shape == (1, module.C, 1)
    assert scores.dtype == torch.float32
    assert scores.requires_grad is False
    assert scores.grad_fn is None

    probe_generator = torch.Generator(device="cuda").manual_seed(7400)
    probe = torch.randn(
        legacy_y.shape,
        dtype=legacy_y.dtype,
        device=legacy_y.device,
        generator=probe_generator,
    )
    legacy_gradients = torch.autograd.grad(
        legacy_y,
        _differentiable(legacy_inputs),
        grad_outputs=probe,
    )
    scored_gradients = torch.autograd.grad(
        scored_y,
        _differentiable(scored_inputs),
        grad_outputs=probe,
    )
    eager_gradients = torch.autograd.grad(
        eager_y,
        _differentiable(eager_inputs),
        grad_outputs=probe,
    )
    assert len(legacy_gradients) == len(scored_gradients) == len(eager_gradients)
    for legacy_gradient, scored_gradient, eager_gradient in zip(
        legacy_gradients, scored_gradients, eager_gradients
    ):
        assert bool(torch.isfinite(legacy_gradient).all())
        assert bool(torch.isfinite(scored_gradient).all())
        assert _relative_mse(scored_gradient, legacy_gradient) < 1e-2
        assert _relative_mse(legacy_gradient, eager_gradient) < 1e-2
        assert _relative_mse(scored_gradient, eager_gradient) < 1e-2


def _selection_fixture(
    module, magnitudes: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    from research.kmd2_ablation.exact_cache import (
        deterministic_topw,
        reference_scan_with_scores,
    )

    steps = module.C
    q = torch.zeros(1, steps, 1, 1, 4, device="cuda")
    k = torch.zeros(1, steps, 1, 4, device="cuda")
    k[..., 0] = 1.0
    v = torch.zeros(1, steps, 1, 64, device="cuda")
    v[..., 0] = torch.tensor(magnitudes, device="cuda").view(1, steps, 1)
    g = torch.ones_like(k)
    beta_e = torch.zeros(1, steps, 1, device="cuda")
    beta_w = torch.ones_like(beta_e)

    _, fast_scores = module._scan_with_update_norm_impl(
        q, k, v, g, beta_e, beta_w, None
    )
    _, reference_scores = reference_scan_with_scores(
        q, k, v, g, beta_e, beta_w
    )
    positions = torch.arange(steps, device="cuda", dtype=torch.int64).view(1, 1, -1)
    valid = torch.ones_like(positions, dtype=torch.bool)
    fast_selected = deterministic_topw(
        fast_scores.permute(0, 2, 1), positions, valid, width=2
    )
    reference_selected = deterministic_topw(
        reference_scores.permute(0, 2, 1), positions, valid, width=2
    )
    return fast_selected, reference_selected


@pytest.mark.cuda
def test_fast_scores_select_separated_updates_deterministically() -> None:
    module = _cuda_fast_scan_module()
    magnitudes = [0.10] * module.C
    magnitudes[1] = 4.0
    magnitudes[3] = 3.0

    fast_selected, reference_selected = _selection_fixture(module, magnitudes)

    expected = [[[1, 3]]]
    assert fast_selected.cpu().tolist() == expected
    assert reference_selected.cpu().tolist() == expected
    torch.testing.assert_close(fast_selected, reference_selected)


@pytest.mark.cuda
def test_fast_scores_break_ties_by_newer_absolute_position() -> None:
    module = _cuda_fast_scan_module()
    magnitudes = [0.10] * module.C
    magnitudes[1] = 4.0
    magnitudes[3] = 4.0

    fast_selected, reference_selected = _selection_fixture(module, magnitudes)

    expected = [[[3, 1]]]
    assert fast_selected.cpu().tolist() == expected
    assert reference_selected.cpu().tolist() == expected
    torch.testing.assert_close(fast_selected, reference_selected)
