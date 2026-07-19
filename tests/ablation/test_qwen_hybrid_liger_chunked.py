from __future__ import annotations

import copy

import pytest
import torch

from tests.ablation.test_qwen_hybrid_triton import _inputs


def _cuda_module():
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as module

    if not module.liger_chunked_four_state_available():
        pytest.skip("CUDA/Triton is unavailable")
    return module


def _batch(source, batch):
    return tuple(
        tensor.repeat((batch,) + (1,) * (tensor.ndim - 1))
        for tensor in source
    )


def _batch_two(source):
    return _batch(source, 2)


def test_true_chunked_dplr_sequence_matches_token_authority_and_vjp():
    _cuda_module()
    from research.kmd2_ablation.qwen_hybrid_chunkwise import (
        _eager_torch_chunk_four_state_segment,
    )
    from research.kmd2_ablation.qwen_hybrid_liger_dplr import (
        true_chunked_dplr_four_state_sequence,
    )

    source = _inputs(
        torch.device("cuda"), tokens=16, update_count=13,
        history=False, heads=8,
    )
    source = (
        *source[:-2],
        torch.tensor(
            [[False, True, False, True]], device="cuda", dtype=torch.bool
        ),
        source[-1],
    )
    actual_floating = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    expected_floating = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    actual = true_chunked_dplr_four_state_sequence(
        *actual_floating, *source[10:]
    )
    expected = _eager_torch_chunk_four_state_segment(
        *expected_floating, *source[10:]
    )
    for actual_index, expected_index in ((0, 0), (1, 2), (2, 3), (3, 4), (4, 5)):
        torch.testing.assert_close(
            actual[actual_index], expected[expected_index],
            atol=3e-5, rtol=3e-4,
        )
    assert torch.equal(actual[5], expected[6])
    assert torch.equal(actual[6], expected[7])

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261031)
    cotangents = tuple(
        torch.randn(
            actual[index].shape,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ) * 0.01
        for index in range(4)
    )
    actual_gradients = torch.autograd.grad(
        actual[:4], actual_floating, grad_outputs=cotangents
    )
    expected_gradients = torch.autograd.grad(
        (expected[0], expected[2], expected[3], expected[4]),
        expected_floating,
        grad_outputs=cotangents,
    )
    for result, reference in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result, reference, atol=2e-4, rtol=2e-3
        )


@pytest.mark.parametrize(
    ("batch", "tokens", "heads", "count", "history"),
    ((1, 17, 8, 3, False), (2, 64, 16, 255, True)),
)
def test_all_triton_wy_factors_match_pytorch_authority(
    batch, tokens, heads, count, history,
):
    _cuda_module()
    from research.kmd2_ablation.qwen_hybrid_chunkwise import (
        _eager_torch_chunk_four_state_segment,
    )
    from research.kmd2_ablation.qwen_hybrid_liger_wy import build_wy_factors

    source = _inputs(
        torch.device("cuda"),
        tokens=tokens,
        update_count=count,
        history=history,
        heads=heads,
    )
    if batch > 1:
        source = _batch(source, batch)
        source = (
            *source[:-2],
            torch.tensor(
                [[True, False, True, False], [False, True, False, True]],
                device="cuda",
                dtype=torch.bool,
            ),
            torch.tensor([count, count + 7], device="cuda", dtype=torch.int64),
        )
    expected = _eager_torch_chunk_four_state_segment(
        *source, _return_factors=True
    )
    actual = build_wy_factors(
        source[1],
        source[2],
        source[3],
        source[4],
        source[5],
        source[6],
        source[7],
        source[8],
        source[9],
        source[10],
        source[11],
    )
    for index in range(6):
        torch.testing.assert_close(
            actual[index], expected[index], atol=3e-5, rtol=3e-4
        )
    assert torch.equal(actual[6], expected[6])
    assert torch.equal(actual[7], expected[7])


@pytest.mark.parametrize(
    ("batch", "tokens", "heads", "count", "history"),
    (
        (1, 17, 8, 3, False),
        (2, 64, 16, 1, True),
        (3, 17, 8, 255, True),
    ),
)
def test_raw_chunked_liger_forward_and_vjp_match_pytorch_wy(
    batch, tokens, heads, count, history,
):
    module = _cuda_module()
    from research.kmd2_ablation.qwen_hybrid_chunkwise import (
        _eager_torch_chunk_four_state_segment,
    )

    source = _inputs(
        torch.device("cuda"),
        tokens=tokens,
        update_count=count,
        history=history,
        heads=heads,
    )
    if batch > 1:
        source = _batch(source, batch)
        # Exercise independent CMS phases and history across batch rows.
        history_rows = [
            [bool((row + lane) % 2) for lane in range(4)]
            for row in range(batch)
        ]
        history_rows[0][0] = history
        source = (
            *source[:-2],
            torch.tensor(
                history_rows,
                dtype=torch.bool,
                device="cuda",
            ),
            count + 7 * torch.arange(batch, dtype=torch.int64, device="cuda"),
        )

    actual_floating = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    expected_floating = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    actual = module.liger_chunked_four_state_segment(
        *actual_floating, *source[10:]
    )
    expected = _eager_torch_chunk_four_state_segment(
        *expected_floating, *source[10:]
    )

    assert actual[1].numel() == 0
    for index in (0, 2, 3, 4, 5):
        torch.testing.assert_close(
            actual[index], expected[index], atol=3e-5, rtol=3e-4
        )
    assert torch.equal(actual[6], expected[6])
    assert torch.equal(actual[7], expected[7])
    assert not actual[5].requires_grad

    torch.manual_seed(20260718 + tokens + batch)
    differentiable_indices = (0, 2, 3, 4)
    cotangents = tuple(
        torch.randn_like(actual[index]) * 0.01
        for index in differentiable_indices
    )
    actual_gradients = torch.autograd.grad(
        tuple(actual[index] for index in differentiable_indices),
        actual_floating,
        grad_outputs=cotangents,
    )
    expected_gradients = torch.autograd.grad(
        tuple(expected[index] for index in differentiable_indices),
        expected_floating,
        grad_outputs=cotangents,
    )
    for result, reference in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("autocast_dtype", (torch.float16, torch.bfloat16))
def test_raw_chunked_liger_autocast_keeps_fp32_recurrence_and_vjp(
    autocast_dtype,
):
    module = _cuda_module()
    from research.kmd2_ablation.qwen_hybrid_chunkwise import (
        _eager_torch_chunk_four_state_segment,
    )

    source = _batch_two(
        _inputs(
            torch.device("cuda"), tokens=17, update_count=255,
            history=True, heads=8,
        )
    )
    source = (
        *source[:-2],
        torch.tensor(
            [[True, True, False, True], [True, False, True, False]],
            dtype=torch.bool,
            device="cuda",
        ),
        torch.tensor([255, 262], dtype=torch.int64, device="cuda"),
    )
    actual_floating = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    expected_floating = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )

    with torch.autocast("cuda", dtype=autocast_dtype):
        actual = module.liger_chunked_four_state_segment(
            *actual_floating, *source[10:]
        )
    assert not torch.is_autocast_enabled("cuda")
    expected = _eager_torch_chunk_four_state_segment(
        *expected_floating, *source[10:]
    )

    for index in (0, 2, 3, 4, 5):
        assert actual[index].dtype == torch.float32
        assert expected[index].dtype == torch.float32
        torch.testing.assert_close(
            actual[index], expected[index], atol=3e-5, rtol=3e-4
        )
    assert torch.equal(actual[6], expected[6])
    assert torch.equal(actual[7], expected[7])

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261001 + int(autocast_dtype == torch.bfloat16))
    indices = (0, 2, 3, 4)
    cotangents = tuple(
        torch.randn(
            actual[index].shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.01
        for index in indices
    )
    actual_gradients = torch.autograd.grad(
        tuple(actual[index] for index in indices),
        actual_floating,
        grad_outputs=cotangents,
    )
    expected_gradients = torch.autograd.grad(
        tuple(expected[index] for index in indices),
        expected_floating,
        grad_outputs=cotangents,
    )
    for result, reference in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result, reference, atol=2e-4, rtol=2e-3
        )


@pytest.mark.parametrize(
    ("input_index", "output_index"),
    (
        (0, 0),  # q receives only the fused-read adjoint.
        (1, 0),  # k receives the compact innovation/factor adjoint.
        (2, 4),  # v receives only the carried previous-value endpoint VJP.
        (5, 0),  # gamma receives prefix, transport, and innovation adjoints.
        (7, 2),  # initial state receives direct and factor-mediated endpoint VJP.
        (8, 3),  # previous key receives only its endpoint VJP.
    ),
)
def test_chunked_liger_partial_input_and_output_vjps_are_exact(
    input_index, output_index,
):
    module = _cuda_module()
    from research.kmd2_ablation.qwen_hybrid_chunkwise import (
        _eager_torch_chunk_four_state_segment,
    )

    source = _inputs(
        torch.device("cuda"),
        tokens=17,
        update_count=13,
        history=False,
        heads=8,
    )
    actual_inputs = list(source)
    expected_inputs = list(source)
    actual_inputs[input_index] = (
        source[input_index].detach().clone().requires_grad_(True)
    )
    expected_inputs[input_index] = (
        source[input_index].detach().clone().requires_grad_(True)
    )
    actual = module.liger_chunked_four_state_segment(*actual_inputs)
    expected = _eager_torch_chunk_four_state_segment(*expected_inputs)
    torch.manual_seed(20260901 + input_index * 10 + output_index)
    cotangent = torch.randn_like(actual[output_index]) * 0.01
    actual_gradient, = torch.autograd.grad(
        actual[output_index],
        (actual_inputs[input_index],),
        grad_outputs=(cotangent,),
    )
    expected_gradient, = torch.autograd.grad(
        expected[output_index],
        (expected_inputs[input_index],),
        grad_outputs=(cotangent,),
    )
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=2e-4, rtol=2e-3
    )


def test_chunked_liger_backward_is_all_triton_and_rematerializes_factors(
    monkeypatch,
):
    module = _cuda_module()
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from research.kmd2_ablation import qwen_hybrid_liger_wy as wy

    source = _inputs(
        torch.device("cuda"),
        tokens=17,
        update_count=7,
        history=True,
        heads=8,
    )
    floating = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )

    build = wy.build_wy_factors
    build_calls = 0

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return build(*args, **kwargs)

    def forbidden_eager(*_args, **_kwargs):
        raise AssertionError("the all-Triton Liger path called PyTorch WY")

    launcher = module._launch_reconstruct_reads_backward
    backward_calls = 0

    def counted_launcher(*args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        return launcher(*args, **kwargs)

    monkeypatch.setattr(wy, "build_wy_factors", counted_build)
    monkeypatch.setattr(
        chunkwise, "_eager_torch_chunk_four_state_segment", forbidden_eager
    )
    monkeypatch.setattr(
        module, "_launch_reconstruct_reads_backward", counted_launcher
    )
    result = module.liger_chunked_four_state_segment(
        *floating, *source[10:]
    )
    loss = result[0].square().mean() + result[2].square().mean()
    loss.backward()
    assert backward_calls == 1
    assert build_calls == 2
    for tensor in floating:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_chunked_liger_dispatch_is_training_only_and_fail_closed():
    module = _cuda_module()
    source = _inputs(
        torch.device("cuda"), tokens=17, update_count=0, history=True
    )
    batch_one_training = (
        source[0].detach().clone().requires_grad_(True), *source[1:]
    )
    assert module.can_use_liger_chunked_four_state(*batch_one_training)
    batch_two = _batch_two(source)
    training = (
        batch_two[0].detach().clone().requires_grad_(True), *batch_two[1:]
    )
    assert module.can_use_liger_chunked_four_state(*training)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        assert not module.can_use_liger_chunked_four_state(*training)
        with pytest.raises(ValueError, match="deterministic algorithms"):
            module.liger_chunked_four_state_segment(*training)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    with torch.no_grad():
        assert not module.can_use_liger_chunked_four_state(*training)
    assert not module.can_use_liger_chunked_four_state(*source)
    assert not module.can_use_liger_chunked_four_state(
        training[0][:, :15],
        *(tensor[:, :15] for tensor in training[1:7]),
        *training[7:],
    )
    assert not module.can_use_liger_chunked_four_state(
        training[0].to(torch.bfloat16), *training[1:]
    )
    assert module.can_use_liger_chunked_four_state(*(_batch_two(training)))
    empty = tuple(tensor[:0] for tensor in training)
    assert not module.can_use_liger_chunked_four_state(*empty)


def test_runtime_toggle_is_recursive_reversible_and_not_checkpointed():
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(2, 2)
            self.first.use_liger_chunked_kernel = False
            self.nested = torch.nn.Sequential(torch.nn.Linear(2, 2))
            self.nested[0].use_liger_chunked_kernel = False

    holder = Holder()
    keys = tuple(holder.state_dict())
    assert liger.set_liger_chunked_training(holder) == 2
    assert holder.first.use_liger_chunked_kernel is True
    assert holder.nested[0].use_liger_chunked_kernel is True
    assert tuple(holder.state_dict()) == keys
    assert liger.set_liger_chunked_training(holder, False) == 2
    assert holder.first.use_liger_chunked_kernel is False
    assert holder.nested[0].use_liger_chunked_kernel is False
    with pytest.raises(TypeError, match="bool"):
        liger.set_liger_chunked_training(holder, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_package_b_default_matches_authority_and_preserves_fallback(
    monkeypatch,
):
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger
    from research.kmd2_ablation import qwen_hybrid_liger_dplr as dplr
    from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise as benchmark
    from tests.ablation.test_qwen_hybrid_chunkwise import (
        _assert_named_parameter_vjps_close,
        _assert_training_tree_close,
        _training_vjp,
    )

    if not liger.liger_chunked_four_state_available():
        pytest.skip("CUDA/Triton is unavailable")
    torch.manual_seed(20260818)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    authority = benchmark._build_package_b(device, dtype)
    default = copy.deepcopy(authority)
    disabled = copy.deepcopy(authority)
    forced = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    disabled.use_liger_chunked_kernel = False
    forced.force_torch_recurrence = True
    for module in (authority, default, disabled, forced):
        module.train()

    generator = benchmark._generator(device, 20260819)
    source = 0.1 * torch.randn(
        2,
        17,
        benchmark.CAMPAIGN_HIDDEN,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    cotangent = torch.randn(
        source.shape, device=device, dtype=torch.float32, generator=generator
    )
    authority_hidden = source.detach().clone().requires_grad_(True)
    default_hidden = source.detach().clone().requires_grad_(True)
    disabled_hidden = source.detach().clone().requires_grad_(True)
    forced_hidden = source.detach().clone().requires_grad_(True)

    authority_result = _training_vjp(authority, authority_hidden, cotangent)
    real_kernel = dplr.true_chunked_dplr_four_state_sequence
    calls = 0

    def counted_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(
        dplr, "true_chunked_dplr_four_state_sequence", counted_kernel
    )
    default_result = _training_vjp(default, default_hidden, cotangent)
    # The recurrence owns the whole sequence once.  Checkpoint recomputation
    # covers only the post-recurrence HOLA/norm tail.
    assert calls == 1
    disabled_result = _training_vjp(disabled, disabled_hidden, cotangent)
    assert calls == 1
    # Forced authority must still override the default backend.
    forced_result = _training_vjp(forced, forced_hidden, cotangent)
    assert calls == 1

    for label, result in (
        ("default", default_result),
        ("disabled", disabled_result),
        ("forced", forced_result),
    ):
        _assert_training_tree_close(
            result,
            authority_result,
            atol=2e-3,
            rtol=8e-3,
            path=label,
        )
    for label, hidden in (
        ("default", default_hidden),
        ("disabled", disabled_hidden),
        ("forced", forced_hidden),
    ):
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert bool(hidden.grad.abs().gt(0).any())
        torch.testing.assert_close(
            hidden.grad,
            authority_hidden.grad,
            atol=4e-3,
            rtol=2e-2,
            msg=lambda message, label=label: f"{label}.input.grad: {message}",
        )
    _assert_named_parameter_vjps_close(
        default, authority, atol=4e-3, rtol=2e-2
    )

    assert authority.state_dict().keys() == default.state_dict().keys()
    assert default.use_liger_chunked_kernel is True
    assert disabled.use_liger_chunked_kernel is False


@pytest.mark.parametrize("batch", (1, 3))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_package_b_default_liger_is_batch_generic(batch, monkeypatch):
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger
    from research.kmd2_ablation import qwen_hybrid_liger_dplr as dplr
    from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise as benchmark
    from tests.ablation.test_qwen_hybrid_chunkwise import (
        _assert_named_parameter_vjps_close,
        _assert_training_tree_close,
        _training_vjp,
    )

    if not liger.liger_chunked_four_state_available():
        pytest.skip("CUDA/Triton is unavailable")
    torch.manual_seed(20261100 + batch)
    device = torch.device("cuda")
    authority = benchmark._build_package_b(device, torch.bfloat16)
    candidate = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    authority.train()
    candidate.train()

    generator = benchmark._generator(device, 20261110 + batch)
    source = 0.1 * torch.randn(
        batch,
        17,
        benchmark.CAMPAIGN_HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cotangent = torch.randn(
        source.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    authority_hidden = source.detach().clone().requires_grad_(True)
    candidate_hidden = source.detach().clone().requires_grad_(True)
    authority_result = _training_vjp(authority, authority_hidden, cotangent)

    real_kernel = dplr.true_chunked_dplr_four_state_sequence
    calls = 0

    def counted_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(
        dplr, "true_chunked_dplr_four_state_sequence", counted_kernel
    )
    candidate_result = _training_vjp(candidate, candidate_hidden, cotangent)
    assert calls == 1
    _assert_training_tree_close(
        candidate_result,
        authority_result,
        atol=2e-3,
        rtol=8e-3,
        path=f"batch_{batch}",
    )
    assert candidate_hidden.grad is not None
    assert authority_hidden.grad is not None
    assert torch.isfinite(candidate_hidden.grad).all()
    torch.testing.assert_close(
        candidate_hidden.grad,
        authority_hidden.grad,
        atol=4e-3,
        rtol=2e-2,
    )
    _assert_named_parameter_vjps_close(
        candidate, authority, atol=4e-3, rtol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_package_b_autocast_keeps_recurrence_fp32_and_matches_authority():
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger
    from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise as benchmark
    from tests.ablation.test_qwen_hybrid_chunkwise import (
        _assert_named_parameter_vjps_close,
        _assert_training_tree_close,
    )

    if not liger.liger_chunked_four_state_available():
        pytest.skip("CUDA/Triton is unavailable")
    torch.manual_seed(20261010)
    device = torch.device("cuda")
    authority = benchmark._build_package_b(device, torch.bfloat16)
    candidate = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    assert liger.set_liger_chunked_training(candidate) == 1
    authority.train()
    candidate.train()

    generator = benchmark._generator(device, 20261011)
    source = 0.1 * torch.randn(
        2,
        17,
        benchmark.CAMPAIGN_HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cotangent = torch.randn(
        source.shape, device=device, dtype=torch.float32, generator=generator
    )
    authority_hidden = source.detach().clone().requires_grad_(True)
    candidate_hidden = source.detach().clone().requires_grad_(True)

    def autocast_vjp(module, hidden):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, cache = module.scan(hidden)
        assert not torch.is_autocast_enabled("cuda")
        (output.float() * cotangent).mean().backward()
        return output, cache

    authority_result = autocast_vjp(authority, authority_hidden)
    candidate_result = autocast_vjp(candidate, candidate_hidden)
    _assert_training_tree_close(
        candidate_result,
        authority_result,
        atol=2e-3,
        rtol=8e-3,
        path="autocast",
    )
    assert candidate_hidden.grad is not None
    assert authority_hidden.grad is not None
    assert torch.isfinite(candidate_hidden.grad).all()
    torch.testing.assert_close(
        candidate_hidden.grad,
        authority_hidden.grad,
        atol=4e-3,
        rtol=2e-2,
    )
    _assert_named_parameter_vjps_close(
        candidate, authority, atol=4e-3, rtol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_package_b_two_chunks_preserve_boundary_and_gradient_carry(
    monkeypatch,
):
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger
    from research.kmd2_ablation import qwen_hybrid_liger_dplr as dplr
    from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise as benchmark
    from tests.ablation.test_qwen_hybrid_chunkwise import (
        _assert_named_parameter_vjps_close,
        _assert_training_tree_close,
        _training_vjp,
    )

    if not liger.liger_chunked_four_state_available():
        pytest.skip("CUDA/Triton is unavailable")
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    authority = benchmark._build_package_b(device, torch.bfloat16)
    candidate = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    assert liger.set_liger_chunked_training(candidate) == 1
    authority.train()
    candidate.train()

    generator = benchmark._generator(device, 20260821)
    source = 0.1 * torch.randn(
        2,
        128,
        benchmark.CAMPAIGN_HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cotangent = torch.randn(
        source.shape, device=device, dtype=torch.float32, generator=generator
    )
    authority_hidden = source.detach().clone().requires_grad_(True)
    candidate_hidden = source.detach().clone().requires_grad_(True)
    authority_result = _training_vjp(authority, authority_hidden, cotangent)

    real_kernel = dplr.true_chunked_dplr_four_state_sequence
    calls = 0

    def counted_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(
        dplr, "true_chunked_dplr_four_state_sequence", counted_kernel
    )
    candidate_result = _training_vjp(candidate, candidate_hidden, cotangent)
    # Both 64-token chunks are owned by one sequence-level recurrence call.
    assert calls == 1
    _assert_training_tree_close(
        candidate_result,
        authority_result,
        atol=2e-3,
        rtol=8e-3,
        path="two_chunk_result",
    )
    assert candidate_hidden.grad is not None
    assert authority_hidden.grad is not None
    assert torch.isfinite(candidate_hidden.grad).all()
    assert bool(candidate_hidden.grad.abs().gt(0).any())
    torch.testing.assert_close(
        candidate_hidden.grad,
        authority_hidden.grad,
        atol=4e-3,
        rtol=2e-2,
    )
    _assert_named_parameter_vjps_close(
        candidate, authority, atol=4e-3, rtol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_package_b_two_optimizer_steps_track_pytorch_authority(
    monkeypatch,
):
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger
    from research.kmd2_ablation import qwen_hybrid_liger_dplr as dplr
    from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise as benchmark
    from tests.ablation.test_qwen_hybrid_chunkwise import (
        _assert_optimizer_state_close,
        _assert_training_tree_close,
        _training_vjp,
    )

    if not liger.liger_chunked_four_state_available():
        pytest.skip("CUDA/Triton is unavailable")
    torch.manual_seed(20260910)
    device = torch.device("cuda")
    authority = benchmark._build_package_b(device, torch.float32)
    candidate = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    assert liger.set_liger_chunked_training(candidate) == 1
    authority.train()
    candidate.train()
    authority_optimizer = torch.optim.AdamW(
        authority.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
    )
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
    )
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in authority.named_parameters()
    }
    real_kernel = dplr.true_chunked_dplr_four_state_sequence
    calls = 0

    def counted_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(
        dplr, "true_chunked_dplr_four_state_sequence", counted_kernel
    )
    for step in range(2):
        generator = benchmark._generator(device, 20260911 + step)
        source = 0.1 * torch.randn(
            2,
            17,
            benchmark.CAMPAIGN_HIDDEN,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        target = torch.randn(
            source.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        authority_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        authority_result = _training_vjp(
            authority, source.detach().clone().requires_grad_(True), target
        )
        before_calls = calls
        candidate_result = _training_vjp(
            candidate, source.detach().clone().requires_grad_(True), target
        )
        assert calls - before_calls == 1
        _assert_training_tree_close(
            candidate_result,
            authority_result,
            atol=3e-5,
            rtol=3e-4,
            path=f"step[{step}]",
        )
        authority_optimizer.step()
        candidate_optimizer.step()
        _assert_optimizer_state_close(
            candidate_optimizer,
            candidate,
            authority_optimizer,
            authority,
            initial_parameters=initial_parameters,
            gradient_atol=2e-5,
            gradient_rtol=2e-3,
            delta_atol=2e-7,
            delta_rtol=2e-3,
            state_atol=2e-8,
            state_rtol=3e-3,
        )
