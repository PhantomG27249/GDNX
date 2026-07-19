from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import pytest
import torch


def _oracle(q, k, v, erase, write, gamma, lam, state, previous_key,
            previous_value, history, update_count):
    periods = update_count.new_tensor((1, 16, 64, 256))
    reads = []
    innovation_sq = []
    count = update_count
    history = history.clone()
    for token in range(q.shape[1]):
        tick_lanes = count[:, None].remainder(periods[None]).eq(0)
        tick = tick_lanes[:, None, :, None, None]
        decayed = gamma[:, token, ..., None] * state
        erased_key = erase[:, token] * k[:, token]
        memory = torch.einsum("bhrk,bhrkv->bhrv", erased_key, decayed)
        homogeneous = torch.where(
            tick,
            decayed - k[:, token, ..., None] * memory[..., None, :],
            decayed,
        )
        current_write = (
            k[:, token, ..., None]
            * (write[:, token] * v[:, token])[..., None, :]
        )
        previous_key_decayed = gamma[:, token] * previous_key
        previous_memory = torch.einsum(
            "bhrk,bhrk->bhr", erased_key, previous_key_decayed
        )
        previous_transported_key = (
            previous_key_decayed - k[:, token] * previous_memory[..., None]
        )
        previous_transported = (
            previous_transported_key[..., None]
            * previous_value[..., None, :]
        )
        effective_lam = torch.where(
            history[:, None, :, None, None],
            lam[:, token, ..., None, None],
            torch.ones_like(lam[:, token, ..., None, None]),
        )
        tick_update = (
            (1.0 - effective_lam) * previous_transported
            + effective_lam * current_write
        )
        state = homogeneous + torch.where(tick, tick_update, 0.0)
        tick_vector = tick_lanes[:, None, :, None]
        previous_key = torch.where(tick_vector, k[:, token], previous_key_decayed)
        previous_value = torch.where(
            tick_vector, write[:, token] * v[:, token], previous_value
        )
        history = history | tick_lanes
        count = count + 1
        reads.append(torch.einsum("bhik,bhjkv->bhijv", q[:, token], state))
        innovation_sq.append((state - decayed).square().sum((-2, -1)))
    return (
        torch.stack(reads, 1), state, previous_key, previous_value,
        torch.stack(innovation_sq, 1),
    )


def _inputs(device, *, tokens, update_count, history, heads=16):
    torch.manual_seed(20260715 + tokens + update_count)
    shape_k = (1, tokens, heads, 4, 32)
    shape_v = (1, tokens, heads, 4, 128)
    q = torch.nn.functional.normalize(torch.randn(shape_k, device=device), dim=-1)
    q = (q * 32 ** -0.5).contiguous()
    k = torch.nn.functional.normalize(torch.randn(shape_k, device=device), dim=-1)
    v = torch.randn(shape_v, device=device) * 0.1
    erase = torch.rand(shape_k, device=device)
    write = torch.rand(shape_v, device=device)
    gamma = 0.99 + 0.009 * torch.rand(shape_k, device=device)
    lam = 0.9 + 0.09 * torch.rand(1, tokens, heads, 4, device=device)
    state = torch.randn(1, heads, 4, 32, 128, device=device) * 0.01
    previous_key = torch.randn(1, heads, 4, 32, device=device) * 0.01
    previous_value = torch.randn(1, heads, 4, 128, device=device) * 0.01
    history_tensor = torch.full(
        (1, 4), history, dtype=torch.bool, device=device
    )
    count_tensor = torch.tensor([update_count], dtype=torch.int64, device=device)
    return (q, k, v, erase, write, gamma, lam, state, previous_key,
            previous_value,
            history_tensor, count_tensor)


def _cuda_module():
    from research.kmd2_ablation import qwen_hybrid_triton as module

    if not module.triton_four_state_segment_available():
        pytest.skip("CUDA/Triton is unavailable")
    return module


@pytest.mark.parametrize(
    ("tokens", "count", "history", "endpoint_loss", "heads"),
    ((64, 0, True, True, 16), (17, 1, False, False, 16),
     (9, 3, False, True, 8)),
)
def test_raw_triton_segment_matches_oracle_forward_and_vjp(
    tokens, count, history, endpoint_loss, heads,
):
    module = _cuda_module()
    device = torch.device("cuda")
    source = _inputs(
        device, tokens=tokens, update_count=count, history=history,
        heads=heads,
    )
    floating_count = 10
    custom_floating = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in source[:floating_count]
    )
    oracle_floating = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in source[:floating_count]
    )
    custom = module.triton_four_state_segment(
        *custom_floating, *source[floating_count:]
    )
    expected = _oracle(
        *oracle_floating, *source[floating_count:]
    )

    torch.testing.assert_close(custom[0], expected[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(custom[1], expected[1], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(custom[2], expected[2], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(custom[3], expected[3], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(custom[4], expected[4], atol=2e-6, rtol=2e-5)
    assert not custom[4].requires_grad

    torch.manual_seed(9182 + tokens)
    read_weight = torch.randn_like(custom[0]) * 0.01
    custom_loss = (custom[0] * read_weight).sum()
    oracle_loss = (expected[0] * read_weight).sum()
    if endpoint_loss:
        state_weight = torch.randn_like(custom[1]) * 0.01
        previous_key_weight = torch.randn_like(custom[2]) * 0.01
        previous_value_weight = torch.randn_like(custom[3]) * 0.01
        custom_loss = (
            custom_loss
            + (custom[1] * state_weight).sum()
            + (custom[2] * previous_key_weight).sum()
            + (custom[3] * previous_value_weight).sum()
        )
        oracle_loss = (
            oracle_loss
            + (expected[1] * state_weight).sum()
            + (expected[2] * previous_key_weight).sum()
            + (expected[3] * previous_value_weight).sum()
        )
    custom_gradients = torch.autograd.grad(custom_loss, custom_floating)
    oracle_gradients = torch.autograd.grad(oracle_loss, oracle_floating)
    for actual, reference in zip(custom_gradients, oracle_gradients, strict=True):
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-4)


def test_raw_triton_dispatch_is_fail_closed():
    module = _cuda_module()
    source = _inputs(torch.device("cuda"), tokens=2, update_count=0, history=True)
    assert module.can_use_triton_four_state_segment(*source)
    assert not module.can_use_triton_four_state_segment(
        source[0].to(torch.bfloat16), *source[1:]
    )
    assert not module.can_use_triton_four_state_segment(
        source[0][:, :, :, :, ::2], *source[1:]
    )
    bad_count = source[-1].to(torch.int32)
    assert not module.can_use_triton_four_state_segment(*source[:-1], bad_count)


def test_raw_triton_public_wrapper_rejects_invalid_shape():
    module = _cuda_module()
    source = _inputs(torch.device("cuda"), tokens=2, update_count=0, history=True)
    with pytest.raises(RuntimeError, match="not dispatchable"):
        module.triton_four_state_segment(
            source[0][:, :, :, :, ::2], *source[1:]
        )


def _assert_canonical_cache_close(actual, expected, path="cache"):
    """Compare every field while respecting the cache's mixed precision."""
    assert type(actual) is type(expected), path
    for field in fields(actual):
        left = getattr(actual, field.name)
        right = getattr(expected, field.name)
        field_path = f"{path}.{field.name}"
        if is_dataclass(left):
            _assert_canonical_cache_close(left, right, field_path)
        elif isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor), field_path
            assert left.shape == right.shape, field_path
            assert left.dtype == right.dtype, field_path
            if not left.is_floating_point():
                assert torch.equal(left, right), field_path
            elif left.dtype == torch.bfloat16:
                torch.testing.assert_close(
                    left, right, atol=2e-3, rtol=8e-3, msg=field_path
                )
            else:
                torch.testing.assert_close(
                    left, right, atol=2e-5, rtol=2e-4, msg=field_path
                )
        else:
            assert left == right, field_path


def test_canonical_module_triton_dispatch_matches_forced_torch_vjp(monkeypatch):
    triton_module = _cuda_module()
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_four_state import (
        QwenFourStateHybrid,
    )

    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    torch.manual_seed(20260715)
    config = SimpleNamespace(
        hidden_size=1024,
        linear_num_value_heads=8,
        linear_num_key_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    native = KMD2NativeAttn(config, layer_idx=0)
    with torch.no_grad():
        for parameter in native.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, std=0.02)
        native.rot_proj.weight.zero_()
        native.rot_proj.bias.fill_(-9.0)
    device = torch.device("cuda")
    native = native.to(device=device, dtype=torch.bfloat16)
    fused = QwenFourStateHybrid.from_native(native)
    # Isolate the older persistent B=1 backend that this test targets.  The
    # batch-generic chunked Liger backend is otherwise the production default
    # and has precedence over this fallback.
    fused.use_liger_chunked_kernel = False
    torch_reference = copy.deepcopy(fused)
    torch_reference.force_torch_recurrence = True

    calls = []
    real_wrapper = triton_module.triton_four_state_segment

    def counted_wrapper(*args, **kwargs):
        calls.append(args[0].shape[1])
        return real_wrapper(*args, **kwargs)

    monkeypatch.setattr(
        triton_module, "triton_four_state_segment", counted_wrapper
    )
    source = (
        torch.randn(1, 17, 1024, device=device, dtype=torch.bfloat16) * 0.1
    )
    fused_input = source.detach().clone().requires_grad_(True)
    reference_input = source.detach().clone().requires_grad_(True)

    fused_output, fused_cache = fused.scan(fused_input)
    assert calls == [17]
    calls_before_reference = len(calls)
    reference_output, reference_cache = torch_reference.scan(reference_input)
    assert len(calls) == calls_before_reference

    torch.testing.assert_close(
        fused_output, reference_output, atol=2e-3, rtol=8e-3
    )
    _assert_canonical_cache_close(fused_cache, reference_cache)

    probe = torch.linspace(
        0.5, 1.5, fused_output.numel(), device=device, dtype=torch.float32
    ).reshape_as(fused_output)
    fused_loss = (
        (fused_output.float() * probe).mean()
        + 0.01 * fused_output.float().square().mean()
    )
    reference_loss = (
        (reference_output.float() * probe).mean()
        + 0.01 * reference_output.float().square().mean()
    )
    parameter_names = (
        "components.trapezoid_proj.bias",
        "components.native_decay_pair",
        "components.output_mixer",
        "hola.gamma_q",
        "hola.gamma_k",
        "hola.sink_logit",
    )
    fused_parameters = dict(fused.named_parameters())
    reference_parameters = dict(torch_reference.named_parameters())
    fused_gradients = torch.autograd.grad(
        fused_loss,
        (fused_input, *(fused_parameters[name] for name in parameter_names)),
    )
    assert len(calls) > calls_before_reference
    assert all(tokens == 17 for tokens in calls)
    calls_before_reference_vjp = len(calls)
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (
            reference_input,
            *(reference_parameters[name] for name in parameter_names),
        ),
    )
    assert len(calls) == calls_before_reference_vjp

    for name, actual, expected in zip(
        ("input", *parameter_names),
        fused_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual, expected, atol=1e-3, rtol=8e-3, msg=name
        )


def test_no_grad_forward_skips_state_traces_and_matches_grad_path():
    """Inference must not allocate compact backward-only FP32 traces."""
    module = _cuda_module()
    device = torch.device("cuda")
    source = _inputs(device, tokens=64, update_count=0, history=True)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        eval_outputs = module.triton_four_state_segment(*source)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    trace_bytes = 64 * 16 * 4 * (32 * 128 + 32 + 128) * 4
    assert peak < trace_bytes // 2, f"no_grad peak {peak} suggests traces were allocated"

    grad_source = tuple(
        tensor.clone().requires_grad_() for tensor in source[:10]
    ) + source[10:]
    grad_outputs = module.triton_four_state_segment(*grad_source)
    for index, (expected, actual) in enumerate(zip(eval_outputs, grad_outputs)):
        assert torch.equal(expected, actual.detach()), f"output {index} differs"
    loss = sum(output.square().sum() for output in grad_outputs[:4])
    loss.backward()
    for index, tensor in enumerate(grad_source[:10]):
        assert tensor.grad is not None, index
        assert bool(torch.isfinite(tensor.grad).all()), index
