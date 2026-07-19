from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch


def _oracle(q, k, v, erase, write, gamma, state):
    outputs = []
    for token in range(q.shape[1]):
        decayed = gamma[:, token, ..., None] * state
        memory = torch.einsum(
            "bhk,bhkv->bhv", erase[:, token] * k[:, token], decayed
        )
        state = decayed + torch.einsum(
            "bhk,bhv->bhkv",
            k[:, token],
            write[:, token] * v[:, token] - memory,
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, token], state))
    return torch.stack(outputs, 1), state


def _inputs(device, *, tokens, heads):
    torch.manual_seed(20260715 + tokens + heads)
    key_shape = (1, tokens, heads, 128)
    q = (
        torch.nn.functional.normalize(torch.randn(key_shape, device=device), dim=-1)
        * 128 ** -0.5
    ).contiguous()
    k = torch.nn.functional.normalize(
        torch.randn(key_shape, device=device), dim=-1
    ).contiguous()
    v = (torch.randn(key_shape, device=device) * 0.1).contiguous()
    erase = torch.rand(key_shape, device=device)
    write = torch.rand(key_shape, device=device)
    gamma = 0.99 + 0.009 * torch.rand(key_shape, device=device)
    state = torch.randn(1, heads, 128, 128, device=device) * 0.01
    return q, k, v, erase, write, gamma, state


def _cuda_module():
    from research.kmd2_ablation import qwen_gdn2_triton as module

    if not module.triton_gdn2_segment_available():
        pytest.skip("CUDA/Triton is unavailable")
    return module


@pytest.mark.parametrize(
    ("tokens", "heads", "endpoint_loss"),
    ((64, 16, True), (17, 8, False)),
)
def test_gdn2_raw_triton_segment_matches_oracle_forward_and_vjp(
    tokens, heads, endpoint_loss,
):
    module = _cuda_module()
    source = _inputs(torch.device("cuda"), tokens=tokens, heads=heads)
    custom_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source
    )
    oracle_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source
    )
    custom = module.triton_gdn2_segment(*custom_inputs)
    expected = _oracle(*oracle_inputs)

    torch.testing.assert_close(custom[0], expected[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(custom[1], expected[1], atol=1e-6, rtol=1e-5)

    torch.manual_seed(9182 + tokens)
    output_weight = torch.randn_like(custom[0]) * 0.01
    custom_loss = (custom[0] * output_weight).sum()
    oracle_loss = (expected[0] * output_weight).sum()
    if endpoint_loss:
        state_weight = torch.randn_like(custom[1]) * 0.01
        custom_loss = custom_loss + (custom[1] * state_weight).sum()
        oracle_loss = oracle_loss + (expected[1] * state_weight).sum()
    actual_gradients = torch.autograd.grad(custom_loss, custom_inputs)
    expected_gradients = torch.autograd.grad(oracle_loss, oracle_inputs)
    for actual, reference in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=5e-4)


def test_gdn2_raw_triton_dispatch_is_fail_closed():
    module = _cuda_module()
    source = _inputs(torch.device("cuda"), tokens=2, heads=16)
    assert module.can_use_triton_gdn2_segment(*source)
    assert not module.can_use_triton_gdn2_segment(
        source[0].to(torch.bfloat16), *source[1:]
    )
    assert not module.can_use_triton_gdn2_segment(
        source[0][..., ::2], *source[1:]
    )
    with pytest.raises(RuntimeError, match="not dispatchable"):
        module.triton_gdn2_segment(source[0][..., ::2], *source[1:])


def test_gdn2_canonical_dispatch_matches_forced_reference_vjp(monkeypatch):
    triton_module = _cuda_module()
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_architecture import (
        KMD2ChannelwiseGDN2Attn,
    )

    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    torch.manual_seed(1309)
    config = SimpleNamespace(
        hidden_size=64,
        linear_num_value_heads=8,
        linear_num_key_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    native = KMD2NativeAttn(config, layer_idx=0).to(device="cuda")
    fused = KMD2ChannelwiseGDN2Attn.from_native(native)
    fused.checkpoint_segment_tokens = 64
    reference = copy.deepcopy(fused)
    reference.force_reference_path = True

    calls = []
    real_wrapper = triton_module.triton_gdn2_segment

    def counted_wrapper(*args, **kwargs):
        calls.append(args[0].shape[1])
        return real_wrapper(*args, **kwargs)

    monkeypatch.setattr(triton_module, "triton_gdn2_segment", counted_wrapper)
    source = torch.randn(1, 65, 64, device="cuda") * 0.1
    fused_input = source.detach().clone().requires_grad_(True)
    reference_input = source.detach().clone().requires_grad_(True)
    fused_output = fused(fused_input)
    assert calls == [64, 1]
    calls_before_reference = len(calls)
    reference_output = reference(reference_input)
    assert len(calls) == calls_before_reference
    torch.testing.assert_close(
        fused_output, reference_output, atol=1e-6, rtol=1e-5
    )

    probe = torch.randn_like(fused_output)
    fused_loss = (fused_output * probe).sum()
    reference_loss = (reference_output * probe).sum()
    parameter_names = (
        "erase_proj.weight",
        "write_proj.weight",
        "write_offset",
        "decay_chan",
        "rot_proj.weight",
    )
    fused_parameters = dict(fused.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    fused_gradients = torch.autograd.grad(
        fused_loss,
        (fused_input, *(fused_parameters[name] for name in parameter_names)),
    )
    assert len(calls) > calls_before_reference
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (
            reference_input,
            *(reference_parameters[name] for name in parameter_names),
        ),
    )
    for name, actual, expected in zip(
        ("input", *parameter_names),
        fused_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual, expected, atol=1e-4, rtol=5e-4, msg=name
        )


def test_gdn2_no_grad_forward_skips_backward_trace():
    module = _cuda_module()
    device = torch.device("cuda")
    source = _inputs(device, tokens=64, heads=16)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        eval_outputs = module.triton_gdn2_segment(*source)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    trace_bytes = 64 * 16 * 128 * 128 * 4
    assert peak < trace_bytes // 2, f"no_grad peak {peak} suggests a trace"

    grad_source = tuple(tensor.clone().requires_grad_() for tensor in source)
    grad_outputs = module.triton_gdn2_segment(*grad_source)
    for expected, actual in zip(eval_outputs, grad_outputs, strict=True):
        assert torch.equal(expected, actual.detach())
    sum(output.square().sum() for output in grad_outputs).backward()
    for index, tensor in enumerate(grad_source):
        assert tensor.grad is not None, index
        assert bool(torch.isfinite(tensor.grad).all()), index
