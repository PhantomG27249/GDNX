from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _independent_math_oracle(module, hidden, boundary, valid):
    """Literal Package-A construction; recurrence comes only from the math oracle."""
    import torch.nn.functional as F
    from research.kmd2_ablation.qwen_hybrid_math import (
        apply_complex_rotation, braided_decay, shared_state_step,
    )
    B, T = hidden.shape[:2]
    tail = hidden.new_zeros(B, module.conv_k - 1, hidden.shape[-1])
    projected = [[] for _ in range(4)]; signals = []
    for token in range(T):
        reset = (boundary[:, token] & valid[:, token])[:, None, None]
        tail = torch.where(reset, torch.zeros_like(tail), tail)
        window = torch.cat((tail, hidden[:, token:token + 1]), 1)
        rank_signals = []
        for rank in range(4):
            weights = (module.components.q_weight[rank], module.components.k_weight[rank],
                       module.components.v_weight[rank])
            qkv = torch.cat([F.linear(window, weight) for weight in weights], -1)
            mixed = F.silu(F.conv1d(qkv.transpose(1, 2), module.components.conv1d.weight,
                                    groups=qkv.shape[-1])).transpose(1, 2)[:, 0]
            projected[rank].append(torch.split(mixed, (module.key_dim, module.key_dim, module.value_dim), -1))
            rank_signals.append(mixed)
        signals.append(torch.stack(rank_signals).mean(0))
        shifted = torch.cat((tail[:, 1:], hidden[:, token:token + 1]), 1)
        tail = torch.where(valid[:, token, None, None], shifted, tail)
    q = torch.stack([torch.stack([x[0] for x in lane], 1).reshape(B, T, module.H, module.dk)
                     for lane in projected], 3)
    k = torch.stack([torch.stack([x[1] for x in lane], 1).reshape(B, T, module.H, module.dk)
                     for lane in projected], 3)
    v = torch.stack([torch.stack([x[2] for x in lane], 1).reshape(B, T, module.H, module.dv)
                     for lane in projected], 3)
    route = torch.einsum("btc,cd->btd", torch.stack(signals, 1), torch.cat((
        module.components.q_weight[0], module.components.k_weight[0], module.components.v_weight[0]), 0))
    _, _, _, erase, write, z = module.components.project_inputs(hidden)
    q, k = module.components.affine_qk(q, k)
    q = F.normalize(module._adjacent_complex_layout(q).float(), dim=-1, eps=1e-6) * module.dk ** -.5
    k = F.normalize(module._adjacent_complex_layout(k).float(), dim=-1, eps=1e-6)
    erase = module._adjacent_complex_layout(erase.float().sigmoid())
    write, v = write.float().sigmoid(), v.float()
    probabilities = module.components.braid_probabilities(route).float()
    residual = module.components.braid_residual.float()[None, None].expand_as(probabilities)
    gamma = braided_decay(module.components.native_decay(hidden).float(), probabilities, residual,
                          hidden.new_tensor((64., 512., 4096., 32768.)).float())
    gamma = module._adjacent_complex_layout(gamma)
    theta = F.softplus(module.rot_proj(hidden)).reshape(B, T, module.H, module.dk // 2).float()
    theta = theta + module.components.phase_logits(route).float()
    state = torch.zeros(B, module.H, module.dk, module.dv)
    phase = torch.zeros(B, module.H, module.dk // 2)
    previous_value = torch.zeros(B, module.H, 4, module.dv)
    previous_write = torch.zeros(B, module.H, 4, module.dk, module.dv)
    has_history = torch.zeros(B, dtype=torch.bool)
    outputs = []
    for token in range(T):
        reset = (boundary[:, token] & valid[:, token])[:, None, None]
        state = torch.where(reset[..., None], torch.zeros_like(state), state)
        phase = torch.where(reset, torch.zeros_like(phase), phase)
        previous_value = torch.where(reset[..., None], torch.zeros_like(previous_value), previous_value)
        previous_write = torch.where(reset[..., None, None], torch.zeros_like(previous_write), previous_write)
        history = valid[:, token] & ~boundary[:, token] & has_history
        phase = torch.where(valid[:, token, None, None], phase + theta[:, token], phase)
        expanded = phase[:, :, None].expand(-1, -1, 4, -1)
        qr = apply_complex_rotation(q[:, token], expanded)
        kr = apply_complex_rotation(k[:, token], expanded)
        rho = module.components.lookahead_gate.float()[None, :, :, None] * history[:, None, None, None]
        vv = v[:, token] + rho * (v[:, token] - previous_value)
        next_state, _, write_delta = shared_state_step(
            state, kr, vv, erase[:, token], write[:, token], gamma[:, token],
            module.components.c.float()[None].expand(B, -1, -1),
            module.components.d.float()[None].expand(B, -1, -1), qr,
            module.components.output_mixer.float(), previous_write=previous_write,
            trap_rho=module.components.trapezoid_gate.float(), history_active=history,
        )
        active = valid[:, token, None, None]
        state = torch.where(active[..., None], next_state, state)
        previous_write = torch.where(active[..., None, None], write_delta, previous_write)
        previous_value = torch.where(active[..., None], v[:, token], previous_value)
        has_history |= valid[:, token]
        reads = torch.einsum("bhrk,bhkv->bhrv", qr, state)
        normalized = torch.stack([
            module.components.norm(reads[:, :, rank].reshape(-1, module.dv),
                                   z[:, token, :, rank].reshape(-1, module.dv)).reshape(B, module.H, module.dv)
            for rank in range(4)
        ], 2)
        mixed = torch.einsum("hrvw,bhrw->bhv", module.components.output_mixer, normalized)
        out = module.components.out_proj(mixed.flatten(1))
        outputs.append(torch.where(valid[:, token, None], out, torch.zeros_like(out)))
    return torch.stack(outputs, 1)


def _native(monkeypatch, dtype=torch.float32, *, key_head_dim=4):
    from gdn3.kmd2_native import KMD2NativeAttn
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    return KMD2NativeAttn(SimpleNamespace(
        hidden_size=16, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=key_head_dim, linear_value_head_dim=4,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6,
    ), layer_idx=3).to(dtype=dtype)


def test_shared_module_matches_oracle(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid

    torch.manual_seed(41)
    native = _native(monkeypatch)
    module = QwenSharedBraidHybrid.from_native(native)
    # 2026-07-14: the oracle does not model the HOLA cache read, whose warm
    # gate sigmoid(-4) contributes ~2e-3 at init by design ("Option B").
    # Compare the recurrence with the cache disabled; HOLA has its own tests.
    module.active_feature_flags = {"cache_policy": "none"}
    x = torch.randn(2, 5, 16, requires_grad=True)
    boundary = torch.tensor([[True, False, False, True, False], [True, False, True, False, False]])
    valid = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    actual = module(x, boundary=boundary, valid=valid)
    expected = _independent_math_oracle(module, x, boundary, valid)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    neutral = torch.randn(2, 4, 16)
    torch.testing.assert_close(module(neutral), native(neutral), atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("q_weight", "k_weight", "v_weight", "erase_weight", "write_weight", "z_weight"):
        grad = getattr(module.components, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum(dim=tuple(range(1, grad.ndim))).gt(0).all(), name
    expected_bytes = (2 * module.H * module.dk * module.dv * 4
                      + 2 * module.H * 4 * module.dk * module.dv * 4 + 2)
    assert module.recurrent_state_bytes(batch_size=2) == expected_bytes
    assert torch.equal(module.components.conv1d.weight, native.conv1d.weight)
    # Restore the default feature flags: the cache-carry assertions below
    # exercise the live HOLA path again.
    module.active_feature_flags = {}
    cached = module(x.detach(), use_cache=True)
    torch.testing.assert_close(cached, module(x.detach()))
    assert module.last_recurrent_cache.hola_state is not None

    native_bf16 = _native(monkeypatch, torch.bfloat16)
    bf16 = QwenSharedBraidHybrid.from_native(native_bf16)
    low = neutral.to(torch.bfloat16)
    torch.testing.assert_close(bf16(low), native_bf16(low), atol=2e-2, rtol=2e-2)


def test_shared_every_chunk_partition_matches_decode(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid

    torch.manual_seed(73)
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    x = torch.randn(2, 6, 16)
    boundary = torch.tensor([[True, False, False, True, False, False], [True, False, True, False, False, True]])
    valid = torch.tensor([[True, True, True, True, True, False], [True, True, True, True, True, True]])
    whole, _ = module.scan(x, boundary=boundary, valid=valid)
    for partition in ((1, 1, 1, 1, 1, 1), (2, 4), (3, 1, 2), (5, 1)):
        cache = None; outputs = []; start = 0
        for width in partition:
            out, cache = module.scan(x[:, start:start+width], boundary=boundary[:, start:start+width],
                                     valid=valid[:, start:start+width], initial_cache=cache)
            outputs.append(out); start += width
        torch.testing.assert_close(torch.cat(outputs, 1), whole, atol=1e-6, rtol=1e-5)


def test_shared_boundary_forces_lookahead_and_trapezoid_inactive(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    torch.manual_seed(91)
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    with torch.no_grad():
        module.components.lookahead_gate.fill_(0.8)
        module.components.trapezoid_gate.fill_(0.7)
    valid = torch.tensor([True, True, False, True])
    boundary = torch.tensor([True, False, False, True])
    has_history = torch.tensor([True, False, True, True])
    assert torch.equal(module.history_active(boundary, valid, has_history),
                       torch.tensor([False, False, False, False]))
    assert torch.equal(module.history_active(torch.zeros(4, dtype=torch.bool), valid,
                                             torch.ones(4, dtype=torch.bool)), valid)


def test_shared_manifest_is_exact_and_active_controls_receive_gradients(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    torch.manual_seed(109)
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    manifest = module.transformation_manifest()
    assert set(manifest["parameters"]) == set(module.state_dict())
    assert "rot_proj.weight" in manifest["parameters"] and "rot_proj.bias" in manifest["parameters"]
    with torch.no_grad():
        c = module.components
        c.lookahead_gate.fill_(0.2); c.trapezoid_gate.fill_(0.2)
        c.alpha_q.fill_(0.1); c.beta_q.fill_(0.1); c.alpha_k.fill_(0.1); c.beta_k.fill_(0.1)
        c.braid_residual.copy_(torch.linspace(-.2, .2, c.braid_residual.numel()).reshape_as(c.braid_residual))
        c.phase_proj.weight.normal_(0, .02)
    output = module(torch.randn(2, 4, 16)).square().mean()
    output.backward()
    names = ("output_mixer", "alpha_q", "beta_q", "alpha_k", "beta_k", "d_q", "d_k", "b_q", "b_k",
             "braid_router.weight", "braid_residual", "phase_proj.weight", "lookahead_gate", "trapezoid_gate")
    parameters = dict(module.named_parameters())
    for name in names:
        full = name if name.startswith("rot_proj") else f"components.{name}"
        assert parameters[full].grad is not None and parameters[full].grad.abs().sum() > 0, full
    assert module.components.output_mixer.grad.abs().sum(dim=(0, 2, 3)).gt(0).all()
    assert module.rot_proj.weight.grad is not None and module.rot_proj.weight.grad.abs().sum() > 0


def test_hybrid_manifests_name_the_executed_reference_loop_not_deferred_kernels(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.qwen_hybrid_math import (
        REFERENCE_IMPLEMENTATION,
        RUNTIME_BACKEND,
    )
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid

    for module in (QwenSharedBraidHybrid.from_native(_native(monkeypatch)),
                   QwenFourStateHybrid.from_native(_native(monkeypatch, key_head_dim=8))):
        manifest = module.transformation_manifest()
        assert module.actual_implementation_identity() == REFERENCE_IMPLEMENTATION
        assert module.scan_implementation == REFERENCE_IMPLEMENTATION
        assert manifest["implementation"] == REFERENCE_IMPLEMENTATION
        assert manifest["scan_implementation"] == REFERENCE_IMPLEMENTATION
        serialized = repr(manifest).lower()
        assert "hybrid_r4_scan" not in serialized
        if type(module) is QwenFourStateHybrid:
            # The semantic oracle and the actual runtime dispatch are separate
            # machine-readable fields; the fused Triton segment recurrence is
            # live production behavior, not deferred, and must be declared.
            assert manifest["runtime_backend"] == RUNTIME_BACKEND
            assert "triton" in manifest["runtime_backend"]
        else:
            assert "triton" not in serialized


def test_shared_math_oracle_is_simultaneous_not_sequential():
    from research.kmd2_ablation.qwen_hybrid_math import identity_output_mixer, shared_state_step
    B, H, K, V = 1, 1, 2, 1
    state = torch.tensor([[[[1.], [2.]]]])
    key = torch.tensor([[[[1., 0.], [0., 1.], [1., 1.], [1., -1.]]]])
    query = key.clone(); value = torch.ones(B, H, 4, V)
    erase = torch.ones_like(key); write = torch.ones_like(value)
    coeff = torch.full((B, H, 4), .25); gamma = torch.ones(B, H, K)
    mixer = identity_output_mixer("shared", H, V, dtype=torch.float32)
    simultaneous, _, _ = shared_state_step(state, key, value, erase, write, gamma, coeff, coeff, query, mixer)
    sequential = state.clone()
    for rank in range(4):
        memory = torch.einsum("bhk,bhkv->bhv", key[:, :, rank], sequential)
        sequential = sequential - .25 * key[:, :, rank, :, None] * memory[:, :, None]
        sequential = sequential + .25 * key[:, :, rank, :, None] * value[:, :, rank, None]
    assert not torch.allclose(simultaneous, sequential)


def test_shared_convolution_isolates_boundaries_and_invalid_tokens(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    torch.manual_seed(151)
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    base = torch.randn(1, 7, 16)
    boundary = torch.tensor([[True, False, False, True, False, False, False]])
    valid = torch.tensor([[True, True, False, True, True, True, True]])
    changed_prefix = base.clone(); changed_prefix[:, :3] += 100
    changed_padding = base.clone(); changed_padding[:, 2] -= 200
    reference = module(base, boundary=boundary, valid=valid)
    prefix = module(changed_prefix, boundary=boundary, valid=valid)
    padding = module(changed_padding, boundary=boundary, valid=valid)
    torch.testing.assert_close(prefix[:, 3:], reference[:, 3:], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(padding[:, 3:], reference[:, 3:], atol=1e-6, rtol=1e-5)


def test_shared_empty_chunk_preserves_cache_exactly(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    _, cache = module.scan(torch.randn(2, 3, 16))
    empty = torch.empty(2, 0, 16)
    output, after = module.scan(empty, initial_cache=cache)
    assert output.shape == (2, 0, 16)
    assert after is cache


def test_shared_initial_empty_chunk_returns_initialized_cache(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch, torch.bfloat16))
    empty = torch.empty(2, 0, 16, dtype=torch.bfloat16)
    output, cache = module.scan(empty)
    assert output.shape == (2, 0, 16) and output.dtype == torch.bfloat16
    assert cache.state.shape == (2, module.H, module.dk, module.dv)
    assert cache.phase.shape == (2, module.H, module.dk // 2)
    assert cache.previous_value.shape == (2, module.H, 4, module.dv)
    assert cache.previous_write.shape == (2, module.H, 4, module.dk, module.dv)
    assert cache.conv_tail.shape == (2, module.conv_k - 1, 16)
    assert cache.has_history.shape == (2,)
    assert cache.state.dtype == cache.phase.dtype == cache.previous_value.dtype == torch.float32
    assert cache.previous_write.dtype == torch.float32
    assert cache.conv_tail.dtype == torch.bfloat16 and cache.has_history.dtype == torch.bool
    assert all(t.device == empty.device for t in (cache.state, cache.phase, cache.previous_value,
                                                   cache.previous_write, cache.conv_tail, cache.has_history))
def test_shared_active_hola_path_has_parameter_gradients(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    torch.manual_seed(211)
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    with torch.no_grad():
        module.components.cache_gate_logit.fill_(torch.logit(torch.tensor(0.2)))
    x = torch.randn(1, 4, 16, requires_grad=True)
    module(x).square().sum().backward()
    for parameter in (module.components.cache_gate_logit, module.hola_output_mixer,
                      module.hola.gamma_q, module.hola.gamma_k, module.hola.sink_logit):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_shared_hola_cross_mixer_does_not_collapse_rin_before_mixing(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    reads = torch.zeros(1, module.H, 4, 4, module.dv)
    reads[:, :, 0, 0] = 2
    reads[:, :, 0, 1] = 7
    with torch.no_grad():
        module.hola_output_mixer.zero_()
        eye = torch.eye(module.dv)
        module.hola_output_mixer[:, 0, 0] = eye
        module.hola_output_mixer[:, 0, 1] = 3 * eye
    mixed = module._mix_hola_reads(reads)
    torch.testing.assert_close(mixed, torch.full_like(mixed, 23))


def test_shared_scan_validates_hola_contract_once_not_per_token(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    module = QwenSharedBraidHybrid.from_native(_native(monkeypatch))
    calls = {"inputs": 0, "state": 0}
    for name, key in (("_validate_inputs", "inputs"), ("_validate_state", "state")):
        original = getattr(module.hola, name)
        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(module.hola, name, counted)
    module(torch.randn(1, 7, 16))
    assert calls == {"inputs": 1, "state": 1}
