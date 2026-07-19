from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch


def _native(monkeypatch, dtype=torch.float32):
    from gdn3.kmd2_native import KMD2NativeAttn
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    return KMD2NativeAttn(SimpleNamespace(
        hidden_size=16, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=4,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6,
    ), layer_idx=3).to(dtype=dtype)


def _assert_dataclass_tensors_close(actual, expected, *, atol, rtol):
    assert type(actual) is type(expected)
    for field in fields(actual):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if is_dataclass(left):
            _assert_dataclass_tensors_close(
                left, right, atol=atol, rtol=rtol,
            )
        elif isinstance(left, torch.Tensor):
            if left.is_floating_point():
                torch.testing.assert_close(
                    left, right, atol=atol, rtol=rtol,
                    msg=lambda message, name=field.name: f"{name}: {message}",
                )
            else:
                assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name


def test_braid_and_cms_use_the_existing_mimo_rank_axis(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    components = module.components
    hidden = torch.zeros(3, 5, 16)
    gamma = components.decay_gamma(hidden)
    assert gamma.shape == (3, 5, module.H, 4, module.dk)
    rate = -gamma.log()
    # The configured ratios are native-relative horizon multipliers and live
    # in log-rate space.
    torch.testing.assert_close(rate[..., 0, :] / rate[..., 1, :], torch.full_like(rate[..., 0, :], 16.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rate[..., 1, :] / rate[..., 2, :], torch.full_like(rate[..., 1, :], 4.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rate[..., 2, :] / rate[..., 3, :], torch.full_like(rate[..., 2, :], 4.0), rtol=1e-5, atol=1e-5)
    assert not hasattr(components, "state_braid_router")
    assert not hasattr(components, "state_braid_gate")
    assert module.update_periods == (1, 16, 64, 256)
    manifest = module.transformation_manifest()
    assert manifest["state_count"] == manifest["write_paths"] == manifest["transition_paths"] == 4
    assert manifest["resources"]["timescale_axis"] == "mimo_rank"
    assert manifest["resources"]["cadence_or_cms_updates"] is True
    assert manifest["resources"]["update_periods"] == (1, 16, 64, 256)


def test_compact_bands_match_one_native_state_and_factorize_history(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    native = _native(monkeypatch)
    module = QwenFourStateHybrid.from_native(native)
    assert module.dk == native.dk // 4
    cache = module._initial_cache(torch.zeros(2, 1, 16))
    assert cache.states.numel() == 2 * native.H * native.dk * native.dv
    assert cache.previous_key.shape == (2, native.H, 4, native.dk // 4)
    assert cache.previous_value.shape == (2, native.H, 4, native.dv)
    report = module.resource_report(batch_size=2)
    assert report["read_compute_native_equivalents"] == 4
    assert report["output_cross_reads_per_token"] == 16
    assert report["trapezoid_history"] == "exact_key_value_outer_product_factors"


def test_four_state_rejects_nonpartitionable_key_width_before_components(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    native = _native(monkeypatch)
    native.dk = 6
    construction_calls = []

    def unexpected_component_construction(*args, **kwargs):
        construction_calls.append((args, kwargs))
        raise AssertionError("full-width Package B fallback must not be constructed")

    monkeypatch.setattr(
        HybridComponents, "from_native", unexpected_component_construction,
    )
    with pytest.raises(ValueError, match="four compact complex-pair bands"):
        QwenFourStateHybrid.from_native(native)
    assert construction_calls == []


def test_noncanonical_compact_model_keeps_complete_torch_fallback(monkeypatch):
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    assert module.components.compact_key_bands
    assert module.dk == 2
    recurrence_calls = []
    original = chunkwise.torch_chunk_four_state_segment

    def counted(*args, **kwargs):
        recurrence_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(chunkwise, "torch_chunk_four_state_segment", counted)
    output, cache = module.scan(torch.randn(2, 17, 16))
    assert output.shape == (2, 17, 16)
    assert cache.update_count.tolist() == [17, 17]
    assert recurrence_calls == []


def test_first_braid_lane_is_pair_tied_native_gdn_decay(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    native = _native(monkeypatch)
    with torch.no_grad():
        native.decay_chan.copy_(torch.tensor([
            [0.2, -0.1, 0.4, 0.3, -0.2, 0.1, 0.0, 0.5],
            [-0.2, 0.1, 0.0, 0.5, 0.2, -0.1, 0.4, 0.3],
        ]))
    module = QwenFourStateHybrid.from_native(native)
    hidden = torch.randn(2, 3, 16)
    components = module.components
    a = torch.einsum("btd,hd->bth", hidden, native.in_proj_a.weight)
    native_log_gamma = (
        -native.A_log.exp()[None, None]
        * torch.nn.functional.softplus(a + native.dt_bias[None, None])
    )
    pair_channel = 0.5 * (
        native.decay_chan[:, : native.dk // 2]
        + native.decay_chan[:, native.dk // 2 :]
    ).reshape(native.H, 4, module.dk // 2)
    pair_gamma = (native_log_gamma[..., None, None] + pair_channel[None, None]).exp()
    expected = torch.cat((pair_gamma, pair_gamma), -1).clamp(
        min=2.0 ** -24, max=1.0
    )
    got = components.decay_gamma(hidden)
    torch.testing.assert_close(got, expected ** torch.tensor(
        (1.0, 1 / 16, 1 / 64, 1 / 256), dtype=expected.dtype
    )[None, None, None, :, None])
    rates = got.log().neg()
    unclamped = rates[..., 1, :] > 0
    assert bool(unclamped.any())
    torch.testing.assert_close(
        rates[..., 0, :][unclamped] / rates[..., 1, :][unclamped],
        torch.full_like(rates[..., 0, :][unclamped], 16.0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_cms_ticks_follow_valid_token_positions(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    positions = torch.tensor([0, 1, 15, 16, 63, 64, 255, 256], dtype=torch.int64)
    got = module.lane_update_mask(positions, torch.ones_like(positions, dtype=torch.bool))
    expected = torch.tensor([
        [True, True, True, True],
        [True, False, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, False, False, False],
        [True, True, True, False],
        [True, False, False, False],
        [True, True, True, True],
    ])
    assert torch.equal(got, expected)


def test_off_tick_lanes_only_decay_state_and_trapezoid_history(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    hidden = torch.randn(1, 2, 16)
    _, first = module.scan(
        hidden[:, :1], boundary=torch.tensor([[True]]), valid=torch.tensor([[True]])
    )
    _, second = module.scan(
        hidden[:, 1:], boundary=torch.tensor([[False]]), valid=torch.tensor([[True]]),
        initial_cache=first,
    )
    gamma = module._adjacent_complex_layout(
        module.components.decay_gamma(hidden[:, 1:]).float()
    )[:, 0]
    torch.testing.assert_close(
        second.states[:, :, 1:],
        gamma[:, :, 1:, :, None] * first.states[:, :, 1:],
    )
    torch.testing.assert_close(
        second.previous_key[:, :, 1:],
        gamma[:, :, 1:] * first.previous_key[:, :, 1:],
    )
    torch.testing.assert_close(second.previous_value[:, :, 1:], first.previous_value[:, :, 1:])


def test_decay_is_pair_tied_and_commutes_with_complex_rotations(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.qwen_hybrid_math import apply_complex_rotation
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    gamma = module._adjacent_complex_layout(module.components.decay_gamma(torch.randn(2, 3, 16)))
    torch.testing.assert_close(gamma[..., 0::2], gamma[..., 1::2])
    x = torch.randn(2, module.H, 4, module.dk)
    phase = torch.randn(2, module.H, 4, module.dk // 2)
    left = apply_complex_rotation(x * gamma[:, 0], phase)
    right = apply_complex_rotation(x, phase) * gamma[:, 0]
    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
    # An old non-pair-tied channel decay must fail the same equivalence.
    bad = gamma[:, 0].clone(); bad[..., 0] *= .5
    assert not torch.allclose(apply_complex_rotation(x * bad, phase), apply_complex_rotation(x, phase) * bad)


def test_trapezoid_is_token_dependent_and_reuses_full_gdn_transition(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import (
        QwenFourStateHybrid, gdn_homogeneous_transition,
    )
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    with torch.no_grad():
        module.components.trapezoid_proj.weight.normal_()
    hidden = torch.randn(2, 3, 16)
    lam = module.components.trapezoid_lambda(hidden)
    assert lam.shape == (2, 3, module.H, 4)
    assert not torch.equal(lam[:, 0], lam[:, 1])
    state = torch.randn(2, module.H, 4, module.dk, module.dv)
    previous = torch.randn_like(state)
    key = torch.nn.functional.normalize(torch.randn(2, module.H, 4, module.dk), dim=-1)
    erase = torch.rand_like(key)
    gamma = module._adjacent_complex_layout(module.components.decay_gamma(hidden))[:, 0]
    # Both operands go through exactly A_t, including current erase.
    got_state = gdn_homogeneous_transition(state, key, erase, gamma)
    got_previous = gdn_homogeneous_transition(previous, key, erase, gamma)
    def literal(x):
        decayed = gamma[..., None] * x
        memory = torch.einsum("bhrk,bhrkv->bhrv", erase * key, decayed)
        return decayed - key[..., None] * memory[..., None, :]
    torch.testing.assert_close(got_state, literal(state))
    torch.testing.assert_close(got_previous, literal(previous))


def test_factorized_trapezoid_matches_independent_dense_history_oracle():
    """The carried p/c factors reconstruct the literal dense endpoint."""
    from research.kmd2_ablation.qwen_hybrid_four_state import (
        gdn_homogeneous_transition,
    )

    torch.manual_seed(2119)
    B, T, H, R, K, V = 2, 7, 2, 4, 3, 2
    key = torch.nn.functional.normalize(torch.randn(B, T, H, R, K), dim=-1)
    value = torch.randn(B, T, H, R, V)
    erase = torch.rand(B, T, H, R, K)
    write = torch.rand(B, T, H, R, V)
    gamma = 0.8 + 0.19 * torch.rand(B, T, H, R, K)
    lam = 0.1 + 0.8 * torch.rand(B, T, H, R)
    factor_state = torch.randn(B, H, R, K, V)
    dense_state = factor_state.clone()
    previous_key = torch.randn(B, H, R, K)
    previous_value = torch.randn(B, H, R, V)
    dense_history = previous_key[..., None] * previous_value[..., None, :]
    has_history = torch.tensor(
        [[False, True, False, True], [True, False, True, False]]
    )
    count = torch.tensor([0, 15], dtype=torch.int64)
    periods = count.new_tensor((1, 16, 64, 256))

    for token in range(T):
        tick_lanes = count[:, None].remainder(periods[None]).eq(0)
        tick = tick_lanes[:, None, :, None, None]
        key_t = key[:, token]
        erased_key = erase[:, token] * key_t
        current_value = write[:, token] * value[:, token]
        current_write = key_t[..., None] * current_value[..., None, :]
        effective_lam = torch.where(
            has_history[:, None, :], lam[:, token], torch.ones_like(lam[:, token])
        )[..., None, None]

        # Authoritative factorized endpoint path.
        factor_decayed = gamma[:, token, ..., None] * factor_state
        factor_homogeneous = gdn_homogeneous_transition(
            factor_state, key_t, erase[:, token], gamma[:, token]
        )
        previous_key_decayed = gamma[:, token] * previous_key
        previous_projection = torch.einsum(
            "bhrk,bhrk->bhr", erased_key, previous_key_decayed
        )
        transported_key = (
            previous_key_decayed - key_t * previous_projection[..., None]
        )
        factor_update = (
            (1.0 - effective_lam)
            * transported_key[..., None]
            * previous_value[..., None, :]
            + effective_lam * current_write
        )
        factor_state = torch.where(
            tick, factor_homogeneous + factor_update, factor_decayed
        )
        previous_key = torch.where(
            tick_lanes[:, None, :, None], key_t, previous_key_decayed
        )
        previous_value = torch.where(
            tick_lanes[:, None, :, None], current_value, previous_value
        )

        # Independent oracle: carry and transition the full KxV endpoint.
        dense_decayed = gamma[:, token, ..., None] * dense_state
        dense_memory = torch.einsum(
            "bhrk,bhrkv->bhrv", erased_key, dense_decayed
        )
        dense_homogeneous = (
            dense_decayed - key_t[..., None] * dense_memory[..., None, :]
        )
        dense_history_decayed = gamma[:, token, ..., None] * dense_history
        dense_history_memory = torch.einsum(
            "bhrk,bhrkv->bhrv", erased_key, dense_history_decayed
        )
        dense_history_transported = (
            dense_history_decayed
            - key_t[..., None] * dense_history_memory[..., None, :]
        )
        dense_update = (
            (1.0 - effective_lam) * dense_history_transported
            + effective_lam * current_write
        )
        dense_state = torch.where(
            tick, dense_homogeneous + dense_update, dense_decayed
        )
        dense_history = torch.where(
            tick, current_write, dense_history_decayed
        )

        torch.testing.assert_close(factor_state, dense_state, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            previous_key[..., None] * previous_value[..., None, :],
            dense_history,
            atol=1e-6,
            rtol=1e-6,
        )
        has_history = has_history | tick_lanes
        count = count + 1


def test_full_and_arbitrary_chunk_scans_match_with_boundaries_and_hola(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    torch.manual_seed(223)
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    x = torch.randn(2, 6, 16)
    boundary = torch.tensor([[True, False, False, True, False, False], [True, False, True, False, False, True]])
    valid = torch.tensor([[True, True, False, True, True, False], [True, True, True, True, True, True]])
    whole, whole_cache = module.scan(x, boundary=boundary, valid=valid)
    for cuts in product((1, 2, 3), repeat=5):
        if sum(cuts) != 6:
            continue
        cache = None; pieces = []; start = 0
        for width in cuts:
            out, cache = module.scan(x[:, start:start+width], boundary=boundary[:, start:start+width],
                                     valid=valid[:, start:start+width], initial_cache=cache)
            pieces.append(out); start += width
        torch.testing.assert_close(torch.cat(pieces, 1), whole, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(cache.states, whole_cache.states)
        torch.testing.assert_close(cache.previous_key, whole_cache.previous_key)
        torch.testing.assert_close(cache.previous_value, whole_cache.previous_value)
        assert torch.equal(cache.has_history, whole_cache.has_history)
        assert torch.equal(cache.update_count, whole_cache.update_count)
        assert cache.hola_state.next_position.equal(whole_cache.hola_state.next_position)
    assert torch.equal(whole_cache.update_count, torch.tensor([2, 1]))


def test_boundary_free_fast_path_matches_forced_reference_cache_and_gradients(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    torch.manual_seed(829)
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    module.checkpoint_segment_tokens = None
    calls = []
    fast_projection = module._project_convolved_fast
    reference_projection = module._project_convolved

    def counted_fast(*args, **kwargs):
        calls.append("fast")
        return fast_projection(*args, **kwargs)

    def counted_reference(*args, **kwargs):
        calls.append("reference")
        return reference_projection(*args, **kwargs)

    monkeypatch.setattr(module, "_project_convolved_fast", counted_fast)
    monkeypatch.setattr(module, "_project_convolved", counted_reference)
    hidden = torch.randn(1, 4, 16, requires_grad=True)
    boundary = torch.zeros(1, 4, dtype=torch.bool)
    valid = torch.ones(1, 4, dtype=torch.bool)

    fast_output, fast_cache = module.scan(
        hidden, boundary=boundary, valid=valid,
    )
    assert calls == ["fast"]
    module.force_reference_path = True
    reference_output, reference_cache = module.scan(
        hidden, boundary=boundary, valid=valid,
    )
    assert calls == ["fast", "reference"]

    # Compact lazy decay changes only FP32 multiplication grouping versus the
    # eager masked oracle; keep the resulting few-ulp envelope explicit.
    torch.testing.assert_close(
        fast_output, reference_output, atol=5e-6, rtol=1e-5,
    )
    _assert_dataclass_tensors_close(
        fast_cache, reference_cache, atol=5e-6, rtol=1e-5,
    )

    weights = torch.linspace(
        0.5, 1.5, fast_output.numel(), dtype=fast_output.dtype,
    ).reshape_as(fast_output)
    fast_loss = (fast_output * weights).sum() + 0.01 * fast_output.square().sum()
    reference_loss = (
        (reference_output * weights).sum()
        + 0.01 * reference_output.square().sum()
    )
    named_parameters = tuple(module.named_parameters())
    gradient_targets = (hidden, *(p for _, p in named_parameters))
    fast_gradients = torch.autograd.grad(
        fast_loss, gradient_targets, allow_unused=True,
    )
    reference_gradients = torch.autograd.grad(
        reference_loss, gradient_targets, allow_unused=True,
    )
    gradient_names = ("hidden", *(n for n, _ in named_parameters))
    # Vectorized convolution, cumulative rotation and causal HOLA batching
    # change only FP32 accumulation order.  This remains well inside the
    # segmented-checkpoint training envelope pinned below.
    for name, fast_gradient, reference_gradient in zip(
            gradient_names, fast_gradients, reference_gradients):
        assert fast_gradient is not None, name
        assert reference_gradient is not None, name
        torch.testing.assert_close(
            fast_gradient, reference_gradient, atol=5e-4, rtol=2e-3,
            msg=lambda message, name=name: f"{name} gradient: {message}",
        )
    assert fast_gradients[0] is not None and fast_gradients[0].abs().sum() > 0


def test_segmented_checkpoint_matches_plain_loop_forward_cache_and_gradients(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    torch.manual_seed(719)
    segmented = QwenFourStateHybrid.from_native(_native(monkeypatch))
    plain = copy.deepcopy(segmented)
    segmented.checkpoint_segment_tokens = 16
    plain.checkpoint_segment_tokens = None
    source = torch.randn(1, 65, 16)
    segmented_input = source.clone().requires_grad_()
    plain_input = source.clone().requires_grad_()

    segmented_output, segmented_cache = segmented.scan(segmented_input)
    plain_output, plain_cache = plain.scan(plain_input)

    torch.testing.assert_close(
        segmented_output, plain_output, atol=1e-6, rtol=1e-6,
    )
    _assert_dataclass_tensors_close(
        segmented_cache, plain_cache, atol=1e-7, rtol=1e-5,
    )
    weights = torch.linspace(
        0.25, 1.25, segmented_output.numel(), dtype=segmented_output.dtype,
    ).reshape_as(segmented_output)
    (segmented_output * weights).sum().backward()
    (plain_output * weights).sum().backward()
    torch.testing.assert_close(
        segmented_input.grad, plain_input.grad, atol=1e-3, rtol=2e-4,
    )
    for (name, segmented_parameter), (plain_name, plain_parameter) in zip(
        segmented.named_parameters(), plain.named_parameters(), strict=True,
    ):
        assert name == plain_name
        if segmented_parameter.grad is None and plain_parameter.grad is None:
            continue
        assert segmented_parameter.grad is not None, name
        assert plain_parameter.grad is not None, name
        torch.testing.assert_close(
            segmented_parameter.grad, plain_parameter.grad,
            atol=1e-3, rtol=2e-4,
            msg=lambda message, name=name: f"{name} gradient: {message}",
        )


def test_hola_gate_v1_checkpoint_fails_closed(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    state = module.state_dict()
    key = "components.cache_gate_logit"
    state["components.cache_gate"] = torch.zeros_like(state.pop(key))
    with pytest.raises(RuntimeError, match="amplitude schema v1"):
        module.load_state_dict(state, strict=True)
    assert torch.allclose(module.components.cache_gate_amplitude,
                          torch.full_like(module.components.cache_gate_amplitude, torch.sigmoid(torch.tensor(-4.0))))


def test_hola_path_is_live_and_uses_versioned_logit(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    with torch.no_grad():
        module.components.cache_gate_logit.fill_(torch.logit(torch.tensor(.2)))
    x = torch.randn(1, 5, 16, requires_grad=True)
    module(x, use_cache=True).square().sum().backward()
    assert module.last_recurrent_cache.hola_state is not None
    for parameter in (module.components.cache_gate_logit, module.hola.gamma_q,
                      module.hola.gamma_k, module.hola.sink_logit):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_transformers_decoder_layer_kwargs_are_accepted_fail_closed(monkeypatch):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    hidden = torch.randn(1, 5, 16)
    output = module(
        hidden,
        cache_params=None,
        output_hidden_states=True,
        use_cache=False,
    )
    assert output.shape == hidden.shape
    with pytest.raises(ValueError, match="cache_params"):
        module(hidden, cache_params={"state": torch.ones(1)})


@pytest.mark.parametrize("mutation", ("shape", "dtype", "nan", "clock"))
def test_cache_validation_remains_fail_closed(monkeypatch, mutation):
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    empty = torch.empty(2, 0, 16)
    _, cache = module.scan(empty)
    if mutation == "shape":
        cache = replace(cache, states=cache.states[:, :, :3])
    elif mutation == "dtype":
        cache = replace(cache, phase=cache.phase.double())
    elif mutation == "nan":
        bad = cache.states.clone(); bad[0, 0, 0, 0, 0] = torch.nan
        cache = replace(cache, states=bad)
    else:
        cache = replace(cache, update_count=torch.tensor([-1, 0]))
    with pytest.raises((TypeError, ValueError)):
        module.scan(empty, initial_cache=cache)


def test_no_grad_selects_segment_mixing_and_matches_global(monkeypatch):
    """Inference must not retain full-sequence reads (32K memory regression)."""
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    torch.manual_seed(1301)
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    module.checkpoint_segment_tokens = 16
    hidden = torch.randn(1, 65, 16)

    mixer_calls: list[int] = []
    original_einsum = torch.einsum

    def counting_einsum(equation, *operands):
        if equation.startswith("hijvw,bthijw"):
            mixer_calls.append(operands[1].shape[1])
        return original_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", counting_einsum)
    with torch.no_grad():
        eval_output, eval_cache = module.scan(hidden)
    # Segment mixing applies the [T-batched] mixer contraction never; the
    # whole-sequence global contraction would show one call with T=65.
    assert 65 not in mixer_calls

    mixer_calls.clear()
    grad_input = hidden.clone().requires_grad_()
    train_output, train_cache = module.scan(grad_input)
    assert 65 in mixer_calls  # training keeps global accumulation

    torch.testing.assert_close(
        train_output.detach(), eval_output, atol=1e-6, rtol=1e-6,
    )
    _assert_dataclass_tensors_close(
        train_cache, eval_cache, atol=0.0, rtol=0.0,
    )
    train_output.square().sum().backward()
    assert grad_input.grad is not None
    assert bool(torch.isfinite(grad_input.grad).all())


def test_no_grad_global_mix_stays_available_as_diagnostic(monkeypatch):
    """force_segment_mixing still forces the fallback under gradients."""
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    torch.manual_seed(1303)
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    module.checkpoint_segment_tokens = 16
    module.force_segment_mixing = True
    hidden = torch.randn(1, 33, 16, requires_grad=True)
    output, _ = module.scan(hidden)
    output.square().sum().backward()
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())


def test_braid_rate_ratios_survive_decay_saturation(monkeypatch):
    """The shared log rate is bounded once, so 1:16:64:256 holds at the floor."""
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid

    torch.manual_seed(1307)
    module = QwenFourStateHybrid.from_native(_native(monkeypatch))
    components = module.components
    hidden = torch.randn(2, 7, 16, dtype=torch.float64) * 0.3

    # Far below the floor: previously ONLY lane 0 hit 2^-24, breaking the
    # first ratio (16 -> ~8.7); the shared-rate clamp keeps all four exact.
    with torch.no_grad():
        components.native_decay_pair.fill_(-30.0)
    gamma = components.decay_gamma(hidden)
    assert float(gamma.min()) >= 2.0 ** -24
    rate = -gamma.log()
    torch.testing.assert_close(
        rate[..., 0, :] / rate[..., 1, :],
        torch.full_like(rate[..., 0, :], 16.0), rtol=1e-5, atol=1e-5,
    )
    torch.testing.assert_close(
        rate[..., 1, :] / rate[..., 3, :],
        torch.full_like(rate[..., 1, :], 16.0), rtol=1e-5, atol=1e-5,
    )

    # Above the ceiling: every lane saturates together at gamma=1 rather
    # than lane-dependently.
    with torch.no_grad():
        components.native_decay_pair.fill_(10.0)
    saturated = components.decay_gamma(hidden)
    assert torch.equal(saturated, torch.ones_like(saturated))

    # Interior values are untouched: ratios exact, gradients live.
    with torch.no_grad():
        components.native_decay_pair.zero_()
    components.native_decay_pair.requires_grad_(True)
    interior = components.decay_gamma(hidden)
    assert float(interior.max()) < 1.0 and float(interior.min()) > 2.0 ** -24
    interior.sum().backward()
    gradient = components.native_decay_pair.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0.0
    components.native_decay_pair.requires_grad_(False)
