from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch

from gdn3.kmd2_native import KMD2NativeAttn
from research.kmd2_ablation.config import CacheConfig


CACHE_PARAMETER_BASENAMES = {
    "cache_gamma_q",
    "cache_gamma_k",
    "cache_sink_logit",
    "cache_amplitude",
}


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )


def _cache_config(*, storage_dtype: str = "fp32") -> CacheConfig:
    return CacheConfig(
        width=2,
        block_size=2,
        read="rmsnorm",
        storage_dtype=storage_dtype,
    )


def _native(monkeypatch: pytest.MonkeyPatch, *, r_out: int = 4) -> KMD2NativeAttn:
    monkeypatch.setenv("GDN3_KMD2_ROUT", str(r_out))
    torch.manual_seed(5000 + r_out)
    return KMD2NativeAttn(_model_config(), layer_idx=7)


def _inherited_named_parameters(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if name.rsplit(".", 1)[-1] not in CACHE_PARAMETER_BASENAMES
    }


@pytest.mark.parametrize("r_out", [1, 4])
def test_exact_cache_subclass_deep_clones_every_inherited_member_without_ambient_rout(
    monkeypatch: pytest.MonkeyPatch,
    r_out: int,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    native = _native(monkeypatch, r_out=r_out)
    native.register_buffer("transfer_probe", torch.arange(3, dtype=torch.float64))
    native.rot_proj.bias.requires_grad_(False)
    native.eval()
    native.conv1d.train()
    native.transfer_metadata = {"nested": ["preserve", r_out]}

    # from_native must use the installed layer, never the process-wide setting.
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4" if r_out == 1 else "1")
    exact = KMD2ExactCacheAttn.from_native(
        native,
        model_config=_model_config(),
        cache_config=_cache_config(),
    )

    assert issubclass(KMD2ExactCacheAttn, KMD2NativeAttn)
    assert KMD2ExactCacheAttn.forward is KMD2NativeAttn.forward
    assert exact.r_out == native.r_out == r_out
    assert exact.layer_idx == native.layer_idx == 7
    assert exact.training is native.training is False
    assert exact.conv1d.training is native.conv1d.training is True
    assert exact.transfer_metadata == native.transfer_metadata
    assert exact.transfer_metadata is not native.transfer_metadata

    native_parameters = dict(native.named_parameters())
    exact_inherited = _inherited_named_parameters(exact)
    assert tuple(exact_inherited) == tuple(native_parameters)
    for name, source in native_parameters.items():
        target = exact_inherited[name]
        torch.testing.assert_close(target, source, rtol=0.0, atol=0.0)
        assert target.data_ptr() != source.data_ptr()
        assert target.device == source.device
        assert target.dtype == source.dtype
        assert target.requires_grad is source.requires_grad

    native_buffers = dict(native.named_buffers())
    exact_buffers = dict(exact.named_buffers())
    assert tuple(exact_buffers) == tuple(native_buffers)
    for name, source in native_buffers.items():
        target = exact_buffers[name]
        torch.testing.assert_close(target, source, rtol=0.0, atol=0.0)
        assert target.data_ptr() != source.data_ptr()
        assert target.device == source.device
        assert target.dtype == source.dtype

    new_names = set(dict(exact.named_parameters())) - set(native_parameters)
    assert new_names == CACHE_PARAMETER_BASENAMES
    assert exact.cache_gamma_q.dtype == torch.float32
    assert exact.cache_gamma_k.dtype == torch.float32
    assert exact.cache_sink_logit.dtype == torch.float32
    assert exact.cache_amplitude.dtype == torch.float32
    torch.testing.assert_close(exact.cache_gamma_q, torch.ones(4))
    torch.testing.assert_close(exact.cache_gamma_k, torch.ones(4))
    torch.testing.assert_close(exact.cache_sink_logit, torch.zeros(2))
    torch.testing.assert_close(exact.cache_amplitude, torch.zeros(2))

    with torch.no_grad():
        exact.in_proj_qkv.weight.add_(1.0)
        exact.transfer_probe.add_(1.0)
        exact.transfer_metadata["nested"].append("changed")
    assert not torch.equal(exact.in_proj_qkv.weight, native.in_proj_qkv.weight)
    assert not torch.equal(exact.transfer_probe, native.transfer_probe)
    assert native.transfer_metadata == {"nested": ["preserve", r_out]}


def test_from_native_rejects_wrong_type_and_double_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    native = _native(monkeypatch)
    exact = KMD2ExactCacheAttn.from_native(
        native,
        model_config=_model_config(),
        cache_config=_cache_config(),
    )

    with pytest.raises(TypeError, match="KMD2NativeAttn"):
        KMD2ExactCacheAttn.from_native(
            torch.nn.Linear(2, 2),
            model_config=_model_config(),
            cache_config=_cache_config(),
        )
    with pytest.raises(ValueError, match="already"):
        KMD2ExactCacheAttn.from_native(
            exact,
            model_config=_model_config(),
            cache_config=_cache_config(),
        )


@pytest.mark.parametrize("r_out", [1, 4])
def test_zero_amplitude_preserves_full_forward_and_inherited_gradients_but_opens_gate(
    monkeypatch: pytest.MonkeyPatch,
    r_out: int,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    native = _native(monkeypatch, r_out=r_out)
    exact = KMD2ExactCacheAttn.from_native(
        native,
        model_config=_model_config(),
        cache_config=_cache_config(),
    )
    native.train()
    exact.train()

    generator = torch.Generator().manual_seed(5200 + r_out)
    hidden = torch.randn(2, 5, 12, generator=generator)
    native_hidden = hidden.detach().clone().requires_grad_(True)
    exact_hidden = hidden.detach().clone().requires_grad_(True)
    native_output = native(native_hidden)
    exact_output = exact(exact_hidden)

    torch.testing.assert_close(exact_output, native_output, rtol=0.0, atol=0.0)
    probe = torch.randn(native_output.shape, generator=generator)
    native_output.backward(probe)
    exact_output.backward(probe)
    torch.testing.assert_close(
        exact_hidden.grad,
        native_hidden.grad,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    native_gradients = {
        name: parameter.grad for name, parameter in native.named_parameters()
    }
    exact_gradients = {
        name: parameter.grad for name, parameter in _inherited_named_parameters(exact).items()
    }
    assert tuple(exact_gradients) == tuple(native_gradients)
    for name, native_gradient in native_gradients.items():
        exact_gradient = exact_gradients[name]
        assert (native_gradient is None) is (exact_gradient is None), name
        if native_gradient is not None:
            torch.testing.assert_close(
                exact_gradient,
                native_gradient,
                rtol=1.0e-6,
                atol=1.0e-7,
                msg=name,
            )

    amplitude_gradient = exact.cache_amplitude.grad
    assert amplitude_gradient is not None
    assert bool(torch.isfinite(amplitude_gradient).all())
    assert float(amplitude_gradient.abs().sum()) > 0.0


def test_from_native_does_not_mutate_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    native = _native(monkeypatch, r_out=4)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "sentinel")
    before = copy.copy(os.environ.get("GDN3_KMD2_ROUT"))

    KMD2ExactCacheAttn.from_native(
        native,
        model_config=_model_config(),
        cache_config=_cache_config(),
    )

    assert os.environ.get("GDN3_KMD2_ROUT") == before


def _scan_inputs(
    *,
    r_out: int,
    seed: int,
    device: torch.device | str = "cpu",
    key_dim: int = 4,
    value_dim: int = 3,
    steps: int = 5,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    batch, heads = 2, 2
    q = (
        torch.randn(
            batch,
            steps,
            heads,
            r_out,
            key_dim,
            generator=generator,
            device=device,
        )
        * 0.2
    ).requires_grad_(True)
    raw_k = torch.randn(
        batch,
        steps,
        heads,
        key_dim,
        generator=generator,
        device=device,
    )
    k = torch.nn.functional.normalize(raw_k, dim=-1).requires_grad_(True)
    v = (
        torch.randn(
            batch,
            steps,
            heads,
            value_dim,
            generator=generator,
            device=device,
        )
        * 0.3
    ).requires_grad_(True)
    g = (
        0.82
        + 0.16
        * torch.rand(
            batch,
            steps,
            heads,
            key_dim,
            generator=generator,
            device=device,
        )
    ).requires_grad_(True)
    beta_e = (
        0.1
        + 0.5
        * torch.rand(
            batch,
            steps,
            heads,
            generator=generator,
            device=device,
        )
    ).requires_grad_(True)
    beta_w = (
        0.2
        + 0.5
        * torch.rand(
            batch,
            steps,
            heads,
            generator=generator,
            device=device,
        )
    ).requires_grad_(True)
    return q, k, v, g, beta_e, beta_w


def _clone_leaves(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)


def _oracle_scan_with_cache(
    inputs: tuple[torch.Tensor, ...],
    *,
    out_mix: torch.Tensor | None,
    cache_config: CacheConfig,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    sink_logit: torch.Tensor,
    amplitude: torch.Tensor,
):
    from research.kmd2_ablation.exact_cache import (
        cache_read_blocks,
        merge_persistent_cache,
        reference_scan_with_scores,
    )

    q, k, v, g, beta_e, beta_w = inputs
    y_state, scores = reference_scan_with_scores(
        q,
        k,
        v,
        g,
        beta_e,
        beta_w,
        out_mix=out_mix,
    )
    if out_mix is None:
        q_eff = q[..., 0, :]
    else:
        q_eff = torch.einsum("bthrd,hr->bthd", q, out_mix)
    batch, steps, heads = k.shape[:3]
    positions = torch.arange(steps, device=k.device, dtype=torch.int64).view(
        1, steps
    ).expand(batch, steps)
    valid = torch.ones(batch, steps, device=k.device, dtype=torch.bool)
    storage_dtype = (
        torch.float32
        if cache_config.storage_dtype == "fp32"
        else torch.bfloat16
    )
    state = None
    cache_outputs = []
    for start in range(0, steps, cache_config.block_size):
        stop = min(steps, start + cache_config.block_size)
        block = slice(start, stop)
        cache_output, _ = cache_read_blocks(
            q_eff=q_eff[:, block],
            query_positions=positions[:, block],
            state=state,
            block_k=k[:, block],
            block_v=v[:, block],
            block_scores=scores[:, block],
            block_positions=positions[:, block],
            block_valid=valid[:, block],
            config=cache_config,
            gamma_q=gamma_q,
            gamma_k=gamma_k,
            sink_logit=sink_logit,
        )
        cache_outputs.append(cache_output)
        state = merge_persistent_cache(
            state=state,
            block_k=k[:, block],
            block_v=v[:, block],
            block_scores=scores[:, block],
            block_positions=positions[:, block],
            block_valid=valid[:, block],
            width=cache_config.width,
            storage_dtype=storage_dtype,
        )
    y_cache = torch.cat(cache_outputs, dim=1)
    assert state is not None
    return (
        y_state + amplitude.view(1, 1, heads, 1) * y_cache,
        y_state,
        y_cache,
        scores,
        state,
    )


def _all_tensor_members_detached(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.requires_grad is False and value.grad_fn is None
    if isinstance(value, tuple):
        return all(_all_tensor_members_detached(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return all(
            _all_tensor_members_detached(getattr(value, name))
            for name in value.__dataclass_fields__
        )
    return True


@pytest.mark.parametrize("r_out", [1, 4])
@pytest.mark.parametrize("storage_dtype", ["fp32", "bf16"])
def test_reference_cache_scan_matches_independent_block_oracle_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
    r_out: int,
    storage_dtype: str,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    native = _native(monkeypatch, r_out=r_out)
    cache_config = _cache_config(storage_dtype=storage_dtype)
    exact = KMD2ExactCacheAttn.from_native(native, _model_config(), cache_config)
    with torch.no_grad():
        exact.cache_gamma_q.copy_(torch.tensor([0.8, 1.1, 1.3, 0.7]))
        exact.cache_gamma_k.copy_(torch.tensor([1.2, 0.9, 0.6, 1.4]))
        exact.cache_sink_logit.copy_(torch.tensor([-0.3, 0.2]))
        exact.cache_amplitude.copy_(torch.tensor([0.35, 0.55]))
        if r_out == 4:
            exact.out_mix.copy_(
                torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
                )
            )
            native.out_mix.copy_(exact.out_mix)

    initial = _scan_inputs(r_out=r_out, seed=5400 + r_out)
    actual_inputs = _clone_leaves(initial)
    expected_inputs = _clone_leaves(initial)
    expected_gamma_q = exact.cache_gamma_q.detach().clone().requires_grad_(True)
    expected_gamma_k = exact.cache_gamma_k.detach().clone().requires_grad_(True)
    expected_sink = exact.cache_sink_logit.detach().clone().requires_grad_(True)
    expected_amplitude = exact.cache_amplitude.detach().clone().requires_grad_(True)
    expected_out_mix = (
        None
        if r_out == 1
        else exact.out_mix.detach().clone().requires_grad_(True)
    )

    actual = exact._scan(*actual_inputs)
    expected, expected_state_output, expected_cache_output, expected_scores, state = (
        _oracle_scan_with_cache(
            expected_inputs,
            out_mix=expected_out_mix,
            cache_config=cache_config,
            gamma_q=expected_gamma_q,
            gamma_k=expected_gamma_k,
            sink_logit=expected_sink,
            amplitude=expected_amplitude,
        )
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
    native_state = native._scan(*_clone_leaves(initial))
    torch.testing.assert_close(
        expected_state_output, native_state, rtol=1.0e-5, atol=1.0e-6
    )
    assert float(expected_cache_output.detach().abs().sum()) > 0.0

    diagnostics = exact.last_cache_diagnostics
    assert diagnostics is not None
    assert len(diagnostics.blocks) == 3
    assert _all_tensor_members_detached(diagnostics)
    torch.testing.assert_close(diagnostics.update_scores, expected_scores)
    torch.testing.assert_close(diagnostics.final_selected_positions, state.positions)
    torch.testing.assert_close(diagnostics.final_selected_scores, state.scores)
    torch.testing.assert_close(diagnostics.final_selected_valid, state.valid)
    assert diagnostics.persistent_bytes == state.nbytes
    assert not hasattr(exact, "cache_state")
    assert not hasattr(exact, "persistent_cache")
    assert not hasattr(exact, "_cache_state")

    generator = torch.Generator().manual_seed(5500 + r_out)
    probe = torch.randn(actual.shape, generator=generator)
    actual_targets = [
        *actual_inputs,
        exact.cache_gamma_q,
        exact.cache_gamma_k,
        exact.cache_sink_logit,
        exact.cache_amplitude,
    ]
    expected_targets = [
        *expected_inputs,
        expected_gamma_q,
        expected_gamma_k,
        expected_sink,
        expected_amplitude,
    ]
    if r_out == 4:
        actual_targets.append(exact.out_mix)
        assert expected_out_mix is not None
        expected_targets.append(expected_out_mix)
    actual_gradients = torch.autograd.grad(actual, actual_targets, probe)
    expected_gradients = torch.autograd.grad(expected, expected_targets, probe)
    assert len(actual_gradients) == len(expected_gradients)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        assert bool(torch.isfinite(actual_gradient).all())
        assert float(actual_gradient.abs().sum()) > 0.0


def test_cache_state_is_discarded_between_full_recompute_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    native = _native(monkeypatch, r_out=1)
    exact = KMD2ExactCacheAttn.from_native(
        native, _model_config(), _cache_config(storage_dtype="bf16")
    )
    exact.cache_amplitude.data.fill_(0.5)
    inputs = _scan_inputs(r_out=1, seed=5600)

    first = exact._scan(*_clone_leaves(inputs))
    second = exact._scan(*_clone_leaves(inputs))

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not any(
        name in vars(exact)
        for name in ("cache_state", "persistent_cache", "_cache_state")
    )


def _diagnostic_tensor_elements(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, tuple):
        return sum(_diagnostic_tensor_elements(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return sum(
            _diagnostic_tensor_elements(getattr(value, name))
            for name in value.__dataclass_fields__
        )
    return 0


def test_published_qwen_diagnostics_are_compact_across_many_cache_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    native = _native(monkeypatch, r_out=1)
    cache_config = CacheConfig(
        width=8,
        block_size=8,
        read="rmsnorm",
        storage_dtype="fp32",
    )
    exact = KMD2ExactCacheAttn.from_native(native, _model_config(), cache_config)
    inputs = _scan_inputs(r_out=1, seed=5650, steps=65)
    exact._scan(*inputs)
    diagnostics = exact.last_cache_diagnostics
    assert diagnostics is not None
    assert len(diagnostics.blocks) == 9

    batch, steps, heads = 2, 65, 2
    final_width = cache_config.width
    expected_elements = batch * steps * heads  # update_scores
    expected_elements += 3 * batch * heads * final_width
    for block_index, block in enumerate(diagnostics.blocks):
        start = block_index * cache_config.block_size
        stop = min(steps, start + cache_config.block_size)
        assert (block.block_start, block.block_stop) == (start, stop)
        assert not hasattr(block, "attention_weights")
        assert not hasattr(block, "candidate_valid")
        assert not hasattr(block, "hit_ready_positions")
        expected_elements += block.persistent_selected_positions.numel()
        expected_elements += 4 * batch * (stop - start) * heads
    assert _diagnostic_tensor_elements(diagnostics) == expected_elements

    # The retired full candidate expansions scaled with every query times W+C.
    old_candidate_lower_bound = sum(
        3
        * batch
        * (min(steps, start + cache_config.block_size) - start)
        * heads
        * (cache_config.width + cache_config.block_size)
        for start in range(0, steps, cache_config.block_size)
    )
    assert expected_elements < old_candidate_lower_bound // 2


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64, torch.float32])
def test_full_recompute_guard_accepts_only_exact_dense_call_contract(
    mask_dtype: torch.dtype,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        validate_full_recompute_call,
    )

    input_ids = torch.arange(10).view(2, 5)
    positions = torch.arange(5).view(1, 5).expand(2, 5)
    validate_full_recompute_call(
        input_ids=input_ids,
        attention_mask=torch.ones(2, 5, dtype=mask_dtype),
        position_ids=positions,
        use_cache=False,
        cache_params=None,
        past_key_values=(),
        cache_position=torch.empty(0, dtype=torch.int64),
        cu_seqlens=[],
        segment_ids={},
        reset_mask=None,
    )
    validate_full_recompute_call(inputs_embeds=torch.zeros(2, 5, 12))


@pytest.mark.parametrize(
    ("mask", "code"),
    [
        (torch.ones(2, 5, 1), "attention_mask_shape"),
        (torch.ones(5), "attention_mask_shape"),
        (torch.ones(2, 4), "attention_mask_shape"),
        (torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]), "padding_unsupported"),
        (torch.full((2, 5), 1.0 + 1.0e-7), "padding_unsupported"),
    ],
)
def test_full_recompute_guard_rejects_malformed_or_padding_masks(
    mask: torch.Tensor,
    code: str,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        FullRecomputeCallError,
        validate_full_recompute_call,
    )

    with pytest.raises(FullRecomputeCallError) as caught:
        validate_full_recompute_call(
            input_ids=torch.zeros(2, 5, dtype=torch.int64),
            attention_mask=mask,
        )
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("positions", "code"),
    [
        ([1, 2, 3, 4, 5], "position_offset"),
        ([0, 1, 3, 4, 5], "position_gap"),
        ([0, 1, 1, 2, 3], "position_duplicate"),
        ([0, 0, 1, 2, 3], "position_duplicate"),
        ([0, 2, 1, 3, 4], "position_decreasing"),
        ([0, 1, 2, 0, 1], "position_reset"),
    ],
)
def test_full_recompute_guard_rejects_each_invalid_position_semantic(
    positions: list[int],
    code: str,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        FullRecomputeCallError,
        validate_full_recompute_call,
    )

    position_ids = torch.tensor(positions).view(1, 5).expand(2, 5)
    with pytest.raises(FullRecomputeCallError) as caught:
        validate_full_recompute_call(
            input_ids=torch.zeros(2, 5, dtype=torch.int64),
            position_ids=position_ids,
        )
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"use_cache": True}, "use_cache_unsupported"),
        ({"use_cache": 0}, "use_cache_malformed"),
        ({"cache_params": {"state": 1}}, "cross_call_cache_unsupported"),
        ({"cache_state": torch.ones(1)}, "cross_call_cache_unsupported"),
        ({"past_key_values": (torch.ones(1),)}, "incremental_decode_unsupported"),
        ({"past_key_value": torch.ones(1)}, "incremental_decode_unsupported"),
        ({"cache_position": torch.tensor([0])}, "cache_position_unsupported"),
        ({"cu_seqlens": torch.tensor([0, 5])}, "packing_unsupported"),
        ({"segment_ids": torch.zeros(1, 5)}, "segments_unsupported"),
        ({"reset_mask": torch.zeros(1, 5)}, "reset_unsupported"),
        ({"packing": True}, "packing_unsupported"),
        ({"decode": True}, "incremental_decode_unsupported"),
    ],
)
def test_full_recompute_guard_rejects_nonempty_unsupported_fields_distinctly(
    kwargs: dict[str, object],
    code: str,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        FullRecomputeCallError,
        validate_full_recompute_call,
    )

    with pytest.raises(FullRecomputeCallError) as caught:
        validate_full_recompute_call(
            input_ids=torch.zeros(1, 5, dtype=torch.int64), **kwargs
        )
    assert caught.value.code == code


def test_guarded_forward_validates_before_model_and_forces_cache_off() -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        FullRecomputeCallError,
        guarded_model_forward,
    )

    class SpyModel:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "called"

    model = SpyModel()
    input_ids = torch.zeros(2, 5, dtype=torch.int64)
    result = guarded_model_forward(
        model,
        input_ids,
        attention_mask=torch.ones(2, 5),
        position_ids=torch.arange(5).view(1, 5).expand(2, 5),
    )
    assert result == "called"
    assert len(model.calls) == 1
    assert model.calls[0][1]["use_cache"] is False

    with pytest.raises(FullRecomputeCallError) as caught:
        guarded_model_forward(
            model,
            input_ids,
            attention_mask=torch.tensor(
                [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
            ),
        )
    assert caught.value.code == "padding_unsupported"
    assert len(model.calls) == 1


def test_full_recompute_guard_rejects_unvalidated_extra_positional_arguments() -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        FullRecomputeCallError,
        validate_full_recompute_call,
    )

    with pytest.raises(FullRecomputeCallError) as caught:
        validate_full_recompute_call(
            torch.zeros(2, 5, dtype=torch.int64),
            torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]),
        )
    assert caught.value.code == "positional_arguments_unsupported"


class _FakeQwenBlock(torch.nn.Module):
    def __init__(self, linear_attn: torch.nn.Module) -> None:
        super().__init__()
        self.linear_attn = linear_attn


class _FakeQwenBackbone(torch.nn.Module):
    def __init__(self, layers: list[_FakeQwenBlock]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class _FakeQwenModel(torch.nn.Module):
    def __init__(self, config: object, layer_count: int = 3) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeQwenBackbone(
            [_FakeQwenBlock(torch.nn.Linear(2, 2)) for _ in range(layer_count)]
        )


class _FakeNativeManager:
    def __init__(
        self,
        model: _FakeQwenModel,
        *,
        indices: tuple[int, ...] = (0, 2),
        install_native: bool = True,
        raise_on_apply: bool = False,
    ) -> None:
        self.model = model
        self.indices = indices
        self.install_native = install_native
        self.raise_on_apply = raise_on_apply
        self.upgraded_layers: list[int] = []
        self.events: list[str] = []
        self.post_apply_tensors: dict[str, torch.Tensor] = {}

    def apply_upgrade(self) -> list[int]:
        self.events.append("apply")
        assert os.environ.get("GDN3_KMD2_NATIVE") == "1"
        if self.raise_on_apply:
            raise RuntimeError("manager failure")
        if self.install_native:
            for index in self.indices:
                if 0 <= index < len(self.model.model.layers):
                    self.model.model.layers[index].linear_attn = KMD2NativeAttn(
                        self.model.config,
                        layer_idx=index,
                    )
        self.upgraded_layers = list(self.indices)
        self.post_apply_tensors = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        return list(self.indices)


def _native_checkpoint_key(index: int, name: str) -> str:
    return f"model.layers.{index}.linear_attn.{name}"


def test_installer_orders_native_checkpoint_conversion_and_optional_resume_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.kmd2_ablation.qwen_exact_cache as qwen_cache

    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    monkeypatch.setenv("GDN3_KMD2_NATIVE", "sentinel")
    model = _FakeQwenModel(_model_config())
    preserved = model.model.layers[1].linear_attn
    manager = _FakeNativeManager(model)
    checkpoint = {
        _native_checkpoint_key(0, "in_proj_qkv.weight"): torch.full(
            (22, 12), 0.125
        ),
        _native_checkpoint_key(2, "bw_off"): torch.tensor([0.2, -0.1]),
    }
    resume_token = {"opaque": "resume"}

    original_from_native = qwen_cache.KMD2ExactCacheAttn.from_native.__func__

    def recording_from_native(cls, native, model_config, cache_config):
        manager.events.append(f"convert:{native.layer_idx}")
        if native.layer_idx == 0:
            torch.testing.assert_close(
                native.in_proj_qkv.weight,
                checkpoint[_native_checkpoint_key(0, "in_proj_qkv.weight")],
            )
        return original_from_native(cls, native, model_config, cache_config)

    monkeypatch.setattr(
        qwen_cache.KMD2ExactCacheAttn,
        "from_native",
        classmethod(recording_from_native),
    )

    def recording_resume(model_arg, checkpoint_arg, expected_job_id, optimizer=None):
        manager.events.append("resume")
        assert model_arg is model
        assert checkpoint_arg is resume_token
        assert expected_job_id == "job-123"
        assert optimizer is None
        assert all(
            isinstance(model.model.layers[index].linear_attn, qwen_cache.KMD2ExactCacheAttn)
            for index in (0, 2)
        )

    monkeypatch.setattr(qwen_cache, "strict_load_cache_resume", recording_resume)

    upgraded = qwen_cache.load_native_then_install(
        model=model,
        manager=manager,
        model_config=_model_config(),
        cache_config=_cache_config(),
        native_checkpoint=checkpoint,
        cache_resume=resume_token,
        expected_job_id="job-123",
    )

    assert upgraded == (0, 2)
    assert manager.events == ["apply", "convert:0", "convert:2", "resume"]
    assert os.environ.get("GDN3_KMD2_NATIVE") == "sentinel"
    assert model.model.layers[1].linear_attn is preserved
    for index in upgraded:
        assert isinstance(
            model.model.layers[index].linear_attn,
            qwen_cache.KMD2ExactCacheAttn,
        )
    torch.testing.assert_close(
        model.model.layers[0].linear_attn.in_proj_qkv.weight,
        checkpoint[_native_checkpoint_key(0, "in_proj_qkv.weight")],
    )
    torch.testing.assert_close(
        model.model.layers[2].linear_attn.bw_off,
        checkpoint[_native_checkpoint_key(2, "bw_off")],
    )


@pytest.mark.parametrize(
    "bad_checkpoint",
    [
        {
            _native_checkpoint_key(0, "in_proj_qkv.weight"): torch.full(
                (22, 12), 0.125
            ),
            _native_checkpoint_key(2, "bw_off"): torch.zeros(3),
        },
        {_native_checkpoint_key(0, "not_a_parameter"): torch.zeros(1)},
        {_native_checkpoint_key(1, "weight"): torch.zeros(2, 2)},
    ],
)
def test_installer_prevalidates_every_checkpoint_tensor_before_any_copy_or_replacement(
    monkeypatch: pytest.MonkeyPatch,
    bad_checkpoint: dict[str, torch.Tensor],
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import load_native_then_install

    monkeypatch.setenv("GDN3_KMD2_NATIVE", "before-failure")
    model = _FakeQwenModel(_model_config())
    manager = _FakeNativeManager(model)

    with pytest.raises((KeyError, ValueError), match="checkpoint"):
        load_native_then_install(
            model,
            manager,
            _model_config(),
            _cache_config(),
            bad_checkpoint,
        )

    assert os.environ.get("GDN3_KMD2_NATIVE") == "before-failure"
    for name, expected in manager.post_apply_tensors.items():
        torch.testing.assert_close(model.state_dict()[name], expected)
    assert all(
        type(model.model.layers[index].linear_attn) is KMD2NativeAttn
        for index in (0, 2)
    )


@pytest.mark.parametrize(
    ("indices", "install_native", "error_match"),
    [
        ((), True, "at least one"),
        ((0, 0), True, "duplicate"),
        ((0, 3), True, "range"),
        ((0, 2), False, "KMD2NativeAttn"),
    ],
)
def test_installer_rejects_untrustworthy_manager_indices_and_actual_layer_types(
    monkeypatch: pytest.MonkeyPatch,
    indices: tuple[int, ...],
    install_native: bool,
    error_match: str,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import load_native_then_install

    monkeypatch.delenv("GDN3_KMD2_NATIVE", raising=False)
    model = _FakeQwenModel(_model_config())
    manager = _FakeNativeManager(
        model,
        indices=indices,
        install_native=install_native,
    )

    with pytest.raises((TypeError, ValueError), match=error_match):
        load_native_then_install(
            model,
            manager,
            _model_config(),
            _cache_config(),
            None,
        )

    assert "GDN3_KMD2_NATIVE" not in os.environ


def test_installer_restores_native_mode_environment_when_manager_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import load_native_then_install

    monkeypatch.setenv("GDN3_KMD2_NATIVE", "restore-me")
    model = _FakeQwenModel(_model_config())
    manager = _FakeNativeManager(model, raise_on_apply=True)

    with pytest.raises(RuntimeError, match="manager failure"):
        load_native_then_install(
            model,
            manager,
            _model_config(),
            _cache_config(),
            None,
        )
    assert os.environ.get("GDN3_KMD2_NATIVE") == "restore-me"


def _exact_cache_model(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeQwenModel:
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    model = _FakeQwenModel(_model_config())
    for index in (0, 2):
        native = KMD2NativeAttn(_model_config(), layer_idx=index)
        model.model.layers[index].linear_attn = KMD2ExactCacheAttn.from_native(
            native,
            _model_config(),
            _cache_config(),
        )
    return model


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert tuple(actual) == tuple(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def test_cache_optimizer_has_stable_named_group_shared_schedule_and_projection_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        build_cache_optimizer,
        named_cache_parameters,
    )

    model = _exact_cache_model(monkeypatch)
    scheduler = object()
    scheduler_spec = {"name": "cosine", "warmup_steps": 7}
    cache_config = _cache_config()
    named = named_cache_parameters(model)
    optimizer = build_cache_optimizer(
        model,
        cache_config,
        betas=(0.8, 0.95),
        eps=1.0e-7,
        scheduler=scheduler,
        scheduler_spec=scheduler_spec,
    )

    assert tuple(name for name, _ in named) == tuple(
        sorted(name for name, _ in named)
    )
    assert len(named) == 8
    assert [parameter for _, parameter in named] == optimizer.param_groups[0]["params"]
    assert optimizer.param_groups[0]["lr"] == cache_config.lr_cache
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1.0e-7
    assert optimizer._kmd2_shared_scheduler is scheduler
    assert optimizer._kmd2_scheduler_spec == scheduler_spec
    assert optimizer._kmd2_scheduler_spec is not scheduler_spec

    amplitudes = [
        parameter
        for name, parameter in named
        if name.endswith(".cache_amplitude")
    ]
    with torch.no_grad():
        amplitudes[0].copy_(torch.tensor([1.2, -0.2]))
        amplitudes[1].copy_(torch.tensor([-4.0, 3.0]))
    for _, parameter in named:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(amplitudes[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(amplitudes[1], torch.tensor([0.0, 1.0]))


def _initialized_cache_optimizer(model, cache_config):
    from research.kmd2_ablation.qwen_exact_cache import (
        build_cache_optimizer,
        named_cache_parameters,
    )

    optimizer = build_cache_optimizer(
        model,
        cache_config,
        betas=(0.85, 0.97),
        eps=1.0e-8,
        scheduler=object(),
        scheduler_spec={"name": "linear", "total_steps": 11},
    )
    with torch.no_grad():
        for name, parameter in named_cache_parameters(model):
            if name.endswith(".cache_amplitude"):
                parameter.fill_(0.35)
    loss = sum(
        parameter.square().sum()
        for _, parameter in named_cache_parameters(model)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def test_cache_resume_round_trips_exact_ordered_model_and_optimizer_state_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        CACHE_RESUME_SCHEMA_VERSION,
        build_cache_resume,
        named_cache_parameters,
        save_cache_resume,
        strict_load_cache_resume,
    )

    cache_config = _cache_config()
    model = _exact_cache_model(monkeypatch)
    optimizer = _initialized_cache_optimizer(model, cache_config)
    envelope = build_cache_resume(model, optimizer, job_id="resume-job")
    expected_names = [name for name, _ in named_cache_parameters(model)]

    assert envelope["schema_version"] == CACHE_RESUME_SCHEMA_VERSION
    assert envelope["job_id"] == "resume-job"
    assert envelope["cache_parameter_names"] == expected_names
    assert len(envelope["cache_tensors"]) == len(expected_names)
    assert envelope["optimizer_parameter_names"] == [expected_names]
    assert envelope["scheduler_spec"] == {
        "name": "linear",
        "total_steps": 11,
    }

    with torch.no_grad():
        for _, parameter in named_cache_parameters(model):
            parameter.add_(0.2)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.zero_()

    strict_load_cache_resume(
        model,
        copy.deepcopy(envelope),
        expected_job_id="resume-job",
        optimizer=optimizer,
    )
    for (_, actual), expected in zip(
        named_cache_parameters(model), envelope["cache_tensors"]
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)
    _assert_nested_equal(optimizer.state_dict(), envelope["optimizer_state"])

    path = tmp_path / "cache-resume.pt"
    save_cache_resume(path, model, optimizer, job_id="resume-job")
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []

    fresh_model = _exact_cache_model(monkeypatch)
    fresh_optimizer = _initialized_cache_optimizer(fresh_model, cache_config)
    strict_load_cache_resume(
        fresh_model,
        path,
        expected_job_id="resume-job",
        optimizer=fresh_optimizer,
    )
    for (_, actual), expected in zip(
        named_cache_parameters(fresh_model), envelope["cache_tensors"]
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)
    _assert_nested_equal(fresh_optimizer.state_dict(), envelope["optimizer_state"])


def _mutate_resume_envelope(envelope: dict[str, object], case: str) -> None:
    if case == "schema":
        envelope["schema_version"] = 999
    elif case == "job":
        envelope["job_id"] = "wrong-job"
    elif case == "missing_root":
        envelope.pop("cache_tensors")
    elif case == "unexpected_root":
        envelope["surprise"] = True
    elif case == "missing_name":
        envelope["cache_parameter_names"].pop()
        envelope["cache_tensors"].pop()
    elif case == "reordered_names":
        envelope["cache_parameter_names"][0], envelope["cache_parameter_names"][1] = (
            envelope["cache_parameter_names"][1],
            envelope["cache_parameter_names"][0],
        )
    elif case == "shape":
        envelope["cache_tensors"][0] = torch.zeros(1)
    elif case == "dtype":
        envelope["cache_tensors"][0] = envelope["cache_tensors"][0].double()
    elif case == "nonfinite":
        envelope["cache_tensors"][0].view(-1)[0] = torch.nan
    elif case == "amplitude_range":
        index = next(
            index
            for index, name in enumerate(envelope["cache_parameter_names"])
            if name.endswith(".cache_amplitude")
        )
        envelope["cache_tensors"][index].view(-1)[0] = 1.01
    elif case == "optimizer_order":
        names = envelope["optimizer_parameter_names"][0]
        names[0], names[1] = names[1], names[0]
    elif case == "optimizer_nonfinite":
        state = envelope["optimizer_state"]["state"]
        first = next(iter(state.values()))
        first["exp_avg"].view(-1)[0] = torch.inf
    elif case == "scheduler_spec":
        envelope["scheduler_spec"] = {"name": "different"}
    else:  # pragma: no cover - test table is exhaustive
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "job",
        "missing_root",
        "unexpected_root",
        "missing_name",
        "reordered_names",
        "shape",
        "dtype",
        "nonfinite",
        "amplitude_range",
        "optimizer_order",
        "optimizer_nonfinite",
        "scheduler_spec",
    ],
)
def test_failed_strict_resume_rejects_corruption_without_model_or_optimizer_mutation(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import (
        build_cache_resume,
        named_cache_parameters,
        strict_load_cache_resume,
    )

    model = _exact_cache_model(monkeypatch)
    optimizer = _initialized_cache_optimizer(model, _cache_config())
    envelope = build_cache_resume(model, optimizer, job_id="expected-job")
    _mutate_resume_envelope(envelope, case)
    parameter_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in named_cache_parameters(model)
    }
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

    with pytest.raises((KeyError, TypeError, ValueError), match="resume"):
        strict_load_cache_resume(
            model,
            envelope,
            expected_job_id="expected-job",
            optimizer=optimizer,
        )

    for name, parameter in named_cache_parameters(model):
        torch.testing.assert_close(
            parameter, parameter_snapshot[name], rtol=0.0, atol=0.0
        )
    _assert_nested_equal(optimizer.state_dict(), optimizer_snapshot)


def _relative_mse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).square().mean()
    denominator = expected.float().square().mean().clamp_min(1.0e-12)
    return float((numerator / denominator).detach().cpu())


@pytest.mark.cuda
def test_true_fast_qwen_cache_scan_matches_reference_across_scan_and_cache_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the true KMD-2 fast-cache integration")

    from gdn3 import kmd2_fast_scan, kmd2_native
    from research.kmd2_ablation.exact_cache import reference_scan_with_scores
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn

    model_config = SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=2,
        rms_norm_eps=1.0e-6,
    )
    cache_config = CacheConfig(
        width=3,
        block_size=5,
        read="rmsnorm",
        storage_dtype="bf16",
    )
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    native = KMD2NativeAttn(model_config, layer_idx=0).cuda()
    exact = KMD2ExactCacheAttn.from_native(
        native,
        model_config,
        cache_config,
    )
    with torch.no_grad():
        exact.out_mix.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4]], device="cuda"))
        exact.cache_gamma_q.copy_(
            torch.tensor(
                [0.7, 0.9, 1.1, 1.3, 1.2, 0.8, 1.4, 0.6],
                device="cuda",
            )
        )
        exact.cache_gamma_k.copy_(
            torch.tensor(
                [1.2, 1.0, 0.8, 0.6, 0.7, 1.3, 0.9, 1.1],
                device="cuda",
            )
        )
        exact.cache_sink_logit.fill_(-0.2)
        exact.cache_amplitude.fill_(0.4)

    steps = kmd2_fast_scan.C + 1
    generator = torch.Generator(device="cuda").manual_seed(5900)
    q = (
        torch.randn(1, steps, 1, 4, 8, generator=generator, device="cuda")
        * 0.15
    )
    k = torch.nn.functional.normalize(
        torch.randn(1, steps, 1, 8, generator=generator, device="cuda"),
        dim=-1,
    )
    v = (
        torch.randn(1, steps, 1, 64, generator=generator, device="cuda")
        * 0.12
    )
    # Well-separated write magnitudes make the selected-index gate meaningful.
    v[:, 1] *= 8.0
    v[:, 7] *= 6.0
    v[:, 13] *= 4.0
    g = 0.97 + 0.02 * torch.rand(
        1, steps, 1, 8, generator=generator, device="cuda"
    )
    beta_e = 0.1 + 0.25 * torch.rand(
        1, steps, 1, generator=generator, device="cuda"
    )
    beta_w = 0.3 + 0.25 * torch.rand(
        1, steps, 1, generator=generator, device="cuda"
    )
    initial = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (q, k, v, g, beta_e, beta_w)
    )

    state_actual_inputs = _clone_leaves(initial)
    state_expected_inputs = _clone_leaves(initial)
    state_expected_out_mix = exact.out_mix.detach().clone().requires_grad_(True)

    cache_actual_inputs = _clone_leaves(initial)
    cache_expected_inputs = _clone_leaves(initial)
    cache_expected_out_mix = exact.out_mix.detach().clone().requires_grad_(True)
    cache_expected_gamma_q = (
        exact.cache_gamma_q.detach().clone().requires_grad_(True)
    )
    cache_expected_gamma_k = (
        exact.cache_gamma_k.detach().clone().requires_grad_(True)
    )
    cache_expected_sink = (
        exact.cache_sink_logit.detach().clone().requires_grad_(True)
    )
    cache_expected_amplitude = (
        exact.cache_amplitude.detach().clone().requires_grad_(True)
    )

    actual_inputs = _clone_leaves(initial)
    expected_inputs = _clone_leaves(initial)
    expected_out_mix = exact.out_mix.detach().clone().requires_grad_(True)
    expected_gamma_q = exact.cache_gamma_q.detach().clone().requires_grad_(True)
    expected_gamma_k = exact.cache_gamma_k.detach().clone().requires_grad_(True)
    expected_sink = exact.cache_sink_logit.detach().clone().requires_grad_(True)
    expected_amplitude = exact.cache_amplitude.detach().clone().requires_grad_(True)

    monkeypatch.setattr(kmd2_native, "_FAST_SCAN", True)
    actual_state_output, actual_scores = exact._native_state_and_scores(
        *state_actual_inputs
    )
    expected_state_output, expected_state_scores = reference_scan_with_scores(
        *state_expected_inputs,
        out_mix=state_expected_out_mix,
    )

    cache_actual_state, _ = exact._native_state_and_scores(*cache_actual_inputs)
    cache_actual_combined = exact._scan(*cache_actual_inputs)
    actual_cache_output = (cache_actual_combined - cache_actual_state) / (
        exact.cache_amplitude.detach().view(1, 1, 1, 1)
    )
    (
        _,
        _,
        expected_cache_output,
        _,
        _,
    ) = _oracle_scan_with_cache(
        cache_expected_inputs,
        out_mix=cache_expected_out_mix,
        cache_config=cache_config,
        gamma_q=cache_expected_gamma_q,
        gamma_k=cache_expected_gamma_k,
        sink_logit=cache_expected_sink,
        amplitude=cache_expected_amplitude,
    )

    actual = exact._scan(*actual_inputs)
    (
        expected,
        _,
        _,
        expected_scores,
        expected_state,
    ) = _oracle_scan_with_cache(
        expected_inputs,
        out_mix=expected_out_mix,
        cache_config=cache_config,
        gamma_q=expected_gamma_q,
        gamma_k=expected_gamma_k,
        sink_logit=expected_sink,
        amplitude=expected_amplitude,
    )

    assert _relative_mse(actual_state_output, expected_state_output) < 2.0e-3
    assert _relative_mse(actual_scores, expected_state_scores) < 2.0e-3
    assert _relative_mse(actual_cache_output, expected_cache_output) < 2.0e-3
    assert _relative_mse(actual, expected) < 2.0e-3
    diagnostics = exact.last_cache_diagnostics
    assert diagnostics is not None
    assert len(diagnostics.blocks) == 4
    assert _relative_mse(diagnostics.update_scores, expected_scores) < 2.0e-3
    torch.testing.assert_close(
        diagnostics.final_selected_positions,
        expected_state.positions,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        diagnostics.final_selected_valid,
        expected_state.valid,
        rtol=0.0,
        atol=0.0,
    )

    state_probe = torch.randn(actual.shape, generator=generator, device="cuda")
    state_actual_targets = [*state_actual_inputs, exact.out_mix]
    state_expected_targets = [*state_expected_inputs, state_expected_out_mix]
    actual_state_gradients = torch.autograd.grad(
        actual_state_output,
        state_actual_targets,
        state_probe,
    )
    expected_state_gradients = torch.autograd.grad(
        expected_state_output,
        state_expected_targets,
        state_probe,
    )
    for actual_gradient, expected_gradient in zip(
        actual_state_gradients, expected_state_gradients
    ):
        assert _relative_mse(actual_gradient, expected_gradient) < 1.0e-2

    cache_probe = torch.randn(actual.shape, generator=generator, device="cuda")
    cache_actual_targets = [
        *cache_actual_inputs[:3],
        exact.cache_gamma_q,
        exact.cache_gamma_k,
        exact.cache_sink_logit,
        exact.out_mix,
    ]
    cache_expected_targets = [
        *cache_expected_inputs[:3],
        cache_expected_gamma_q,
        cache_expected_gamma_k,
        cache_expected_sink,
        cache_expected_out_mix,
    ]
    actual_cache_gradients = torch.autograd.grad(
        actual_cache_output,
        cache_actual_targets,
        cache_probe,
    )
    expected_cache_gradients = torch.autograd.grad(
        expected_cache_output,
        cache_expected_targets,
        cache_probe,
    )
    for actual_gradient, expected_gradient in zip(
        actual_cache_gradients, expected_cache_gradients
    ):
        assert _relative_mse(actual_gradient, expected_gradient) < 1.0e-2

    probe = torch.randn(actual.shape, generator=generator, device="cuda")
    actual_targets = [
        *actual_inputs,
        exact.cache_gamma_q,
        exact.cache_gamma_k,
        exact.cache_sink_logit,
        exact.cache_amplitude,
        exact.out_mix,
    ]
    expected_targets = [
        *expected_inputs,
        expected_gamma_q,
        expected_gamma_k,
        expected_sink,
        expected_amplitude,
        expected_out_mix,
    ]
    actual_gradients = torch.autograd.grad(actual, actual_targets, probe)
    expected_gradients = torch.autograd.grad(expected, expected_targets, probe)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        assert bool(torch.isfinite(actual_gradient).all())
        assert _relative_mse(actual_gradient, expected_gradient) < 1.0e-2
