from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch


def _explicit_true_mimo_scan(q, k, v, g, beta_e, beta_w, z, out_mix, initial_state):
    """Independent literal equation oracle; intentionally does not use production helpers."""
    rank = q.shape[3]
    state = initial_state
    outputs = []
    for token in range(q.shape[1]):
        state_bar = g[:, token].unsqueeze(-1) * state
        memory = torch.einsum("bhrd,bhdv->bhrv", k[:, token], state_bar)
        erase = torch.einsum(
            "bhrd,bhrv->bhdv", k[:, token],
            (beta_e[:, token] / rank).unsqueeze(-1) * memory,
        )
        write = torch.einsum(
            "bhrd,bhrv->bhdv", k[:, token],
            beta_w[:, token].unsqueeze(-1) * v[:, token],
        )
        state = state_bar - erase + write
        read = torch.einsum("bhrd,bhdv->bhrv", q[:, token], state)
        outputs.append(
            (out_mix[:, token] * (read * torch.nn.functional.silu(z[:, token]))).sum(dim=2)
        )
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("rank", [2, 4])
def test_true_mimo_scan_matches_independent_fp64_forward_and_all_gradients(rank):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan

    torch.manual_seed(700 + rank)
    B, T, H, dk, dv = 2, 3, 2, 3, 4
    shapes = (
        (B, T, H, rank, dk), (B, T, H, rank, dk),
        (B, T, H, rank, dv), (B, T, H, dk),
        (B, T, H, rank), (B, T, H, rank),
        (B, T, H, rank, dv), (B, T, H, rank, dv),
        (B, H, dk, dv),
    )
    operands = [torch.randn(s, dtype=torch.float64, requires_grad=True) for s in shapes]
    # Positive gates keep this representative of the public scan contract.
    operands[4] = torch.rand(shapes[4], dtype=torch.float64, requires_grad=True)
    operands[5] = torch.rand(shapes[5], dtype=torch.float64, requires_grad=True)
    actual = true_mimo_sequence_scan(*operands[:-1], initial_state=operands[-1])
    expected = _explicit_true_mimo_scan(*operands)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), operands, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), operands)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("rank", [2, 4])
def test_true_mimo_scan_is_invariant_to_common_rank_permutation(rank):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    torch.manual_seed(800 + rank)
    B, T, H, dk, dv = 1, 3, 2, 3, 4
    q = torch.randn(B, T, H, rank, dk)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, rank, dv)
    g = torch.rand(B, T, H, dk)
    be, bw = torch.rand(B, T, H, rank), torch.rand(B, T, H, rank)
    z, mix = torch.randn_like(v), torch.randn_like(v)
    state = torch.randn(B, H, dk, dv)
    baseline = true_mimo_sequence_scan(q, k, v, g, be, bw, z, mix, state)
    p = torch.randperm(rank)
    permuted = true_mimo_sequence_scan(q[:, :, :, p], k[:, :, :, p], v[:, :, :, p], g,
                                       be[:, :, :, p], bw[:, :, :, p], z[:, :, :, p],
                                       mix[:, :, :, p], state)
    torch.testing.assert_close(permuted, baseline)


def _valid_mimo_scan_operands(*, rank=2, dtype=torch.float32, device="cpu"):
    B, T, H, dk, dv = 1, 2, 2, 3, 4
    return [
        torch.randn(B, T, H, rank, dk, dtype=dtype, device=device),
        torch.randn(B, T, H, rank, dk, dtype=dtype, device=device),
        torch.randn(B, T, H, rank, dv, dtype=dtype, device=device),
        torch.rand(B, T, H, dk, dtype=dtype, device=device),
        torch.rand(B, T, H, rank, dtype=dtype, device=device),
        torch.rand(B, T, H, rank, dtype=dtype, device=device),
        torch.randn(B, T, H, rank, dv, dtype=dtype, device=device),
        torch.randn(B, T, H, rank, dv, dtype=dtype, device=device),
        torch.randn(B, H, dk, dv, dtype=dtype, device=device),
    ]


@pytest.mark.parametrize("index", range(9))
def test_true_mimo_scan_rejects_every_malformed_operand(index):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands()
    args[index] = args[index][..., :-1]
    with pytest.raises(ValueError, match="shape"):
        true_mimo_sequence_scan(*args[:-1], initial_state=args[-1])


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_true_mimo_scan_rejects_unsupported_rank(rank):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands(rank=rank)
    with pytest.raises(ValueError, match="R.*2.*4"):
        true_mimo_sequence_scan(*args[:-1], initial_state=args[-1])


@pytest.mark.parametrize("index", range(9))
def test_true_mimo_scan_rejects_nonfinite_every_operand(index):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands()
    args[index].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        true_mimo_sequence_scan(*args[:-1], initial_state=args[-1])


def test_true_mimo_scan_rejects_nontensor_nonfloating_and_dtype_mismatch():
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands()
    bad = list(args); bad[2] = object()
    with pytest.raises(TypeError, match="tensor"):
        true_mimo_sequence_scan(*bad[:-1], initial_state=bad[-1])


@pytest.mark.parametrize("index", [4, 5])
def test_true_mimo_scan_rejects_negative_scalar_gates(index):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands()
    args[index].reshape(-1)[0] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        true_mimo_sequence_scan(*args[:-1], initial_state=args[-1])


def test_true_mimo_scan_performs_only_one_python_scalar_synchronization(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands()
    original = torch.Tensor.__bool__
    calls = 0
    def counted(tensor):
        nonlocal calls
        calls += 1
        return original(tensor)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted)
    true_mimo_sequence_scan(*args[:-1], initial_state=args[-1])
    assert calls == 1


@pytest.mark.parametrize("tokens", [2, 7])
def test_true_mimo_scan_casts_each_full_operand_once_not_per_token(monkeypatch, tokens):
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands(dtype=torch.float16)
    args[:8] = [tensor[:, :1].expand(-1, tokens, *([-1] * (tensor.ndim - 2))).clone()
                for tensor in args[:8]]
    original = torch.Tensor.to
    calls = 0
    def counted(tensor, *to_args, **to_kwargs):
        nonlocal calls
        calls += 1
        return original(tensor, *to_args, **to_kwargs)
    monkeypatch.setattr(torch.Tensor, "to", counted)
    true_mimo_sequence_scan(*args[:-1], initial_state=args[-1])
    assert calls == 9
    bad = list(args); bad[2] = bad[2].to(torch.int64)
    with pytest.raises(TypeError, match="floating"):
        true_mimo_sequence_scan(*bad[:-1], initial_state=bad[-1])
    bad = list(args); bad[2] = bad[2].double()
    with pytest.raises(ValueError, match="dtype"):
        true_mimo_sequence_scan(*bad[:-1], initial_state=bad[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cross-device operand")
def test_true_mimo_scan_rejects_device_mismatch():
    from research.kmd2_ablation.qwen_architecture import true_mimo_sequence_scan
    args = _valid_mimo_scan_operands()
    args[2] = args[2].cuda()
    with pytest.raises(ValueError, match="device"):
        true_mimo_sequence_scan(*args[:-1], initial_state=args[-1])


def _canonical_native(monkeypatch, *, dtype=torch.float32):
    from gdn3.kmd2_native import KMD2NativeAttn
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    config = SimpleNamespace(
        hidden_size=16, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6,
    )
    return KMD2NativeAttn(config, layer_idx=3).to(dtype=dtype)


def _manual_rout_4_module(module, x, attention_mask=None):
    """Literal shared-state widening oracle; no production forward/scan helper."""
    F = torch.nn.functional
    B, T, _ = x.shape
    H, dk, dv = module.H, module.dk, module.dv
    mixed = F.silu(module.conv1d(module.in_proj_qkv(x).transpose(1, 2))[:, :, :T]).transpose(1, 2)
    query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], -1)
    q = F.normalize(query.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6) * dk ** -.5
    k = F.normalize(key.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6)
    v = value.reshape(B, T, H, dv).float()
    z = module.in_proj_z(x)
    b, a = module.in_proj_b(x).float(), module.in_proj_a(x).float()
    be, bw = torch.sigmoid(b), torch.sigmoid(b + module.bw_off.float())
    g = (-module.A_log.float().exp() * F.softplus(a + module.dt_bias.float()))
    g = (g.unsqueeze(-1) + module.decay_chan.float()).exp().clamp(max=1.0)
    theta = F.softplus(module.rot_proj(x)).reshape(B, T, H, dk // 2).float().cumsum(1)
    cos, sin = theta.cos(), theta.sin()
    def rotate(tensor):
        first, second = tensor[..., :dk // 2], tensor[..., dk // 2:]
        return torch.cat((first * cos.unsqueeze(-2) - second * sin.unsqueeze(-2),
                          first * sin.unsqueeze(-2) + second * cos.unsqueeze(-2)), -1)
    q = rotate(q.unsqueeze(3) * (1 + module.q_slot_scale.float())[None, None])
    k_first, k_second = k[..., :dk // 2], k[..., dk // 2:]
    k = torch.cat((k_first * cos - k_second * sin, k_first * sin + k_second * cos), -1)
    state = torch.zeros(B, H, dk, dv)
    outputs = []
    for token in range(T):
        state = state * g[:, token].unsqueeze(-1)
        memory = torch.einsum("bhd,bhdv->bhv", k[:, token], state)
        state = state - torch.einsum("bhd,bhv->bhdv", k[:, token], be[:, token, :, None] * memory)
        state = state + torch.einsum("bhd,bhv->bhdv", k[:, token], bw[:, token, :, None] * v[:, token])
        reads = torch.einsum("bhrd,bhdv->bhrv", q[:, token], state)
        outputs.append(torch.einsum("hr,bhrv->bhv", module.out_mix.float(), reads))
    y = torch.stack(outputs, 1)
    y = module.norm(y.reshape(-1, dv).to(z.dtype), z.reshape(-1, dv)).reshape(B, T, module.value_dim)
    out = module.out_proj(y)
    if attention_mask is not None:
        out = out * (attention_mask.unsqueeze(-1) if attention_mask.ndim == 2 else attention_mask)
    return out


def test_rout_4_exact_init_full_oracle_gradients_state_and_active_effect(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2SharedQueryWideningAttn
    torch.manual_seed(804)
    native = _canonical_native(monkeypatch)
    module = KMD2SharedQueryWideningAttn.from_native(native, 4)
    assert module.q_slot_scale.shape == (native.H, 4, native.dk)
    assert module.out_mix.shape == (native.H, 4)
    assert torch.count_nonzero(module.q_slot_scale) == 0
    torch.testing.assert_close(module.out_mix[:, 0], torch.ones(native.H))
    assert torch.count_nonzero(module.out_mix[:, 1:]) == 0
    assert not any(token in name for name in module.state_dict() for token in ("mimo", "rankwise", "mimo_k", "mimo_v", "mimo_z"))
    inherited = native.state_dict()
    assert module.transformation_manifest() == {"copied": tuple(inherited), "transformed": (), "new": ("q_slot_scale", "out_mix")}
    assert module.architecture_classification == "control"
    assert module.promotable is False
    assert module.identity_at_initialization is True
    assert module.recurrent_state_bytes(3) == 3 * native.H * native.dk * native.dv * 4

    x1 = torch.randn(2, 4, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 0., 1., 0.]])
    actual, expected = module(x1, attention_mask=mask), _manual_rout_4_module(module, x2, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, [x1, *module.parameters()], probe, retain_graph=True)
    oracle_grads = torch.autograd.grad(expected, [x2, *module.parameters()], probe)
    for name, left, right in zip(["hidden", *dict(module.named_parameters())], actual_grads, oracle_grads):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4, msg=lambda m: f"{name}: {m}")
    named_grads = dict(zip(dict(module.named_parameters()), actual_grads[1:]))
    assert torch.count_nonzero(named_grads["q_slot_scale"][:, 1:]) == 0
    assert torch.count_nonzero(named_grads["q_slot_scale"][:, 0]) > 0
    assert torch.count_nonzero(named_grads["out_mix"]) > 0
    native_x = x1.detach().clone().requires_grad_(True)
    native_out = native(native_x, attention_mask=mask)
    torch.testing.assert_close(actual, native_out, atol=0, rtol=0)
    native_grads = torch.autograd.grad(native_out, [native_x, *native.parameters()], probe)
    actual_by_name = dict(zip(dict(module.named_parameters()), actual_grads[1:]))
    native_by_name = dict(zip(dict(native.named_parameters()), native_grads[1:]))
    torch.testing.assert_close(actual_grads[0], native_grads[0], atol=0, rtol=0)
    for name in native_by_name:
        torch.testing.assert_close(actual_by_name[name], native_by_name[name], atol=0, rtol=0)
    baseline = module(x1.detach())
    with torch.no_grad():
        module.q_slot_scale[:, 2].add_(.2)
        module.out_mix[:, 2].fill_(.3)
    assert not torch.allclose(module(x1.detach()), baseline)


def test_rout_4_builder_is_strict_and_fails_closed_to_reference(monkeypatch):
    from gdn3 import kmd2_native
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import KMD2SharedQueryWideningAttn, QwenArchitectureConfig, build_qwen_architecture
    config = QwenArchitectureConfig("rout-4", registry_sha256(), architecture_record("rout-4"))
    monkeypatch.setattr(kmd2_native, "_FAST_SCAN", True)
    built = build_qwen_architecture(_canonical_native(monkeypatch), config)
    assert type(built) is KMD2SharedQueryWideningAttn and built.width == 4
    assert built.implementation_reference == "qwen_architecture.KMD2SharedQueryWideningAttn.reference_fp32"
    assert built.implementation_path == "reference_fp32_fast_scan_fail_closed"
    with pytest.raises(ValueError, match="width.*4"):
        KMD2SharedQueryWideningAttn.from_native(_canonical_native(monkeypatch), 3)


def test_rout_4_conversion_declares_exact_width_metadata(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2SharedQueryWideningAttn
    module = KMD2SharedQueryWideningAttn.from_native(_canonical_native(monkeypatch))
    assert type(module.output_width) is int and module.output_width == 4
    with pytest.raises(AttributeError):
        module.output_width = 3
    assert type(module.r_out) is int and module.r_out == 4
    assert "output_width" not in module.state_dict()
    assert module.transformation_manifest()["new"] == ("q_slot_scale", "out_mix")


@pytest.mark.parametrize("kwargs", [
    {"use_cache": True},
    {"past_key_values": (torch.zeros(1),)},
    {"past_key_value": (torch.zeros(1),)},
    {"cache_position": torch.tensor([0])},
])
def test_rout_4_forward_rejects_populated_incremental_cache_kwargs(monkeypatch, kwargs):
    from research.kmd2_ablation.qwen_architecture import KMD2SharedQueryWideningAttn
    module = KMD2SharedQueryWideningAttn.from_native(_canonical_native(monkeypatch))
    with pytest.raises(ValueError, match="shared_query_widening_cache_unsupported"):
        module(torch.randn(1, 2, 16), **kwargs)


def test_rout_4_forward_allows_empty_cache_kwargs(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2SharedQueryWideningAttn
    module = KMD2SharedQueryWideningAttn.from_native(_canonical_native(monkeypatch))
    result = module(torch.randn(1, 2, 16), use_cache=False, past_key_values=None,
                    past_key_value=None, cache_position=None)
    assert result.shape == (1, 2, 16)


def _manual_true_mimo_module(module, x, attention_mask=None):
    """Independent whole-module oracle: no module forward or scan helper."""
    B, T, _ = x.shape
    H, dk, dv, rank = module.H, module.dk, module.dv, module.rank
    mixed = torch.nn.functional.silu(
        module.conv1d(module.in_proj_qkv(x).transpose(1, 2))[:, :, :T]
    ).transpose(1, 2)
    q0, k0, v0 = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], -1)
    q0, k0 = q0.reshape(B, T, H, dk), k0.reshape(B, T, H, dk)
    q = torch.einsum("bthd,hrde->bthre", q0.float(), module.mimo_q_transform.float())
    k = torch.einsum("bthd,hrde->bthre", k0.float(), module.mimo_k_transform.float())
    q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6) * dk ** -0.5
    k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
    theta = torch.nn.functional.softplus(module.rot_proj(x)).view(B, T, H, dk // 2).float().cumsum(1)
    cos, sin = theta.cos().unsqueeze(3), theta.sin().unsqueeze(3)
    def rotate(tensor):
        a, b = tensor[..., :dk // 2], tensor[..., dk // 2:]
        return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)
    q, k = rotate(q), rotate(k)
    v = v0.reshape(B, T, H, dv).float().unsqueeze(3) * module.mimo_v[None, None]
    z = module.in_proj_z(x).reshape(B, T, H, dv).float().unsqueeze(3) * module.mimo_z[None, None]
    b = module.in_proj_b(x).float()
    be = torch.sigmoid(b).unsqueeze(-1).expand(-1, -1, -1, rank)
    bw = torch.sigmoid(b + module.bw_off).unsqueeze(-1).expand_as(be)
    a = module.in_proj_a(x).float()
    g = (-module.A_log.float().exp() * torch.nn.functional.softplus(a + module.dt_bias.float()))
    g = (g.unsqueeze(-1) + module.decay_chan).exp().clamp(max=1.0)
    state = torch.zeros(B, H, dk, dv)
    ys = []
    for token in range(T):
        decayed = g[:, token].unsqueeze(-1) * state
        memory = torch.einsum("bhrd,bhdv->bhrv", k[:, token], decayed)
        state = decayed - torch.einsum("bhrd,bhrv->bhdv", k[:, token], be[:, token].unsqueeze(-1) * memory / rank)
        state = state + torch.einsum("bhrd,bhrv->bhdv", k[:, token], bw[:, token].unsqueeze(-1) * v[:, token])
        read = torch.einsum("bhrd,bhdv->bhrv", q[:, token], state)
        ys.append((module.mimo_out[None] * read * torch.nn.functional.silu(z[:, token])).sum(2))
    y = torch.stack(ys, 1)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + module.norm.variance_epsilon)
    y = y * module.norm.weight.float()
    out = module.out_proj(y.reshape(B, T, module.value_dim).to(x.dtype))
    if attention_mask is not None:
        out = out * (attention_mask.unsqueeze(-1) if attention_mask.ndim == 2 else attention_mask)
    return out


@pytest.mark.parametrize("rank", [2, 4])
def test_true_mimo_module_exact_factors_oracle_gradients_and_invariants(monkeypatch, rank):
    from research.kmd2_ablation.qwen_architecture import KMD2TrueMIMOAttn
    torch.manual_seed(810 + rank)
    native = _canonical_native(monkeypatch)
    module = KMD2TrueMIMOAttn.from_native(native, rank)
    eye = torch.eye(native.dk).expand(native.H, rank, -1, -1)
    torch.testing.assert_close(module.mimo_q_transform, eye)
    torch.testing.assert_close(module.mimo_k_transform, eye)
    assert module.mimo_v.shape == module.mimo_z.shape == module.mimo_out.shape == (native.H, rank, native.dv)
    torch.testing.assert_close(module.mimo_v, torch.full_like(module.mimo_v, 1 / rank))
    torch.testing.assert_close(module.mimo_z, torch.ones_like(module.mimo_z))
    torch.testing.assert_close(module.mimo_out, torch.full_like(module.mimo_out, 1 / rank))
    assert not hasattr(module, "q_slot_scale") and not hasattr(module, "out_mix")
    assert module.conv1d.padding_mode == native.conv1d.padding_mode
    inherited = native.state_dict()
    for name, value in inherited.items(): torch.testing.assert_close(module.state_dict()[name], value)
    x1 = torch.randn(2, 4, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 0., 1., 0.]])
    actual, expected = module(x1, attention_mask=mask), _manual_true_mimo_module(module, x2, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    targets = [x1, *module.parameters()]
    actual_grads = torch.autograd.grad(actual, targets, probe, retain_graph=True)
    oracle_targets = [x2, *module.parameters()]
    oracle_grads = torch.autograd.grad(expected, oracle_targets, probe)
    for name, left, right in zip(["hidden", *dict(module.named_parameters())], actual_grads, oracle_grads):
        assert left is not None and torch.isfinite(left).all(), name
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4, msg=lambda m: f"{name}: {m}")
    assert torch.count_nonzero(actual[mask == 0]) == 0
    baseline = module(x1.detach())
    with torch.no_grad(): module.mimo_q_transform[0, 0].add_(torch.randn_like(module.mimo_q_transform[0, 0]) * .2)
    assert not torch.allclose(module(x1.detach()), baseline)
    permutation = torch.arange(rank - 1, -1, -1)
    before = module(x1.detach())
    with torch.no_grad():
        for name in ("mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out"):
            parameter = getattr(module, name); parameter.copy_(parameter[:, permutation])
    torch.testing.assert_close(module(x1.detach()), before, atol=2e-6, rtol=2e-5)
    manifest = module.transformation_manifest()
    assert manifest == {"copied": tuple(inherited), "transformed": (), "new": ("mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out")}
    assert module.architecture_classification == "cold_redesign"
    assert module.identity_at_initialization is False
    assert module.implementation_reference == "qwen_architecture.KMD2TrueMIMOAttn.reference_fp32"


@pytest.mark.parametrize("rank", [1, 3])
def test_true_mimo_module_rejects_non_genuine_rank(monkeypatch, rank):
    from research.kmd2_ablation.qwen_architecture import KMD2TrueMIMOAttn
    with pytest.raises(ValueError, match="rank.*2 or 4"):
        KMD2TrueMIMOAttn.from_native(_canonical_native(monkeypatch), rank)


def test_true_mimo_module_rejects_widening_gdn2_cache_and_fast_scan(monkeypatch):
    from gdn3 import kmd2_native
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn, KMD2TrueMIMOAttn
    native = _canonical_native(monkeypatch)
    widened = _canonical_native(monkeypatch)
    widened.r_out = 4
    with pytest.raises(TypeError, match="canonical R1"):
        KMD2TrueMIMOAttn.from_native(widened, 2)
    with pytest.raises(TypeError, match="canonical R1"):
        KMD2TrueMIMOAttn.from_native(KMD2ChannelwiseGDN2Attn.from_native(native), 2)
    module = KMD2TrueMIMOAttn.from_native(native, 2)
    with pytest.raises(ValueError, match="cache"):
        module(torch.randn(1, 2, 16), use_cache=True)
    monkeypatch.setattr(kmd2_native, "_FAST_SCAN", True)
    with pytest.raises(ValueError, match="fast scan"):
        KMD2TrueMIMOAttn.from_native(native, 2)


def test_true_mimo_conversion_rejects_non_current_rotation_before_clone(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2TrueMIMOAttn
    native = _canonical_native(monkeypatch)
    original_deepcopy = __import__("copy").deepcopy
    calls = []
    monkeypatch.setattr("research.kmd2_ablation.qwen_architecture.copy.deepcopy", lambda value: calls.append(value) or original_deepcopy(value))
    with pytest.raises(ValueError, match="true_mimo_rotation_mode_unsupported"):
        KMD2TrueMIMOAttn.from_native(native, 2, rotation_mode="off")
    assert calls == []


@pytest.mark.parametrize("rank", [2, 4])
def test_true_mimo_bf16_qk_transforms_match_explicit_fp32_oracle_and_gradients(monkeypatch, rank):
    from research.kmd2_ablation.qwen_architecture import KMD2TrueMIMOAttn
    torch.manual_seed(920 + rank)
    module = KMD2TrueMIMOAttn.from_native(_canonical_native(monkeypatch, dtype=torch.bfloat16), rank)
    with torch.no_grad():
        module.mimo_q_transform.add_(torch.randn_like(module.mimo_q_transform) * 0.07)
        module.mimo_k_transform.add_(torch.randn_like(module.mimo_k_transform) * 0.07)
    x_actual = torch.randn(1, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    x_oracle = x_actual.detach().clone().requires_grad_(True)
    original_einsum = torch.einsum
    transform_dtypes = []
    def observed_einsum(equation, *operands):
        if equation == "bthd,hrde->bthre": transform_dtypes.append(tuple(item.dtype for item in operands))
        return original_einsum(equation, *operands)
    monkeypatch.setattr(torch, "einsum", observed_einsum)
    actual = module(x_actual)
    monkeypatch.setattr(torch, "einsum", original_einsum)
    assert transform_dtypes == [(torch.float32, torch.float32)] * 2
    expected = _manual_true_mimo_module(module, x_oracle)
    torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)
    probe = torch.randn_like(actual)
    names = ("mimo_q_transform", "mimo_k_transform", "in_proj_qkv.weight", "conv1d.weight")
    parameters = dict(module.named_parameters())
    actual_grads = torch.autograd.grad(actual, [x_actual, *(parameters[name] for name in names)], probe, retain_graph=True)
    oracle_grads = torch.autograd.grad(expected, [x_oracle, *(parameters[name] for name in names)], probe)
    for name, left, right in zip(("hidden", *names), actual_grads, oracle_grads):
        assert torch.isfinite(left).all(), name
        torch.testing.assert_close(left, right, atol=3.2e-2, rtol=3.2e-2, msg=lambda m: f"{name}: {m}")
    assert module.implementation_reference.endswith("reference_fp32")


def test_gdn2_conversion_has_exact_row_copy_and_manifest(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn
    native = _canonical_native(monkeypatch)
    with torch.no_grad():
        native.in_proj_b.weight.copy_(torch.arange(32).reshape(2, 16))
        native.bw_off.copy_(torch.tensor([-0.5, 0.75]))
    converted = KMD2ChannelwiseGDN2Attn.from_native(native)
    assert converted.erase_proj.weight.shape == (8, 16)
    assert converted.write_proj.weight.shape == (8, 16)
    for h in range(2):
        assert torch.equal(converted.erase_proj.weight[h*4:(h+1)*4], native.in_proj_b.weight[h].expand(4, -1))
        assert torch.equal(converted.write_proj.weight[h*4:(h+1)*4], native.in_proj_b.weight[h].expand(4, -1))
    assert torch.equal(converted.write_offset, native.bw_off)
    native_state = native.state_dict()
    converted_state = converted.state_dict()
    assert tuple(name for name in converted_state if name in native_state) == tuple(native_state)
    for name, value in native_state.items():
        assert torch.equal(converted_state[name], value), name
    assert torch.equal(converted.conv1d.weight, native.conv1d.weight)
    assert converted.training is native.training
    assert converted.transformation_manifest() == {
        "copied": tuple(native_state),
        "transformed": (
            ("in_proj_b.weight", "erase_proj.weight", "row_copy_dk"),
            ("in_proj_b.weight", "write_proj.weight", "row_copy_dv"),
            ("bw_off", "write_offset", "copy"),
        ),
        "new": (),
    }


def test_gdn2_full_forward_and_chain_rule_match_native_at_init(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn
    torch.manual_seed(19)
    native = _canonical_native(monkeypatch)
    converted = KMD2ChannelwiseGDN2Attn.from_native(native)
    x1 = torch.randn(2, 5, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = native(x1), converted(x2)
    torch.testing.assert_close(y2, y1, atol=1e-6, rtol=1e-5)
    y1.sum().backward(); y2.sum().backward()
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-6, rtol=1e-5)
    native_parameters = dict(native.named_parameters())
    converted_parameters = dict(converted.named_parameters())
    for name in native_parameters.keys() - {"in_proj_b.weight", "bw_off"}:
        left, right = native_parameters[name].grad, converted_parameters[name].grad
        assert left is not None and right is not None, name
        torch.testing.assert_close(right, left, atol=1e-6, rtol=1e-5, msg=lambda msg: f"{name}: {msg}")
    for h in range(native.H):
        transformed = (converted.erase_proj.weight.grad[h*native.dk:(h+1)*native.dk].sum(0)
                       + converted.write_proj.weight.grad[h*native.dv:(h+1)*native.dv].sum(0))
        torch.testing.assert_close(transformed, native.in_proj_b.weight.grad[h], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(converted.write_offset.grad, native.bw_off.grad, atol=1e-6, rtol=1e-5)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in native.named_parameters())
    for name, parameter in converted.named_parameters():
        if name in {"in_proj_b.weight", "bw_off"}:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 parity requires CUDA support")
def test_gdn2_bf16_initialization_output_parity(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn
    native = _canonical_native(monkeypatch, dtype=torch.bfloat16).cuda()
    converted = KMD2ChannelwiseGDN2Attn.from_native(native)
    x = torch.randn(1, 5, 16, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(converted(x), native(x), atol=2e-2, rtol=2e-2)


def test_gdn2_erase_and_write_gates_are_independent(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn
    module = KMD2ChannelwiseGDN2Attn.from_native(_canonical_native(monkeypatch))
    x = torch.randn(1, 2, 16)
    erase0, write0 = module._factor_logits(x)
    with torch.no_grad(): module.erase_proj.weight[0, 0].add_(1)
    erase1, write1 = module._factor_logits(x)
    assert not torch.equal(erase0, erase1)
    assert torch.equal(write0, write1)
    with torch.no_grad():
        module.in_proj_b.weight.normal_(); module.bw_off.normal_()
    erase2, write2 = module._factor_logits(x)
    assert torch.equal(erase1, erase2)
    assert torch.equal(write1, write2)


def test_gdn2_fp32_recurrence_matches_independent_oracle_forward_and_gradients(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn
    module = KMD2ChannelwiseGDN2Attn.from_native(_canonical_native(monkeypatch))
    B, T, H, dk, dv = 1, 4, module.H, module.dk, module.dv
    inputs = [torch.randn(B, T, H, 1, dk), torch.randn(B, T, H, dk),
              torch.randn(B, T, H, dv), torch.rand(B, T, H, dk),
              torch.rand(B, T, H, dk), torch.rand(B, T, H, dv)]
    q, k, v, g, erase, write = [tensor.requires_grad_() for tensor in inputs]
    actual = module._scan_channelwise(q, k, v, g, erase, write)
    oracle_inputs = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v, g, erase, write)]
    oq, ok, ov, og, oe, ow = oracle_inputs
    state = torch.zeros(B, H, dk, dv)
    expected = []
    for t in range(T):
        decayed = og[:, t].unsqueeze(-1) * state
        memory = torch.einsum("bhd,bhdv->bhv", oe[:, t] * ok[:, t], decayed)
        state = decayed + torch.einsum("bhd,bhv->bhdv", ok[:, t], ow[:, t] * ov[:, t] - memory)
        expected.append(torch.einsum("bhd,bhdv->bhv", oq[:, t, :, 0], state))
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward(); expected.square().sum().backward()
    for operand, oracle in zip((q, k, v, g, erase, write), oracle_inputs):
        torch.testing.assert_close(operand.grad, oracle.grad, atol=2e-5, rtol=2e-5)


def test_channelwise_gdn2_public_update_matches_independent_fp64_sequence_oracle():
    from research.kmd2_ablation.architecture import channelwise_gdn2_update

    torch.manual_seed(37)
    B, T, H, dk, dv = 2, 4, 2, 3, 5
    inputs = (
        torch.randn(B, T, H, dk, dtype=torch.float64),
        torch.randn(B, T, H, dk, dtype=torch.float64),
        torch.randn(B, T, H, dv, dtype=torch.float64),
        torch.rand(B, T, H, dk, dtype=torch.float64),
        torch.rand(B, T, H, dk, dtype=torch.float64),
        torch.rand(B, T, H, dv, dtype=torch.float64),
    )
    q, k, v, g, erase, write = [tensor.requires_grad_() for tensor in inputs]
    oracle_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    oq, ok, ov, og, oe, ow = oracle_inputs

    state = torch.zeros(B, H, dk, dv, dtype=torch.float64)
    actual_outputs = []
    for t in range(T):
        state = channelwise_gdn2_update(
            state * g[:, t].unsqueeze(-1),
            k[:, t].unsqueeze(2), v[:, t].unsqueeze(2),
            erase[:, t].unsqueeze(2), write[:, t].unsqueeze(2),
        )
        actual_outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, t], state))
    actual = torch.stack(actual_outputs, dim=1)

    oracle_state = torch.zeros(B, H, dk, dv, dtype=torch.float64)
    expected_outputs = []
    for t in range(T):
        decayed = oracle_state * og[:, t].unsqueeze(-1)
        memory = torch.einsum("bhd,bhdv->bhv", oe[:, t] * ok[:, t], decayed)
        oracle_state = decayed + torch.einsum(
            "bhd,bhv->bhdv", ok[:, t], ow[:, t] * ov[:, t] - memory
        )
        expected_outputs.append(torch.einsum("bhd,bhdv->bhv", oq[:, t], oracle_state))
    expected = torch.stack(expected_outputs, dim=1)

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for operand, oracle in zip((q, k, v, g, erase, write), oracle_inputs):
        torch.testing.assert_close(operand.grad, oracle.grad, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize(
    "operand,shape,message",
    [
        ("q", (1, 2, 2, 4), "q_shape_invalid"),
        ("k", (1, 2, 2, 1, 4), "k_shape_invalid"),
        ("v", (1, 2, 2, 5), "v_shape_invalid"),
        ("g", (1, 2, 2, 5), "g_shape_invalid"),
        ("erase", (1, 2, 2, 5), "erase_shape_invalid"),
        ("write", (1, 2, 2, 5), "write_shape_invalid"),
    ],
)
def test_gdn2_recurrence_rejects_each_malformed_operand(monkeypatch, operand, shape, message):
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn
    module = KMD2ChannelwiseGDN2Attn.from_native(_canonical_native(monkeypatch))
    values = {
        "q": torch.randn(1, 2, 2, 1, 4), "k": torch.randn(1, 2, 2, 4),
        "v": torch.randn(1, 2, 2, 4), "g": torch.rand(1, 2, 2, 4),
        "erase": torch.rand(1, 2, 2, 4), "write": torch.rand(1, 2, 2, 4),
    }
    values[operand] = torch.randn(shape)
    with pytest.raises(ValueError, match=message):
        module._scan_channelwise(**values)


def test_gdn2_conversion_rejects_tensor_name_collisions(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import KMD2ChannelwiseGDN2Attn
    native = _canonical_native(monkeypatch)
    native.erase_proj = torch.nn.Linear(16, 8, bias=False)
    with pytest.raises(ValueError, match="collision"):
        KMD2ChannelwiseGDN2Attn.from_native(native)


def test_qwen_architecture_config_is_frozen_and_canonical():
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig

    config = QwenArchitectureConfig(
        "gdn2-channel-r1", registry_sha256(), architecture_record("gdn2-channel-r1")
    )
    with pytest.raises(FrozenInstanceError):
        config.arm_id = "stock"  # type: ignore[misc]


@pytest.mark.parametrize(("arm", "rank"), [("mimo-r2", 2), ("mimo-r4", 4)])
def test_qwen_architecture_config_and_default_builder_route_true_mimo(monkeypatch, arm, rank):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import KMD2TrueMIMOAttn, QwenArchitectureConfig, build_qwen_architecture
    config = QwenArchitectureConfig(arm, registry_sha256(), architecture_record(arm))
    built = build_qwen_architecture(_canonical_native(monkeypatch), config)
    assert type(built) is KMD2TrueMIMOAttn
    assert built.rank == rank


@pytest.mark.parametrize("arm", [
    "rot-off", "rot-constant", "rot-noncumulative", "rot-fixed-rope",
    "rot-moving-frame-oracle",
])
def test_qwen_architecture_config_and_default_builder_route_rotation_controls(monkeypatch, arm):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import (
        KMD2RotationControlAttn, QwenArchitectureConfig, build_qwen_architecture,
    )
    config = QwenArchitectureConfig(arm, registry_sha256(), architecture_record(arm))
    built = build_qwen_architecture(_canonical_native(monkeypatch), config)
    assert type(built) is KMD2RotationControlAttn
    assert built.rotation_mode == architecture_record(arm).rotation_mode
    assert built.transformation_manifest()["new"] == {
        "rot-constant": ("rotation_rate",),
        "rot-fixed-rope": ("inv_freq",),
    }.get(arm, ())
    assert "inv_freq" in dict(built.named_buffers()) if arm == "rot-fixed-rope" else True


@pytest.mark.parametrize("arm", [
    "rot-off", "rot-constant", "rot-noncumulative", "rot-fixed-rope",
    "rot-moving-frame-oracle",
])
def test_rotation_manifest_accounts_for_complete_state_inventory(monkeypatch, arm):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture
    module = build_qwen_architecture(
        _canonical_native(monkeypatch),
        QwenArchitectureConfig(arm, registry_sha256(), architecture_record(arm)),
    )
    manifest = module.transformation_manifest()
    transformed_targets = tuple(item[1] for item in manifest["transformed"])
    inventory = manifest["copied"] + transformed_targets + manifest["new"]
    assert len(inventory) == len(set(inventory))
    assert set(inventory) == set(module.state_dict())


def _manual_rotation_module(module, hidden_states, attention_mask):
    """Literal full-module oracle; intentionally avoids production rotation/scan helpers."""
    F = torch.nn.functional
    B, T, _ = hidden_states.shape
    H, dk, dv = module.H, module.dk, module.dv
    mixed = module.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed = F.silu(module.conv1d(mixed)[:, :, :T]).transpose(1, 2)
    query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], -1)
    q = F.normalize(query.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6) * dk ** -.5
    k = F.normalize(key.reshape(B, T, H, dk).float(), dim=-1, eps=1e-6)
    v = value.reshape(B, T, H, dv).float()
    z = module.in_proj_z(hidden_states)
    b, a = module.in_proj_b(hidden_states).float(), module.in_proj_a(hidden_states).float()
    beta_e = torch.sigmoid(b)
    beta_w = torch.sigmoid(b + module.bw_off.float())
    g_head = -module.A_log.float().exp() * F.softplus(a + module.dt_bias.float())
    g = (g_head.unsqueeze(-1) + module.decay_chan.float()).exp().clamp(max=1.0)
    if module.rotation_mode in {"noncumulative", "moving-frame-oracle"}:
        projected = module.rot_proj(hidden_states).reshape(B, T, H, dk // 2).float()
        delta = F.softplus(projected)
        phase = delta if module.rotation_mode == "noncumulative" else delta.cumsum(1)
    elif module.rotation_mode == "constant":
        positions = torch.arange(1, T + 1, device=q.device, dtype=q.dtype)
        phase = (positions[None, :, None, None] * module.rotation_rate.float()[None, None]).expand(B, -1, -1, -1)
    elif module.rotation_mode == "fixed-rope":
        positions = torch.arange(T, device=q.device, dtype=q.dtype)
        phase = (positions[None, :, None, None] * module.inv_freq.float()[None, None, None]).expand(B, -1, H, -1)
    else:
        phase = torch.zeros(B, T, H, dk // 2, device=q.device, dtype=q.dtype)

    state = torch.zeros(B, H, dk, dv, device=q.device, dtype=q.dtype)
    previous = torch.zeros(B, H, dk // 2, device=q.device, dtype=q.dtype)
    reads = []
    for token in range(T):
        qt, kt = q[:, token], k[:, token]
        if module.rotation_mode == "moving-frame-oracle":
            angle = previous - phase[:, token]
            first, second = state[:, :, :dk // 2], state[:, :, dk // 2:]
            cosine, sine = angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1)
            state = torch.cat((first * cosine - second * sine, first * sine + second * cosine), 2)
            previous = phase[:, token]
        else:
            cosine, sine = phase[:, token].cos(), phase[:, token].sin()
            def rotate(x):
                first, second = x[..., :dk // 2], x[..., dk // 2:]
                return torch.cat((first * cosine - second * sine, first * sine + second * cosine), -1)
            qt, kt = rotate(qt), rotate(kt)
        state = state * g[:, token].unsqueeze(-1)
        memory = torch.einsum("bhd,bhdv->bhv", kt, state)
        state = state - torch.einsum("bhd,bhv->bhdv", kt, beta_e[:, token, :, None] * memory)
        state = state + torch.einsum("bhd,bhv->bhdv", kt, beta_w[:, token, :, None] * v[:, token])
        reads.append(torch.einsum("bhd,bhdv->bhv", qt, state))
    y = torch.stack(reads, 1).reshape(-1, dv).to(z.dtype)
    gate = z.reshape(-1, dv)
    y32 = y.float()
    y = (module.norm.weight * (y32 * torch.rsqrt(y32.square().mean(-1, keepdim=True) + module.norm.variance_epsilon)).to(y.dtype))
    y = (y * F.silu(gate.float())).to(y.dtype).reshape(B, T, module.value_dim)
    output = module.out_proj(y)
    return output * attention_mask.unsqueeze(-1)


@pytest.mark.parametrize("arm", [
    "rot-off", "rot-constant", "rot-noncumulative", "rot-fixed-rope",
    "rot-moving-frame-oracle",
])
def test_rotation_full_module_matches_independent_manual_oracle_and_all_gradients(monkeypatch, arm):
    import copy
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture
    config = QwenArchitectureConfig(arm, registry_sha256(), architecture_record(arm),
                                    diagnostic_training=arm == "rot-moving-frame-oracle")
    actual = build_qwen_architecture(_canonical_native(monkeypatch).double(), config)
    oracle = copy.deepcopy(actual)
    x = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
    ox = x.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
    actual_output, oracle_output = actual(x, attention_mask=mask), _manual_rotation_module(oracle, ox, mask)
    torch.testing.assert_close(actual_output, oracle_output, atol=2e-7, rtol=2e-6)
    actual_output.square().sum().backward(); oracle_output.square().sum().backward()
    torch.testing.assert_close(x.grad, ox.grad, atol=2e-7, rtol=2e-6)
    for (name, parameter), (oracle_name, oracle_parameter) in zip(actual.named_parameters(), oracle.named_parameters()):
        assert name == oracle_name
        assert (parameter.grad is None) == (oracle_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, oracle_parameter.grad, atol=3e-7, rtol=3e-6)
    if arm in {"rot-off", "rot-constant", "rot-fixed-rope"}:
        assert actual.rot_proj.weight.grad is actual.rot_proj.bias.grad is None


@pytest.mark.parametrize("arm", [
    "rot-off", "rot-constant", "rot-noncumulative", "rot-fixed-rope",
    "rot-moving-frame-oracle",
])
def test_rotation_control_forward_has_active_effect_and_preserves_native_state(monkeypatch, arm):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture
    native = _canonical_native(monkeypatch).double()
    before = {name: value.detach().clone() for name, value in native.state_dict().items()}
    built = build_qwen_architecture(native, QwenArchitectureConfig(
        arm, registry_sha256(), architecture_record(arm),
        diagnostic_training=arm == "rot-moving-frame-oracle",
    ))
    for name, value in before.items():
        torch.testing.assert_close(built.state_dict()[name], value)
    x = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
    output = built(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    if arm in {"rot-noncumulative", "rot-moving-frame-oracle"}:
        assert built.rot_proj.weight.grad is not None
    if arm == "rot-constant":
        assert built.rotation_rate.grad is not None


def test_moving_frame_diagnostic_training_explicitly_controls_rotation_trainability(monkeypatch):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture
    record = architecture_record("rot-moving-frame-oracle")
    frozen = build_qwen_architecture(_canonical_native(monkeypatch), QwenArchitectureConfig(
        "rot-moving-frame-oracle", registry_sha256(), record, diagnostic_training=False))
    trained = build_qwen_architecture(_canonical_native(monkeypatch), QwenArchitectureConfig(
        "rot-moving-frame-oracle", registry_sha256(), record, diagnostic_training=True))
    assert not frozen.rot_proj.weight.requires_grad and not frozen.rot_proj.bias.requires_grad
    assert trained.rot_proj.weight.requires_grad and trained.rot_proj.bias.requires_grad


@pytest.mark.parametrize("arm", ["rot-off", "rot-constant", "rot-fixed-rope"])
def test_rotation_control_unread_projector_modes_never_call_rot_proj(monkeypatch, arm):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture
    module = build_qwen_architecture(_canonical_native(monkeypatch), QwenArchitectureConfig(
        arm, registry_sha256(), architecture_record(arm)))
    class RaisingProjector(torch.nn.Module):
        def forward(self, _x):
            raise AssertionError("preserved rot_proj must remain unread")
    module.rot_proj = RaisingProjector()
    output = module(torch.randn(1, 3, 16))
    assert output.shape == (1, 3, 16)


@pytest.mark.parametrize("arm", [
    "rot-off", "rot-constant", "rot-noncumulative", "rot-fixed-rope",
    "rot-moving-frame-oracle",
])
def test_rotation_module_forward_bypasses_public_validators_and_tensor_bool_sync(monkeypatch, arm):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    import research.kmd2_ablation.qwen_architecture as implementation
    module = implementation.build_qwen_architecture(
        _canonical_native(monkeypatch),
        implementation.QwenArchitectureConfig(arm, registry_sha256(), architecture_record(arm)),
    )
    def forbidden(*_args, **_kwargs):
        raise AssertionError("module hot path called a public validated helper")
    monkeypatch.setattr(implementation, "rotation_phase", forbidden)
    monkeypatch.setattr(implementation, "paired_rotate", forbidden)
    monkeypatch.setattr(implementation, "moving_frame_scan", forbidden)
    original_bool = torch.Tensor.__bool__
    bool_calls = 0
    def counted_bool(tensor):
        nonlocal bool_calls
        bool_calls += 1
        return original_bool(tensor)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    with torch.no_grad():
        output = module(torch.randn(1, 3, 16))
    assert output.shape == (1, 3, 16)
    assert bool_calls == 0


def test_architecture_transaction_restores_modules_and_all_parameter_flags(monkeypatch):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import (
        QwenArchitectureConfig,
        QwenArchitectureInstallError,
        install_qwen_architecture,
    )

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = torch.nn.Linear(2, 2)
            self.linear_attn.r_out = 1

    model = torch.nn.Module()
    model.model = SimpleNamespace(layers=[Block(), Block()])
    model.outside = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    originals = tuple(layer.linear_attn for layer in model.model.layers)
    flags = {p: p.requires_grad for p in model.parameters()}
    config = QwenArchitectureConfig(
        "gdn2-channel-r1", registry_sha256(), architecture_record("gdn2-channel-r1")
    )

    def factory(native, _config):
        return torch.nn.Linear(2, 2)

    def fail(*_args):
        raise RuntimeError("configure failed")

    with pytest.raises(QwenArchitectureInstallError) as error:
        install_qwen_architecture(
            model, (0, 1), config, factory=factory, configure_trainables=fail,
            declared_trainables=("outside",), native_type=torch.nn.Linear,
            expected_type=torch.nn.Linear, expected_indices=(0, 1),
        )
    assert error.value.code == "architecture_install_failed"
    assert tuple(layer.linear_attn for layer in model.model.layers) == originals
    assert {p: p.requires_grad for p in model.parameters()} == flags


def test_builder_rejects_wrong_concrete_class_and_mixed_dtype_replacement():
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture

    class Native(torch.nn.Linear):
        pass
    class Expected(torch.nn.Linear):
        pass
    native = Native(2, 2)
    native.r_out = 1
    config = QwenArchitectureConfig("gdn2-channel-r1", registry_sha256(), architecture_record("gdn2-channel-r1"))
    with pytest.raises(TypeError, match="exact expected architecture class"):
        build_qwen_architecture(native, config, factory=lambda *_: torch.nn.Linear(2, 2), native_type=Native, expected_type=Expected)
    def mixed(*_):
        result = Expected(2, 2)
        result.register_buffer("mixed", torch.ones(1, dtype=torch.float64))
        return result
    with pytest.raises(ValueError, match="dtype/device"):
        build_qwen_architecture(native, config, factory=mixed, native_type=Native, expected_type=Expected)


@pytest.mark.parametrize("failure_stage", ["prepare", "swap", "final"])
def test_transaction_rolls_back_at_every_stage_and_emits_exact_order(failure_stage):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, QwenArchitectureInstallError, install_qwen_architecture

    class Native(torch.nn.Linear):
        pass
    class Replacement(torch.nn.Linear):
        pass
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear_attn = Native(2, 2); self.linear_attn.r_out = 1
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([Block(), Block()]); self.outside = torch.nn.Parameter(torch.ones(1))
    model = Model(); originals = tuple(x.linear_attn for x in model.model.layers); flags = {p: p.requires_grad for p in model.parameters()}; events = []
    calls = 0
    def factory(native, _config):
        nonlocal calls; calls += 1
        if failure_stage == "prepare" and calls == 2: raise RuntimeError("second prepare")
        result = Replacement(2, 2); result.load_state_dict(native.state_dict()); return result
    def verify(_model, _indices):
        if failure_stage == "final": raise RuntimeError("final verify")
    def event(name):
        events.append(name)
        if failure_stage == "swap" and name == "configure_trainables":
            model.model.layers[1].linear_attn = originals[1]
    config = QwenArchitectureConfig("gdn2-channel-r1", registry_sha256(), architecture_record("gdn2-channel-r1"))
    with pytest.raises(QwenArchitectureInstallError):
        install_qwen_architecture(model, (0, 1), config, factory=factory, expected_type=Replacement, expected_indices=(0, 1), native_type=Native, configure_trainables=lambda *_: None, verify_conversion=verify, event=event)
    assert tuple(x.linear_attn for x in model.model.layers) == originals
    assert {p: p.requires_grad for p in model.parameters()} == flags
    if failure_stage == "final":
        assert events == ["prepare_replacements", "swap_replacements", "configure_trainables", "verify_conversion"]


def test_second_swap_verification_failure_restores_first_swap_and_flags():
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, QwenArchitectureInstallError, install_qwen_architecture
    class Native(torch.nn.Linear): pass
    class Replacement(torch.nn.Linear): pass
    class Block(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear_attn = Native(2, 2); self.linear_attn.r_out = 1
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.layers_owner = torch.nn.Module(); self.layers_owner.layers = torch.nn.ModuleList([Block(), Block()]); self.model = self.layers_owner; self.outside = torch.nn.Parameter(torch.ones(1))
    model = Model(); originals = tuple(layer.linear_attn for layer in model.model.layers); flags = {p: p.requires_grad for p in model.parameters()}; verified = []
    def factory(native, _config):
        result = Replacement(2, 2); result.load_state_dict(native.state_dict()); return result
    def swap_verify(_model, index, _replacement):
        verified.append(index)
        if index == 1: raise RuntimeError("second swap verification")
    config = QwenArchitectureConfig("gdn2-channel-r1", registry_sha256(), architecture_record("gdn2-channel-r1"))
    with pytest.raises(QwenArchitectureInstallError, match="second swap verification"):
        install_qwen_architecture(model, (0, 1), config, factory=factory, expected_type=Replacement, expected_indices=(0, 1), native_type=Native, swap_verifier=swap_verify)
    assert verified == [0, 1]
    assert tuple(layer.linear_attn for layer in model.model.layers) == originals
    assert {p: p.requires_grad for p in model.parameters()} == flags


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
def test_transaction_rolls_back_and_preserves_base_exception_identity(control_flow):
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, install_qwen_architecture
    class Native(torch.nn.Linear): pass
    class Replacement(torch.nn.Linear): pass
    class Block(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear_attn = Native(2, 2); self.linear_attn.r_out = 1
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([Block()]); self.outside = torch.nn.Parameter(torch.ones(1))
    model = Model(); original = model.model.layers[0].linear_attn; flags = {p: p.requires_grad for p in model.parameters()}
    def factory(native, _config): result = Replacement(2, 2); result.load_state_dict(native.state_dict()); return result
    config = QwenArchitectureConfig("gdn2-channel-r1", registry_sha256(), architecture_record("gdn2-channel-r1"))
    with pytest.raises(control_flow) as caught:
        install_qwen_architecture(model, (0,), config, factory=factory, expected_type=Replacement, expected_indices=(0,), native_type=Native, configure_trainables=lambda *_: (_ for _ in ()).throw(control_flow()))
    assert type(caught.value) is control_flow
    assert model.model.layers[0].linear_attn is original
    assert {p: p.requires_grad for p in model.parameters()} == flags


def test_transaction_fingerprint_snapshot_does_not_clone_prepared_state(monkeypatch):
    import research.kmd2_ablation.qwen_architecture as module
    calls = []
    original = module._fingerprint_state_tensor
    def fingerprint(tensor):
        calls.append(tensor)
        result = original(tensor)
        assert not isinstance(result, torch.Tensor)
        return result
    monkeypatch.setattr(module, "_fingerprint_state_tensor", fingerprint)
    # Reuse the success-shaped transaction with a large state tensor.
    class Native(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.arange(65536.0)); self.r_out = 1
    class Replacement(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.empty(65536))
    class Block(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear_attn = Native()
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([Block()])
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    config = module.QwenArchitectureConfig("gdn2-channel-r1", registry_sha256(), architecture_record("gdn2-channel-r1"))
    model = Model()
    def factory(native, _config): result = Replacement(); result.load_state_dict(native.state_dict()); return result
    module.install_qwen_architecture(model, (0,), config, factory=factory, expected_type=Replacement, expected_indices=(0,), native_type=Native)
    assert len(calls) == 2


def test_rotation_control_phase_and_paired_rotation_fp64_equations():
    from research.kmd2_ablation.qwen_architecture import paired_rotate, rotation_phase

    raw = torch.tensor([[[[0.0, 1.0]], [[2.0, -1.0]], [[0.5, 0.25]]]], dtype=torch.float64,
                       requires_grad=True)
    resets = torch.tensor([[True, False, True]])
    delta = torch.nn.functional.softplus(raw)
    expected_current = torch.stack((delta[:, 0], delta[:, 0] + delta[:, 1], delta[:, 2]), dim=1)
    current = rotation_phase(raw, "current", reset_mask=resets)
    torch.testing.assert_close(current, expected_current)
    torch.testing.assert_close(rotation_phase(raw, "moving-frame-oracle", reset_mask=resets), expected_current)
    torch.testing.assert_close(rotation_phase(raw, "noncumulative"), delta)
    torch.testing.assert_close(rotation_phase(raw, "off"), torch.zeros_like(raw))
    rate = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    expected_constant = torch.arange(1, 4, dtype=torch.float64)[None, :, None, None] * rate[None, None]
    torch.testing.assert_close(rotation_phase(raw, "constant", rotation_rate=rate), expected_constant)
    inv = torch.tensor([1.0, 0.01], dtype=torch.float64)
    expected_rope = torch.arange(3, dtype=torch.float64)[None, :, None, None] * inv[None, None, None]
    torch.testing.assert_close(rotation_phase(raw, "fixed-rope"), expected_rope)
    wide = raw.detach().expand(2, -1, 3, -1).clone()
    assert rotation_phase(wide, "fixed-rope").shape == wide.shape
    assert rotation_phase(wide, "constant", rotation_rate=rate.expand(3, -1)).shape == wide.shape
    current.sum().backward()
    multiplicity = torch.tensor([2., 1., 1.], dtype=torch.float64)[None, :, None, None]
    torch.testing.assert_close(raw.grad, torch.sigmoid(raw.detach()) * multiplicity)

    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float64, requires_grad=True)
    phi = torch.tensor([[[[0.2, -0.3]]]], dtype=torch.float64, requires_grad=True)
    a, b = x[..., :2], x[..., 2:]
    expected = torch.cat((a * phi.cos() - b * phi.sin(), a * phi.sin() + b * phi.cos()), -1)
    actual = paired_rotate(x, phi)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert x.grad is not None and phi.grad is not None


def test_rotation_control_moving_frame_matches_independent_fp64_loop_and_gradients():
    from research.kmd2_ablation.qwen_architecture import moving_frame_scan

    torch.manual_seed(911)
    B, T, H, dk, dv = 2, 4, 2, 4, 3
    values = (
        torch.randn(B, T, H, dk, dtype=torch.float64),
        torch.randn(B, T, H, dk, dtype=torch.float64),
        torch.randn(B, T, H, dv, dtype=torch.float64),
        torch.rand(B, T, H, dk, dtype=torch.float64),
        torch.rand(B, T, H, dtype=torch.float64),
        torch.rand(B, T, H, dtype=torch.float64),
        torch.randn(B, T, H, dk // 2, dtype=torch.float64),
    )
    actual_inputs = [x.requires_grad_() for x in values]
    oracle_inputs = [x.detach().clone().requires_grad_() for x in values]
    resets = torch.tensor([[True, False, False, True], [True, False, True, False]])
    actual = moving_frame_scan(*actual_inputs, reset_mask=resets)
    oq, ok, ov, og, obe, obw, ophi = oracle_inputs
    state = torch.zeros(B, H, dk, dv, dtype=torch.float64)
    previous = torch.zeros(B, H, dk // 2, dtype=torch.float64)
    outputs = []
    for t in range(T):
        reset = resets[:, t, None, None]
        state = torch.where(reset[..., None], torch.zeros_like(state), state)
        previous = torch.where(reset, torch.zeros_like(previous), previous)
        da, db = state[:, :, :dk // 2], state[:, :, dk // 2:]
        angle = previous - ophi[:, t]
        state = torch.cat((da * angle.cos()[..., None] - db * angle.sin()[..., None],
                           da * angle.sin()[..., None] + db * angle.cos()[..., None]), dim=2)
        state = state * og[:, t].unsqueeze(-1)
        memory = torch.einsum("bhd,bhdv->bhv", ok[:, t], state)
        state = state - torch.einsum("bhd,bhv->bhdv", ok[:, t], obe[:, t, :, None] * memory)
        state = state + torch.einsum("bhd,bhv->bhdv", ok[:, t], obw[:, t, :, None] * ov[:, t])
        outputs.append(torch.einsum("bhd,bhdv->bhv", oq[:, t], state))
        previous = ophi[:, t]
    expected = torch.stack(outputs, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
    actual.square().sum().backward(); expected.square().sum().backward()
    for left, right in zip(actual_inputs, oracle_inputs):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-10, rtol=2e-9)


def test_rotation_control_transport_sign_and_global_equivalence():
    from research.kmd2_ablation.qwen_architecture import moving_frame_scan, paired_rotate

    dtype = torch.float64
    q = torch.tensor([[[[1., 0.]], [[1., 0.]]]], dtype=dtype)
    k = torch.tensor([[[[1., 0.]], [[0., 1.]]]], dtype=dtype)
    v = torch.tensor([[[[2.]], [[0.]]]], dtype=dtype)
    g = torch.ones(1, 2, 1, 2, dtype=dtype)
    be = torch.zeros(1, 2, 1, dtype=dtype)
    bw = torch.ones(1, 2, 1, dtype=dtype)
    phi = torch.tensor([[[[0.2]], [[0.7]]]], dtype=dtype)
    local = moving_frame_scan(q, k, v, g, be, bw, phi)
    # Token two reads the first row after transport by 0.2 - 0.7.
    torch.testing.assert_close(local[:, 1], torch.tensor([[[2. * torch.cos(torch.tensor(-0.5, dtype=dtype))]]]))

    # With isotropic pair decay, local moving coordinates and globally rotated
    # q/k are the same recurrence expressed in two bases.
    qg, kg = paired_rotate(q, phi), paired_rotate(k, phi)
    state = torch.zeros(1, 1, 2, 1, dtype=dtype); global_out = []
    for t in range(2):
        state = state * g[:, t].unsqueeze(-1)
        memory = torch.einsum("bhd,bhdv->bhv", kg[:, t], state)
        state = state - torch.einsum("bhd,bhv->bhdv", kg[:, t], be[:, t, :, None] * memory)
        state = state + torch.einsum("bhd,bhv->bhdv", kg[:, t], bw[:, t, :, None] * v[:, t])
        global_out.append(torch.einsum("bhd,bhdv->bhv", qg[:, t], state))
    torch.testing.assert_close(local, torch.stack(global_out, 1), atol=1e-12, rtol=1e-10)


def test_rotation_control_public_helpers_reject_invalid_contracts():
    from research.kmd2_ablation.qwen_architecture import moving_frame_scan, paired_rotate, rotation_phase

    with pytest.raises(ValueError, match="even"):
        paired_rotate(torch.ones(1, 3), torch.ones(1, 1))
    with pytest.raises(ValueError, match="mode"):
        rotation_phase(torch.ones(1, 1, 1, 1), "mystery")
    args = [torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 1),
            torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1), torch.ones(1, 1, 1), torch.zeros(1, 1, 1, 1)]
    args[1][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        moving_frame_scan(*args)
    args[1] = torch.ones(1, 1, 1, 2)
    args[4][0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        moving_frame_scan(*args)


def test_rotation_control_public_validation_has_one_bool_sync_per_call(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import moving_frame_scan, paired_rotate, rotation_phase

    original = torch.Tensor.__bool__
    calls = []
    def counted(tensor):
        calls.append(tensor)
        return original(tensor)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted)
    x = torch.ones(1, 1, 1, 2)
    phi = torch.zeros(1, 1, 1, 1)
    paired_rotate(x, phi)
    assert len(calls) == 1
    calls.clear(); rotation_phase(phi, "current")
    assert len(calls) == 1
    calls.clear()
    moving_frame_scan(x, x, torch.ones(1, 1, 1, 1), x, torch.ones(1, 1, 1),
                      torch.ones(1, 1, 1), phi)
    assert len(calls) == 1


def test_rotation_control_low_precision_casts_each_full_operand_once_outside_scan(monkeypatch):
    from research.kmd2_ablation.qwen_architecture import moving_frame_scan

    original = torch.Tensor.to
    calls = []
    def counted(tensor, *args, **kwargs):
        calls.append(tensor)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", counted)
    T = 5
    args = [torch.ones(1, T, 1, 2, dtype=torch.bfloat16),
            torch.ones(1, T, 1, 2, dtype=torch.bfloat16),
            torch.ones(1, T, 1, 1, dtype=torch.bfloat16),
            torch.ones(1, T, 1, 2, dtype=torch.bfloat16),
            torch.ones(1, T, 1, dtype=torch.bfloat16),
            torch.ones(1, T, 1, dtype=torch.bfloat16),
            torch.zeros(1, T, 1, 1, dtype=torch.bfloat16)]
    moving_frame_scan(*args)
    assert len(calls) == len(args)
    assert all(actual is expected for actual, expected in zip(calls, args))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rotation_control_low_precision_matches_explicit_fp32_forward_and_gradients(dtype):
    from research.kmd2_ablation.qwen_architecture import moving_frame_scan, paired_rotate, rotation_phase

    torch.manual_seed(1201)
    raw32 = (torch.randn(1, 3, 1, 2) * .1).requires_grad_()
    raw_low = raw32.detach().to(dtype).requires_grad_()
    phase32 = rotation_phase(raw32, "current")
    phase_low = rotation_phase(raw_low, "current")
    assert phase_low.dtype == torch.float32
    torch.testing.assert_close(phase_low, phase32, atol=2e-3, rtol=2e-3)
    phase_low.sum().backward(); phase32.sum().backward()
    torch.testing.assert_close(raw_low.grad.float(), raw32.grad, atol=3e-3, rtol=3e-3)

    shapes = ((1, 3, 1, 4), (1, 3, 1, 4), (1, 3, 1, 2), (1, 3, 1, 4),
              (1, 3, 1), (1, 3, 1), (1, 3, 1, 2))
    base = [((torch.rand(s) - .5) * .3) for s in shapes]
    base[3] = torch.sigmoid(base[3]); base[4] = torch.sigmoid(base[4]); base[5] = torch.sigmoid(base[5])
    fp32 = [x.requires_grad_() for x in base]
    low = [x.detach().to(dtype).requires_grad_() for x in base]
    out32, out_low = moving_frame_scan(*fp32), moving_frame_scan(*low)
    assert out_low.dtype == torch.float32
    torch.testing.assert_close(out_low, out32, atol=4e-3, rtol=4e-3)
    out_low.sum().backward(); out32.sum().backward()
    for low_tensor, fp32_tensor in zip(low, fp32):
        torch.testing.assert_close(low_tensor.grad.float(), fp32_tensor.grad, atol=6e-3, rtol=6e-3)

    rotated = paired_rotate(low[0], low[6])
    assert rotated.dtype == torch.float32
@pytest.mark.parametrize(
    ("arm_id", "expected_type"),
    [
        ("trapezoid", "KMD2TrapezoidAttn"),
        ("lookahead", "KMD2LookaheadAttn"),
        ("qk-bc-additive", "KMD2BCBiasAttn"),
        ("qk-diagonal", "KMD2DiagonalQKAttn"),
    ],
)
def test_qwen_incremental_architecture_arms_build_exact_production_types(
    monkeypatch, arm_id: str, expected_type: str
) -> None:
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture

    native = _canonical_native(monkeypatch)
    replacement = build_qwen_architecture(
        native, QwenArchitectureConfig(arm_id, registry_sha256(), architecture_record(arm_id))
    )
    assert type(replacement).__name__ == expected_type
    manifest = replacement.transformation_manifest()
    assert set(manifest) == {"copied", "transformed", "new"}
    assert set(manifest["copied"]) | set(manifest["new"]) == set(replacement.state_dict())
    assert not manifest["transformed"]


@pytest.mark.parametrize("arm_id", ["trapezoid", "lookahead", "qk-bc-additive", "qk-diagonal"])
def test_qwen_incremental_architecture_arms_reject_cache(monkeypatch, arm_id: str) -> None:
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture

    replacement = build_qwen_architecture(
        _canonical_native(monkeypatch),
        QwenArchitectureConfig(arm_id, registry_sha256(), architecture_record(arm_id)),
    )
    hidden = torch.randn(1, 2, replacement.in_proj_qkv.in_features)
    with pytest.raises(ValueError, match="do not support cache"):
        replacement(hidden, use_cache=True)


def test_qwen_shared_hybrid_uses_canonical_builder(monkeypatch) -> None:
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid

    arm = "gdn2-mimo-r4-braid-shared-hola-w64"
    built = build_qwen_architecture(
        _canonical_native(monkeypatch),
        QwenArchitectureConfig(arm, registry_sha256(), architecture_record(arm)),
    )
    assert type(built) is QwenSharedBraidHybrid
    assert built.transformation_manifest()["package"] == "shared"


def test_builder_dispatches_four_state_hybrid(monkeypatch):
    from types import SimpleNamespace
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import QwenArchitectureConfig, build_qwen_architecture
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    native = KMD2NativeAttn(SimpleNamespace(
        hidden_size=16, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=4,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6,
    ), layer_idx=3)
    arm = "gdn2-mimo-r4-braid-four-state-hola-w64"
    built = build_qwen_architecture(native, QwenArchitectureConfig(arm, registry_sha256(), architecture_record(arm)))
    assert type(built) is QwenFourStateHybrid
    assert built.transformation_manifest()["state_count"] == 4
