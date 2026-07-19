from __future__ import annotations

import copy

import pytest
import torch


def _native(dtype=torch.float64):
    torch.manual_seed(31)
    hidden, heads, key_width, value_width = 7, 2, 8, 3
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.H, self.dk, self.dv = heads, key_width, value_width
            self.key_dim, self.value_dim, self.conv_k, self.r_out = heads*key_width, heads*value_width, 3, 1
            self.in_proj_qkv = torch.nn.Linear(hidden, heads*(2*key_width+value_width), bias=False, dtype=dtype)
            self.in_proj_b = torch.nn.Linear(hidden, heads, bias=False, dtype=dtype)
            self.in_proj_z = torch.nn.Linear(hidden, heads*value_width, bias=False, dtype=dtype)
            self.in_proj_a = torch.nn.Linear(hidden, heads, bias=False, dtype=dtype)
            self.conv1d = torch.nn.Conv1d(heads*(2*key_width+value_width), heads*(2*key_width+value_width),
                                          3, groups=heads*(2*key_width+value_width), bias=False, padding=2, dtype=dtype)
            self.out_proj = torch.nn.Linear(heads*value_width, hidden, bias=False, dtype=dtype)
            self.dt_bias = torch.nn.Parameter(torch.randn(heads, dtype=dtype))
            self.A_log = torch.nn.Parameter(torch.randn(heads, dtype=dtype))
            self.norm = torch.nn.LayerNorm(value_width, elementwise_affine=True, bias=False, dtype=dtype)
            self.rot_proj = torch.nn.Linear(hidden, heads*(key_width//2), dtype=dtype)
            self.decay_chan = torch.nn.Parameter(torch.randn(heads, key_width, dtype=dtype))
            self.bw_off = torch.nn.Parameter(torch.randn(heads, dtype=dtype))
    return Native()


def test_conversion_preserves_source_and_replication():
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    native = _native(); source = copy.deepcopy(native.state_dict())
    for package in ("shared", "four_state"):
        component = HybridComponents.from_native(native, package=package)
        x = torch.randn(2, 5, 7, dtype=torch.float64)
        q, k, v, erase, write, z = component.project_inputs(x)
        expected_key_width = 8 if package == "shared" else 2
        assert q.shape == k.shape == erase.shape == (2, 5, 2, 4, expected_key_width)
        assert v.shape == write.shape == z.shape == (2, 5, 2, 4, 3)
        assert component.native_decay_weight.shape == (2, 7)
        assert component.native_decay_topology == "single_shared_content_rate"
        if package == "shared":
            assert component.native_decay_chan is not None
            assert component.native_decay_chan.shape == (2, 8)
            assert component.native_decay_pair is None
        else:
            assert component.native_decay_chan is None
            assert component.native_decay_pair is not None
            assert component.native_decay_pair.shape == (2, 4, 1)
            native_q = native.in_proj_qkv.weight[: native.key_dim].reshape(
                native.H, native.dk, 7
            )
            for rank in range(4):
                expected_q = torch.cat((
                    native_q[:, rank:rank + 1],
                    native_q[:, native.dk // 2 + rank:native.dk // 2 + rank + 1],
                ), 1)
                torch.testing.assert_close(
                    component.q_weight[rank].reshape(native.H, component.dk, 7),
                    expected_q,
                )
            assert component.conv_channel_indices.shape == (
                4, 2 * native.H * component.dk + native.value_dim
            )
            expected_pair = 0.5 * (
                native.decay_chan[:, : native.dk // 2]
                + native.decay_chan[:, native.dk // 2 :]
            ).reshape(2, 4, 1)
            torch.testing.assert_close(component.native_decay_pair, expected_pair)
        assert component.phase_logits(x).shape == ((2, 5, 2, 4) if package == "shared" else (2, 5, 2, 4, 1))
    for key, value in native.state_dict().items():
        assert torch.equal(value, source[key])


def test_four_state_decay_has_no_second_lane_axis_or_router():
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    component = HybridComponents.from_native(_native(), package="four_state")
    x = torch.zeros(2, 5, 7, dtype=torch.float64)
    gamma = component.decay_gamma(x)
    assert gamma.shape == (2, 5, 2, 4, 2)
    assert not hasattr(component, "state_braid_router")
    assert component.braid_residual is None
    assert component.decay_residual_topology == "fixed_log_rate_multipliers"
    # Exercise the unsaturated regime; the production safety clamp still
    # intentionally bounds a positive native channel residual at gamma=1.
    with torch.no_grad():
        component.native_decay_pair.zero_()
    gamma = component.decay_gamma(x)
    rates = -gamma.log()
    expected = torch.tensor((1., 1/16, 1/64, 1/256), dtype=gamma.dtype)
    torch.testing.assert_close(rates / rates[..., :1, :], expected[None, None, None, :, None].expand_as(rates))
    assert component.trapezoid_lambda(x).shape == (2, 5, 2, 4)


def test_four_state_decay_stores_one_identifiable_parameter_per_complex_pair():
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    component = HybridComponents.from_native(_native(), package="four_state")
    assert "native_decay_pair" in dict(component.named_parameters())
    assert "native_decay_chan" not in dict(component.named_parameters())
    with torch.no_grad():
        component.native_decay_pair.copy_(torch.tensor([
            [[0.1], [-0.2], [0.3], [-0.4]],
            [[0.2], [-0.1], [0.4], [-0.3]],
        ], dtype=torch.float64))
    gamma = component.decay_gamma(torch.zeros(1, 1, 7, dtype=torch.float64))
    torch.testing.assert_close(gamma[..., :1], gamma[..., 1:])


def test_cache_gate_schema_is_explicit_and_affine_is_identity_initialized():
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    component = HybridComponents.from_native(_native(), package="four_state")
    assert "cache_gate_logit" in dict(component.named_parameters())
    assert "cache_gate" not in dict(component.named_parameters())
    torch.testing.assert_close(component.cache_gate_logit, torch.full_like(component.cache_gate_logit, -4.0))
    q = torch.randn(2, 3, 2, 4, component.dk, dtype=torch.float64)
    k = torch.randn_like(q)
    qa, ka = component.affine_qk(q, k)
    torch.testing.assert_close(qa, q); torch.testing.assert_close(ka, k)


@pytest.mark.parametrize("mutation", ("missing", "unexpected", "shape", "dtype", "conv"))
def test_conversion_rejects_bad_native_transactionally(mutation):
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    native = _native(); before = copy.deepcopy(native.state_dict())
    if mutation == "missing": del native._modules["in_proj_a"]
    elif mutation == "unexpected": native.extra = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    elif mutation == "shape": native.decay_chan = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
    elif mutation == "dtype": native.dt_bias = torch.nn.Parameter(native.dt_bias.float())
    else: native.conv1d.groups = 1
    with pytest.raises((TypeError, ValueError)):
        HybridComponents.from_native(native, package="four_state")
    for key, value in native.state_dict().items():
        if key in before and value.shape == before[key].shape and value.dtype == before[key].dtype:
            assert torch.equal(value, before[key])
