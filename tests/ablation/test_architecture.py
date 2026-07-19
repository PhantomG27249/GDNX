import pytest
import torch

import research.kmd2_ablation.architecture as architecture_module
from dataclasses import replace


def test_frozen_registry_exports_are_available():
    assert len(architecture_module.STAGE1_IDS) == 30
    assert len(architecture_module.expand_stage2()) == 248


def test_maximum_hybrid_records_are_exact():
    expected = {
        "gdn2-mimo-r4-braid-shared-hola-w64": ("shared", "shared", "shared_state_outputs", 4),
        "gdn2-mimo-r4-braid-four-state-hola-w64": ("mimo_rank_contributions", "per_rank", "full_cross_state", 16),
    }
    for arm_id, (state_topology, phase_topology, read_topology, read_paths) in expected.items():
        record = architecture_module.architecture_record(arm_id)
        assert record.core_family == "gdn2_channelwise"
        assert record.mimo_rank == 4
        assert (record.input_rank, record.output_rank) == (4, 4)
        assert record.output_width == 4
        assert record.gate_mode == "channelwise"
        assert record.timescales == ((1, 16, 64, 256)
                                     if state_topology == "mimo_rank_contributions"
                                     else (64, 512, 4096, 32768))
        assert record.state_topology == state_topology
        assert record.rotation_mode == "mamba3_complex_input_dependent_cumulative"
        assert record.phase_topology == phase_topology
        assert record.update_paths == 4
        assert record.read_topology == read_topology
        assert record.read_paths == read_paths
        assert record.convolution_on is True
        assert record.state_input_mode == "trapezoid"
        assert record.lookahead is (state_topology == "shared")
        assert isinstance(record.qk_mode, architecture_module.QKMode)
        assert record.qk_contract == ("affine_diagonal_plus_additive_identity_init" if state_topology == "shared" else "gdn_unit_directional_affine_postrenorm_v1")
        assert record.transition_paths == (None if state_topology == "shared" else 4)
        assert record.cache.read_init == "hola_gate_logit_v2_minus4"
        assert record.cache.enabled is True
        assert (record.cache.width, record.cache.block_size) == (64, 256)
        assert record.cache.score == "active_update_frobenius"
        assert record.cache.read == "rmsnorm_rank_aware"
        assert record.cache_scope == "per_head_shared_across_ranks"
        if state_topology == "mimo_rank_contributions":
            assert "cms_periodic_updates" in record.compatibility.requires_semantics
            assert "cms_periodic_updates" not in record.compatibility.forbidden_families

        for drift in (
            {"convolution_on": False}, {"mimo_rank": 2},
            {"timescales": (1, 16, 64)},
            {"rotation_mode": "current"}, {"phase_topology": "shared" if phase_topology == "per_rank" else "per_rank"},
            {"input_rank": 2}, {"output_rank": 2},
            {"read_topology": "full_cross_state" if read_topology == "shared_state_outputs" else "shared_state_outputs"},
            {"read_paths": 16 if read_paths == 4 else 4},
            {"qk_contract": "none"},
            {"cache_scope": "per_state"}, {"update_paths": 16},
        ):
            with pytest.raises(ValueError):
                replace(record, **drift)


def test_compatibility_fails_closed_for_rank_width_gate_and_cache_crosses():
    incompatible = (
        ("mimo-r2", "rout-4"), ("mimo-r4", "rout-4"),
        ("gdn2-channel-r1", "mimo-r2"),
        ("gdn2-channel-r1", "cache-surprise-w64"),
        ("cache-surprise-w64", "mimo-r4"),
    )
    for left, right in incompatible:
        assert architecture_module.architecture_record(left).compatible_with(right) is False
    assert architecture_module.architecture_record("mimo-r2").compatibility.forbidden_families
    assert architecture_module.architecture_record("gdn2-channel-r1").compatibility.requires_semantics


def test_factorial_hash_excludes_status_but_commits_complete_scientific_identity():
    cell = architecture_module.expand_stage2()[0]
    common = dict(pair_id=cell.pair_id, baseline_id=cell.baseline_id,
                  left_arm_id=cell.left_arm_id, right_arm_id=cell.right_arm_id,
                  cell=cell.cell, enabled_arm_ids=cell.enabled_arm_ids)
    original = architecture_module.factorial_config_sha256(**common, incompatible_reason=cell.incompatible_reason)
    assert original == cell.config_sha256
    assert architecture_module.factorial_config_sha256(**{**common, "baseline_id": "stock"}, incompatible_reason=cell.incompatible_reason) != original
    assert architecture_module.factorial_config_sha256(**common, incompatible_reason="changed") != original


def test_factorial_cell_rejects_mutable_and_nonsensical_replacements():
    cell = architecture_module.expand_stage2()[0]
    with pytest.raises(TypeError):
        replace(cell, enabled_arm_ids=[])
    for changes in (
        {"pair_id": "unknown"}, {"cell": "22"}, {"left_arm_id": "unknown"},
        {"baseline_id": "unknown"}, {"status": "selected"},
        {"config_sha256": "ABC"}, {"incompatible_reason": "surprise"},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(cell, **changes)


def test_cache_semantics_and_source_root_are_exact():
    assert architecture_module.architecture_record("stock").baseline_id == "stock"
    expected = {
        "cache-surprise-w64": ("||k||_2 * ||beta_w*v - beta_e*m||_2", "unit_l2", 0, 64),
        "cache-coupled-w64": ("||k||_2 * beta_w * ||v-m||_2", "unit_l2", 0, 64),
        "cache-residual-w64": ("||k||_2 * ||v-m||_2", "unit_l2", 0, 64),
        "cache-write-value-w64": ("||k||_2 * beta_w * ||v||_2", "unit_l2", 0, 64),
        "cache-recency-w64": ("token_position", "unit_l2", 0, 64),
        "cache-reservoir-w64": ("SHA256(seed:batch:token:head:position)[0:3]_big_endian / 16777216", "unit_l2", 11, 64),
        "cache-unbounded-oracle": ("unbounded_oracle", "unit_l2", 0, None),
    }
    for arm_id, values in expected.items():
        cache = architecture_module.architecture_record(arm_id).cache
        assert (cache.score, cache.read, cache.selector_seed, cache.width) == values


def test_architecture_value_objects_reject_mutable_and_nonsensical_construction():
    with pytest.raises(TypeError):
        architecture_module.Compatibility(forbidden_arm_ids=[])
    with pytest.raises(TypeError):
        architecture_module.QKMode("diagonal", 0.0, 0.0, [-0.5, 0.5])
    with pytest.raises(ValueError):
        architecture_module.CacheArchitecture(width=-1)
    with pytest.raises(TypeError):
        replace(architecture_module.architecture_record("mimo-r2"), target_layers=[0])
    with pytest.raises(ValueError):
        replace(architecture_module.architecture_record("mimo-r2"), mimo_rank=0)


def test_cache_selector_formulas_are_authenticated_exact_semantics():
    expected = {
        "cache-surprise-w64": "||k||_2 * ||beta_w*v - beta_e*m||_2",
        "cache-coupled-w64": "||k||_2 * beta_w * ||v-m||_2",
        "cache-residual-w64": "||k||_2 * ||v-m||_2",
        "cache-write-value-w64": "||k||_2 * beta_w * ||v||_2",
        "cache-recency-w64": "token_position",
        "cache-reservoir-w64": "SHA256(seed:batch:token:head:position)[0:3]_big_endian / 16777216",
    }
    for arm_id, score in expected.items():
        assert architecture_module.architecture_record(arm_id).cache.score == score


def test_known_arm_replacement_rejects_every_noncanonical_field():
    mimo = architecture_module.architecture_record("mimo-r2")
    for changes in (
        {"state_key_dim": 64}, {"classification": "addition"},
        {"rotation_mode": "off"}, {"mimo_rank": 4},
    ):
        with pytest.raises(ValueError, match="canonical frozen record"):
            replace(mimo, **changes)
    cache = architecture_module.architecture_record("cache-surprise-w64")
    with pytest.raises(ValueError, match="canonical frozen record"):
        replace(cache, cache=replace(cache.cache, score="token_position"))
    qk = architecture_module.architecture_record("qk-diagonal")
    with pytest.raises(ValueError, match="canonical frozen record"):
        replace(qk, qk_mode="none")
import research.kmd2_ablation.tiny_backend as tiny_backend_module
from research.kmd2_ablation.architecture import (
    channelwise_gdn2_update,
    true_mimo_update,
)


def _leaf(shape, generator):
    return torch.randn(
        shape, dtype=torch.float64, generator=generator
    ).requires_grad_()


def _mimo_oracle(state, key, value, beta_e, beta_w):
    rank = key.shape[-2]
    rows = []
    for batch in range(state.shape[0]):
        heads = []
        for head in range(state.shape[1]):
            result = state[batch, head]
            if rank == 1:
                memory = key[batch, head, 0] @ result
                update = (
                    beta_w[batch, head, 0] * value[batch, head, 0]
                    - beta_e[batch, head, 0] * memory
                )
                result = result + torch.outer(key[batch, head, 0], update)
            else:
                for slot in range(rank):
                    memory = key[batch, head, slot] @ state[batch, head]
                    result = result - torch.outer(
                        key[batch, head, slot],
                        beta_e[batch, head, slot] * memory / rank,
                    )
                    result = result + torch.outer(
                        key[batch, head, slot],
                        beta_w[batch, head, slot]
                        * value[batch, head, slot],
                    )
            heads.append(result)
        rows.append(torch.stack(heads))
    return torch.stack(rows)


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_true_mimo_matches_explicit_fp64_oracle_forward_and_gradients(rank):
    generator = torch.Generator().manual_seed(1100 + rank)
    operands = (
        _leaf((2, 3, 4, 5), generator),
        _leaf((2, 3, rank, 4), generator),
        _leaf((2, 3, rank, 5), generator),
        torch.rand(
            (2, 3, rank), dtype=torch.float64, generator=generator
        ).requires_grad_(),
        torch.rand(
            (2, 3, rank), dtype=torch.float64, generator=generator
        ).requires_grad_(),
    )
    actual = true_mimo_update(*operands)
    expected = _mimo_oracle(*operands)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    weights = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), operands, retain_graph=True
    )
    expected_grads = torch.autograd.grad((expected * weights).sum(), operands)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=1e-10, rtol=1e-8
        )


def _channel_oracle(state, key, value, erase, write):
    rows = []
    for batch in range(state.shape[0]):
        heads = []
        for head in range(state.shape[1]):
            k = key[batch, head, 0]
            memory = (erase[batch, head, 0] * k) @ state[batch, head]
            update = write[batch, head, 0] * value[batch, head, 0] - memory
            heads.append(state[batch, head] + torch.outer(k, update))
        rows.append(torch.stack(heads))
    return torch.stack(rows)


def test_channelwise_gdn2_matches_nonuniform_fp64_oracle_and_gradients():
    generator = torch.Generator().manual_seed(1201)
    operands = (
        _leaf((2, 2, 3, 4), generator),
        _leaf((2, 2, 1, 3), generator),
        _leaf((2, 2, 1, 4), generator),
        torch.rand(
            (2, 2, 1, 3), dtype=torch.float64, generator=generator
        ).requires_grad_(),
        torch.rand(
            (2, 2, 1, 4), dtype=torch.float64, generator=generator
        ).requires_grad_(),
    )
    actual = channelwise_gdn2_update(*operands)
    expected = _channel_oracle(*operands)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    weights = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), operands, retain_graph=True
    )
    expected_grads = torch.autograd.grad((expected * weights).sum(), operands)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=1e-10, rtol=1e-8
        )


def test_uniform_channel_gates_equal_scalar_rank_one():
    generator = torch.Generator().manual_seed(1202)
    state = torch.randn((2, 2, 3, 4), dtype=torch.float64, generator=generator)
    key = torch.randn((2, 2, 1, 3), dtype=torch.float64, generator=generator)
    value = torch.randn((2, 2, 1, 4), dtype=torch.float64, generator=generator)
    erase = torch.rand((2, 2, 1), dtype=torch.float64, generator=generator)
    write = torch.rand((2, 2, 1), dtype=torch.float64, generator=generator)
    scalar = true_mimo_update(state, key, value, erase, write)
    channel = channelwise_gdn2_update(
        state,
        key,
        value,
        erase[..., None].expand(-1, -1, -1, 3),
        write[..., None].expand(-1, -1, -1, 4),
    )
    torch.testing.assert_close(channel, scalar, atol=1e-10, rtol=1e-8)


def test_true_mimo_is_invariant_to_common_rank_permutation():
    generator = torch.Generator().manual_seed(1203)
    state = torch.randn((2, 2, 3, 4), dtype=torch.float64, generator=generator)
    key = torch.randn((2, 2, 4, 3), dtype=torch.float64, generator=generator)
    value = torch.randn((2, 2, 4, 4), dtype=torch.float64, generator=generator)
    erase = torch.rand((2, 2, 4), dtype=torch.float64, generator=generator)
    write = torch.rand((2, 2, 4), dtype=torch.float64, generator=generator)
    permutation = torch.tensor([2, 0, 3, 1])
    expected = true_mimo_update(state, key, value, erase, write)
    actual = true_mimo_update(
        state,
        key[..., permutation, :],
        value[..., permutation, :],
        erase[..., permutation],
        write[..., permutation],
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


def test_channelwise_rejects_rank_greater_than_one():
    state = torch.zeros(1, 1, 2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="rank R=1"):
        channelwise_gdn2_update(
            state,
            torch.zeros(1, 1, 2, 2, dtype=torch.float64),
            torch.zeros(1, 1, 2, 3, dtype=torch.float64),
            torch.zeros(1, 1, 2, 2, dtype=torch.float64),
            torch.zeros(1, 1, 2, 3, dtype=torch.float64),
        )


def test_tiny_validated_update_does_not_repeat_public_validation(monkeypatch):
    state = torch.zeros(1, 1, 2, 3, dtype=torch.float64)
    key = torch.ones(1, 1, 1, 2, dtype=torch.float64)
    value = torch.ones(1, 1, 1, 3, dtype=torch.float64)
    gate = torch.ones(1, 1, 1, dtype=torch.float64)

    def repeated_validation(*args, **kwargs):
        raise AssertionError("shared public validation repeated")

    monkeypatch.setattr(architecture_module, "_validate_common", repeated_validation)
    actual = tiny_backend_module.true_mimo_update(state, key, value, gate, gate)
    assert actual.shape == state.shape
