from __future__ import annotations

import torch
import pytest

from research.kmd2_ablation.qwen_hybrid_math import (
    apply_complex_rotation,
    braided_decay,
    causal_lookahead,
    cumulative_phase,
    exact_outer_score,
    four_state_transition,
    four_state_step,
    identity_output_mixer,
    shared_state_step,
)


def test_four_state_transition_braids_independently_decayed_states_over_sources() -> None:
    states = torch.arange(1, 25, dtype=torch.float64).reshape(1, 1, 4, 2, 3)
    gamma = torch.tensor([[[[.9, .8], [.7, .6], [.5, .4], [.3, .2]]]], dtype=torch.float64)
    decayed = gamma[..., None] * states
    logits = torch.tensor([[[[3., 1., -1., -2.], [-2., 2., 0., 1.],
                             [0., -1., 4., 2.], [1., 3., -2., 0.]]]], dtype=torch.float64)
    probabilities = logits.softmax(-1)
    gate = torch.tensor([[.05, .10, .20, .25]], dtype=torch.float64)
    expected = (1 - gate[None, :, :, None, None]) * decayed + gate[None, :, :, None, None] * torch.einsum(
        "bhdl,bhlkv->bhdkv", probabilities, decayed)
    torch.testing.assert_close(four_state_transition(decayed, probabilities, gate), expected)
    assert torch.equal(four_state_transition(decayed, probabilities, torch.zeros_like(gate)), decayed)


def _t(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64)


def test_shared_r4_matches_oracle_with_unequal_dims() -> None:
    torch.manual_seed(1)
    b, h, r, kdim, vdim = 2, 2, 4, 3, 5
    state, key, value = _t(b, h, kdim, vdim), _t(b, h, r, kdim), _t(b, h, r, vdim)
    erase, write = torch.sigmoid(_t(b, h, r, kdim)), torch.sigmoid(_t(b, h, r, vdim))
    gamma, query = torch.sigmoid(_t(b, h, kdim)), _t(b, h, r, kdim)
    c = torch.softmax(_t(b, h, r), dim=-1)
    d = torch.sigmoid(_t(b, h, r)) / 4
    mixer = torch.eye(vdim, dtype=torch.float64).expand(h, r, vdim, vdim).clone() / 4

    got_state, got_output, got_write = shared_state_step(
        state, key, value, erase, write, gamma, c, d, query, mixer
    )
    decayed = gamma[..., None] * state
    erased_value = torch.einsum("bhrk,bhkv->bhrv", erase * key, decayed)
    erase_delta = torch.einsum("bhr,bhrk,bhrv->bhkv", c, key, erased_value)
    write_delta = torch.einsum("bhr,bhrk,bhrv->bhkv", d, key, write * value)
    expected_state = decayed - erase_delta + write_delta
    reads = torch.einsum("bhrk,bhkv->bhrv", query, expected_state)
    expected_output = torch.einsum("hrvw,bhrw->bhv", mixer, reads)

    torch.testing.assert_close(got_state, expected_state)
    torch.testing.assert_close(got_output, expected_output)
    torch.testing.assert_close(got_write, key[..., None] * (write * value)[..., None, :])
    assert torch.autograd.gradcheck(
        lambda s, kk, vv: shared_state_step(s, kk, vv, erase, write, gamma, c, d, query, mixer)[0],
        (state.requires_grad_(), key.requires_grad_(), value.requires_grad_()),
    )


def test_four_state_is_exactly_four_paths() -> None:
    torch.manual_seed(2)
    b, h, r, kdim, vdim = 1, 1, 4, 3, 2
    states, key, value = _t(b, h, r, kdim, vdim), _t(b, h, r, kdim), _t(b, h, r, vdim)
    erase, write = torch.sigmoid(_t(b, h, r, kdim)), torch.sigmoid(_t(b, h, r, vdim))
    gamma, query = torch.sigmoid(_t(b, h, r, kdim)), _t(b, h, r, kdim)
    mixer = torch.eye(vdim, dtype=torch.float64).expand(h, r, r, vdim, vdim).clone() / 16

    got_states, got_output, active = four_state_step(
        states, key, value, erase, write, gamma, query, mixer
    )
    decayed = gamma[..., None] * states
    erased = torch.einsum("bhrk,bhrkv->bhrv", erase * key, decayed)
    expected = decayed - key[..., :, None] * erased[..., None, :] + key[..., :, None] * (write * value)[..., None, :]
    reads = torch.einsum("bhik,bhjkv->bhijv", query, expected)
    expected_output = torch.einsum("hijvw,bhijw->bhv", mixer, reads)

    torch.testing.assert_close(got_states, expected)
    torch.testing.assert_close(got_output, expected_output)
    assert active.shape == (b, h, 4, kdim, vdim)
    assert reads.shape[2:4] == (4, 4)


def test_real_rotation_matches_explicit_complex() -> None:
    torch.manual_seed(3)
    x, phase = _t(2, 3, 8), _t(2, 3, 4)
    got = apply_complex_rotation(x, phase)
    z = torch.complex(x[..., 0::2], x[..., 1::2])
    expected_z = z * torch.exp(torch.complex(torch.zeros_like(phase), phase))
    expected = torch.stack((expected_z.real, expected_z.imag), dim=-1).flatten(-2)
    torch.testing.assert_close(got, expected)


def test_cumulative_phase_chunks_match_decode() -> None:
    theta = torch.tensor([[[0.1, 0.2], [0.3, -0.1], [8.0, 8.0], [0.4, 0.2], [0.5, 0.1]]])
    boundary = torch.tensor([[True, False, False, True, False]])
    valid = torch.tensor([[True, True, False, True, True]])
    full, final = cumulative_phase(theta, boundary, valid)
    carry = None
    pieces = []
    for lo, hi in ((0, 2), (2, 4), (4, 5)):
        chunk, carry = cumulative_phase(theta[:, lo:hi], boundary[:, lo:hi], valid[:, lo:hi], carry)
        pieces.append(chunk)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=0, atol=0)
    torch.testing.assert_close(carry, final, rtol=0, atol=0)
    torch.testing.assert_close(full[:, 2], full[:, 1])  # padding neither advances nor resets
    torch.testing.assert_close(full[:, 3], theta[:, 3])  # boundary resets before current theta


def test_low_memory_score_matches_materialized() -> None:
    torch.manual_seed(4)
    left, right = _t(2, 7, 3), _t(2, 7, 5)
    signs = torch.tensor([1, -1, 1, -1, 1, -1, 1], dtype=torch.float64)
    right = right * signs[None, :, None]
    materialized = torch.einsum("bnk,bnv->bkv", left, right)
    torch.testing.assert_close(exact_outer_score(left, right), torch.linalg.vector_norm(materialized, dim=(-2, -1)))
    assert torch.autograd.gradcheck(exact_outer_score, (left.requires_grad_(), right.requires_grad_()))


def test_simultaneous_differs_from_sequential() -> None:
    state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]]], dtype=torch.float64)
    value = torch.zeros(1, 1, 4, 2, dtype=torch.float64)
    erase = torch.ones_like(key)
    write = torch.zeros_like(value)
    gamma = torch.ones(1, 1, 2, dtype=torch.float64)
    weights = torch.full((1, 1, 4), 0.25, dtype=torch.float64)
    query = key.clone()
    mixer = torch.eye(2, dtype=torch.float64).expand(1, 4, 2, 2).clone() / 4
    simultaneous = shared_state_step(state, key, value, erase, write, gamma, weights, weights, query, mixer)[0]
    sequential = state.clone()
    for lane in range(4):
        read = torch.einsum("bhk,bhkv->bhv", key[:, :, lane], sequential)
        sequential = sequential - 0.25 * key[:, :, lane, :, None] * read[..., None, :]
    assert not torch.allclose(simultaneous, sequential)


def test_braid_identity_and_strict_validation() -> None:
    floor = 2.0**-24
    native = torch.tensor([[[1.0, 0.75, floor]]], dtype=torch.float64)
    pi = torch.softmax(_t(1, 1, 3, 4), dim=-1)
    zero = torch.zeros_like(pi)
    tau = torch.tensor([64.0, 512.0, 4096.0, 32768.0], dtype=torch.float64)
    got = braided_decay(native, pi, zero, tau)
    assert torch.equal(got, native)
    interior = torch.tensor([[[0.8, 0.75, 0.6]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda n, z: braided_decay(n, pi, z, tau), (interior, zero.requires_grad_()))
    for bad_native in (torch.tensor([1.0e-9], dtype=torch.float64), torch.tensor([1.1], dtype=torch.float64)):
        try:
            braided_decay(bad_native, torch.full((1, 4), 0.25, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64), tau)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid native decay accepted")
    bad_pi = pi.clone()
    bad_pi[..., 0] += 0.1
    try:
        braided_decay(native, bad_pi, zero, tau)
    except ValueError:
        pass
    else:
        raise AssertionError("non-simplex routing accepted")
    extreme = torch.tensor([[[[-1.0e6] * 4, [1.0e6] * 4]]], dtype=torch.float64)
    native_extreme = torch.tensor([[[1.0, 0.5]]], dtype=torch.float64)
    pi_extreme = torch.full_like(extreme, 0.25)
    saturated = braided_decay(native_extreme, pi_extreme, extreme, tau)
    assert torch.equal(saturated, torch.tensor([[[1.0, floor]]], dtype=torch.float64))
    assert bool(torch.isfinite(saturated).all())


def test_lookahead_trapezoid_mixers_and_gradchecks() -> None:
    value, previous = _t(1, 2, 4, 3), _t(1, 2, 4, 3)
    rho = torch.sigmoid(_t(1, 2, 4))
    active = torch.tensor([[[True, False, True, True], [False, True, True, False]]])
    got = causal_lookahead(value, previous, rho, active)
    expected = value + active[..., None] * rho[..., None] * (value - previous)
    torch.testing.assert_close(got, expected)
    assert torch.autograd.gradcheck(causal_lookahead, (value.requires_grad_(), previous.requires_grad_(), rho.requires_grad_(), active))
    torch.testing.assert_close(identity_output_mixer("shared", 2, 3, dtype=torch.float64), torch.eye(3, dtype=torch.float64).expand(2, 4, 3, 3) / 4)
    # "Option A" (2026-07-15): the Package B warm-start mixer reads only the
    # native lane-0 state with the lane-0 query so conversion is output-preserving.
    expected_four_state = torch.zeros(2, 4, 4, 3, 3, dtype=torch.float64)
    expected_four_state[:, 0, 0] = torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(identity_output_mixer("four_state", 2, 3, dtype=torch.float64), expected_four_state)

    b, h, kdim, vdim = 1, 1, 2, 2
    state, key, val = _t(b, h, kdim, vdim), _t(b, h, 4, kdim), _t(b, h, 4, vdim)
    erase, write = torch.sigmoid(_t(b, h, 4, kdim)), torch.sigmoid(_t(b, h, 4, vdim))
    gamma, weights, query = torch.sigmoid(_t(b, h, kdim)), torch.full((b, h, 4), 0.25, dtype=torch.float64), _t(b, h, 4, kdim)
    mixer = identity_output_mixer("shared", h, vdim, dtype=torch.float64)
    prior_write, trap = _t(b, h, 4, kdim, vdim), torch.sigmoid(_t(h, 4))
    history = torch.ones(b, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda s, q, p, tr: shared_state_step(
            s, key, val, erase, write, gamma, weights, weights, q, mixer,
            previous_write=p, trap_rho=tr, history_active=history)[:2],
        (state.requires_grad_(), query.requires_grad_(), prior_write.requires_grad_(), trap.requires_grad_()),
    )
    states = _t(b, h, 4, kdim, vdim)
    gamma4 = torch.sigmoid(_t(b, h, 4, kdim))
    mixer4 = identity_output_mixer("four_state", h, vdim, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda ss, q: four_state_step(ss, key, val, erase, write, gamma4, q, mixer4)[:2],
        (states.requires_grad_(), query.detach().requires_grad_()),
    )


def test_shared_trapezoid_preserves_rank_directions_and_gate_gradients() -> None:
    """Package A applies each rank gate before reducing the four write directions."""
    b, h, r, kdim, vdim = 2, 1, 4, 2, 3
    state = torch.zeros(b, h, kdim, vdim, dtype=torch.float64)
    key = torch.arange(1, 1 + b*h*r*kdim, dtype=torch.float64).reshape(b, h, r, kdim) / 10
    value = torch.arange(1, 1 + b*h*r*vdim, dtype=torch.float64).reshape(b, h, r, vdim) / 7
    erase = torch.zeros_like(key)
    write = torch.full_like(value, .6)
    gamma = torch.tensor([[[.8, .7]], [[.6, .5]]], dtype=torch.float64)
    c = torch.zeros(b, h, r, dtype=torch.float64)
    d = torch.tensor([[[.1, .2, .3, .4]], [[.4, .3, .2, .1]]], dtype=torch.float64)
    query = torch.ones_like(key)
    mixer = identity_output_mixer("shared", h, vdim, dtype=torch.float64)
    previous = torch.flip(key[..., None] * value[..., None, :], dims=(2,)).requires_grad_()
    rho = torch.tensor([[.15, .35, .55, .75]], dtype=torch.float64, requires_grad=True)
    history = torch.tensor([True, False])

    got, _, current = shared_state_step(
        state, key, value, erase, write, gamma, c, d, query, mixer,
        previous_write=previous, trap_rho=rho, history_active=history,
    )
    rank_write = key[..., None] * (write * value)[..., None, :]
    native = torch.einsum("bhr,bhrkv->bhkv", d, rank_write)
    trap = torch.einsum(
        "bhr,bhr,bhrkv->bhkv", d, rho.expand(b, -1, -1),
        gamma[:, :, None, :, None] * previous - rank_write,
    )
    expected = native + history[:, None, None, None] * trap
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert current.shape == (b, h, r, kdim, vdim)
    torch.testing.assert_close(current, rank_write, rtol=0, atol=0)

    gate_grad = torch.autograd.grad(got[0].sum(), rho)[0]
    assert gate_grad.shape == (h, r)
    assert torch.unique(gate_grad).numel() == r
    perturbed = rho.detach().clone(); perturbed[0, 2] += .1
    changed = shared_state_step(
        state, key, value, erase, write, gamma, c, d, query, mixer,
        previous_write=previous.detach(), trap_rho=perturbed, history_active=history,
    )[0]
    assert not torch.equal(changed[0], got.detach()[0])
    assert torch.equal(changed[1], got.detach()[1])


def test_shared_trapezoid_zero_gate_is_bit_exact_and_requires_history_mask() -> None:
    b, h, r, kdim, vdim = 1, 2, 4, 2, 2
    tensors = {
        "state": _t(b, h, kdim, vdim), "key": _t(b, h, r, kdim),
        "value": _t(b, h, r, vdim), "erase": torch.sigmoid(_t(b, h, r, kdim)),
        "write": torch.sigmoid(_t(b, h, r, vdim)), "gamma": torch.sigmoid(_t(b, h, kdim)),
        "erase_weight": torch.sigmoid(_t(b, h, r)), "write_weight": torch.sigmoid(_t(b, h, r)),
        "query": _t(b, h, r, kdim),
    }
    mixer = identity_output_mixer("shared", h, vdim, dtype=torch.float64)
    native = shared_state_step(**tensors, output_mixer=mixer)
    zero = shared_state_step(
        **tensors, output_mixer=mixer,
        previous_write=_t(b, h, r, kdim, vdim),
        trap_rho=torch.zeros(h, r, dtype=torch.float64),
        history_active=torch.ones(b, dtype=torch.bool),
    )
    assert torch.equal(zero[0], native[0])
    assert torch.equal(zero[1], native[1])
    with pytest.raises(ValueError, match="history_active"):
        shared_state_step(
            **tensors, output_mixer=mixer,
            previous_write=_t(b, h, r, kdim, vdim),
            trap_rho=torch.zeros(h, r, dtype=torch.float64),
        )


def test_rotation_phase_gradchecks_and_invalid_inputs() -> None:
    x, phase = _t(2, 4), _t(2, 2)
    assert torch.autograd.gradcheck(apply_complex_rotation, (x.requires_grad_(), phase.requires_grad_()))
    theta = _t(1, 3, 2).requires_grad_()
    boundary = torch.tensor([[True, False, False]])
    valid = torch.ones_like(boundary)
    assert torch.autograd.gradcheck(lambda t: cumulative_phase(t, boundary, valid)[0], (theta,))
    invalid_calls = (
        lambda: apply_complex_rotation(torch.empty(2, 0, dtype=torch.float64), torch.empty(2, 0, dtype=torch.float64)),
        lambda: cumulative_phase(theta, [[True, False, False]], valid),
        lambda: causal_lookahead(_t(1, 2), _t(1, 2), torch.tensor([0.5], dtype=torch.float64), torch.tensor([[True, False]])),
        lambda: causal_lookahead(_t(1, 2), _t(1, 2), torch.tensor([1.1], dtype=torch.float64), torch.tensor([[True, True]])),
    )
    for call in invalid_calls:
        try:
            call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("strict invalid input accepted")
