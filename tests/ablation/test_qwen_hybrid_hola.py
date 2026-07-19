import torch
import dataclasses
import inspect
import pytest


def test_per_head_top64_can_diverge() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    cache = HybridHOLACache(width=2, block_size=4, heads=2, rank_in=4, key_dim=3, value_dim=2)
    k = torch.randn(1, 3, 2, 4, 3)
    v = torch.randn(1, 3, 2, 4, 2)
    scores = torch.tensor([[[9., 1.], [2., 8.], [7., 6.]]])
    state = cache.admit(None, k, v, scores, torch.arange(3)[None], torch.ones(1, 3, dtype=torch.bool))
    assert state.block_positions[0, 0, :3].tolist() == [0, 1, 2]
    assert state.block_positions[0, 1, :3].tolist() == [0, 1, 2]
    assert state.block_scores[0, 0, :3].tolist() == [9., 2., 7.]
    assert state.block_scores[0, 1, :3].tolist() == [1., 8., 6.]


def test_hola_admits_before_read_and_first_token_reads_itself() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    cache = HybridHOLACache(width=64, block_size=256, heads=1, rank_in=4, key_dim=2, value_dim=2)
    q = torch.ones(1, 1, 1, 1, 2)
    k = torch.ones(1, 1, 1, 4, 2)
    v = torch.ones(1, 1, 1, 4, 2)
    out, state = cache.scan(q, k, v, torch.ones(1, 1, 1), valid=torch.ones(1, 1, dtype=torch.bool))
    assert not torch.equal(out, torch.zeros_like(out))
    assert state.valid.sum().item() == 0
    assert state.block_valid.sum().item() == 1


def test_hola_boundary_resets_before_current_admission_and_invalid_is_noop() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    cache = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=4, key_dim=2, value_dim=2)
    q = torch.ones(1, 3, 1, 1, 2)
    k = torch.ones(1, 3, 1, 4, 2)
    v = torch.tensor([1., 9., 7.]).view(1, 3, 1, 1, 1).expand(-1, -1, -1, 4, 2)
    out, state = cache.scan(
        q, k, v, torch.ones(1, 3, 1),
        valid=torch.tensor([[True, True, False]]),
        boundary=torch.tensor([[False, True, True]]),
    )
    assert out[0, 1].mean() > out[0, 0].mean()
    assert torch.equal(out[:, 2], torch.zeros_like(out[:, 2]))
    assert state.current_epoch.item() == 1
    assert state.admission_count.item() == 2


def test_composed_hola_kernel_gradient_parity() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    cache = HybridHOLACache(width=3, block_size=8, heads=1, rank_in=4, key_dim=2, value_dim=3,
                            storage_dtype=torch.float32)
    q = torch.randn(1, 4, 1, 4, 2, requires_grad=True)
    k = torch.randn(1, 4, 1, 4, 2, requires_grad=True)
    v = torch.randn(1, 4, 1, 4, 3, requires_grad=True)
    score = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    actual, _ = cache.scan(q, k, v, score)
    expected = []
    for t in range(4):
        qq = q[:, t].float()
        qq = qq * torch.rsqrt(qq.square().mean(-1, keepdim=True) + 1e-6)
        qq = qq * cache.gamma_q.float()[None]
        kk = k[:, :t + 1].permute(0, 2, 1, 3, 4).float()
        kk = kk * torch.rsqrt(kk.square().mean(-1, keepdim=True) + 1e-6)
        kk = kk * cache.gamma_k.float()[None, :, None]
        logits = torch.einsum("bhok,bhtik->bhoit", qq, kk) * 2 ** -0.5
        sink = cache.sink_logit.float()[None, :, None, :, None]
        weights = torch.softmax(torch.cat((logits, sink.expand(*logits.shape[:-1], 1)), -1), -1)[..., :-1]
        expected.append(torch.einsum("bhoit,bhtiv->bhoiv", weights, v[:, :t + 1].permute(0, 2, 1, 3, 4).float()))
    expected = torch.stack(expected, 1)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    torch.testing.assert_close(actual, expected)
    for left, right in zip(ga, ge):
        torch.testing.assert_close(left, right)
    actual.square().sum().backward()
    for parameter in (cache.gamma_q, cache.gamma_k, cache.sink_logit):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_uniform_rmsnorm_gamma_scaling_changes_cache_sharpness() -> None:
    """Regression: pre-norm gamma would cancel this uniform rescaling exactly."""
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    cache = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=1,
                            key_dim=2, value_dim=1, storage_dtype=torch.float32)
    keys = torch.tensor([[[[[1., 0.]]], [[[0., 1.]]]]])
    values = torch.tensor([[[[[1.]]], [[[0.]]]]])
    state = cache.admit(None, keys, values, torch.ones(1, 2, 1),
                        torch.arange(2)[None], torch.ones(1, 2, dtype=torch.bool))
    query = torch.tensor([[[[1., 0.]]]])
    with torch.no_grad():
        cache.gamma_q.fill_(1.0); cache.gamma_k.fill_(1.0)
        unit = cache.read(state, query)
        cache.gamma_q.fill_(2.0); cache.gamma_k.fill_(2.0)
        sharp = cache.read(state, query)
    assert sharp.item() > unit.item() + 0.05
    assert sharp.item() > 0.95


def test_c_th_token_reads_full_block_before_promotion() -> None:
    """The low-score C-th token must remain visible for its own causal read."""
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    cache = HybridHOLACache(width=1, block_size=2, heads=1, rank_in=4,
                            key_dim=2, value_dim=1, storage_dtype=torch.float32)
    with torch.no_grad():
        cache.gamma_q.fill_(8.0); cache.gamma_k.fill_(8.0)
    query = torch.tensor([1., 0., 0., 1.]).reshape(1, 2, 1, 1, 2)
    keys = query.expand(-1, -1, -1, 4, -1).clone()
    values = torch.tensor([0., 10.]).view(1, 2, 1, 1, 1).expand(-1, -1, -1, 4, -1)
    output, state = cache.scan(query, keys, values, torch.tensor([[[10.], [1.]]]))
    assert output[0, 1].mean().item() > 9.9
    # Promotion still occurs after the read and retains only the high-score item.
    assert state.positions.item() == 0
    assert state.block_count.item() == 0


def test_cache_bytes_include_metadata_and_workspace() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    cache = HybridHOLACache(width=64, block_size=256, heads=2, rank_in=4, key_dim=8, value_dim=6,
                            storage_dtype=torch.bfloat16)
    report = cache.resource_report(batch_size=3)
    assert report["persistent_bytes"] == sum(report["persistent"].values())
    assert report["workspace_bytes"] == sum(report["workspace"].values())
    assert report["persistent"]["scores"] > 0
    assert report["persistent"]["positions"] > 0
    assert report["persistent"]["valid"] > 0
    assert report["workspace"]["block_keys"] > 0
    state = cache._empty(3, torch.device("cpu"))
    assert state.nbytes == report["allocated_bytes"]


def test_score_helpers_use_committed_matrix_norms_without_key_factor() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import (
        four_state_exact_update_score, shared_exact_update_score)
    update = torch.tensor([[[[3., 4.], [0., 0.]]]])
    assert shared_exact_update_score(update).item() == 5
    lanes = torch.stack((update, -update, torch.zeros_like(update), torch.zeros_like(update)), 2)
    torch.testing.assert_close(four_state_exact_update_score(lanes), torch.tensor([[5 * 2 ** .5]]))


def test_four_state_normalized_score_removes_cms_tick_cadence_bias() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import (
        four_state_exact_update_score, four_state_normalized_update_score,
    )

    # Equal per-lane innovations occur on the 1/16/64/256-token cadence with
    # one through four lanes ticking.  The normalized score must measure the
    # same content surprise instead of growing by sqrt(ticking lanes).
    tick_lanes = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    per_head_update = torch.tensor([
        [[3.0, 4.0], [0.0, 0.0]],
        [[5.0, 0.0], [0.0, 12.0]],
    ])
    lane_updates = (
        per_head_update[None, :, None]
        * tick_lanes[:, None, :, None, None]
    )
    per_lane_norm = torch.tensor([5.0, 13.0]).expand(4, -1)

    normalized = four_state_normalized_update_score(lane_updates, tick_lanes)
    torch.testing.assert_close(normalized, per_lane_norm)
    raw = four_state_exact_update_score(lane_updates)
    expected_raw = per_lane_norm * tick_lanes.sum(-1).sqrt()[:, None]
    torch.testing.assert_close(raw, expected_raw)


def test_recency_and_surprise_promote_different_survivors() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    args = dict(width=1, block_size=3, heads=1, rank_in=4, key_dim=1, value_dim=1)
    k = torch.arange(3.).reshape(1, 3, 1, 1, 1).expand(-1, -1, -1, 4, -1)
    v = k.clone(); score = torch.tensor([[[10.], [1.], [2.]]]); pos = torch.arange(3)[None]
    valid = torch.ones(1, 3, dtype=torch.bool)
    surprise = HybridHOLACache(**args).admit(None, k, v, score, pos, valid)
    recency = HybridHOLACache(**args, policy="recency").admit(None, k, v, score, pos, valid)
    assert surprise.positions.item() == 0
    assert recency.positions.item() == 2


def test_chunk_decode_reset_and_padding_continuation_parity() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    torch.manual_seed(91)
    cache = HybridHOLACache(width=3, block_size=2, heads=2, rank_in=4, key_dim=3, value_dim=2)
    q = torch.randn(2, 6, 2, 4, 3)
    k = torch.randn(2, 6, 2, 4, 3)
    v = torch.randn(2, 6, 2, 4, 2)
    score = torch.randn(2, 6, 2).abs()
    boundary = torch.tensor([[True, False, False, True, False, False],
                             [True, False, False, False, False, False]])
    valid = torch.tensor([[True, True, True, True, True, False],
                          [True, True, False, False, False, False]])
    full, full_state = cache.scan(q, k, v, score, boundary=boundary, valid=valid)
    parts, state = [], None
    for start, stop in ((0, 2), (2, 3), (3, 6)):
        out, state = cache.scan(q[:, start:stop], k[:, start:stop], v[:, start:stop],
                                score[:, start:stop], boundary=boundary[:, start:stop],
                                valid=valid[:, start:stop], initial_state=state)
        parts.append(out)
    torch.testing.assert_close(torch.cat(parts, 1), full)
    assert state is not None
    torch.testing.assert_close(state.keys, full_state.keys)
    assert torch.equal(state.positions, full_state.positions)


def test_empty_scan_still_strictly_validates_continuation_state() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=3, heads=1, rank_in=4, key_dim=2, value_dim=2)
    state = cache._empty(1, torch.device("cpu"))
    stale = dataclasses.replace(state, block_count=torch.full((1, 1), 4, dtype=torch.int64))
    q = torch.empty(1, 0, 1, 4, 2); k = q.clone(); v = q.clone()
    with pytest.raises(ValueError, match="block_count.*range"):
        cache.scan(q, k, v, torch.empty(1, 0, 1), initial_state=stale)


@pytest.mark.parametrize("mutation,message", [
    (lambda s: dataclasses.replace(s, keys=s.keys[..., :1]), "keys shape"),
    (lambda s: dataclasses.replace(s, block_values=s.block_values.double()), "block_values dtype"),
    (lambda s: dataclasses.replace(s, scores=s.scores.fill_(float("nan"))), "scores must be finite"),
    (lambda s: dataclasses.replace(s, block_count=torch.ones(1, 1, dtype=torch.int32)), "block_count dtype"),
    (lambda s: dataclasses.replace(s, block_count=torch.ones(1, 1, dtype=torch.int64)), "block_valid.*count"),
])
def test_malformed_or_stale_hola_state_is_rejected(mutation, message) -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=3, heads=1, rank_in=4, key_dim=2, value_dim=2)
    state = mutation(cache._empty(1, torch.device("cpu")))
    q = torch.empty(1, 0, 1, 4, 2)
    with pytest.raises((TypeError, ValueError), match=message):
        cache.scan(q, q, q, torch.empty(1, 0, 1), initial_state=state)


def test_hola_per_token_hot_path_has_no_python_tensor_sync() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    source = "\n".join(inspect.getsource(getattr(HybridHOLACache, name))
                       for name in ("_run", "_admit_unchecked", "_promotion_transform", "read"))
    assert "bool(" not in source
    assert ".item(" not in source
    assert ".tolist(" not in source
    promotion_control = inspect.getsource(HybridHOLACache._promote_if_complete)
    assert "bool(rows.any())" in promotion_control
    assert HybridHOLACache.implementation_reference.endswith("pytorch_reference_host_sync")


def test_survivor_selection_runs_once_per_completed_block(monkeypatch) -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=4, key_dim=2, value_dim=2)
    monkeypatch.setattr(torch, "cond", lambda pred, true_fn, false_fn, operands:
                        true_fn(*operands) if pred else false_fn(*operands))
    calls = 0
    original = cache._select_survivors
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(cache, "_select_survivors", counted)
    q = torch.randn(1, 4, 1, 4, 2); k = torch.randn_like(q); v = torch.randn_like(q)
    cache.scan(q, k, v, torch.randn(1, 4, 1).abs())
    assert calls == 1


def test_entire_promotion_transform_runs_zero_times_before_c_and_once_at_c(monkeypatch) -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=4, key_dim=2, value_dim=2)
    monkeypatch.setattr(torch, "cond", lambda pred, true_fn, false_fn, operands:
                        true_fn(*operands) if pred else false_fn(*operands))
    calls = 0
    original = cache._promotion_transform
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(cache, "_promotion_transform", counted)
    state = cache._empty(1, torch.device("cpu"))
    for token in range(4):
        k = torch.randn(1, 1, 1, 4, 2); v = torch.randn_like(k)
        state = cache.admit(state, k, v, torch.ones(1, 1, 1), torch.tensor([[token]]),
                            torch.ones(1, 1, dtype=torch.bool))
        assert calls == (1 if token == 3 else 0)


def test_unchecked_step_has_no_validation_allocation_or_python_sync() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    source = inspect.getsource(HybridHOLACache.step_unchecked)
    for forbidden in ("_validate", "_empty(", "bool(", ".item(", ".tolist("):
        assert forbidden not in source


def test_step_fast_matches_unchecked_across_promotion_and_gradients() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    torch.manual_seed(613)
    batch, tokens, heads, rank, key_dim, value_dim = 2, 5, 2, 4, 2, 3
    cache = HybridHOLACache(
        width=2, block_size=3, heads=heads, rank_in=rank,
        key_dim=key_dim, value_dim=value_dim, storage_dtype=torch.float32,
    )
    query = torch.randn(
        batch, tokens, heads, rank, key_dim, requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(
        batch, tokens, heads, rank, value_dim, requires_grad=True,
    )
    # Unique scores make survivor selection deterministic for every row/head.
    score = torch.arange(
        1, batch * tokens * heads + 1, dtype=torch.float32,
    ).reshape(batch, tokens, heads)
    valid = torch.ones(batch, dtype=torch.bool)
    boundary = torch.zeros(batch, dtype=torch.bool)
    fast_state = cache._empty(batch, torch.device("cpu"))
    reference_state = cache._empty(batch, torch.device("cpu"))
    block_fill = 0
    fast_outputs, reference_outputs = [], []

    for token in range(tokens):
        fast_output, fast_state, block_fill = cache.step_fast(
            fast_state, query[:, token], key[:, token], value[:, token],
            score[:, token], fast_state.next_position, block_fill,
        )
        reference_output, reference_state = cache.step_unchecked(
            reference_state, query[:, token], key[:, token], value[:, token],
            score[:, token], reference_state.next_position, valid, boundary,
        )
        fast_outputs.append(fast_output)
        reference_outputs.append(reference_output)

    fast_output = torch.stack(fast_outputs, 1)
    reference_output = torch.stack(reference_outputs, 1)
    assert block_fill == tokens % cache.block_size
    assert torch.equal(
        fast_state.block_count,
        torch.full_like(fast_state.block_count, tokens % cache.block_size),
    )
    assert fast_state.valid.sum().item() == batch * heads * cache.width

    # Both paths execute the same FP32 cache math; step_fast replaces admission
    # masking with indexed scatter and promotion sync with a mirrored fill
    # counter, so 1e-6 is a tight bound.
    torch.testing.assert_close(
        fast_output, reference_output, atol=1e-6, rtol=1e-6,
    )
    for field in dataclasses.fields(fast_state):
        fast_value = getattr(fast_state, field.name)
        reference_value = getattr(reference_state, field.name)
        assert isinstance(fast_value, torch.Tensor)
        if fast_value.is_floating_point():
            torch.testing.assert_close(
                fast_value, reference_value, atol=1e-6, rtol=1e-6,
            )
        else:
            assert torch.equal(fast_value, reference_value), field.name

    weights = torch.linspace(
        0.25, 1.25, fast_output.numel(), dtype=fast_output.dtype,
    ).reshape_as(fast_output)
    fast_loss = (fast_output * weights).sum() + 0.01 * fast_output.square().sum()
    reference_loss = (
        (reference_output * weights).sum()
        + 0.01 * reference_output.square().sum()
    )
    named_parameters = tuple(cache.named_parameters())
    gradient_targets = (query, key, value, *(p for _, p in named_parameters))
    fast_gradients = torch.autograd.grad(fast_loss, gradient_targets)
    reference_gradients = torch.autograd.grad(reference_loss, gradient_targets)
    gradient_names = ("query", "key", "value", *(n for n, _ in named_parameters))
    # Preserve the documented <=1e-5 backward envelope while keeping absolute
    # error one order tighter for the sync-only step specialization.
    for name, fast_gradient, reference_gradient in zip(
            gradient_names, fast_gradients, reference_gradients):
        assert fast_gradient.abs().sum() > 0, name
        torch.testing.assert_close(
            fast_gradient, reference_gradient, atol=1e-6, rtol=1e-5,
            msg=lambda message, name=name: f"{name} gradient: {message}",
        )


def test_scan_fast_batches_causal_reads_across_promotions_with_gradient_parity() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache

    torch.manual_seed(947)
    batch, tokens, heads, rank, key_dim, value_dim = 2, 7, 2, 4, 3, 2
    cache = HybridHOLACache(
        width=2, block_size=3, heads=heads, rank_in=rank,
        key_dim=key_dim, value_dim=value_dim, storage_dtype=torch.float32,
    )
    query = torch.randn(
        batch, tokens, heads, rank, key_dim, requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(
        batch, tokens, heads, rank, value_dim, requires_grad=True,
    )
    score = torch.arange(
        1, batch * tokens * heads + 1, dtype=torch.float32,
    ).reshape(batch, tokens, heads)

    batched_state = cache._empty(batch, torch.device("cpu"))
    reference_state = cache._empty(batch, torch.device("cpu"))
    batched_output, batched_state, batched_fill = cache.scan_fast(
        batched_state, query, key, value, score, 0,
    )
    reference_outputs = []
    reference_fill = 0
    for token in range(tokens):
        output, reference_state, reference_fill = cache.step_fast(
            reference_state, query[:, token], key[:, token], value[:, token],
            score[:, token], reference_state.next_position, reference_fill,
        )
        reference_outputs.append(output)
    reference_output = torch.stack(reference_outputs, 1)

    assert batched_fill == reference_fill == tokens % cache.block_size
    torch.testing.assert_close(
        batched_output, reference_output, atol=1e-6, rtol=1e-6,
    )
    for field in dataclasses.fields(batched_state):
        batched_value = getattr(batched_state, field.name)
        reference_value = getattr(reference_state, field.name)
        if batched_value.is_floating_point():
            torch.testing.assert_close(
                batched_value, reference_value, atol=1e-6, rtol=1e-6,
            )
        else:
            assert torch.equal(batched_value, reference_value), field.name

    weights = torch.linspace(
        0.25, 1.25, batched_output.numel(), dtype=batched_output.dtype,
    ).reshape_as(batched_output)
    targets = (query, key, value, *cache.parameters())
    batched_gradients = torch.autograd.grad(
        (batched_output * weights).sum(), targets, retain_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        (reference_output * weights).sum(), targets,
    )
    for batched_gradient, reference_gradient in zip(
            batched_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(
            batched_gradient, reference_gradient, atol=1e-6, rtol=1e-5,
        )


def test_lazy_epoch_reset_reuses_all_cache_payload_storage() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=4, key_dim=2, value_dim=2)
    state = cache._empty(1, torch.device("cpu"))
    payloads = ("keys", "values", "scores", "positions", "valid", "epochs",
                "block_keys", "block_values", "block_scores", "block_positions",
                "block_valid", "block_epochs")
    pointers = {name: getattr(state, name).data_ptr() for name in payloads}
    ordinary = cache._advance_epoch_unchecked(state, torch.zeros(1, dtype=torch.bool))
    reset = cache._advance_epoch_unchecked(ordinary, torch.ones(1, dtype=torch.bool))
    assert all(getattr(ordinary, name).data_ptr() == pointers[name] for name in payloads)
    assert all(getattr(reset, name).data_ptr() == pointers[name] for name in payloads)
    assert reset.current_epoch.item() == 1 and reset.block_count.item() == 0
    source = inspect.getsource(HybridHOLACache._advance_epoch_unchecked)
    assert "keys" in source and "zeros_like(state.keys)" not in source


def test_epoch_overflow_fails_closed_at_scan_boundary() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=4, key_dim=2, value_dim=2)
    state = cache._empty(1, torch.device("cpu"))
    state = dataclasses.replace(state, current_epoch=torch.full_like(state.current_epoch, torch.iinfo(torch.int64).max))
    q = torch.randn(1, 1, 1, 4, 2)
    with pytest.raises(ValueError, match="epoch overflow"):
        cache.scan(q, q, q, torch.ones(1, 1, 1), boundary=torch.ones(1, 1, dtype=torch.bool),
                   initial_state=state)


def test_epoch_overflow_accounts_for_all_chunk_boundaries() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=4, key_dim=2, value_dim=2)
    state = cache._empty(1, torch.device("cpu"))
    state = dataclasses.replace(state, current_epoch=torch.full_like(
        state.current_epoch, torch.iinfo(torch.int64).max - 1))
    q = torch.randn(1, 3, 1, 4, 2)
    boundary = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="epoch overflow"):
        cache.scan(q, q, q, torch.ones(1, 3, 1), boundary=boundary, initial_state=state)


def test_promotion_crossing_preserves_exact_selected_kv_gradients() -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=1, block_size=2, heads=1, rank_in=4, key_dim=2, value_dim=2,
                            storage_dtype=torch.float32)
    q = torch.randn(1, 3, 1, 4, 2, requires_grad=True)
    k = torch.randn(1, 3, 1, 4, 2, requires_grad=True)
    v = torch.randn(1, 3, 1, 4, 2, requires_grad=True)
    score = torch.tensor([[[9.], [1.], [0.]]])
    actual, state = cache.scan(q, k, v, score)
    qq = q[:, 2].float()
    qq = qq * torch.rsqrt(qq.square().mean(-1, keepdim=True) + 1e-6)
    qq = qq * cache.gamma_q.float()[None]
    kk = k[:, [0, 2]].permute(0, 2, 1, 3, 4).float()
    kk = kk * torch.rsqrt(kk.square().mean(-1, keepdim=True) + 1e-6)
    kk = kk * cache.gamma_k.float()[None, :, None]
    logits = torch.einsum("bhok,bhnik->bhoin", qq, kk) * 2 ** -0.5
    sink = cache.sink_logit.float()[None, :, None, :, None]
    weights = torch.softmax(torch.cat((logits, sink.expand(*logits.shape[:-1], 1)), -1), -1)[..., :-1]
    expected = torch.einsum("bhoin,bhniv->bhoiv", weights, v[:, [0, 2]].permute(0, 2, 1, 3, 4).float())
    torch.testing.assert_close(actual[:, 2], expected)
    actual_grad = torch.autograd.grad(actual[:, 2].sum(), (k, v), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (k, v), allow_unused=True)
    for left, right in zip(actual_grad, expected_grad):
        if right is None:
            right = torch.zeros_like(left)
        torch.testing.assert_close(left, right)
    assert actual_grad[0][:, 0].abs().sum() > 0
    assert actual_grad[1][:, 0].abs().sum() > 0
    assert state.positions.item() == 0


@pytest.mark.parametrize("kind", ["persistent", "block"])
def test_future_epoch_slot_is_rejected_before_read(kind) -> None:
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    cache = HybridHOLACache(width=2, block_size=3, heads=1, rank_in=4, key_dim=2, value_dim=2)
    state = cache._empty(1, torch.device("cpu"))
    if kind == "persistent":
        valid = state.valid.clone(); valid[..., 0] = True
        positions = state.positions.clone(); positions[..., 0] = 0
        epochs = state.epochs.clone(); epochs[..., 0] = 1
        state = dataclasses.replace(state, valid=valid, positions=positions, epochs=epochs)
    else:
        valid = state.block_valid.clone(); valid[..., 0] = True
        positions = state.block_positions.clone(); positions[..., 0] = 0
        epochs = state.block_epochs.clone(); epochs[..., 0] = 1
        state = dataclasses.replace(state, block_valid=valid, block_positions=positions,
                                    block_epochs=epochs, block_count=torch.ones_like(state.block_count))
    q = torch.randn(1, 1, 1, 4, 2)
    with pytest.raises(ValueError, match=f"future {kind} epoch"):
        cache.scan(q, q, q, torch.ones(1, 1, 1), initial_state=state)
