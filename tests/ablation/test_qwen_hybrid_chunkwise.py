from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import math
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch


def test_package_b_benchmark_cli_smoke_and_json_schema():
    script = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "kmd2_ablation"
        / "scripts"
        / "benchmark_qwen_chunkwise.py"
    )
    assert script.is_file(), "complete-Package-B benchmark CLI is missing"
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "inference",
            "--candidate",
            "auto",
            "--lengths",
            "32",
            "64",
            "128",
            "256",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--json",
        ],
        cwd=script.parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["schema_version"] == "1.0.0"
    assert report["control_id"] == "package-b-hola-w64"
    assert report["mode"] == "inference"
    assert report["candidate"] == "auto"
    assert [record["length"] for record in report["records"]] == [32, 64, 128, 256]
    for record in report["records"]:
        assert record["complete_module"] is True
        assert set(record["arms"]) == {
            "forced_authority",
            "current_auto",
            "candidate",
        }
        for arm in record["arms"].values():
            assert math.isfinite(arm["median_ms"]) and arm["median_ms"] > 0
            assert math.isfinite(arm["tokens_per_second"])
            assert arm["tokens_per_second"] > 0
            assert type(arm["peak_bytes"]) is int and arm["peak_bytes"] >= 0
            assert type(arm["incremental_peak_bytes"]) is int
            assert 0 <= arm["incremental_peak_bytes"] <= arm["peak_bytes"]
            assert arm["correct"] is True
        assert record["gate"]["correctness"] is True
        assert set(record["gate"]) >= {
            "throughput_gain",
            "memory_reduction",
            "throughput_win",
            "memory_win",
            "other_metric_within_tolerance",
            "passed",
        }


def _benchmark_cli_module():
    try:
        from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise
    except ImportError as error:
        pytest.fail(f"complete-Package-B benchmark CLI is missing: {error}")
    return benchmark_qwen_chunkwise


def _assert_training_tree_close(actual, expected, *, atol, rtol, path="value"):
    if is_dataclass(actual) or is_dataclass(expected):
        assert type(actual) is type(expected), path
        for field in fields(actual):
            _assert_training_tree_close(
                getattr(actual, field.name),
                getattr(expected, field.name),
                atol=atol,
                rtol=rtol,
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(actual, dict) or isinstance(expected, dict):
        assert isinstance(actual, dict) and isinstance(expected, dict), path
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_training_tree_close(
                actual[key], expected[key], atol=atol, rtol=rtol,
                path=f"{path}.{key}",
            )
        return
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_training_tree_close(
                left, right, atol=atol, rtol=rtol,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor), path
        if actual.is_floating_point():
            torch.testing.assert_close(
                actual, expected, atol=atol, rtol=rtol,
                msg=lambda message: f"{path}: {message}",
            )
        else:
            assert torch.equal(actual, expected), path
        return
    assert actual == expected, path


def _assert_training_tree_equal(actual, expected, path="value"):
    if is_dataclass(actual) or is_dataclass(expected):
        assert type(actual) is type(expected), path
        for field in fields(actual):
            _assert_training_tree_equal(
                getattr(actual, field.name), getattr(expected, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor), path
        assert torch.equal(actual, expected), path
        return
    assert actual == expected, path


def _assert_named_parameter_vjps_close(
    actual, expected, *, atol, rtol,
):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, parameter in actual_parameters.items():
        reference = expected_parameters[name]
        assert (parameter.grad is None) is (reference.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad,
                reference.grad,
                atol=atol,
                rtol=rtol,
                msg=lambda message, name=name: f"{name}.grad: {message}",
            )


def _training_vjp(module, hidden, cotangent):
    output, cache = module.scan(hidden)
    (output.float() * cotangent).mean().backward()
    return output, cache


def _training_continuation_vjp(
    module, hidden, cotangent, continuation_hidden, continuation_cotangent,
):
    output, cache = module.scan(hidden)
    continuation_output, continuation_cache = module.scan(
        continuation_hidden, initial_cache=cache,
    )
    loss = (
        (output.float() * cotangent).mean()
        + (continuation_output.float() * continuation_cotangent).mean()
    )
    loss.backward()
    return output, cache, continuation_output, continuation_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("tokens", (17, 64))
@pytest.mark.parametrize(
    ("dtype", "forward_atol", "forward_rtol", "vjp_atol", "vjp_rtol"),
    (
        (torch.float32, 3e-5, 3e-4, 2e-4, 2e-3),
        # Recurrent arithmetic remains FP32, while bf16 projections, mixer,
        # HOLA storage, and their parameter gradients incur bf16 rounding.
        (torch.bfloat16, 2e-3, 8e-3, 4e-3, 2e-2),
    ),
)
def test_private_package_b_rematerialized_training_full_vjp_fp32_bf16(
    tokens, dtype, forward_atol, forward_rtol, vjp_atol, vjp_rtol,
):
    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("rematerialized", "training")
    assert adapter.available
    torch.manual_seed(20260717 + tokens)
    device = torch.device("cuda")
    authority = benchmark._build_package_b(device, dtype)
    candidate = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    authority.train()
    candidate.train()
    assert authority._active("cache_policy") == "hola_exact_outer_w64"
    assert candidate._active("cache_policy") == "hola_exact_outer_w64"

    source = 0.1 * torch.randn(2, tokens, 1024, device=device, dtype=dtype)
    authority_hidden = source.detach().clone().requires_grad_(True)
    candidate_hidden = source.detach().clone().requires_grad_(True)
    cotangent = torch.randn_like(source).float()
    continuation_source = 0.1 * torch.randn(
        2, 1, 1024, device=device, dtype=dtype,
    )
    authority_continuation_hidden = (
        continuation_source.detach().clone().requires_grad_(True)
    )
    candidate_continuation_hidden = (
        continuation_source.detach().clone().requires_grad_(True)
    )
    continuation_cotangent = torch.randn_like(continuation_source).float()

    (
        authority_output,
        authority_cache,
        authority_continuation_output,
        authority_continuation_cache,
    ) = _training_continuation_vjp(
        authority,
        authority_hidden,
        cotangent,
        authority_continuation_hidden,
        continuation_cotangent,
    )
    with adapter.arm_context("candidate", integrated=False) as calls:
        (
            candidate_output,
            candidate_cache,
            candidate_continuation_output,
            candidate_continuation_cache,
        ) = _training_continuation_vjp(
            candidate,
            candidate_hidden,
            cotangent,
            candidate_continuation_hidden,
            continuation_cotangent,
        )
    # T=17 and T=64 are one eligible checkpoint segment. The private wrapper
    # executes once in forward and once in nonreentrant recomputation. The T=1
    # continuation is intentionally ineligible and stays on the complete loop.
    assert calls.count == 2

    torch.testing.assert_close(
        candidate_output,
        authority_output,
        atol=forward_atol,
        rtol=forward_rtol,
    )
    _assert_training_tree_close(
        candidate_cache,
        authority_cache,
        atol=forward_atol,
        rtol=forward_rtol,
        path="cache",
    )
    torch.testing.assert_close(
        candidate_continuation_output,
        authority_continuation_output,
        atol=forward_atol,
        rtol=forward_rtol,
    )
    _assert_training_tree_close(
        candidate_continuation_cache,
        authority_continuation_cache,
        atol=forward_atol,
        rtol=forward_rtol,
        path="continuation_cache",
    )
    torch.testing.assert_close(
        candidate_hidden.grad,
        authority_hidden.grad,
        atol=vjp_atol,
        rtol=vjp_rtol,
    )
    torch.testing.assert_close(
        candidate_continuation_hidden.grad,
        authority_continuation_hidden.grad,
        atol=vjp_atol,
        rtol=vjp_rtol,
    )
    _assert_named_parameter_vjps_close(
        candidate, authority, atol=vjp_atol, rtol=vjp_rtol,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_production_multisegment_bf16_gradients_cross_cache_and_match_authority(
    monkeypatch,
):
    """Exercise two WY segments, a fallback tail, and differentiable cache carry."""
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    benchmark = _benchmark_cli_module()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(20260811)
    authority = benchmark._build_package_b(device, dtype)
    candidate = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    candidate.use_liger_chunked_kernel = False
    authority.train()
    candidate.train()

    generator = benchmark._generator(device, 20260812)
    source = 0.1 * torch.randn(
        2, 129, benchmark.CAMPAIGN_HIDDEN,
        device=device, dtype=dtype, generator=generator,
    )
    continuation_source = 0.1 * torch.randn(
        2, 1, benchmark.CAMPAIGN_HIDDEN,
        device=device, dtype=dtype, generator=generator,
    )
    cotangent = torch.randn(
        source.shape, device=device, dtype=torch.float32, generator=generator,
    )
    continuation_cotangent = torch.randn(
        continuation_source.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    authority_hidden = source.detach().clone().requires_grad_(True)
    candidate_hidden = source.detach().clone().requires_grad_(True)
    authority_continuation = (
        continuation_source.detach().clone().requires_grad_(True)
    )
    candidate_continuation = (
        continuation_source.detach().clone().requires_grad_(True)
    )

    authority_result = _training_continuation_vjp(
        authority,
        authority_hidden,
        cotangent,
        authority_continuation,
        continuation_cotangent,
    )
    production = chunkwise.rematerialized_torch_chunk_four_state_segment
    production_calls = 0

    def counted_production(*args, **kwargs):
        nonlocal production_calls
        production_calls += 1
        return production(*args, **kwargs)

    monkeypatch.setattr(
        chunkwise,
        "rematerialized_torch_chunk_four_state_segment",
        counted_production,
    )
    candidate_result = _training_continuation_vjp(
        candidate,
        candidate_hidden,
        cotangent,
        candidate_continuation,
        continuation_cotangent,
    )
    # Two eligible 64-token segments execute once in forward and once during
    # non-reentrant checkpoint recomputation.  The one-token tail and the
    # continuation deliberately retain the token-loop authority fallback.
    assert production_calls == 4

    for name, actual, expected in zip(
        ("output", "cache", "continuation_output", "continuation_cache"),
        candidate_result,
        authority_result,
        strict=True,
    ):
        _assert_training_tree_close(
            actual, expected, atol=2e-3, rtol=8e-3, path=name,
        )
    for name, actual, expected in (
        ("input", candidate_hidden.grad, authority_hidden.grad),
        (
            "continuation_input",
            candidate_continuation.grad,
            authority_continuation.grad,
        ),
    ):
        assert actual is not None and expected is not None, name
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
        assert bool(actual.abs().gt(0).any()), name
        assert bool(expected.abs().gt(0).any()), name
        torch.testing.assert_close(
            actual, expected, atol=4e-3, rtol=2e-2,
            msg=lambda message, name=name: f"{name}.grad: {message}",
        )

    candidate_parameters = dict(candidate.named_parameters())
    authority_parameters = dict(authority.named_parameters())
    assert candidate_parameters.keys() == authority_parameters.keys()
    nonzero_gradient_names = set()
    for name, parameter in candidate_parameters.items():
        reference = authority_parameters[name]
        assert (parameter.grad is None) is (reference.grad is None), name
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.isfinite(reference.grad).all(), name
        if bool(reference.grad.abs().gt(0).any()):
            assert bool(parameter.grad.abs().gt(0).any()), name
            nonzero_gradient_names.add(name)
        torch.testing.assert_close(
            parameter.grad,
            reference.grad,
            atol=4e-3,
            rtol=2e-2,
            msg=lambda message, name=name: f"{name}.grad: {message}",
        )

    # Require live learning signals through every core differentiable subsystem,
    # not merely one arbitrary parameter somewhere in the layer.
    required_nonzero = {
        "components.q_weight",
        "components.k_weight",
        "components.v_weight",
        "components.erase_weight",
        "components.write_weight",
        "components.z_weight",
        "components.native_decay_weight",
        "components.native_A_log",
        "components.native_dt_bias",
        "components.native_decay_pair",
        "components.phase_proj.weight",
        "components.trapezoid_proj.weight",
        "components.output_mixer",
        "components.conv1d.weight",
        "components.norm.weight",
        "components.out_proj.weight",
        "rot_proj.weight",
    }
    assert required_nonzero <= nonzero_gradient_names, (
        "missing nonzero gradients: "
        + ", ".join(sorted(required_nonzero - nonzero_gradient_names))
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("tokens", (32, 64, 128, 256))
def test_production_rematerialized_full_module_exact_selection_opportunities(
    tokens, monkeypatch,
):
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    benchmark = _benchmark_cli_module()
    torch.manual_seed(20260730 + tokens)
    module = benchmark._build_package_b(torch.device("cuda"), torch.bfloat16)
    module.use_liger_chunked_kernel = False
    module.train()
    hidden = 0.1 * torch.randn(
        2, tokens, 1024, device="cuda", dtype=torch.bfloat16,
    )
    hidden.requires_grad_(True)
    production = chunkwise.rematerialized_torch_chunk_four_state_segment
    calls = 0

    def counted_production(*args, **kwargs):
        nonlocal calls
        calls += 1
        return production(*args, **kwargs)

    monkeypatch.setattr(
        chunkwise,
        "rematerialized_torch_chunk_four_state_segment",
        counted_production,
    )
    output, _cache = module.scan(hidden)
    output.float().square().mean().backward()

    assert calls == 2 * math.ceil(tokens / 64)


def _assert_optimizer_state_close(
    actual_optimizer,
    actual_module,
    expected_optimizer,
    expected_module,
    *,
    initial_parameters=None,
    gradient_atol,
    gradient_rtol,
    delta_atol,
    delta_rtol,
    state_atol,
    state_rtol,
):
    actual_parameters = dict(actual_module.named_parameters())
    expected_parameters = dict(expected_module.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    _assert_named_parameter_vjps_close(
        actual_module,
        expected_module,
        atol=gradient_atol,
        rtol=gradient_rtol,
    )
    if initial_parameters is not None:
        assert initial_parameters.keys() == actual_parameters.keys()
        for name in actual_parameters:
            actual_delta = actual_parameters[name].detach() - initial_parameters[name]
            expected_delta = expected_parameters[name].detach() - initial_parameters[name]
            torch.testing.assert_close(
                actual_delta,
                expected_delta,
                atol=delta_atol,
                rtol=delta_rtol,
                msg=lambda message, name=name: f"{name}.delta: {message}",
            )
    for name in actual_parameters:
        _assert_training_tree_close(
            actual_optimizer.state[actual_parameters[name]],
            expected_optimizer.state[expected_parameters[name]],
            atol=state_atol,
            rtol=state_rtol,
            path=f"optimizer.state.{name}",
        )
    actual_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in actual_optimizer.param_groups
    ]
    expected_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in expected_optimizer.param_groups
    ]
    _assert_training_tree_close(
        actual_groups,
        expected_groups,
        atol=state_atol,
        rtol=state_rtol,
        path="optimizer.param_groups",
    )


def test_optimizer_state_parity_rejects_zeroed_candidate_gradients():
    authority = torch.nn.Linear(2, 1, bias=False)
    candidate = copy.deepcopy(authority)
    authority_optimizer = torch.optim.AdamW(authority.parameters(), lr=1e-4)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-4)
    authority.weight.grad = torch.full_like(authority.weight, 1e-2)
    candidate.weight.grad = torch.zeros_like(candidate.weight)

    with pytest.raises(AssertionError, match="grad"):
        _assert_optimizer_state_close(
            candidate_optimizer,
            candidate,
            authority_optimizer,
            authority,
            gradient_atol=0.0,
            gradient_rtol=0.0,
            delta_atol=0.0,
            delta_rtol=0.0,
            state_atol=0.0,
            state_rtol=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_package_b_training_two_optimizer_steps_exact_state():
    benchmark = _benchmark_cli_module()
    torch.manual_seed(20260719)
    device = torch.device("cuda")
    dtype = torch.float32
    authority = benchmark._build_package_b(device, dtype)
    candidate = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    authority.train()
    candidate.train()
    authority_optimizer = torch.optim.AdamW(
        authority.parameters(), lr=1e-4, betas=(0.9, 0.95),
        weight_decay=0.01, foreach=False,
    )
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=1e-4, betas=(0.9, 0.95),
        weight_decay=0.01, foreach=False,
    )
    adapter = benchmark._resolve_candidate("rematerialized", "training")
    shared_initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in authority.named_parameters()
    }

    for step in range(2):
        generator = torch.Generator(device=device).manual_seed(20260720 + step)
        source = 0.1 * torch.randn(
            2, 17, 1024, device=device, dtype=dtype, generator=generator,
        )
        target = torch.randn(
            2, 17, 1024, device=device, dtype=torch.float32, generator=generator,
        )
        authority_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        authority_output, authority_cache = _training_vjp(
            authority, source.detach().clone().requires_grad_(True), target,
        )
        with adapter.arm_context("candidate", integrated=False) as calls:
            candidate_output, candidate_cache = _training_vjp(
                candidate, source.detach().clone().requires_grad_(True), target,
            )
        assert calls.count == 2, step
        torch.testing.assert_close(
            candidate_output, authority_output, atol=3e-5, rtol=3e-4,
        )
        _assert_training_tree_close(
            candidate_cache, authority_cache, atol=3e-5, rtol=3e-4,
            path=f"step[{step}].cache",
        )
        _assert_named_parameter_vjps_close(
            candidate, authority, atol=1e-11, rtol=3e-4,
        )
        authority_optimizer.step()
        candidate_optimizer.step()
        _assert_optimizer_state_close(
            candidate_optimizer,
            candidate,
            authority_optimizer,
            authority,
            initial_parameters=shared_initial_parameters,
            gradient_atol=1e-11,
            gradient_rtol=3e-4,
            delta_atol=2e-8,
            delta_rtol=3e-4,
            state_atol=2e-12,
            state_rtol=3e-4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_package_b_schema_checkpoint_resume_compatibility(tmp_path):
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_checkpoint import (
        QWEN_CHECKPOINT_SCHEMA_VERSION,
        QwenCheckpointMetadata,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )
    from research.kmd2_ablation.qwen_hybrid_four_state import (
        FourStateHybridCache,
        QwenFourStateHybrid,
    )
    from research.kmd2_ablation.qwen_variants import (
        maximum_control_contract,
        resolve_maximum_hybrid_variant,
    )

    benchmark = _benchmark_cli_module()
    assert benchmark.CONTROL_ID == "package-b-hola-w64"
    contract = maximum_control_contract(benchmark.CONTROL_ID)
    assert contract.control_id == "package-b-hola-w64"
    assert contract.module_kind == "package_b"
    assert contract.topology == "four_state"
    assert contract.input_rank == contract.output_rank == 4
    assert contract.cache_policy == "hola_exact_outer_w64"
    assert resolve_maximum_hybrid_variant(benchmark.CONTROL_ID) == {
        "control_id": "package-b-hola-w64",
        "architecture": "package_b",
        "mimo_rank": 4,
        "state_count": 4,
        "cache": "hola_exact_outer_w64",
        "convolution": True,
        "contract_sha256": (
            "2dfbdb9bb6fa13baa93573acac505d175f081ea1c8774c0846f1c065ce9a11b8"
        ),
        "trainable_components": (
            "q", "k", "v", "erase", "write", "output_mixer", "braid",
            "trapezoid", "affine_qk", "rotation", "cache",
        ),
    }
    assert is_dataclass(FourStateHybridCache)
    assert FourStateHybridCache.__dataclass_params__.frozen
    assert tuple(field.name for field in fields(FourStateHybridCache)) == (
        "states", "phase", "previous_key", "previous_value", "conv_tail",
        "has_history", "update_count", "hola_state",
    )
    assert tuple(inspect.signature(QwenFourStateHybrid.from_native).parameters) == (
        "native",
    )
    assert tuple(inspect.signature(QwenFourStateHybrid.scan).parameters) == (
        "self", "hidden_states", "boundary", "valid", "initial_cache",
    )
    assert tuple(inspect.signature(QwenFourStateHybrid.forward).parameters) == (
        "self", "hidden_states", "attention_mask", "boundary", "valid", "kwargs",
    )

    device = torch.device("cuda")
    torch.manual_seed(20260731)
    authority_layer = benchmark._build_package_b(device, torch.float32)
    authority_layer.train()
    assert authority_layer._active("cache_policy") == "hola_exact_outer_w64"
    assert authority_layer.maximum_control_contract == contract

    def schema(module, *, parameters):
        tensors = module.named_parameters() if parameters else module.state_dict().items()
        rows = [
            (name, tuple(tensor.shape), str(tensor.dtype), tensor.numel())
            for name, tensor in tensors
        ]
        digest = hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return len(rows), sum(row[3] for row in rows), digest

    assert schema(authority_layer, parameters=True) == (
        33,
        39_968_368,
        "b97116a3246127faac3be14c834dfea5a76b8b775d0e34b86285524b802f2635",
    )
    assert schema(authority_layer, parameters=False) == (
        36,
        42_589_812,
        "719e018fff895b006f11496052465d484cfcd68ed17dd80deaa9c3e3de2dab9b",
    )

    def checkpoint_optimizer(model):
        named = tuple(model.named_parameters())
        return torch.optim.AdamW(
            [{
                "params": [parameter for _name, parameter in named],
                "parameter_names": tuple(name for name, _parameter in named),
            }],
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            foreach=False,
        )

    authority = torch.nn.ModuleDict({"package_b": authority_layer})
    authority_optimizer = checkpoint_optimizer(authority)
    authority_scheduler = torch.optim.lr_scheduler.StepLR(
        authority_optimizer, step_size=1, gamma=0.9,
    )
    boundary = torch.tensor(
        [[True, False], [True, False]], device=device, dtype=torch.bool,
    )
    valid = torch.tensor(
        [[True, True], [True, False]], device=device, dtype=torch.bool,
    )

    def train_step(layer, optimizer, scheduler, hidden, target):
        optimizer.zero_grad(set_to_none=True)
        output, cache = layer.scan(hidden, boundary=boundary, valid=valid)
        (output.float() * target).mean().backward()
        optimizer.step()
        scheduler.step()
        return output, cache

    warm_generator = torch.Generator(device=device).manual_seed(20260732)
    warm_hidden = 0.05 * torch.randn(
        2, 2, 1024, device=device, generator=warm_generator,
    )
    warm_target = torch.randn(
        2, 2, 1024, device=device, generator=warm_generator,
    )
    train_step(
        authority_layer, authority_optimizer, authority_scheduler,
        warm_hidden, warm_target,
    )
    authority_optimizer.zero_grad(set_to_none=True)

    metadata = QwenCheckpointMetadata(
        job_id="package-b-schema-resume",
        pairing_id="b" * 64,
        arm="surprise",
        step=1,
        tokens_seen=3,
        source_hashes={"package-b-runtime": "1" * 64},
        data_identity={"sha256": "2" * 64, "row_count": 2},
        example_ids=("row0", "row1"),
        promotion_config={"width": 64, "block_size": 256, "policy": "exact_outer"},
        architecture_arm_id="gdn2-mimo-r4-braid-four-state-hola-w64",
        architecture_registry_sha256=registry_sha256(),
        auxiliary_identity={"maximum_control": "package-b-hola-w64"},
    )
    checkpoint = tmp_path / "package-b-hola-w64.pt"
    save_qwen_checkpoint(
        checkpoint,
        model=authority,
        optimizer=authority_optimizer,
        scheduler=authority_scheduler,
        metadata=metadata,
        target_module_names=("package_b",),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == QWEN_CHECKPOINT_SCHEMA_VERSION == 3
    assert payload["metadata"]["arm"] == "surprise"
    assert payload["metadata"]["auxiliary_identity"] == {
        "maximum_control": "package-b-hola-w64",
    }
    assert payload["target_module_names"] == ["package_b"]

    torch.manual_seed(20260733)
    resumed_layer = benchmark._build_package_b(device, torch.float32)
    strict_state = {
        name.removeprefix("package_b."): tensor
        for name, tensor in payload["model_state"].items()
    }
    incompatible = resumed_layer.load_state_dict(strict_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert schema(resumed_layer, parameters=True) == schema(
        authority_layer, parameters=True,
    )
    assert schema(resumed_layer, parameters=False) == schema(
        authority_layer, parameters=False,
    )

    resumed = torch.nn.ModuleDict({"package_b": resumed_layer})
    resumed_optimizer = checkpoint_optimizer(resumed)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=1, gamma=0.9,
    )
    resume_state = load_qwen_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expectation=QwenResumeExpectation.from_metadata(metadata),
        target_module_names=("package_b",),
    )
    assert (resume_state.step, resume_state.tokens_seen) == (1, 3)
    for name, authority_parameter in authority_layer.named_parameters():
        assert torch.equal(
            dict(resumed_layer.named_parameters())[name], authority_parameter,
        ), f"pre_step.{name}"
    _assert_optimizer_state_close(
        resumed_optimizer,
        resumed,
        authority_optimizer,
        authority,
        gradient_atol=0.0,
        gradient_rtol=0.0,
        delta_atol=0.0,
        delta_rtol=0.0,
        state_atol=0.0,
        state_rtol=0.0,
    )
    _assert_training_tree_close(
        resumed_scheduler.state_dict(), authority_scheduler.state_dict(),
        atol=0.0, rtol=0.0, path="pre_step.scheduler",
    )

    next_generator = torch.Generator(device=device).manual_seed(20260734)
    next_hidden = 0.05 * torch.randn(
        2, 2, 1024, device=device, generator=next_generator,
    )
    next_target = torch.randn(
        2, 2, 1024, device=device, generator=next_generator,
    )

    @contextmanager
    def deterministic_cuda_step():
        algorithms = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ):
                yield
        finally:
            torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)

    with deterministic_cuda_step():
        authority_optimizer.zero_grad(set_to_none=True)
        resumed_optimizer.zero_grad(set_to_none=True)
        authority_output, authority_cache = authority_layer.scan(
            next_hidden, boundary=boundary, valid=valid,
        )
        resumed_output, resumed_cache = resumed_layer.scan(
            next_hidden.clone(), boundary=boundary, valid=valid,
        )
        assert torch.equal(resumed_output, authority_output)
        _assert_training_tree_equal(resumed_cache, authority_cache, path="cache")
        (authority_output.float() * next_target).mean().backward()
        (resumed_output.float() * next_target.clone()).mean().backward()
        for name, authority_parameter in authority_layer.named_parameters():
            resumed_parameter = dict(resumed_layer.named_parameters())[name]
            assert (resumed_parameter.grad is None) is (authority_parameter.grad is None), name
            if authority_parameter.grad is not None:
                maximum = (resumed_parameter.grad - authority_parameter.grad).abs().max()
                assert torch.equal(resumed_parameter.grad, authority_parameter.grad), (
                    f"gradient.{name}: max_abs={float(maximum)}"
                )
        authority_optimizer.step()
        resumed_optimizer.step()
        authority_scheduler.step()
        resumed_scheduler.step()
    for name, authority_parameter in authority_layer.named_parameters():
        assert torch.equal(
            dict(resumed_layer.named_parameters())[name], authority_parameter,
        ), name
    _assert_optimizer_state_close(
        resumed_optimizer,
        resumed,
        authority_optimizer,
        authority,
        gradient_atol=0.0,
        gradient_rtol=0.0,
        delta_atol=0.0,
        delta_rtol=0.0,
        state_atol=0.0,
        state_rtol=0.0,
    )
    _assert_training_tree_close(
        resumed_scheduler.state_dict(), authority_scheduler.state_dict(),
        atol=0.0, rtol=0.0, path="scheduler",
    )


@pytest.mark.parametrize("mode", ("inference", "training", "decode"))
@pytest.mark.parametrize(
    "candidate",
    ("auto", "rematerialized", "decode", "direct-reads", "mixer", "hola"),
)
def test_benchmark_cli_parser_covers_all_modes_and_candidates(mode, candidate):
    benchmark = _benchmark_cli_module()
    options = benchmark._parse_options(
        [
            "--mode",
            mode,
            "--candidate",
            candidate,
            "--lengths",
            "32",
            "64",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ]
    )
    assert options.mode == mode
    assert options.candidate == candidate
    assert options.lengths == [32, 64]

    resolved = benchmark._resolve_candidate(candidate, mode)
    assert resolved.name == candidate
    expected_available = (
        candidate == "auto"
        or (candidate == "rematerialized" and mode == "training")
        or (candidate == "decode" and mode == "decode")
    )
    assert resolved.available is expected_available
    if not expected_available:
        assert "unavailable" in resolved.reason


def test_benchmark_cli_decode_batches_and_private_adapter_are_fail_closed():
    benchmark = _benchmark_cli_module()
    options = benchmark._parse_options(
        [
            "--mode",
            "decode",
            "--candidate",
            "auto",
            "--batch-sizes",
            "1",
            "2",
        ]
    )
    assert options.batch_sizes == [1, 2]

    class Holder:
        def dispatch(self, value):
            return f"auto:{value}"

        def production(self, value):
            return f"production:{value}"

    holder = Holder()
    original_dispatch = holder.dispatch
    original_production = holder.production

    def prototype(value):
        return f"private:{value}"

    def fallback(value):
        return f"fallback:{value}"

    adapter = benchmark._CandidateAdapter(
        name="test-only",
        modes=frozenset({"inference"}),
        private_target=(holder, "dispatch"),
        private_prototype=prototype,
        production_target=(holder, "production"),
        production_fallback=fallback,
    )
    with adapter.arm_context("candidate", integrated=False) as calls:
        assert holder.dispatch("x") == "private:x"
        assert calls.count == 1
    assert holder.dispatch("x") == original_dispatch("x")

    with adapter.arm_context("current_auto", integrated=True) as calls:
        assert holder.production("x") == "fallback:x"
        assert calls.count == 0
    assert holder.production("x") == original_production("x")

    with adapter.arm_context("forced_authority", integrated=True) as calls:
        assert holder.production("x") == "production:x"
        assert calls.count == 1
    assert holder.production("x") == original_production("x")

    with adapter.arm_context("candidate", integrated=True) as calls:
        assert holder.production("x") == "production:x"
        assert calls.count == 1
    assert holder.production("x") == original_production("x")


def test_benchmark_candidate_adapter_prefers_arm_specific_context():
    benchmark = _benchmark_cli_module()
    entered = []

    @contextmanager
    def arm_context(arm, integrated):
        entered.append((arm, integrated, "enter"))
        calls = benchmark._CallCounter()
        yield calls
        entered.append((arm, integrated, "exit"))

    adapter = benchmark._CandidateAdapter(
        name="test-only",
        modes=frozenset({"decode"}),
        arm_context_factory=arm_context,
    )
    for integrated in (False, True):
        for arm in benchmark.ARM_NAMES:
            with adapter.arm_context(arm, integrated=integrated) as calls:
                assert calls.count == 0

    expected = []
    for integrated in (False, True):
        for arm in benchmark.ARM_NAMES:
            expected.extend(((arm, integrated, "enter"), (arm, integrated, "exit")))
    assert entered == expected


def test_decode_adapter_contexts_isolate_arms_count_direct_and_restore(monkeypatch):
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from research.kmd2_ablation import qwen_hybrid_four_state as four_state
    from research.kmd2_ablation import qwen_hybrid_triton as triton

    benchmark = _benchmark_cli_module()

    def production_eligibility(*args, **kwargs):
        return True

    def private_eligibility(*args, **kwargs):
        return "private-eligible"

    def direct(*args, **kwargs):
        return "direct"

    def triton_eligibility(*args, **kwargs):
        return "triton-eligible"

    def chunk_segment(*args, **kwargs):
        return "chunk-segment"

    monkeypatch.setattr(
        chunkwise, "_can_use_package_b_decode_step", production_eligibility,
    )
    monkeypatch.setattr(
        chunkwise, "_can_use_torch_four_state_decode_step", private_eligibility,
    )
    monkeypatch.setattr(chunkwise, "_torch_four_state_decode_step", direct)
    monkeypatch.setattr(
        chunkwise, "torch_chunk_four_state_segment", chunk_segment,
    )
    monkeypatch.setattr(
        triton, "can_use_triton_four_state_segment", triton_eligibility,
    )
    originals = {
        "production": chunkwise._can_use_package_b_decode_step,
        "private": chunkwise._can_use_torch_four_state_decode_step,
        "direct": chunkwise._torch_four_state_decode_step,
        "chunk": chunkwise.torch_chunk_four_state_segment,
        "triton": triton.can_use_triton_four_state_segment,
        "cache": four_state._can_use_torch_chunk_with_cache,
    }
    adapter = benchmark._resolve_candidate("decode", "decode")

    for integrated in (False, True):
        for arm in benchmark.ARM_NAMES:
            with adapter.arm_context(arm, integrated=integrated) as calls:
                production_enabled = chunkwise._can_use_package_b_decode_step()
                expected_enabled = (
                    integrated and arm in ("forced_authority", "candidate")
                ) or (not integrated and arm == "forced_authority")
                assert production_enabled is expected_enabled
                assert calls.count == 0
                assert chunkwise._torch_four_state_decode_step() == "direct"
                assert calls.count == 1
                if not integrated and arm == "candidate":
                    assert four_state._can_use_torch_chunk_with_cache(True) is True
                    with torch.no_grad():
                        assert four_state._can_use_torch_chunk_with_cache(True) is True
                    assert triton.can_use_triton_four_state_segment() is False
                    assert (
                        chunkwise.can_use_torch_chunk_four_state_segment()
                        == "private-eligible"
                    )
                else:
                    assert four_state._can_use_torch_chunk_with_cache(True) is True
                    with torch.no_grad():
                        assert four_state._can_use_torch_chunk_with_cache(True) is False
                    assert (
                        triton.can_use_triton_four_state_segment()
                        == "triton-eligible"
                    )
            assert chunkwise._can_use_package_b_decode_step is originals["production"]
            assert chunkwise._can_use_torch_four_state_decode_step is originals["private"]
            assert chunkwise._torch_four_state_decode_step is originals["direct"]
            assert chunkwise.torch_chunk_four_state_segment is originals["chunk"]
            assert triton.can_use_triton_four_state_segment is originals["triton"]
            assert four_state._can_use_torch_chunk_with_cache is originals["cache"]

    with pytest.raises(RuntimeError, match="restore me"):
        with adapter.arm_context("candidate", integrated=False):
            raise RuntimeError("restore me")
    assert chunkwise._can_use_package_b_decode_step is originals["production"]
    assert chunkwise._torch_four_state_decode_step is originals["direct"]


def test_decode_adapter_contexts_serialize_all_arms():
    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("decode", "decode")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    failures = []

    def candidate_worker():
        try:
            with adapter.arm_context("candidate", integrated=False):
                first_entered.set()
                assert release_first.wait(timeout=5)
        except BaseException as error:
            failures.append(error)

    def baseline_worker():
        try:
            assert first_entered.wait(timeout=5)
            second_attempted.set()
            with adapter.arm_context("current_auto", integrated=False):
                second_entered.set()
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=candidate_worker)
    second = threading.Thread(target=baseline_worker)
    first.start()
    second.start()
    assert second_attempted.wait(timeout=5)
    blocked_while_candidate_active = not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert blocked_while_candidate_active
    assert second_entered.is_set()
    assert not first.is_alive() and not second.is_alive()
    assert failures == []


def test_decode_expected_calls_depend_on_batch_and_integration():
    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("decode", "decode")
    expected = adapter.expected_calls_per_sample
    assert expected is not None
    assert expected(32, 1, False) == 1
    assert expected(256, 2, False) == 1
    assert expected(32, 1, True) == 0
    assert expected(256, 2, True) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("integrated", (False, True))
def test_decode_benchmark_real_cuda_arm_isolation_smoke(integrated):
    benchmark = _benchmark_cli_module()
    arguments = [
        "--mode", "decode",
        "--candidate", "decode",
        "--batch-sizes", "1", "2",
        "--lengths", "32",
        "--warmup", "0",
        "--iterations", "1",
    ]
    if integrated:
        arguments.append("--integrated")
    report = benchmark._run_benchmark(benchmark._parse_options(arguments))

    assert len(report["records"]) == 2
    for record in report["records"]:
        batch = record["batch_size"]
        expected_candidate_calls = int(not integrated or batch == 2)
        assert record["gate"]["correctness"] is True
        assert record["arms"]["forced_authority"]["call_count"] == 0
        assert record["arms"]["current_auto"]["call_count"] == 0
        assert (
            record["arms"]["candidate"]["call_count"]
            == expected_candidate_calls
        )
        assert record["gate"]["claimable"] is bool(expected_candidate_calls)
        assert (
            record["gate"]["selection_proof"]["candidate_selected"]
            is bool(expected_candidate_calls)
        )


def test_production_rematerialized_cache_seam_is_training_only_and_scoped():
    from research.kmd2_ablation import qwen_hybrid_four_state as four_state

    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("rematerialized", "training")
    production = four_state._can_use_torch_chunk_with_cache
    assert production(False) is True
    assert production(True) is True
    with torch.no_grad():
        assert production(False) is False
        assert production(True) is False

    with adapter.arm_context("candidate", integrated=False) as calls:
        assert four_state._can_use_torch_chunk_with_cache(False) is True
        assert four_state._can_use_torch_chunk_with_cache(True) is True
        assert calls.count == 0

    assert four_state._can_use_torch_chunk_with_cache is production
    assert four_state._can_use_torch_chunk_with_cache(True) is True


def test_benchmark_callable_replacement_serializes_overlapping_threads():
    benchmark = _benchmark_cli_module()

    class Holder:
        def dispatch(self):
            return "original"

    holder = Holder()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    failures = []

    def first_worker():
        try:
            with benchmark._replace_callable(
                (holder, "dispatch"), lambda: "first", count=False,
            ):
                assert holder.dispatch() == "first"
                first_entered.set()
                assert release_first.wait(timeout=5)
        except BaseException as error:
            failures.append(error)

    def second_worker():
        try:
            assert first_entered.wait(timeout=5)
            second_attempted.set()
            with benchmark._replace_callable(
                (holder, "dispatch"), lambda: "second", count=False,
            ):
                second_entered.set()
                assert holder.dispatch() == "second"
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    second.start()
    assert second_attempted.wait(timeout=5)
    blocked_while_first_active = not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert blocked_while_first_active
    assert second_entered.is_set()
    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert holder.dispatch() == "original"


def test_benchmark_callable_replacement_is_reentrant_and_restores_nested_hooks():
    benchmark = _benchmark_cli_module()

    class Holder:
        def dispatch(self):
            return "original"

    holder = Holder()
    with benchmark._replace_callable(
        (holder, "dispatch"), lambda: "outer", count=False,
    ):
        assert holder.dispatch() == "outer"
        with benchmark._replace_callable(
            (holder, "dispatch"), lambda: "inner", count=False,
        ):
            assert holder.dispatch() == "inner"
        assert holder.dispatch() == "outer"
    assert holder.dispatch() == "original"


def test_benchmark_cli_decode_cache_population_accepts_staged_and_promoted_hola():
    benchmark = _benchmark_cli_module()

    def cache(*, counts, visible, updates=256, history=True):
        batch, heads = counts.shape
        return SimpleNamespace(
            has_history=torch.full((batch, 4), history, dtype=torch.bool),
            update_count=torch.full((batch,), updates, dtype=torch.int64),
            hola_state=SimpleNamespace(
                block_count=counts,
                valid=visible,
                epochs=torch.where(
                    visible,
                    torch.zeros(batch, heads, 2, dtype=torch.int64),
                    torch.full((batch, heads, 2), -1, dtype=torch.int64),
                ),
                current_epoch=torch.zeros(batch, heads, dtype=torch.int64),
            ),
        )

    staged = cache(
        counts=torch.ones(2, 3, dtype=torch.int64),
        visible=torch.zeros(2, 3, 2, dtype=torch.bool),
    )
    promoted = cache(
        counts=torch.zeros(2, 3, dtype=torch.int64),
        visible=torch.ones(2, 3, 2, dtype=torch.bool),
    )
    mixed = cache(
        counts=torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int64),
        visible=torch.tensor(
            [
                [[False, False], [True, False], [False, False]],
                [[True, False], [False, False], [True, False]],
            ]
        ),
    )
    empty = cache(
        counts=torch.zeros(2, 3, dtype=torch.int64),
        visible=torch.zeros(2, 3, 2, dtype=torch.bool),
    )
    no_recurrent_history = cache(
        counts=torch.ones(2, 3, dtype=torch.int64),
        visible=torch.zeros(2, 3, 2, dtype=torch.bool),
        updates=0,
        history=False,
    )

    assert benchmark._decode_cache_is_populated(staged)
    assert benchmark._decode_cache_is_populated(promoted)
    assert benchmark._decode_cache_is_populated(mixed)
    assert not benchmark._decode_cache_is_populated(empty)
    assert not benchmark._decode_cache_is_populated(no_recurrent_history)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_benchmark_cli_decode_length_256_accepts_promoted_hola_cache():
    script = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "kmd2_ablation"
        / "scripts"
        / "benchmark_qwen_chunkwise.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "decode",
            "--candidate",
            "auto",
            "--batch-sizes",
            "1",
            "2",
            "--lengths",
            "256",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--json",
        ],
        cwd=script.parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["mode"] == "decode"
    assert [record["batch_size"] for record in report["records"]] == [1, 2]
    assert [record["length"] for record in report["records"]] == [256, 256]
    assert all(record["gate"]["correctness"] for record in report["records"])


def _benchmark_arm_metrics(*, calls=0, throughput=100.0, incremental_peak=1000):
    return {
        "median_ms": 1.0,
        "tokens_per_second": throughput,
        "peak_bytes": 100_000,
        "incremental_peak_bytes": incremental_peak,
        "correct": True,
        "call_count": calls,
    }


def test_benchmark_cli_gate_requires_named_candidate_selection_proof():
    benchmark = _benchmark_cli_module()
    arms = {
        "forced_authority": _benchmark_arm_metrics(),
        "current_auto": _benchmark_arm_metrics(),
        "candidate": _benchmark_arm_metrics(calls=0, throughput=110.0),
    }

    missing_calls = benchmark._gate(arms, claimable=True, iterations=2)
    assert missing_calls["throughput_win"] is True
    assert missing_calls["selection_proof"] == {
        "required": True,
        "minimum_candidate_calls": 2,
        "candidate_calls": 0,
        "expected_candidate_calls": 2,
        "observed_candidate_calls": 0,
        "forced_authority_calls": 0,
        "current_auto_calls": 0,
        "candidate_selected": False,
        "other_arms_clean": True,
        "proved": False,
    }
    assert missing_calls["passed"] is False

    arms["candidate"]["call_count"] = 2
    selected = benchmark._gate(arms, claimable=True, iterations=2)
    assert selected["selection_proof"]["proved"] is True
    assert selected["passed"] is True

    arms["forced_authority"]["call_count"] = 1
    contaminated = benchmark._gate(arms, claimable=True, iterations=2)
    assert contaminated["selection_proof"]["other_arms_clean"] is False
    assert contaminated["passed"] is False

    aggregate = benchmark._gate(arms, claimable=False, iterations=2)
    assert aggregate["selection_proof"]["required"] is False
    assert aggregate["passed"] is False


def test_benchmark_cli_gate_requires_exact_candidate_selection_opportunities():
    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("rematerialized", "training")
    assert adapter.expected_calls_per_sample is not None
    assert adapter.expected_calls_per_sample(32, 1, False) == 2
    assert adapter.expected_calls_per_sample(64, 2, False) == 2
    assert adapter.expected_calls_per_sample(128, 1, True) == 4
    assert adapter.expected_calls_per_sample(256, 2, True) == 8
    arms = {
        "forced_authority": _benchmark_arm_metrics(),
        "current_auto": _benchmark_arm_metrics(),
        "candidate": _benchmark_arm_metrics(calls=7, throughput=110.0),
    }

    partial = benchmark._gate(
        arms, claimable=True, iterations=2, expected_candidate_calls=8,
    )
    assert partial["selection_proof"]["expected_candidate_calls"] == 8
    assert partial["selection_proof"]["observed_candidate_calls"] == 7
    assert partial["selection_proof"]["candidate_selected"] is False
    assert partial["passed"] is False

    arms["candidate"]["call_count"] = 9
    excess = benchmark._gate(
        arms, claimable=True, iterations=2, expected_candidate_calls=8,
    )
    assert excess["selection_proof"]["candidate_selected"] is False
    assert excess["passed"] is False

    arms["candidate"]["call_count"] = 8
    exact = benchmark._gate(
        arms, claimable=True, iterations=2, expected_candidate_calls=8,
    )
    assert exact["selection_proof"]["candidate_selected"] is True
    assert exact["selection_proof"]["proved"] is True
    assert exact["passed"] is True


@pytest.mark.parametrize(
    ("length", "expected_calls"),
    (
        (1, 0),
        (15, 0),
        (16, 2),
        (64, 2),
        (65, 2),
        (79, 2),
        (80, 4),
        (128, 4),
        (129, 4),
    ),
)
def test_rematerialized_training_expected_calls_follow_eligible_segments(
    length, expected_calls,
):
    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("rematerialized", "training")
    assert adapter.expected_calls_per_sample is not None
    assert adapter.expected_calls_per_sample(length, 1, False) == expected_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("length", "expected_calls", "claimable", "candidate_selected"),
    (
        (15, 0, False, False),
        (65, 2, True, True),
    ),
)
def test_private_training_benchmark_boundary_selection_smoke(
    length, expected_calls, claimable, candidate_selected,
):
    script = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "kmd2_ablation"
        / "scripts"
        / "benchmark_qwen_chunkwise.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "training",
            "--candidate",
            "rematerialized",
            "--lengths",
            str(length),
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--json",
        ],
        cwd=script.parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    record = report["records"][0]
    proof = record["gate"]["selection_proof"]
    assert record["arms"]["candidate"]["correct"] is True
    assert record["training_correctness"]["gradient_parity"]["candidate"] is True
    assert proof["expected_candidate_calls"] == expected_calls
    assert proof["observed_candidate_calls"] == expected_calls
    assert record["gate"]["claimable"] is claimable
    assert proof["candidate_selected"] is candidate_selected
    if claimable:
        assert proof["proved"] is True
    else:
        assert record["gate"]["passed"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_production_training_benchmark_integrated_selection_and_authority():
    benchmark = _benchmark_cli_module()
    options = benchmark._parse_options(
        [
            "--mode", "training",
            "--candidate", "rematerialized",
            "--integrated",
            "--lengths", "17",
            "--warmup", "0",
            "--iterations", "1",
        ]
    )
    report = benchmark._run_benchmark(options)
    record = report["records"][0]
    assert record["gate"]["correctness"] is True
    assert record["training_correctness"]["gradient_parity"] == {
        "forced_authority": True,
        "current_auto": True,
        "candidate": True,
    }
    assert record["arms"]["forced_authority"]["call_count"] == 0
    assert record["arms"]["current_auto"]["call_count"] == 0
    assert record["arms"]["candidate"]["call_count"] == 2
    proof = record["gate"]["selection_proof"]
    assert proof["candidate_selected"] is True
    assert proof["forced_authority_calls"] == 0
    assert proof["current_auto_calls"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_benchmark_cli_cuda_incremental_peak_excludes_resident_allocations():
    benchmark = _benchmark_cli_module()
    device = torch.device("cuda")
    resident = torch.empty(4 * 1024 * 1024, device=device, dtype=torch.float32)
    allocated_elements = 1024 * 1024

    class AllocatingModule(torch.nn.Module):
        def scan(self, hidden, *, initial_cache=None):
            del hidden, initial_cache
            return (
                torch.empty(allocated_elements, device=device, dtype=torch.float32),
                SimpleNamespace(),
            )

    runtime = benchmark._ArmRuntime(
        AllocatingModule(), torch.zeros(1, 1, 1, device=device), None
    )
    try:
        elapsed, peak, incremental_peak, calls = benchmark._sample(
            runtime,
            mode="inference",
            arm="candidate",
            adapter=benchmark._resolve_candidate("auto", "inference"),
            integrated=False,
            device=device,
        )
        assert elapsed > 0
        assert calls == 0
        assert incremental_peak >= allocated_elements * 4
        assert peak - incremental_peak >= resident.numel() * resident.element_size()
    finally:
        del resident, runtime
        torch.cuda.empty_cache()


def test_benchmark_cli_text_reports_raw_and_incremental_peak(capsys):
    benchmark = _benchmark_cli_module()
    arms = {
        arm: _benchmark_arm_metrics(incremental_peak=1234)
        for arm in ("forced_authority", "current_auto", "candidate")
    }
    benchmark._print_text(
        {
            "control_id": "package-b-hola-w64",
            "mode": "inference",
            "candidate": "auto",
            "integrated": False,
            "records": [
                {
                    "batch_size": 2,
                    "length": 32,
                    "arms": arms,
                    "gate": {"passed": False},
                }
            ],
        }
    )
    output = capsys.readouterr().out
    assert "peak=100000" in output
    assert "incremental=1234" in output


def test_benchmark_cli_errors_distinguish_unavailable_from_execution(
    monkeypatch, capsys,
):
    benchmark = _benchmark_cli_module()
    assert benchmark.main(["--candidate", "rematerialized", "--json"]) == 2
    unavailable_output = capsys.readouterr()
    unavailable_report = json.loads(unavailable_output.out)
    assert unavailable_output.err == ""
    assert unavailable_report["candidate_available"] is False
    assert unavailable_report["error_kind"] == "candidate_unavailable"

    def execution_error(_options):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(benchmark, "_run_benchmark", execution_error)
    assert benchmark.main(["--candidate", "auto", "--json"]) == 2
    execution_output = capsys.readouterr()
    execution_report = json.loads(execution_output.out)
    assert execution_output.err == ""
    assert execution_report["candidate_available"] is True
    assert execution_report["error_kind"] == "execution_error"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_benchmark_cli_training_smoke_proves_gradient_correctness():
    script = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "kmd2_ablation"
        / "scripts"
        / "benchmark_qwen_chunkwise.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "training",
            "--candidate",
            "auto",
            "--lengths",
            "32",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--json",
        ],
        cwd=script.parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["mode"] == "training"
    assert len(report["records"]) == 1
    record = report["records"][0]
    assert record["complete_module"] is True
    assert record["gate"]["correctness"] is True
    assert record["training_correctness"] == {
        "backward_executed": True,
        "gradient_parity": {
            "forced_authority": True,
            "current_auto": True,
            "candidate": True,
        },
    }
    for arm in record["arms"].values():
        assert arm["correct"] is True
        assert math.isfinite(arm["median_ms"]) and arm["median_ms"] > 0
        assert type(arm["peak_bytes"]) is int and arm["peak_bytes"] > 0
        assert type(arm["incremental_peak_bytes"]) is int
        assert 0 < arm["incremental_peak_bytes"] <= arm["peak_bytes"]


def _prototype():
    try:
        from research.kmd2_ablation.qwen_hybrid_chunkwise import (
            torch_chunk_four_state_segment,
        )
    except ImportError as error:
        pytest.fail(f"chunkwise recurrence prototype is missing: {error}")
    return torch_chunk_four_state_segment


def _rematerialized_prototype():
    try:
        from research.kmd2_ablation.qwen_hybrid_chunkwise import (
            _rematerialized_torch_chunk_four_state_segment,
        )
    except ImportError as error:
        pytest.fail(f"private rematerialized recurrence is missing: {error}")
    return _rematerialized_torch_chunk_four_state_segment


def _decode_step_prototype():
    try:
        from research.kmd2_ablation.qwen_hybrid_chunkwise import (
            _torch_four_state_decode_step,
        )
    except ImportError as error:
        pytest.fail(f"private exact decode step is missing: {error}")
    return _torch_four_state_decode_step


def test_chunkwise_module_docstring_names_training_and_decode_production_seams():
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    assert "Canonical CUDA training segments use the generalized-WY" in chunkwise.__doc__
    assert "narrow B2 cached-decode specialization" in chunkwise.__doc__
    assert "authoritative" in chunkwise.__doc__


def _authoritative_fast_decode_step(
    q, k, v, erase, write, gamma, lam, state,
    previous_key, previous_value, history, update_count,
):
    periods = update_count.new_tensor((1, 16, 64, 256))
    ticks = update_count[:, None].remainder(periods[None]).eq(0)
    tick = ticks[:, None, :, None, None]
    tick_vector = ticks[:, None, :, None]
    q_t, k_t = q[:, 0], k[:, 0]
    v_t, erase_t, write_t = v[:, 0], erase[:, 0], write[:, 0]
    gamma_t = gamma[:, 0, ..., None]

    decayed = gamma_t * state
    erased_key = erase_t * k_t
    memory = torch.einsum("bhrk,bhrkv->bhrv", erased_key, decayed)
    full_homogeneous = decayed - k_t[..., None] * memory[..., None, :]
    homogeneous = torch.where(tick, full_homogeneous, decayed)

    previous_decayed_key = gamma[:, 0] * previous_key
    previous_memory = torch.einsum(
        "bhrk,bhrk->bhr", erased_key, previous_decayed_key,
    )
    previous_transported_key = (
        previous_decayed_key - k_t * previous_memory[..., None]
    )
    previous_transported = (
        previous_transported_key[..., None] * previous_value[..., None, :]
    )
    current_value = write_t * v_t
    current_write = k_t[..., None] * current_value[..., None, :]
    one = state.new_ones(())
    effective_lambda = lam[:, 0, ..., None, None]
    effective_lambda = torch.where(
        history[:, None, :, None, None], effective_lambda, one,
    )
    tick_update = (
        (1.0 - effective_lambda) * previous_transported
        + effective_lambda * current_write
    )
    zero = state.new_zeros(())
    input_update = torch.where(tick, tick_update, zero)
    next_state = homogeneous + input_update
    innovation = next_state - decayed
    innovation_sq = innovation.square().sum((-2, -1)).detach()
    reads = torch.einsum("bhik,bhjkv->bhijv", q_t, next_state)[:, None]
    return (
        reads,
        next_state,
        torch.where(tick_vector, k_t, previous_decayed_key),
        torch.where(tick_vector, current_value, previous_value),
        innovation_sq[:, None],
        history | ticks,
        update_count + 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_private_decode_step_is_bit_exact_to_authoritative_fast_primitive():
    from tests.ablation.test_qwen_hybrid_triton import _inputs

    direct = _decode_step_prototype()
    device = torch.device("cuda")
    for count in (0, 1, 15, 16, 63, 64, 255, 256):
        for history in (False, True):
            source = _inputs(
                device, tokens=1, update_count=count,
                history=history, heads=16,
            )
            actual = direct(*source)
            expected = _authoritative_fast_decode_step(*source)
            for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
                assert torch.equal(got, want), (
                    f"output {index} differs at count={count}, history={history}"
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_package_b_decode_production_is_bit_exact_at_every_cms_phase(monkeypatch):
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    benchmark = _benchmark_cli_module()
    device = torch.device("cuda")
    torch.manual_seed(20260804)
    authority = benchmark._build_package_b(device, torch.bfloat16)
    production = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    source = 0.1 * torch.randn(
        2, 257, 1024, device=device, dtype=torch.bfloat16,
    )
    direct = chunkwise._torch_four_state_decode_step
    calls = 0

    def counted_direct(*args, **kwargs):
        nonlocal calls
        calls += 1
        return direct(*args, **kwargs)

    monkeypatch.setattr(
        chunkwise, "_torch_four_state_decode_step", counted_direct,
    )
    for index, count in enumerate((15, 0, 1, 16, 63, 64, 255, 256), start=1):
        with torch.no_grad():
            if count:
                _, populated = authority.scan(source[:, :count])
            else:
                populated = None
            authority_output, authority_cache = authority.scan(
                source[:, count:count + 1],
                initial_cache=copy.deepcopy(populated),
            )
            production_output, production_cache = production.scan(
                source[:, count:count + 1],
                initial_cache=copy.deepcopy(populated),
            )
        assert calls == index
        assert torch.equal(production_output, authority_output)
        _assert_training_tree_equal(
            production_cache, authority_cache,
            path=f"count_{count}.cache",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_private_decode_step_raw_and_full_package_b_parity():
    torch.manual_seed(20260801)
    direct = _decode_step_prototype()
    names = (
        "q", "k", "v", "erase", "write", "gamma", "lambda",
        "state", "previous_key", "previous_value",
    )
    for counts in (
        torch.tensor([0, 1, 15, 16], dtype=torch.int64),
        torch.tensor([63, 64, 255, 256], dtype=torch.int64),
    ):
        source = _random_inputs(
            batch=4, tokens=1, heads=2, key_dim=3, value_dim=2,
            counts=counts,
        )
        for has_history in (False, True):
            metadata = (
                torch.full((4, 4), has_history, dtype=torch.bool), counts,
            )
            direct_float = tuple(
                tensor.detach().clone().requires_grad_(True)
                for tensor in source[:10]
            )
            eager_float = tuple(
                tensor.detach().clone().requires_grad_(True)
                for tensor in source[:10]
            )
            dense_float = tuple(
                tensor.detach().clone().requires_grad_(True)
                for tensor in source[:10]
            )
            actual = direct(*direct_float, *metadata)
            eager_raw = _prototype()(*eager_float, *metadata)
            eager = (
                eager_raw[0], eager_raw[2], eager_raw[3], eager_raw[4],
                eager_raw[5], eager_raw[6], eager_raw[7],
            )
            dense_raw = _dense_history_oracle(*dense_float, *metadata)
            dense = (
                dense_raw[0], dense_raw[2], dense_raw[3], dense_raw[4],
                dense_raw[6], dense_raw[7], dense_raw[8],
            )

            assert len(actual) == 7
            for got, eager_want, dense_want in zip(
                actual[:5], eager[:5], dense[:5], strict=True,
            ):
                torch.testing.assert_close(got, eager_want, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(got, dense_want, atol=3e-6, rtol=3e-5)
            assert not actual[4].requires_grad
            assert torch.equal(actual[5], eager[5])
            assert torch.equal(actual[6], eager[6])
            assert torch.equal(actual[5], dense[5])
            assert torch.equal(actual[6], dense[6])

            cotangents = tuple(torch.randn_like(output) for output in actual[:4])
            actual_vjp = torch.autograd.grad(
                actual[:4], direct_float, grad_outputs=cotangents,
            )
            eager_vjp = torch.autograd.grad(
                eager[:4], eager_float, grad_outputs=cotangents,
            )
            dense_vjp = torch.autograd.grad(
                dense[:4], dense_float, grad_outputs=cotangents,
            )
            for name, got, eager_want, dense_want in zip(
                names, actual_vjp, eager_vjp, dense_vjp, strict=True,
            ):
                torch.testing.assert_close(
                    got, eager_want, atol=8e-5, rtol=8e-4,
                    msg=lambda message, name=name: f"{name} eager VJP: {message}",
                )
                torch.testing.assert_close(
                    got, dense_want, atol=8e-5, rtol=8e-4,
                    msg=lambda message, name=name: f"{name} dense VJP: {message}",
                )

    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("decode", "decode")
    assert adapter.available
    device = torch.device("cuda")
    for batch in (1, 2):
        authority = benchmark._build_package_b(device, torch.bfloat16)
        authority.force_torch_recurrence = True
        candidate = copy.deepcopy(authority)
        candidate.force_torch_recurrence = False
        prefix = 0.1 * torch.randn(
            batch, 256, 1024, device=device, dtype=torch.bfloat16,
        )
        token = 0.1 * torch.randn(
            batch, 1, 1024, device=device, dtype=torch.bfloat16,
        )
        with torch.no_grad():
            _, populated = authority.scan(prefix)
            authority_output, authority_cache = authority.scan(
                token, initial_cache=copy.deepcopy(populated),
            )
            with adapter.arm_context("candidate", integrated=False) as calls:
                candidate_output, candidate_cache = candidate.scan(
                    token, initial_cache=copy.deepcopy(populated),
                )
        assert calls.count == 1
        torch.testing.assert_close(
            candidate_output, authority_output, atol=2e-3, rtol=8e-3,
        )
        _assert_training_tree_close(
            candidate_cache, authority_cache, atol=2e-3, rtol=8e-3,
            path=f"batch_{batch}.cache",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("batch", (1, 2))
def test_private_decode_full_cache_phases_fallbacks_and_no_growth(batch):
    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("decode", "decode")
    device = torch.device("cuda")
    torch.manual_seed(20260802 + batch)
    authority = benchmark._build_package_b(device, torch.bfloat16)
    authority.force_torch_recurrence = True
    candidate = copy.deepcopy(authority)
    candidate.force_torch_recurrence = False
    source = 0.1 * torch.randn(
        batch, 257, 1024, device=device, dtype=torch.bfloat16,
    )

    def tensor_numel(value):
        if is_dataclass(value):
            return sum(tensor_numel(getattr(value, field.name)) for field in fields(value))
        return value.numel() if isinstance(value, torch.Tensor) else 0

    populated = None
    staged_populated = None
    start = 0
    for count in (1, 15, 16, 63, 64, 255, 256):
        with torch.no_grad():
            _, populated = authority.scan(
                source[:, start:count], initial_cache=populated,
            )
        start = count
        if count == 255:
            staged_populated = copy.deepcopy(populated)
        assert torch.equal(
            populated.update_count,
            torch.full((batch,), count, device=device, dtype=torch.int64),
        )
        before_numel = tensor_numel(populated)
        token = source[:, count:count + 1]
        with torch.no_grad():
            authority_output, authority_cache = authority.scan(
                token, initial_cache=copy.deepcopy(populated),
            )
            with adapter.arm_context("candidate", integrated=False) as calls:
                candidate_output, candidate_cache = candidate.scan(
                    token, initial_cache=copy.deepcopy(populated),
                )
        assert calls.count == 1
        torch.testing.assert_close(
            candidate_output, authority_output, atol=2e-3, rtol=8e-3,
        )
        _assert_training_tree_close(
            candidate_cache, authority_cache, atol=2e-3, rtol=8e-3,
            path=f"B{batch}.count{count}.cache",
        )
        assert tensor_numel(candidate_cache) == before_numel
        assert torch.equal(candidate_cache.update_count, populated.update_count + 1)
        assert torch.equal(
            candidate_cache.hola_state.next_position,
            populated.hola_state.next_position + 1,
        )

    token = source[:, 256:257]
    fallback_cases = [
        (
            torch.ones(batch, 1, device=device, dtype=torch.bool),
            torch.ones(batch, 1, device=device, dtype=torch.bool),
        ),
        (
            torch.zeros(batch, 1, device=device, dtype=torch.bool),
            torch.zeros(batch, 1, device=device, dtype=torch.bool),
        ),
    ]
    if batch == 2:
        fallback_cases.append((
            torch.tensor([[True], [False]], device=device),
            torch.ones(2, 1, device=device, dtype=torch.bool),
        ))
    for boundary, valid in fallback_cases:
        with torch.no_grad():
            authority_output, authority_cache = authority.scan(
                token, boundary=boundary, valid=valid,
                initial_cache=copy.deepcopy(populated),
            )
            with adapter.arm_context("candidate", integrated=False) as calls:
                candidate_output, candidate_cache = candidate.scan(
                    token, boundary=boundary, valid=valid,
                    initial_cache=copy.deepcopy(populated),
                )
        assert calls.count == 0
        torch.testing.assert_close(
            candidate_output, authority_output, atol=0.0, rtol=0.0,
        )
        _assert_training_tree_close(
            candidate_cache, authority_cache, atol=0.0, rtol=0.0,
            path="fallback.cache",
        )

    if batch == 2:
        assert staged_populated is not None
        nonuniform = copy.deepcopy(staged_populated)
        hola = nonuniform.hola_state
        block_count = hola.block_count.clone()
        block_valid = hola.block_valid.clone()
        block_count[0, 0] -= 1
        block_valid[0, 0, block_count[0, 0]] = False
        nonuniform = replace(
            nonuniform,
            hola_state=replace(
                hola, block_count=block_count, block_valid=block_valid,
            ),
        )
        with torch.no_grad():
            authority_output, authority_cache = authority.scan(
                source[:, 255:256], initial_cache=copy.deepcopy(nonuniform),
            )
            with adapter.arm_context("candidate", integrated=False) as calls:
                candidate_output, candidate_cache = candidate.scan(
                    source[:, 255:256], initial_cache=copy.deepcopy(nonuniform),
                )
        assert calls.count == 0
        torch.testing.assert_close(
            candidate_output, authority_output, atol=0.0, rtol=0.0,
        )
        _assert_training_tree_close(
            candidate_cache, authority_cache, atol=0.0, rtol=0.0,
            path="nonuniform.cache",
        )

    candidate.force_torch_recurrence = True
    with adapter.arm_context("candidate", integrated=False) as calls:
        with torch.no_grad():
            candidate.scan(token, initial_cache=copy.deepcopy(populated))
    assert calls.count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_package_b_decode_production_dispatch_call_selection(monkeypatch):
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    benchmark = _benchmark_cli_module()
    device = torch.device("cuda")
    torch.manual_seed(20260805)
    model = benchmark._build_package_b(device, torch.bfloat16)
    source = 0.1 * torch.randn(
        3, 33, 1024, device=device, dtype=torch.bfloat16,
    )
    caches = {}
    with torch.no_grad():
        for batch in (1, 2, 3):
            _, caches[batch] = model.scan(source[:batch, :32])

    direct = chunkwise._torch_four_state_decode_step
    calls = 0

    def counted_direct(*args, **kwargs):
        nonlocal calls
        calls += 1
        return direct(*args, **kwargs)

    monkeypatch.setattr(
        chunkwise, "_torch_four_state_decode_step", counted_direct,
    )

    token = source[:2, 32:33]
    model.force_torch_recurrence = True
    with torch.no_grad():
        authority_output, authority_cache = model.scan(
            token, initial_cache=copy.deepcopy(caches[2]),
        )
    assert calls == 0

    model.force_torch_recurrence = False
    with torch.no_grad():
        candidate_output, candidate_cache = model.scan(
            token, initial_cache=copy.deepcopy(caches[2]),
        )
    assert calls == 1
    torch.testing.assert_close(
        candidate_output, authority_output, atol=2e-3, rtol=8e-3,
    )
    _assert_training_tree_close(
        candidate_cache, authority_cache, atol=2e-3, rtol=8e-3,
        path="production.B2.cache",
    )

    fallback_cases = (
        (1, {}, False),
        (3, {}, False),
        (2, {"boundary": torch.ones(2, 1, device=device, dtype=torch.bool)}, False),
        (2, {"valid": torch.zeros(2, 1, device=device, dtype=torch.bool)}, False),
        (2, {}, True),
    )
    for batch, scan_kwargs, forced in fallback_cases:
        model.force_torch_recurrence = forced
        before = calls
        with torch.no_grad():
            model.scan(
                source[:batch, 32:33],
                initial_cache=copy.deepcopy(caches[batch]),
                **scan_kwargs,
            )
        assert calls == before

    model.force_torch_recurrence = False
    before = calls
    model.scan(
        token.detach().clone().requires_grad_(True),
        initial_cache=copy.deepcopy(caches[2]),
    )
    assert calls == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_package_b_decode_cuda_fp16_autocast_falls_back_bit_exactly(monkeypatch):
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from tests.ablation.test_qwen_hybrid_triton import _inputs

    device = torch.device("cuda")
    canonical = _inputs(
        device, tokens=1, update_count=15, history=True, heads=16,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        assert not chunkwise._can_use_torch_four_state_decode_step(*canonical)

    benchmark = _benchmark_cli_module()
    torch.manual_seed(20260808)
    authority = benchmark._build_package_b(device, torch.bfloat16)
    production = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    source = 0.1 * torch.randn(
        2, 17, 1024, device=device, dtype=torch.bfloat16,
    )
    with torch.no_grad():
        _, populated = authority.scan(source[:, :16])

    direct = chunkwise._torch_four_state_decode_step
    calls = 0

    def counted_direct(*args, **kwargs):
        nonlocal calls
        calls += 1
        return direct(*args, **kwargs)

    monkeypatch.setattr(
        chunkwise, "_torch_four_state_decode_step", counted_direct,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        authority_output, authority_cache = authority.scan(
            source[:, 16:17], initial_cache=copy.deepcopy(populated),
        )
        production_output, production_cache = production.scan(
            source[:, 16:17], initial_cache=copy.deepcopy(populated),
        )
    assert calls == 0
    assert torch.equal(production_output, authority_output)
    _assert_training_tree_equal(
        production_cache, authority_cache, path="autocast.cache",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_private_decode_grad_path_falls_back_with_full_input_parameter_vjp():
    benchmark = _benchmark_cli_module()
    adapter = benchmark._resolve_candidate("decode", "decode")
    device = torch.device("cuda")
    torch.manual_seed(20260804)
    authority = benchmark._build_package_b(device, torch.float32)
    authority.force_torch_recurrence = True
    candidate = copy.deepcopy(authority)
    prefix = 0.1 * torch.randn(1, 16, 1024, device=device)
    with torch.no_grad():
        _, populated = authority.scan(prefix)
    token = 0.1 * torch.randn(1, 1, 1024, device=device)
    authority_hidden = token.detach().clone().requires_grad_(True)
    candidate_hidden = token.detach().clone().requires_grad_(True)
    cotangent = torch.randn_like(token)

    authority_output, authority_cache = authority.scan(
        authority_hidden, initial_cache=copy.deepcopy(populated),
    )
    (authority_output * cotangent).sum().backward()
    with adapter.arm_context("candidate", integrated=False) as calls:
        candidate_output, candidate_cache = candidate.scan(
            candidate_hidden, initial_cache=copy.deepcopy(populated),
        )
        (candidate_output * cotangent).sum().backward()
    assert calls.count == 0
    torch.testing.assert_close(
        candidate_output, authority_output, atol=0.0, rtol=0.0,
    )
    _assert_training_tree_close(
        candidate_cache, authority_cache, atol=0.0, rtol=0.0,
        path="grad.cache",
    )
    torch.testing.assert_close(
        candidate_hidden.grad, authority_hidden.grad, atol=0.0, rtol=0.0,
    )
    _assert_named_parameter_vjps_close(
        candidate, authority, atol=0.0, rtol=0.0,
    )


def test_private_decode_step_validation_partial_vjp_and_autocast():
    from research.kmd2_ablation.qwen_hybrid_chunkwise import (
        _can_use_torch_four_state_decode_step,
    )

    direct = _decode_step_prototype()
    with pytest.raises(ValueError, match="exactly one token"):
        direct(*_hand_inputs(2))
    inputs = _hand_inputs(1)
    with pytest.raises(TypeError, match="FP32"):
        direct(inputs[0].bfloat16(), *inputs[1:])

    q_only = list(inputs[:10])
    q_only[0] = q_only[0].detach().clone().requires_grad_(True)
    q_only[7] = q_only[7].detach().clone().requires_grad_(True)
    result = direct(*q_only, *inputs[10:])
    unused, state_gradient = torch.autograd.grad(
        result[1].sum(), (q_only[0], q_only[7]), allow_unused=True,
        retain_graph=True,
    )
    assert unused is None
    assert state_gradient is not None
    q_gradient, = torch.autograd.grad(result[0].sum(), (q_only[0],))
    assert bool(q_gradient.ne(0).any())

    direct_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in inputs[:10]
    )
    eager_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in inputs[:10]
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = direct(*direct_float, *inputs[10:])
    eager_raw = _prototype()(*eager_float, *inputs[10:])
    assert not torch.is_autocast_enabled("cpu")
    eager = (
        eager_raw[0], eager_raw[2], eager_raw[3], eager_raw[4],
        eager_raw[5], eager_raw[6], eager_raw[7],
    )
    for got, want in zip(actual[:5], eager[:5], strict=True):
        torch.testing.assert_close(got, want, atol=2e-3, rtol=8e-3)
    cotangents = tuple(torch.randn_like(output) for output in actual[:4])
    actual_vjp = torch.autograd.grad(
        actual[:4], direct_float, grad_outputs=cotangents,
    )
    eager_vjp = torch.autograd.grad(
        eager[:4], eager_float, grad_outputs=cotangents,
    )
    for got, want in zip(actual_vjp, eager_vjp, strict=True):
        torch.testing.assert_close(got, want, atol=2e-3, rtol=8e-3)

    if torch.cuda.is_available():
        from tests.ablation.test_qwen_hybrid_triton import _inputs

        canonical = _inputs(
            torch.device("cuda"), tokens=1, update_count=15,
            history=True, heads=16,
        )
        assert not _can_use_torch_four_state_decode_step(*canonical)
        with torch.no_grad():
            assert _can_use_torch_four_state_decode_step(*canonical)
            assert not _can_use_torch_four_state_decode_step(
                canonical[0].bfloat16(), *canonical[1:],
            )
            assert not _can_use_torch_four_state_decode_step(
                *(tensor.repeat((3,) + (1,) * (tensor.ndim - 1))
                  for tensor in canonical),
            )


def test_rematerialized_eight_output_compatibility_shim_is_exact():
    benchmark = _benchmark_cli_module()
    source = _hand_inputs(17)

    def floating_copies():
        return tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in source[:10]
        )

    reference_inputs = floating_copies()
    actual_inputs = floating_copies()
    metadata = tuple(tensor.detach().clone() for tensor in source[10:])
    reference = _rematerialized_prototype()(*reference_inputs, *metadata)
    actual = benchmark._rematerialized_eight_output_compatibility(
        *actual_inputs, *metadata,
    )

    assert len(actual) == 8
    placeholder = actual[1]
    assert placeholder.shape == (0,)
    assert placeholder.dtype == actual[0].dtype
    assert placeholder.device == actual[0].device
    assert placeholder._base is None
    assert placeholder.untyped_storage().nbytes() == 0
    for got, expected in zip(
        (actual[0], *actual[2:]), reference, strict=True,
    ):
        if got.is_floating_point():
            torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)
        else:
            assert torch.equal(got, expected)

    actual_endpoints = (actual[0], actual[2], actual[3], actual[4])
    reference_endpoints = (
        reference[0], reference[1], reference[2], reference[3],
    )
    generator = torch.Generator().manual_seed(20260731)
    cotangents = tuple(
        torch.randn(
            output.shape, dtype=output.dtype, device=output.device,
            generator=generator,
        )
        for output in actual_endpoints
    )
    actual_gradients = torch.autograd.grad(
        actual_endpoints, actual_inputs, grad_outputs=cotangents,
    )
    reference_gradients = torch.autograd.grad(
        reference_endpoints, reference_inputs, grad_outputs=cotangents,
    )
    for got, expected in zip(
        actual_gradients, reference_gradients, strict=True,
    ):
        torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)


def _hand_inputs(tokens: int) -> tuple[torch.Tensor, ...]:
    B, H, R, K, V = 1, 1, 4, 1, 1
    time = torch.arange(tokens, dtype=torch.float32)[None, :, None, None, None]
    lane = torch.arange(R, dtype=torch.float32)[None, None, None, :, None]
    q = torch.ones(B, tokens, H, R, K)
    k = 0.2 + 0.03 * time + 0.01 * lane
    v = 0.5 + 0.1 * time + 0.02 * lane
    erase = torch.zeros_like(k)
    write = torch.ones(B, tokens, H, R, V)
    gamma = 0.7 + 0.04 * time + 0.02 * lane
    lam = torch.ones(B, tokens, H, R)
    state = (1.0 + 0.1 * lane[:, 0]).unsqueeze(-1)
    previous_key = (0.3 + 0.01 * lane[:, 0]).clone()
    previous_value = (0.4 + 0.02 * lane[:, 0]).clone()
    history = torch.zeros(B, R, dtype=torch.bool)
    update_count = torch.zeros(B, dtype=torch.int64)
    return (
        q, k, v, erase, write, gamma, lam, state,
        previous_key, previous_value, history, update_count,
    )


def _dense_history_oracle(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Direct token recurrence carrying a full KxV endpoint, never WY factors."""
    (
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    ) = inputs
    periods = update_count.new_tensor((1, 16, 64, 256))
    dense_history = previous_key[..., None] * previous_value[..., None, :]
    factor_key = previous_key
    factor_value = previous_value
    reads = []
    state_trace = []
    innovations = []
    innovation_sq = []
    count = update_count
    history_out = history.clone()
    for token in range(q.shape[1]):
        tick_lanes = count[:, None].remainder(periods[None]).eq(0)
        tick = tick_lanes[:, None, :, None, None]
        key_t = k[:, token]
        erased_key = erase[:, token] * key_t
        current_value = write[:, token] * v[:, token]
        current_write = key_t[..., None] * current_value[..., None, :]

        decayed = gamma[:, token, ..., None] * state
        memory = torch.einsum("bhrk,bhrkv->bhrv", erased_key, decayed)
        homogeneous = decayed - key_t[..., None] * memory[..., None, :]
        dense_history_decayed = gamma[:, token, ..., None] * dense_history
        history_memory = torch.einsum(
            "bhrk,bhrkv->bhrv", erased_key, dense_history_decayed
        )
        history_transported = (
            dense_history_decayed
            - key_t[..., None] * history_memory[..., None, :]
        )
        effective_lam = torch.where(
            history_out[:, None, :],
            lam[:, token],
            torch.ones_like(lam[:, token]),
        )[..., None, None]
        update = (
            (1.0 - effective_lam) * history_transported
            + effective_lam * current_write
        )
        next_state = torch.where(tick, homogeneous + update, decayed)
        innovation = next_state - decayed
        state = next_state
        dense_history = torch.where(tick, current_write, dense_history_decayed)

        factor_key_decayed = gamma[:, token] * factor_key
        tick_vector = tick_lanes[:, None, :, None]
        factor_key = torch.where(tick_vector, key_t, factor_key_decayed)
        factor_value = torch.where(tick_vector, current_value, factor_value)
        history_out = history_out | tick_lanes
        count = count + 1
        state_trace.append(state)
        innovations.append(innovation)
        reads.append(torch.einsum("bhik,bhjkv->bhijv", q[:, token], state))
        innovation_sq.append(innovation.square().sum((-2, -1)))
    return (
        torch.stack(reads, 1),
        torch.stack(state_trace, 1),
        state,
        factor_key,
        factor_value,
        dense_history,
        torch.stack(innovation_sq, 1),
        history_out,
        count,
        torch.stack(innovations, 1),
    )


def test_t1_uses_G_t_t_plus_1_identity_for_current_innovation():
    inputs = _hand_inputs(1)
    q, k, v, _, write, gamma, _, state, _, _, _, count = inputs
    (
        reads, state_trace, state_out, previous_key, previous_value,
        innovation_sq, history_out, count_out,
    ) = _prototype()(*inputs)

    current_value = write[:, 0] * v[:, 0]
    innovation = k[:, 0, ..., None] * current_value[..., None, :]
    expected_state = gamma[:, 0, ..., None] * state + innovation
    expected_reads = torch.einsum(
        "bhik,bhjkv->bhijv", q[:, 0], expected_state
    )

    torch.testing.assert_close(state_trace[:, 0], expected_state)
    torch.testing.assert_close(state_out, expected_state)
    torch.testing.assert_close(reads[:, 0], expected_reads)
    torch.testing.assert_close(previous_key, k[:, 0])
    torch.testing.assert_close(previous_value, current_value)
    torch.testing.assert_close(
        innovation_sq[:, 0], innovation.square().sum((-2, -1))
    )
    assert not innovation_sq.requires_grad
    assert torch.equal(history_out, torch.ones_like(history_out))
    assert torch.equal(count_out, count + 1)


def _assert_expanded_interval_products(tokens: int) -> None:
    inputs = _hand_inputs(tokens)
    q, k, v, _, write, gamma, _, state, _, _, _, count = inputs
    (
        reads, state_trace, state_out, previous_key, previous_value,
        innovation_sq, history_out, count_out,
    ) = _prototype()(*inputs)
    current_value = write * v
    raw_innovation = k[..., None] * current_value[..., None, :]
    tick = torch.zeros(1, tokens, 1, 4, 1, 1, dtype=torch.bool)
    tick[:, 0] = True
    tick[:, 1:, :, 0] = True
    innovation = torch.where(tick, raw_innovation, 0.0)

    # Literal expansions pin G_{t:j}=D_t...D_j and G_{t:t+1}=I.
    first = gamma[:, 0, ..., None] * state + innovation[:, 0]
    expected_trace = [first]
    if tokens >= 2:
        second = (
            gamma[:, 1, ..., None] * gamma[:, 0, ..., None] * state
            + gamma[:, 1, ..., None] * innovation[:, 0]
            + innovation[:, 1]
        )
        expected_trace.append(second)
    if tokens == 3:
        third = (
            gamma[:, 2, ..., None]
            * gamma[:, 1, ..., None]
            * gamma[:, 0, ..., None]
            * state
            + gamma[:, 2, ..., None]
            * gamma[:, 1, ..., None]
            * innovation[:, 0]
            + gamma[:, 2, ..., None] * innovation[:, 1]
            + innovation[:, 2]
        )
        expected_trace.append(third)
    expected_trace_tensor = torch.stack(expected_trace, 1)
    expected_reads = torch.einsum(
        "bthik,bthjkv->bthijv", q, expected_trace_tensor
    )

    torch.testing.assert_close(state_trace, expected_trace_tensor)
    torch.testing.assert_close(state_out, expected_trace_tensor[:, -1])
    torch.testing.assert_close(reads, expected_reads)
    torch.testing.assert_close(
        innovation_sq, innovation.square().sum((-2, -1))
    )

    slow_decay = gamma[:, 1:, :, 1:].prod(1)
    expected_key = k[:, -1].clone()
    expected_value = current_value[:, -1].clone()
    expected_key[:, :, 1:] = slow_decay * k[:, 0, :, 1:]
    expected_value[:, :, 1:] = current_value[:, 0, :, 1:]
    torch.testing.assert_close(previous_key, expected_key)
    torch.testing.assert_close(previous_value, expected_value)
    torch.testing.assert_close(
        previous_key[..., None] * previous_value[..., None, :],
        expected_key[..., None] * expected_value[..., None, :],
    )
    assert torch.equal(history_out, torch.ones_like(history_out))
    assert torch.equal(count_out, count + tokens)


def test_t2_uses_G_1_1_for_prior_and_G_1_2_for_current_innovation():
    _assert_expanded_interval_products(2)


def test_t3_keeps_both_interval_endpoints_across_an_off_tick_gap():
    _assert_expanded_interval_products(3)


def test_chunkwise_wy_matches_independent_dense_history_every_token():
    torch.manual_seed(20260717)
    B, T, H, R, K, V = 2, 7, 2, 4, 3, 2
    q = torch.randn(B, T, H, R, K)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, R, K), dim=-1)
    v = torch.randn(B, T, H, R, V)
    erase = torch.rand(B, T, H, R, K)
    write = torch.rand(B, T, H, R, V)
    gamma = 0.75 + 0.24 * torch.rand(B, T, H, R, K)
    lam = 0.1 + 0.8 * torch.rand(B, T, H, R)
    state = torch.randn(B, H, R, K, V) * 0.1
    previous_key = torch.randn(B, H, R, K) * 0.1
    previous_value = torch.randn(B, H, R, V) * 0.1
    history = torch.tensor(
        [[False, True, False, True], [True, False, True, False]]
    )
    update_count = torch.tensor([0, 15], dtype=torch.int64)
    inputs = (
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, update_count,
    )

    actual = _prototype()(*inputs)
    expected = _dense_history_oracle(*inputs)
    for got, want in zip(actual[:5], expected[:5], strict=True):
        torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        actual[3][..., None] * actual[4][..., None, :],
        expected[5],
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(actual[5], expected[6], atol=2e-6, rtol=2e-5)
    assert not actual[5].requires_grad
    assert torch.equal(actual[6], expected[7])
    assert torch.equal(actual[7], expected[8])

    periods = update_count.new_tensor((1, 16, 64, 256))
    positions = update_count[:, None] + torch.arange(T)[None]
    ticks = positions[:, :, None].remainder(periods[None, None]).eq(0)
    assert torch.equal(actual[5].eq(0).all(2), ~ticks)


def _random_inputs(
    *, batch: int, tokens: int, heads: int, key_dim: int, value_dim: int,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    shape_k = (batch, tokens, heads, 4, key_dim)
    shape_v = (batch, tokens, heads, 4, value_dim)
    q = torch.randn(shape_k)
    k = torch.nn.functional.normalize(torch.randn(shape_k), dim=-1)
    v = 0.2 * torch.randn(shape_v)
    erase = torch.rand(shape_k)
    write = torch.rand(shape_v)
    gamma = 0.97 + 0.029 * torch.rand(shape_k)
    lam = 0.1 + 0.8 * torch.rand(batch, tokens, heads, 4)
    state = 0.05 * torch.randn(batch, heads, 4, key_dim, value_dim)
    previous_key = 0.05 * torch.randn(batch, heads, 4, key_dim)
    previous_value = 0.05 * torch.randn(batch, heads, 4, value_dim)
    history = torch.tensor(
        [[False, True, False, True], [True, False, True, False],
         [False, False, True, True], [True, True, False, False]],
        dtype=torch.bool,
    )[:batch]
    return (
        q, k, v, erase, write, gamma, lam, state, previous_key,
        previous_value, history, counts,
    )


def _run_partitions(
    inputs: tuple[torch.Tensor, ...], widths: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    sequence = inputs[:7]
    state, previous_key, previous_value, history, count = inputs[7:]
    reads = []
    traces = []
    scores = []
    start = 0
    for width in widths:
        stop = start + width
        chunk = tuple(tensor[:, start:stop] for tensor in sequence)
        result = _prototype()(
            *chunk, state, previous_key, previous_value, history, count
        )
        reads.append(result[0])
        traces.append(result[1])
        scores.append(result[5])
        state, previous_key, previous_value = result[2:5]
        history, count = result[6:8]
        start = stop
    assert start == inputs[0].shape[1]
    return (
        torch.cat(reads, 1), torch.cat(traces, 1), state,
        previous_key, previous_value, torch.cat(scores, 1), history, count,
    )


def test_chunk_partitions_cross_every_cms_boundary_and_preserve_carries():
    torch.manual_seed(20260718)
    counts = torch.tensor([0, 15, 63, 255], dtype=torch.int64)
    inputs = _random_inputs(
        batch=4, tokens=257, heads=1, key_dim=2, value_dim=2,
        counts=counts,
    )
    whole = _prototype()(*inputs)
    dense = _dense_history_oracle(*inputs)
    for got, want in zip(whole[:5], dense[:5], strict=True):
        torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(
        whole[3][..., None] * whole[4][..., None, :],
        dense[5],
        atol=3e-5,
        rtol=3e-4,
    )
    torch.testing.assert_close(whole[5], dense[6], atol=3e-5, rtol=3e-4)
    assert torch.equal(whole[6], dense[7])
    assert torch.equal(whole[7], dense[8])

    for widths in (
        (32, 32, 32, 32, 32, 32, 32, 32, 1),
        (64, 64, 64, 64, 1),
        (128, 128, 1),
        (256, 1),
        (17, 47, 65, 128),
    ):
        partitioned = _run_partitions(inputs, widths)
        for got, want in zip(partitioned, whole, strict=True):
            if got.is_floating_point():
                torch.testing.assert_close(got, want, atol=4e-5, rtol=4e-4)
            else:
                assert torch.equal(got, want)

    periods = counts.new_tensor((1, 16, 64, 256))
    positions = counts[:, None] + torch.arange(257)[None]
    ticks = positions[:, :, None].remainder(periods[None, None]).eq(0)
    for period_index, period in enumerate((16, 64, 256), start=1):
        assert bool(ticks[..., period_index].any()), period


def test_hola_score_matches_detached_rms_over_exact_ticking_innovations():
    from research.kmd2_ablation.qwen_hybrid_hola import (
        four_state_normalized_update_score,
    )

    torch.manual_seed(20260719)
    counts = torch.tensor([0, 63], dtype=torch.int64)
    inputs = _random_inputs(
        batch=2, tokens=4, heads=2, key_dim=3, value_dim=2,
        counts=counts,
    )
    actual = _prototype()(*inputs)
    dense = _dense_history_oracle(*inputs)
    periods = counts.new_tensor((1, 16, 64, 256))
    positions = counts[:, None] + torch.arange(4)[None]
    ticks = positions[:, :, None].remainder(periods[None, None]).eq(0)
    tick_count = ticks.sum(-1).to(actual[5].dtype).clamp_min(1.0)
    score = torch.sqrt(actual[5].sum(-1) / tick_count[:, :, None]).detach()
    expected = torch.stack(
        [
            four_state_normalized_update_score(dense[9][:, token], ticks[:, token])
            for token in range(4)
        ],
        1,
    )
    torch.testing.assert_close(score, expected, atol=2e-6, rtol=2e-5)
    assert not actual[5].requires_grad
    assert not score.requires_grad


def test_chunkwise_wy_vjp_matches_independent_dense_history_oracle():
    torch.manual_seed(20260720)
    source = _random_inputs(
        batch=2, tokens=5, heads=2, key_dim=3, value_dim=2,
        counts=torch.tensor([0, 15], dtype=torch.int64),
    )
    custom_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    oracle_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    custom = _prototype()(*custom_float, *source[10:])
    oracle = _dense_history_oracle(*oracle_float, *source[10:])

    read_weight = torch.randn_like(custom[0])
    state_weight = torch.randn_like(custom[2])
    key_weight = torch.randn_like(custom[3])
    value_weight = torch.randn_like(custom[4])
    dense_weight = torch.randn_like(custom[3][..., None] * custom[4][..., None, :])
    custom_loss = (
        (custom[0] * read_weight).sum()
        + (custom[2] * state_weight).sum()
        + (custom[3] * key_weight).sum()
        + (custom[4] * value_weight).sum()
        + (custom[3][..., None] * custom[4][..., None, :] * dense_weight).sum()
    )
    oracle_loss = (
        (oracle[0] * read_weight).sum()
        + (oracle[2] * state_weight).sum()
        + (oracle[3] * key_weight).sum()
        + (oracle[4] * value_weight).sum()
        + (oracle[5] * dense_weight).sum()
    )
    custom_gradients = torch.autograd.grad(custom_loss, custom_float)
    oracle_gradients = torch.autograd.grad(oracle_loss, oracle_float)
    names = (
        "q", "k", "v", "erase", "write", "gamma", "lambda",
        "state", "previous_key", "previous_value",
    )
    for name, got, want in zip(
        names, custom_gradients, oracle_gradients, strict=True
    ):
        torch.testing.assert_close(
            got, want, atol=8e-5, rtol=8e-4,
            msg=lambda message, name=name: f"{name} VJP: {message}",
        )


@pytest.mark.parametrize("tokens", (17, 64))
@pytest.mark.parametrize("has_history", (False, True))
def test_rematerialized_chunk_recurrence_forward_vjp_and_saved_tensors(
    tokens, has_history,
):
    torch.manual_seed(20260722 + tokens + has_history)
    source = _random_inputs(
        batch=1, tokens=tokens, heads=1, key_dim=3, value_dim=2,
        counts=torch.tensor([15], dtype=torch.int64),
    )
    metadata = (
        torch.full((1, 4), has_history, dtype=torch.bool),
        source[11],
    )
    rematerialized_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    eager_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )

    packed = []

    def pack(tensor):
        packed.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = _rematerialized_prototype()(
            *rematerialized_float, *metadata
        )

    originals = (*rematerialized_float, *metadata)
    assert len(packed) == len(originals) == 12
    assert all(got is want for got, want in zip(packed, originals, strict=True))
    assert [(tensor.shape, tensor.dtype) for tensor in packed] == [
        (tensor.shape, tensor.dtype) for tensor in originals
    ]
    final_state = actual[1]
    assert final_state._base is None
    assert final_state.untyped_storage().nbytes() == (
        final_state.numel() * final_state.element_size()
    )

    eager_raw = _prototype()(*eager_float, *metadata)
    eager = (
        eager_raw[0], eager_raw[2], eager_raw[3], eager_raw[4],
        eager_raw[5], eager_raw[6], eager_raw[7],
    )
    dense_raw = _dense_history_oracle(*source[:10], *metadata)
    dense = (
        dense_raw[0], dense_raw[2], dense_raw[3], dense_raw[4],
        dense_raw[6], dense_raw[7], dense_raw[8],
    )
    for got, want in zip(actual[:5], eager[:5], strict=True):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    for got, want in zip(actual[:5], dense[:5], strict=True):
        torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-4)
    assert not actual[4].requires_grad
    assert torch.equal(actual[5], eager[5])
    assert torch.equal(actual[6], eager[6])
    assert torch.equal(actual[5], dense[5])
    assert torch.equal(actual[6], dense[6])

    names = (
        "q", "k", "v", "erase", "write", "gamma", "lambda",
        "state", "previous_key", "previous_value",
    )
    endpoint_cotangents = (
        torch.zeros_like(actual[0]),
        torch.randn_like(actual[1]),
        torch.randn_like(actual[2]),
        torch.randn_like(actual[3]),
    )
    mixed_cotangents = tuple(torch.randn_like(output) for output in actual[:4])
    for index, cotangents in enumerate((endpoint_cotangents, mixed_cotangents)):
        actual_gradients = torch.autograd.grad(
            actual[:4], rematerialized_float, grad_outputs=cotangents,
            retain_graph=index == 0,
        )
        eager_gradients = torch.autograd.grad(
            eager[:4], eager_float, grad_outputs=cotangents,
            retain_graph=index == 0,
        )
        for name, got, want in zip(
            names, actual_gradients, eager_gradients, strict=True
        ):
            # Backward reruns the identical eager operations, so exact FP32
            # equality is required rather than an approximation tolerance.
            torch.testing.assert_close(
                got, want, atol=0.0, rtol=0.0,
                msg=lambda message, name=name: f"{name} VJP: {message}",
            )


def test_rematerialized_chunk_recurrence_partial_gradients_are_exact():
    torch.manual_seed(20260723)
    source = _random_inputs(
        batch=1, tokens=17, heads=1, key_dim=3, value_dim=2,
        counts=torch.tensor([15], dtype=torch.int64),
    )

    q_only = list(source[:10])
    q_only[0] = q_only[0].detach().clone().requires_grad_(True)
    q_only_result = _rematerialized_prototype()(*q_only, *source[10:])
    q_gradient, = torch.autograd.grad(q_only_result[1].sum(), (q_only[0],))

    eager_q_only = list(source[:10])
    eager_q_only[0] = eager_q_only[0].detach().clone().requires_grad_(True)
    eager_q_only[7] = eager_q_only[7].detach().clone().requires_grad_(True)
    eager_q_only_result = _prototype()(*eager_q_only, *source[10:])
    eager_q_gradient, _ = torch.autograd.grad(
        eager_q_only_result[2].sum(), (eager_q_only[0], eager_q_only[7]),
        allow_unused=True, materialize_grads=True,
    )
    torch.testing.assert_close(
        q_gradient, eager_q_gradient, atol=0.0, rtol=0.0
    )

    rematerialized = list(source[:10])
    eager = list(source[:10])
    rematerialized[7] = (
        rematerialized[7].detach().clone().requires_grad_(True)
    )
    eager[7] = eager[7].detach().clone().requires_grad_(True)
    rematerialized_result = _rematerialized_prototype()(
        *rematerialized, *source[10:]
    )
    eager_result = _prototype()(*eager, *source[10:])
    read_cotangent = torch.randn_like(rematerialized_result[0])
    rematerialized_gradient, = torch.autograd.grad(
        rematerialized_result[0], (rematerialized[7],),
        grad_outputs=read_cotangent,
    )
    eager_gradient, = torch.autograd.grad(
        eager_result[0], (eager[7],), grad_outputs=read_cotangent,
    )
    torch.testing.assert_close(
        rematerialized_gradient, eager_gradient, atol=0.0, rtol=0.0
    )


def _assert_rematerialized_autocast_forward_and_vjp(
    device: torch.device, dtype: torch.dtype,
):
    torch.manual_seed(20260724)
    cpu_source = _random_inputs(
        batch=1, tokens=17, heads=1, key_dim=3, value_dim=2,
        counts=torch.tensor([15], dtype=torch.int64),
    )
    source = tuple(tensor.to(device) for tensor in cpu_source)
    assert all(tensor.dtype == torch.float32 for tensor in source[:10])
    rematerialized_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )
    eager_float = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source[:10]
    )

    with torch.autocast(device.type, dtype=dtype):
        actual = _rematerialized_prototype()(
            *rematerialized_float, *source[10:]
        )
    assert not torch.is_autocast_enabled(device.type)
    eager_raw = _prototype()(*eager_float, *source[10:])

    eager = (
        eager_raw[0], eager_raw[2], eager_raw[3], eager_raw[4],
        eager_raw[5], eager_raw[6], eager_raw[7],
    )
    for got, want in zip(actual[:5], eager[:5], strict=True):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    assert torch.equal(actual[5], eager[5])
    assert torch.equal(actual[6], eager[6])

    cotangents = tuple(torch.randn_like(output) for output in actual[:4])
    actual_gradients = torch.autograd.grad(
        actual[:4], rematerialized_float, grad_outputs=cotangents
    )
    eager_gradients = torch.autograd.grad(
        eager[:4], eager_float, grad_outputs=cotangents
    )
    names = (
        "q", "k", "v", "erase", "write", "gamma", "lambda",
        "state", "previous_key", "previous_value",
    )
    for name, got, want in zip(
        names, actual_gradients, eager_gradients, strict=True
    ):
        torch.testing.assert_close(
            got, want, atol=0.0, rtol=0.0,
            msg=lambda message, name=name: f"{name} autocast VJP: {message}",
        )


def test_rematerialized_chunk_recurrence_cpu_autocast_forward_and_vjp():
    _assert_rematerialized_autocast_forward_and_vjp(
        torch.device("cpu"), torch.bfloat16
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rematerialized_chunk_recurrence_cuda_autocast_forward_and_vjp():
    _assert_rematerialized_autocast_forward_and_vjp(
        torch.device("cuda"), torch.float16
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rematerialized_chunk_recurrence_cuda_peak_memory():
    torch.manual_seed(20260723)
    B, T, H, K, V = 1, 64, 2, 32, 64
    cpu_source = _random_inputs(
        batch=B, tokens=T, heads=H, key_dim=K, value_dim=V,
        counts=torch.tensor([15], dtype=torch.int64),
    )
    source = tuple(tensor.cuda() for tensor in cpu_source)
    cotangents = (
        torch.randn(B, T, H, 4, 4, V, device="cuda"),
        torch.randn(B, H, 4, K, V, device="cuda"),
        torch.randn(B, H, 4, K, device="cuda"),
        torch.randn(B, H, 4, V, device="cuda"),
    )

    def measure(rematerialized):
        floating = tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in source[:10]
        )
        metadata = (source[10].clone(), source[11].clone())
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if rematerialized:
            result = _rematerialized_prototype()(*floating, *metadata)
            outputs = result[:4]
        else:
            result = _prototype()(*floating, *metadata)
            outputs = (result[0], result[2], result[3], result[4])
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - baseline
        forward_peak = torch.cuda.max_memory_allocated() - baseline
        gradients = torch.autograd.grad(
            outputs, floating, grad_outputs=cotangents
        )
        assert all(gradient is not None for gradient in gradients)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        gradient_copies = tuple(
            gradient.detach().cpu() for gradient in gradients
        )
        final_state_storage_bytes = outputs[1].untyped_storage().nbytes()

        del gradients, outputs, result, metadata, floating
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "retained_after_forward": retained,
            "forward_phase_peak": forward_peak,
            "total_forward_backward_peak": peak,
            "final_state_storage_bytes": final_state_storage_bytes,
            "gradients": gradient_copies,
        }

    # Warm library workspaces and the caching allocator before either sample.
    measure(False)
    measure(True)
    raw = measure(False)
    rematerialized = measure(True)

    element_bytes = torch.empty((), dtype=torch.float32).element_size()
    diagnostic_trace_bytes = B * T * H * 4 * K * V * element_bytes
    final_state_bytes = B * H * 4 * K * V * element_bytes
    interval_factor_bytes = B * T * T * H * 4 * K * element_bytes
    assert raw["final_state_storage_bytes"] == diagnostic_trace_bytes
    assert rematerialized["final_state_storage_bytes"] == final_state_bytes
    assert (
        raw["retained_after_forward"]
        - rematerialized["retained_after_forward"]
        >= diagnostic_trace_bytes + interval_factor_bytes
    )
    assert (
        raw["forward_phase_peak"] - rematerialized["forward_phase_peak"]
        >= diagnostic_trace_bytes + interval_factor_bytes
    )
    # Exact rematerialization trades retained forward allocations for backward
    # recomputation; keep the end-to-end peaks measured without pretending that
    # recomputation itself is free or structurally proving a CUDA allocation.
    for sample in (raw, rematerialized):
        total_peak = sample["total_forward_backward_peak"]
        assert math.isfinite(total_peak) and total_peak > 0
    for got, want in zip(
        rematerialized["gradients"], raw["gradients"], strict=True
    ):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_chunkwise_recurrence_rejects_non_fp32_and_non_compact_shapes():
    inputs = _hand_inputs(1)
    with pytest.raises(TypeError, match="FP32"):
        _prototype()(inputs[0].bfloat16(), *inputs[1:])
    with pytest.raises(ValueError, match="exactly four lanes"):
        _prototype()(*(tensor[:, :, :, :3] for tensor in inputs[:7]),
                     *(tensor[:, :, :3] for tensor in inputs[7:10]),
                     inputs[10][:, :3], inputs[11])

    rounded = tuple(
        tensor.bfloat16().float() if tensor.is_floating_point() else tensor
        for tensor in inputs
    )
    result = _prototype()(*rounded)
    assert all(tensor.dtype == torch.float32 for tensor in result[:6])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_torch_chunk_dispatch_envelope_is_narrow_and_fail_closed():
    from research.kmd2_ablation.qwen_hybrid_chunkwise import (
        can_use_rematerialized_torch_chunk_four_state_segment,
        can_use_torch_chunk_four_state_segment,
    )
    from tests.ablation.test_qwen_hybrid_triton import _inputs

    one = _inputs(
        torch.device("cuda"), tokens=32, update_count=0,
        history=True, heads=8,
    )
    batch_two = tuple(
        tensor.repeat((2,) + (1,) * (tensor.ndim - 1)) for tensor in one
    )
    assert not can_use_torch_chunk_four_state_segment(*batch_two)
    with torch.no_grad():
        assert can_use_torch_chunk_four_state_segment(*batch_two)
        assert not can_use_torch_chunk_four_state_segment(
            batch_two[0].bfloat16(), *batch_two[1:]
        )
        assert not can_use_torch_chunk_four_state_segment(
            *(tensor[:, :15] for tensor in batch_two[:7]), *batch_two[7:]
        )
        batch_three = tuple(
            tensor.repeat((3,) + (1,) * (tensor.ndim - 1)) for tensor in one
        )
        assert not can_use_torch_chunk_four_state_segment(*batch_three)

    training = list(batch_two)
    training[0] = training[0].detach().clone().requires_grad_(True)
    assert can_use_rematerialized_torch_chunk_four_state_segment(*training)
    assert not can_use_rematerialized_torch_chunk_four_state_segment(*batch_two)
    with torch.no_grad():
        assert not can_use_rematerialized_torch_chunk_four_state_segment(
            *training
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("tokens", "cache_active"),
    ((32, True), (256, True), (32, False)),
)
def test_canonical_batch2_inference_dispatch_matches_complete_torch_path(
    monkeypatch, tokens, cache_active,
):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from research.kmd2_ablation.qwen_hybrid_four_state import (
        QwenFourStateHybrid,
    )
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    torch.manual_seed(20260721)
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
    native = native.to(device="cuda", dtype=torch.bfloat16)
    optimized = QwenFourStateHybrid.from_native(native)
    reference = copy.deepcopy(optimized)
    reference.force_torch_recurrence = True
    assert optimized._active("cache_policy", "hola_exact_outer_w64") != "none"
    if not cache_active:
        optimized.active_feature_flags = {"cache_policy": "none"}
        reference.active_feature_flags = {"cache_policy": "none"}

    calls = []
    real_prototype = chunkwise.torch_chunk_four_state_segment

    def counted_prototype(*args, **kwargs):
        calls.append(args[0].shape)
        return real_prototype(*args, **kwargs)

    monkeypatch.setattr(
        chunkwise, "torch_chunk_four_state_segment", counted_prototype
    )
    source = 0.1 * torch.randn(
        2, tokens, 1024, device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        optimized_output, optimized_cache = optimized.scan(source)
    assert calls == []
    calls_before_reference = len(calls)
    with torch.no_grad():
        reference_output, reference_cache = reference.scan(source)
    assert len(calls) == calls_before_reference

    torch.testing.assert_close(
        optimized_output, reference_output, atol=0.0, rtol=0.0
    )
    _assert_training_tree_close(
        optimized_cache, reference_cache, atol=0.0, rtol=0.0,
        path="cache",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_package_b_decode_token_256_promotion_is_bit_exact(monkeypatch):
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    benchmark = _benchmark_cli_module()
    device = torch.device("cuda")
    torch.manual_seed(20260809)
    authority = benchmark._build_package_b(device, torch.bfloat16)
    production = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    generator = benchmark._generator(
        device, benchmark.DEFAULT_SEED + 1009 * 256 + 97 * 2,
    )
    source = 0.1 * torch.randn(
        2, 256, benchmark.CAMPAIGN_HIDDEN,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    with torch.no_grad():
        _, populated = authority.scan(source[:, :255])
    assert bool(populated.hola_state.next_position.eq(255).all())
    assert bool(populated.hola_state.block_count.eq(255).all())

    direct = chunkwise._torch_four_state_decode_step
    calls = 0

    def counted_direct(*args, **kwargs):
        nonlocal calls
        calls += 1
        return direct(*args, **kwargs)

    monkeypatch.setattr(
        chunkwise, "_torch_four_state_decode_step", counted_direct,
    )
    with torch.no_grad():
        authority_output, authority_cache = authority.scan(
            source[:, 255:256], initial_cache=copy.deepcopy(populated),
        )
        production_output, production_cache = production.scan(
            source[:, 255:256], initial_cache=copy.deepcopy(populated),
        )
    assert calls == 1
    assert torch.equal(production_output, authority_output)
    for name in ("scores", "positions", "keys", "values"):
        assert torch.equal(
            getattr(production_cache.hola_state, name),
            getattr(authority_cache.hola_state, name),
        ), name
    _assert_training_tree_equal(
        production_cache, authority_cache, path="promotion.cache",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_package_b_hola_score_roundoff_keeps_production_on_authority():
    """Compare private WY and production with the forced token-loop authority."""
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from research.kmd2_ablation import qwen_hybrid_four_state as four_state

    benchmark = _benchmark_cli_module()
    seed = benchmark.DEFAULT_SEED
    device = torch.device("cuda")
    torch.manual_seed(20260810)
    authority = benchmark._build_package_b(device, torch.bfloat16)
    private_wy = copy.deepcopy(authority)
    production = copy.deepcopy(authority)
    authority.force_torch_recurrence = True
    generator = benchmark._generator(device, seed + 1009 * 256 + 97 * 2)
    prefix = 0.1 * torch.randn(
        2, 256, benchmark.CAMPAIGN_HIDDEN,
        device=device, dtype=torch.bfloat16, generator=generator,
    )

    recurrence = chunkwise.torch_chunk_four_state_segment
    rematerialized = chunkwise._rematerialized_torch_chunk_four_state_segment
    cache_seam = four_state._can_use_torch_chunk_with_cache
    with torch.no_grad():
        authority_output, authority_cache = authority.scan(prefix)
    with benchmark._rematerialized_training_context() as private_calls:
        with torch.no_grad():
            private_output, private_cache = private_wy.scan(prefix)
    assert private_calls.count == 4
    assert chunkwise.torch_chunk_four_state_segment is recurrence
    assert chunkwise._rematerialized_torch_chunk_four_state_segment is rematerialized
    assert four_state._can_use_torch_chunk_with_cache is cache_seam
    assert cache_seam(True) is True
    with torch.no_grad():
        assert cache_seam(True) is False

    with benchmark._replace_callable(
        (chunkwise, "torch_chunk_four_state_segment"), recurrence, count=True,
    ) as production_calls:
        with torch.no_grad():
            production_output, production_cache = production.scan(prefix)
    assert production_calls.count == 0

    # The generalized-WY path must remain a close mathematical candidate even
    # though its changed FP32 reduction order is not a bit-exact HOLA authority.
    torch.testing.assert_close(
        private_output, authority_output, atol=2e-3, rtol=8e-3,
    )
    for field in fields(authority_cache):
        if field.name != "hola_state":
            _assert_training_tree_close(
                getattr(private_cache, field.name),
                getattr(authority_cache, field.name),
                atol=2e-3,
                rtol=8e-3,
                path=f"private_cache.{field.name}",
            )
    authority_scores = authority_cache.hola_state.scores
    private_scores = private_cache.hola_state.scores
    assert torch.isfinite(authority_scores).all()
    assert torch.isfinite(private_scores).all()
    score_error = (private_scores - authority_scores).abs()
    assert bool(score_error.gt(0).any())
    assert float(score_error.max()) < 1e-5

    # Production is the exact ground-truth path: it must never dispatch WY and
    # must match every output and cache bit, including HOLA ordering and scores.
    torch.testing.assert_close(
        production_output, authority_output, atol=0.0, rtol=0.0,
    )
    _assert_training_tree_equal(
        production_cache, authority_cache, path="production_cache",
    )
