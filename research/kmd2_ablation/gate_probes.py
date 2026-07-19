"""Deterministic, dependency-light scientific gate probes for preflight.

The probes execute the suite-owned recurrence on small tensors.  They are not
registry declarations: identity and active-effect evidence is recomputed for
the requested implementation on every production preflight without importing
Transformers or loading external model tensors.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from .config import ExperimentConfig
from .tiny_backend import TinyFactors, TinyKMD2Cell, TinyKMD2Config, TinyKMD2Model


_WARM_MECHANISMS = {
    "trapezoid",
    "bc_bias",
    "corrected_momentum",
    "causal_lookahead",
    "exact_cache",
}


def _probe_config(config: ExperimentConfig, spec: Any) -> TinyKMD2Config:
    arm_id = spec.arm_id
    is_cache = arm_id.startswith("exact_cache.") and arm_id not in {
        "exact_cache.off",
        "exact_cache.current_block_only",
    }
    cache = None
    if is_cache:
        cache = replace(
            config.cache,
            width=max(1, min(config.cache.width, 2)),
            block_size=2,
            storage_dtype="fp32",
        )
    r_out = 4 if arm_id.startswith("exact_cache.r_out_factorial") else 1
    mimo_rank = 2 if spec.mechanism == "true_mimo" else 1
    key_dim = 4 if spec.mechanism == "state_size" else 2
    return TinyKMD2Config(
        d_model=8,
        heads=1,
        dk=key_dim,
        dv=2,
        layers=1,
        vocab_size=11,
        d_ff=16,
        r_out=r_out,
        mimo_rank=mimo_rank,
        rotation_mode="current",
        convolution_gate_init=1.0,
        rotation_gate_init=1.0,
        channel_decay_gate_init=1.0,
        write_offset_gate_init=1.0,
        trapezoid=spec.mechanism == "trapezoid",
        cache=cache,
        corrected_momentum=spec.mechanism == "corrected_momentum",
        causal_lookahead=spec.mechanism == "causal_lookahead",
        bc_bias_mode="additive" if spec.mechanism == "bc_bias" else "none",
        selector_seed=1729,
        gdn2_decoupled=spec.mechanism == "gdn2_decoupled",
    )


def _factors(
    probe: TinyKMD2Config,
    *,
    gate_name: str | None = None,
    gate_value: float = 0.0,
    gate_requires_grad: bool = False,
    include_cache_coordinates: bool = False,
) -> TinyFactors:
    generator = torch.Generator().manual_seed(1729)
    steps = 4
    q_slots = probe.mimo_rank if probe.mimo_rank > 1 else probe.r_out
    write_slots = probe.mimo_rank
    q = torch.randn(1, steps, probe.heads, q_slots, probe.dk, generator=generator)
    k = torch.randn(
        1, steps, probe.heads, write_slots, probe.dk, generator=generator
    )
    v = torch.randn(
        1, steps, probe.heads, write_slots, probe.dv, generator=generator
    )
    decay = torch.sigmoid(
        torch.randn(1, steps, probe.heads, probe.dk, generator=generator)
    )
    beta_e = torch.sigmoid(
        torch.randn(1, steps, probe.heads, write_slots, generator=generator)
    )
    beta_w = torch.sigmoid(
        torch.randn(1, steps, probe.heads, write_slots, generator=generator)
    )
    out_mix = torch.full(
        (1, steps, probe.heads, q_slots), 1.0 / q_slots, dtype=torch.float32
    )
    optional: dict[str, torch.Tensor] = {}
    if gate_name is not None:
        gate = torch.full(
            (1, steps, probe.heads), gate_value, dtype=torch.float32
        )
        gate.requires_grad_(gate_requires_grad)
        optional[gate_name] = gate
    if include_cache_coordinates:
        optional["cache_q"] = q.detach().clone()
        optional["cache_k"] = k.detach().clone()
    return TinyFactors(
        q=q,
        k=k,
        v=v,
        decay=decay,
        beta_e=beta_e,
        beta_w=beta_w,
        out_mix=out_mix,
        valid=torch.ones(1, steps, dtype=torch.bool),
        positions=torch.arange(steps, dtype=torch.int64).view(1, steps),
        **optional,
    )


def _same_native_result(left: Any, right: Any) -> bool:
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in ("read", "final_state", "scores")
    )


def _finite_nonzero(tensor: torch.Tensor | None) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and bool(torch.isfinite(tensor).all())
        and bool(tensor.abs().sum() > 0)
    )


def _parameter_evidence(
    probe: TinyKMD2Config, spec: Any, *, gate_connected: bool
) -> tuple[list[str], list[str], list[str]]:
    model = TinyKMD2Model(probe, init_seed=1729)
    named = dict(model.named_parameters())

    def resolves(declared: str) -> bool:
        return any(
            actual == declared or actual.endswith("." + declared)
            for actual in named
        )

    missing = [name for name in spec.changed_parameters if not resolves(name)]
    gate_suffixes = {
        "trapezoid": ("rho_head",),
        "corrected_momentum": ("momentum_gamma",),
        "causal_lookahead": ("lookahead_rho",),
        "bc_bias": ("bc_q_amplitude", "bc_k_amplitude"),
        "exact_cache": ("cache_amplitude",),
    }.get(spec.mechanism, ())
    gate_parameters = {
        name: parameter
        for name, parameter in named.items()
        if any(name == suffix or name.endswith("." + suffix) for suffix in gate_suffixes)
    }
    frozen = [
        name
        for name, parameter in gate_parameters.items()
        if not parameter.requires_grad and bool(torch.count_nonzero(parameter) == 0)
    ]
    disconnected = [] if gate_connected else sorted(gate_parameters)
    return sorted(missing), disconnected, sorted(frozen)


def _warm_probe(config: ExperimentConfig, spec: Any) -> dict[str, Any]:
    probe = _probe_config(config, spec)
    native_config = replace(
        probe,
        trapezoid=False,
        cache=None,
        corrected_momentum=False,
        causal_lookahead=False,
        bc_bias_mode="none",
    )
    native_cell = TinyKMD2Cell(native_config)
    future = (
        torch.arange(1, 5, dtype=torch.float32).view(1, 4)
        if probe.cache is not None and probe.cache.score == "future_query_oracle"
        else None
    )
    pre_rotation = (
        probe.cache is not None and probe.cache.coordinate_frame == "pre_rotation"
    )

    if spec.mechanism == "trapezoid":
        gate_name = "trapezoid_rho"
    elif spec.mechanism == "corrected_momentum":
        gate_name = "momentum_gamma"
    elif spec.mechanism == "causal_lookahead":
        gate_name = "lookahead_rho"
    else:
        gate_name = None

    native_factors = _factors(native_config)
    zero_factors = _factors(
        probe,
        gate_name=gate_name,
        gate_value=0.0,
        include_cache_coordinates=pre_rotation,
    )
    native = native_cell(native_factors)
    cell = TinyKMD2Cell(probe)
    zero = cell(zero_factors, future_relevance=future)
    identity_passed = _same_native_result(native, zero)

    active_factors = _factors(
        probe,
        gate_name=gate_name,
        gate_value=0.4,
        gate_requires_grad=gate_name is not None,
        include_cache_coordinates=pre_rotation,
    )
    if spec.mechanism == "bc_bias":
        with torch.no_grad():
            cell.bc_q_amplitude.fill_(0.4)
            cell.bc_k_amplitude.fill_(0.4)
    elif probe.cache is not None:
        with torch.no_grad():
            cell.cache_amplitude.fill_(0.4)
    active = cell(active_factors, future_relevance=future)
    active_delta = max(
        float((active.read - native.read).detach().abs().max()),
        float((active.final_state - native.final_state).detach().abs().max()),
    )
    loss = active.read.square().sum() + active.final_state.square().sum()
    loss.backward()
    if gate_name is not None:
        gate_gradient = getattr(active_factors, gate_name).grad
    elif spec.mechanism == "bc_bias":
        gate_gradient = cell.bc_q_amplitude.grad
    else:
        gate_gradient = cell.cache_amplitude.grad
    gate_connected = _finite_nonzero(gate_gradient)
    missing, disconnected, frozen = _parameter_evidence(
        probe, spec, gate_connected=gate_connected
    )
    details = {
        "kind": "tiny_recurrence_tensor_probe",
        "mechanism": spec.mechanism,
        "identity_passed": identity_passed,
        "active_max_abs_delta": active_delta,
        "gate_gradient_finite_nonzero": gate_connected,
    }
    digest = hashlib.sha256(
        json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "available": True,
        "identity_passed": identity_passed,
        "active_effect_passed": active_delta > 0.0 and gate_connected,
        "missing_parameters": missing,
        "disconnected_parameters": disconnected,
        "frozen_zero_gates": frozen,
        "native_feature_present": False,
        "probe": details | {"sha256": digest},
    }


def _maximum_hybrid_probe(config: ExperimentConfig) -> dict[str, Any]:
    """Run the flagship gate through a converted production module and real cache carry."""
    from gdn3.kmd2_native import KMD2NativeAttn
    from .qwen_hybrid_four_state import FourStateHybridCache, QwenFourStateHybrid
    from .qwen_hybrid_shared import QwenSharedBraidHybrid, SharedHybridCache
    from .qwen_training import package_b_auxiliary_loss

    control_id = config.task.params.get("maximum_control")
    four_state = control_id == "package-b-hola-w64"
    if control_id not in {"package-a-hola-w64", "package-b-hola-w64"}:
        return {"available": False, "reason": "unsupported_maximum_control_probe"}
    prior_rout = os.environ.get("GDN3_KMD2_ROUT")
    os.environ["GDN3_KMD2_ROUT"] = "1"
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            model_config = SimpleNamespace(
                hidden_size=16, linear_num_value_heads=2, linear_num_key_heads=2,
                linear_key_head_dim=(8 if four_state else 4), linear_value_head_dim=4,
                linear_conv_kernel_dim=3, rms_norm_eps=1e-6,
            )
            native = KMD2NativeAttn(model_config, layer_idx=0)
            module = (QwenFourStateHybrid.from_native(native) if four_state
                      else QwenSharedBraidHybrid.from_native(native))
            identity_hidden = torch.randn(2, 4, 16)
            native_identity_output = native(identity_hidden)
            converted_identity_output = module(identity_hidden)
            native_parity_passed = torch.allclose(
                converted_identity_output, native_identity_output,
                atol=1e-6, rtol=1e-6,
            )
            hidden = torch.randn(2, 4, 16, requires_grad=True)
            valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
            boundary = torch.tensor([[True, False, False, False], [True, False, True, False]])
            neutral_forward = module(hidden, boundary=boundary, valid=valid, use_cache=True)
            neutral_repeat = module(hidden, boundary=boundary, valid=valid)
            deterministic_repeat = torch.equal(neutral_forward, neutral_repeat)
            cache = module.last_recurrent_cache
            expected_cache = FourStateHybridCache if four_state else SharedHybridCache
            if type(cache) is not expected_cache or cache.hola_state is None:
                raise RuntimeError("real hybrid probe did not produce its genuine cache carry")
            scan_output, scan_cache = module.scan(
                hidden.detach(), boundary=boundary, valid=valid, initial_cache=None
            )
            if type(scan_cache) is not expected_cache or not bool(torch.isfinite(scan_output).all()):
                raise RuntimeError("real hybrid scan did not execute")
            with torch.no_grad():
                module.components.cache_gate_logit.fill_(torch.logit(torch.tensor(0.25)))
            active = module(hidden, boundary=boundary, valid=valid, use_cache=True)
            active_delta = float((active - neutral_forward).detach().abs().max())
            loss = active.square().mean()
            loss.backward(retain_graph=four_state)
            connected = hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
            projection_gradients = [
                getattr(module.components, name).grad
                for name in ("q_weight", "k_weight", "v_weight", "erase_weight", "write_weight", "z_weight")
            ]
            connected = connected and all(_finite_nonzero(value) for value in projection_gradients)
            lane_specialization = True
            braid_staging = True
            if four_state:
                module.zero_grad(set_to_none=True)
                auxiliary, _ = package_b_auxiliary_loss(
                    module, lambda_spec=1.0, lambda_gate=0.1,
                    successful_updates=0, specialization_updates=1,
                )
                auxiliary.backward()
                q_gradient = module.components.q_weight.grad.reshape(4, -1)
                lane_specialization = torch.unique(q_gradient, dim=0).shape[0] == 4
                gate_gradient = module.components.trapezoid_proj.bias.grad
                # Execute the declared staging: a successful specialization/gate
                # update first breaks lane symmetry, then ordinary loss can reach
                # the token-dependent trapezoid on the following step.
                with torch.no_grad():
                    for name in ("q_weight", "k_weight", "v_weight", "erase_weight",
                                 "write_weight", "z_weight"):
                        parameter = getattr(module.components, name)
                        parameter.add_(parameter.grad, alpha=-0.1)
                module.zero_grad(set_to_none=True)
                routed, _ = module.scan(hidden.detach(), boundary=boundary, valid=valid)
                routed.square().mean().backward()
                trapezoid_gradient = module.components.trapezoid_proj.weight.grad
                braid_staging = _finite_nonzero(gate_gradient) and _finite_nonzero(trapezoid_gradient)
            hola = module.last_recurrent_cache.hola_state
            admissions = int(hola.admission_count.sum().cpu())
            details = {
                "kind": "real_converted_kmd2_hybrid_module",
                "control_id": control_id,
                "native_base_class": type(native).__name__,
                "converted_module_class": type(module).__name__,
                "forward_executed": True,
                "scan_executed": True,
                "cache_schema": "states" if four_state else "state",
                "hola_admissions": admissions,
                "hola_inclusive": admissions > 0 and bool((hola.block_positions >= 0).any()),
                "finite_connected_gradients": connected,
                "lane_specialization_gradients": bool(lane_specialization),
                "braid_staging_passed": bool(braid_staging),
                "native_parity_passed": native_parity_passed,
                "native_parity_expected": not four_state,
                "native_parity_atol": 1e-6,
                "native_parity_rtol": 1e-6,
                "neutral_repeat_deterministic": deterministic_repeat,
                "active_max_abs_delta": active_delta,
                "native_max_abs_delta": float(
                    (converted_identity_output - native_identity_output).detach().abs().max()
                ),
            }
            return {
                "available": True,
                # Package B intentionally changes decay horizons and enables a
                # .5-initialized token trapezoid, so source-output parity is not
                # its identity contract. Conversion determinism is.
                "identity_passed": deterministic_repeat and (native_parity_passed or four_state),
                "active_effect_passed": (
                    active_delta > 0 and connected and lane_specialization
                    and braid_staging and details["hola_inclusive"]
                ),
                "missing_parameters": (), "disconnected_parameters": (),
                "frozen_zero_gates": (), "native_feature_present": False,
                "probe": details,
            }
    finally:
        if prior_rout is None:
            os.environ.pop("GDN3_KMD2_ROUT", None)
        else:
            os.environ["GDN3_KMD2_ROUT"] = prior_rout


def measure_scientific_gates(config: ExperimentConfig, spec: Any) -> dict[str, Any]:
    """Return freshly measured identity/active evidence for one registered arm."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if getattr(spec, "evidence_kind", None) != "addition":
        return {"available": False, "reason": "arm_is_not_an_addition"}
    if spec.mechanism in _WARM_MECHANISMS:
        return _warm_probe(config, spec)
    if spec.mechanism in {"state_size", "true_mimo"}:
        probe = _probe_config(config, spec)
        missing, disconnected, frozen = _parameter_evidence(
            probe, spec, gate_connected=True
        )
        native_state = 1 * 1 * 2 * 2
        active_state = probe.heads * probe.dk * probe.dv
        if probe.mimo_rank > 1:
            active_state += probe.mimo_rank * (probe.dk + probe.dv)
        return {
            "available": True,
            "identity_passed": False,
            "active_effect_passed": active_state != native_state,
            "missing_parameters": missing,
            "disconnected_parameters": disconnected,
            "frozen_zero_gates": frozen,
            "native_feature_present": False,
            "probe": {
                "kind": "cold_redesign_state_shape_probe",
                "native_state_elements": native_state,
                "active_state_elements": active_state,
            },
        }
    if spec.mechanism == "maximum_hybrid":
        return _maximum_hybrid_probe(config)
    if spec.mechanism == "gdn2_decoupled":
        probe = _probe_config(config, spec)
        missing, disconnected, frozen = _parameter_evidence(
            probe, spec, gate_connected=True
        )
        model = TinyKMD2Model(probe, init_seed=1729)
        projector = model.blocks[0].projector
        cell = model.blocks[0].cell
        hidden = torch.randn(
            1, 4, probe.d_model, generator=torch.Generator().manual_seed(1731)
        )
        valid = torch.ones(1, 4, dtype=torch.bool)
        positions = torch.arange(4, dtype=torch.int64).view(1, 4)
        factors = projector(hidden, valid, positions)
        output = cell(factors)
        gradients = torch.autograd.grad(
            output.final_state.square().sum(),
            (projector.erase_proj.weight, projector.write_proj.weight),
        )
        connected = all(_finite_nonzero(gradient) for gradient in gradients)
        shape_correct = (
            factors.beta_e.shape == (1, 4, probe.heads, 1, probe.dk)
            and factors.beta_w.shape == (1, 4, probe.heads, 1, probe.dv)
        )
        details = {
            "kind": "gdn2_channelwise_recurrence_probe",
            "erase_gate_shape": list(factors.beta_e.shape),
            "write_gate_shape": list(factors.beta_w.shape),
            "independent_projection_gradients": connected,
        }
        return {
            "available": True,
            "identity_passed": False,
            "active_effect_passed": shape_correct and connected,
            "missing_parameters": missing,
            "disconnected_parameters": disconnected,
            "frozen_zero_gates": frozen,
            "native_feature_present": False,
            "probe": details,
        }
    return {"available": False, "reason": "unsupported_addition_probe"}


__all__ = ["measure_scientific_gates"]
