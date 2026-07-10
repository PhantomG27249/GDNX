"""Irregular-time analytic integration task for trapezoid ablations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


INTEGRATION_SCHEMA_VERSION = "1.0.0"
INTEGRATION_DECAY_RANGE = (0.05, 2.0)
INTEGRATION_FORCING_RANGE = (-1.5, 1.5)
INTEGRATION_DELTA_VALUES = (1e-7, 1e-4, 0.05, 0.5, 2.5)
INTEGRATION_RK4_STEPS = 4096
INTEGRATION_CURVATURE_THRESHOLDS = (0.25, 0.75)
_FORCING_CURVATURE_STENCIL = (0.0, 0.05, 0.10, 0.55, -0.80)


def analytic_piecewise_linear_step(
    h0: Tensor,
    u0: Tensor,
    u1: Tensor,
    delta: float,
    decay: Tensor,
) -> Tensor:
    """Solve ``dh/dt=-decay*h+u(t)`` for linear endpoint forcing."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    tensors = (h0, u0, u1, decay)
    if any(tensor.dtype != torch.float64 for tensor in tensors):
        raise TypeError("analytic integration inputs must be float64")
    if not (h0.shape == u0.shape == u1.shape == decay.shape):
        raise ValueError("analytic integration vectors must share shape")
    if torch.any(decay <= 0) or not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("analytic integration inputs must be finite with decay > 0")

    delta_tensor = torch.as_tensor(delta, dtype=torch.float64)
    z = decay * delta_tensor
    one_minus_exp = -torch.expm1(-z)
    z_minus_one_minus = z - one_minus_exp
    small = z.abs() < 1.0e-4
    z2 = z * z
    stable_series = z2 * (
        0.5
        + z
        * (
            -1.0 / 6.0
            + z * (1.0 / 24.0 + z * (-1.0 / 120.0 + z / 720.0))
        )
    )
    phi2_numerator = torch.where(small, stable_series, z_minus_one_minus)
    slope = (u1 - u0) / delta_tensor
    return (
        torch.exp(-z) * h0
        + u0 * (one_minus_exp / decay)
        + slope * (phi2_numerator / decay.square())
    )


def rk4_piecewise_linear_oracle(
    h0: Tensor,
    u0: Tensor,
    u1: Tensor,
    delta: float,
    decay: Tensor,
    *,
    steps: int = INTEGRATION_RK4_STEPS,
) -> Tensor:
    """High-resolution validation oracle; generators never train against it."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive int")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    state = h0.to(dtype=torch.float64).clone()
    start = u0.to(dtype=torch.float64)
    end = u1.to(dtype=torch.float64)
    rates = decay.to(dtype=torch.float64)
    dt = delta / steps

    def derivative(value: Tensor, elapsed: float) -> Tensor:
        forcing = start + (end - start) * (elapsed / delta)
        return -rates * value + forcing

    elapsed = 0.0
    for _ in range(steps):
        k1 = derivative(state, elapsed)
        k2 = derivative(state + 0.5 * dt * k1, elapsed + 0.5 * dt)
        k3 = derivative(state + 0.5 * dt * k2, elapsed + 0.5 * dt)
        k4 = derivative(state + dt * k3, elapsed + dt)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elapsed += dt
    return state


def _curvature_bin(current: Tensor, previous: Tensor, before_previous: Tensor) -> int:
    curvature = float((current - 2.0 * previous + before_previous).abs().mean())
    low, high = INTEGRATION_CURVATURE_THRESHOLDS
    if curvature < low:
        return 0
    if curvature < high:
        return 1
    return 2


def generate_irregular_integration(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"components"}
    if unknown:
        raise ValueError(
            f"unknown irregular_integration params: {', '.join(sorted(unknown))}"
        )
    components = params.get("components", 2)
    if type(components) is not int or not 1 <= components <= 16:
        raise ValueError("irregular_integration components must be an int in [1,16]")

    time = _operation_count(length, split)
    if time < 2:
        raise ValueError(
            "irregular_integration requires at least two actual timepoints "
            "to produce a supervised transition"
        )
    continuous = torch.zeros(
        batch_size, time, 2 * components + 1, dtype=torch.float64
    )
    targets = torch.zeros(batch_size, time, components, dtype=torch.float64)
    valid = torch.ones(batch_size, time, dtype=torch.bool)
    query_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries = torch.zeros(batch_size, time, dtype=torch.bool)
    source_spans = torch.full((batch_size, time, 2), -1, dtype=torch.int64)
    delta_bin = torch.full((batch_size, time), -1, dtype=torch.int64)
    curvature_bin = torch.full((batch_size, time), -1, dtype=torch.int64)
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    boundary_tokens = {0}
    if time > 2:
        boundary_tokens.add(time // 2)

    forcing_low, forcing_high = INTEGRATION_FORCING_RANGE
    decay_low, decay_high = INTEGRATION_DECAY_RANGE
    for example in range(batch_size):
        identity, generator = _example_identity(
            "irregular_integration",
            INTEGRATION_SCHEMA_VERSION,
            {"components": components},
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        decay = decay_low + (decay_high - decay_low) * torch.rand(
            components, generator=generator, dtype=torch.float64
        )
        base = 0.4 * torch.rand(
            components, generator=generator, dtype=torch.float64
        ) - 0.2
        signs = 2.0 * torch.randint(
            0, 2, (components,), generator=generator, dtype=torch.int64
        ).to(torch.float64) - 1.0
        forcing = torch.empty(time, components, dtype=torch.float64)
        segment_start = 0
        for token_index in range(time):
            if token_index in boundary_tokens:
                segment_start = token_index
                if token_index:
                    base = 0.4 * torch.rand(
                        components, generator=generator, dtype=torch.float64
                    ) - 0.2
                    signs = 2.0 * torch.randint(
                        0, 2, (components,), generator=generator, dtype=torch.int64
                    ).to(torch.float64) - 1.0
            local_index = token_index - segment_start
            offset = _FORCING_CURVATURE_STENCIL[
                local_index % len(_FORCING_CURVATURE_STENCIL)
            ]
            forcing[token_index] = base + signs * offset
        if torch.any(forcing < forcing_low) or torch.any(forcing > forcing_high):
            raise AssertionError("integration forcing stencil exceeded its declared range")
        continuous[example, :, :components] = forcing
        continuous[example, :, components + 1 :] = decay
        state = torch.zeros(components, dtype=torch.float64)
        previous = forcing[0]
        before_previous = forcing[0]
        segment_start = 0
        for token_index in range(time):
            current = forcing[token_index]
            if token_index in boundary_tokens:
                boundaries[example, token_index] = True
                continuous[example, token_index, components] = 0.0
                state.zero_()
                previous = current
                before_previous = current
                segment_start = token_index
                continue
            bin_index = (token_index + example) % len(INTEGRATION_DELTA_VALUES)
            delta = INTEGRATION_DELTA_VALUES[bin_index]
            continuous[example, token_index, components] = delta
            state = analytic_piecewise_linear_step(
                state, previous, current, delta, decay
            )
            targets[example, token_index] = state
            query_mask[example, token_index] = True
            loss_mask[example, token_index] = True
            source_spans[example, token_index] = torch.tensor(
                [segment_start, token_index], dtype=torch.int64
            )
            delta_bin[example, token_index] = bin_index
            curvature_bin[example, token_index] = _curvature_bin(
                current, previous, before_previous
            )
            before_previous = previous
            previous = current
        metadata.append(
            {
                "actual_length": time,
                "components": components,
                "decay": decay.tolist(),
                "logical_length": length,
                "operation_count": time,
                "schema_version": INTEGRATION_SCHEMA_VERSION,
                "split_multiplier": time // length,
            }
        )

    return EpisodeBatch(
        task="irregular_integration",
        split=split,
        seed=seed,
        example_ids=tuple(example_ids),
        input_ids=None,
        continuous_inputs=continuous,
        direct_factors=None,
        targets=targets,
        valid=valid,
        positions=_segment_positions(valid, boundaries),
        loss_mask=loss_mask,
        query_mask=query_mask,
        boundaries=boundaries,
        source_spans=source_spans,
        strata={"curvature_bin": curvature_bin, "delta_bin": delta_bin},
        metadata=tuple(metadata),
    )


__all__ = [
    "INTEGRATION_CURVATURE_THRESHOLDS",
    "INTEGRATION_DECAY_RANGE",
    "INTEGRATION_DELTA_VALUES",
    "INTEGRATION_FORCING_RANGE",
    "INTEGRATION_RK4_STEPS",
    "INTEGRATION_SCHEMA_VERSION",
    "analytic_piecewise_linear_step",
    "generate_irregular_integration",
    "rk4_piecewise_linear_oracle",
]
