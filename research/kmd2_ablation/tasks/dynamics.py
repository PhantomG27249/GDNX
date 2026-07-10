"""Causal drift and trajectory forecasting tasks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


DYNAMICS_SCHEMA_VERSION = "1.0.0"
DYNAMICS_TOKENS_PER_STEP = 2
DRIFT_SLOPE_RANGE = (0.02, 0.12)
TRAJECTORY_FREQUENCY_RANGE = (0.15, 0.45)
TRAJECTORY_AMPLITUDE_RANGE = (0.5, 1.5)


def _phase_for_step(step: int, steps: int) -> int:
    return min(2, (3 * step) // steps)


def _assemble_dynamics_batch(
    *,
    task: str,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    values: list[Tensor],
    example_ids: list[str],
    metadata: list[dict[str, Any]],
    extra_strata: Mapping[str, Tensor],
    change_steps: list[int],
) -> EpisodeBatch:
    steps = _operation_count(length, split)
    time = 1 + DYNAMICS_TOKENS_PER_STEP * steps
    continuous = torch.zeros(batch_size, time, 3, dtype=torch.float64)
    targets = torch.zeros(batch_size, time, 1, dtype=torch.float64)
    valid = torch.ones(batch_size, time, dtype=torch.bool)
    query_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries[:, 0] = True
    source_spans = torch.full((batch_size, time, 2), -1, dtype=torch.int64)
    phase = torch.full((batch_size, time), -1, dtype=torch.int64)
    causal_lag = torch.full((batch_size, time), -1, dtype=torch.int64)
    overshoot = torch.zeros(batch_size, time, dtype=torch.int64)
    for example, trajectory in enumerate(values):
        continuous[example, 0, 0] = trajectory[0]
        for step in range(steps):
            query_position = 1 + 2 * step
            observation_position = query_position + 1
            normalized_time = (step + 1) / steps
            continuous[example, query_position, 1] = 1.0
            continuous[example, query_position, 2] = normalized_time
            continuous[example, observation_position, 0] = trajectory[step + 1]
            continuous[example, observation_position, 2] = normalized_time
            targets[example, query_position, 0] = trajectory[step + 1]
            query_mask[example, query_position] = True
            loss_mask[example, query_position] = True
            source_spans[example, query_position] = torch.tensor(
                [query_position - 1, query_position]
            )
            phase_value = _phase_for_step(step, steps)
            phase[example, query_position] = phase_value
            target_step = step + 1
            if target_step >= change_steps[example]:
                causal_lag[example, query_position] = target_step - change_steps[example]
            if phase_value == 1 and target_step >= change_steps[example]:
                overshoot[example, query_position] = 1

    combined_strata = {
        "causal_lag": causal_lag,
        "overshoot": overshoot,
        "phase": phase,
        **extra_strata,
    }
    completed_metadata = tuple(
        {
            "actual_length": time,
            "logical_length": length,
            "operation_count": steps,
            "query_before_observation": True,
            "schema_version": DYNAMICS_SCHEMA_VERSION,
            "split_multiplier": steps // length,
            **item,
        }
        for item in metadata
    )
    return EpisodeBatch(
        task=task,
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
        strata=combined_strata,
        metadata=completed_metadata,
    )


def generate_drift_reversal(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    if params:
        raise ValueError("drift_reversal params must be empty")
    steps = _operation_count(length, split)
    reversal_step = max(1, steps // 3)
    low, high = DRIFT_SLOPE_RANGE
    values: list[Tensor] = []
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    change_steps: list[int] = []
    regime = torch.full((batch_size, 1 + 2 * steps), -1, dtype=torch.int64)
    for example in range(batch_size):
        identity, generator = _example_identity(
            "drift_reversal", seed, split, length, example
        )
        example_ids.append(identity)
        base = float(2.0 * torch.rand((), generator=generator).item() - 1.0)
        magnitude = low + (high - low) * float(
            torch.rand((), generator=generator).item()
        )
        direction = -1.0 if int(torch.randint(0, 2, (), generator=generator)) else 1.0
        slope = direction * magnitude
        reversal_value = base + slope * reversal_step
        trajectory = torch.empty(steps + 1, dtype=torch.float64)
        for step in range(steps + 1):
            if step <= reversal_step:
                trajectory[step] = base + slope * step
            else:
                trajectory[step] = reversal_value - slope * (step - reversal_step)
        values.append(trajectory)
        change_steps.append(reversal_step)
        metadata.append(
            {"base": base, "reversal_step": reversal_step, "slope": slope}
        )
        for step in range(steps):
            regime[example, 1 + 2 * step] = _phase_for_step(step, steps)
    return _assemble_dynamics_batch(
        task="drift_reversal",
        batch_size=batch_size,
        length=length,
        seed=seed,
        split=split,
        values=values,
        example_ids=example_ids,
        metadata=metadata,
        extra_strata={"regime": regime},
        change_steps=change_steps,
    )


def _linear_trajectory(
    *, base: float, velocity: float, steps: int, change_step: int, has_change: bool
) -> Tensor:
    result = torch.empty(steps + 1, dtype=torch.float64)
    change_value = base + velocity * change_step
    for step in range(steps + 1):
        if not has_change or step <= change_step:
            result[step] = base + velocity * step
        else:
            result[step] = change_value - 1.5 * velocity * (step - change_step)
    return result


def _sinusoidal_trajectory(
    *,
    base: float,
    amplitude: float,
    frequency: float,
    phase: float,
    new_phase: float,
    steps: int,
    change_step: int,
    has_change: bool,
) -> Tensor:
    result = torch.empty(steps + 1, dtype=torch.float64)
    value_at_change = base + amplitude * math.sin(phase + frequency * change_step)
    new_frequency = min(TRAJECTORY_FREQUENCY_RANGE[1], 1.35 * frequency)
    for step in range(steps + 1):
        if not has_change or step <= change_step:
            value = base + amplitude * math.sin(phase + frequency * step)
        else:
            elapsed = step - change_step
            value = value_at_change + amplitude * (
                math.sin(new_phase + new_frequency * elapsed) - math.sin(new_phase)
            )
        result[step] = value
    return result


def generate_trajectory(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    if params:
        raise ValueError("trajectory params must be empty")
    steps = _operation_count(length, split)
    change_step = max(1, steps // 3)
    values: list[Tensor] = []
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    change_steps: list[int] = []
    time = 1 + 2 * steps
    trajectory_type = torch.full((batch_size, time), -1, dtype=torch.int64)
    change_case = torch.full((batch_size, time), -1, dtype=torch.int64)
    for example in range(batch_size):
        identity, generator = _example_identity("trajectory", seed, split, length, example)
        example_ids.append(identity)
        mode = "linear" if example % 2 == 0 else "sinusoidal"
        has_change = (example // 2) % 2 == 1
        base = float(2.0 * torch.rand((), generator=generator).item() - 1.0)
        if mode == "linear":
            low, high = DRIFT_SLOPE_RANGE
            velocity = low + (high - low) * float(
                torch.rand((), generator=generator).item()
            )
            if int(torch.randint(0, 2, (), generator=generator)):
                velocity = -velocity
            trajectory = _linear_trajectory(
                base=base,
                velocity=velocity,
                steps=steps,
                change_step=change_step,
                has_change=has_change,
            )
            item_metadata: dict[str, Any] = {
                "base": base,
                "has_change_point": has_change,
                "mode": mode,
                "velocity": velocity,
            }
        else:
            amp_low, amp_high = TRAJECTORY_AMPLITUDE_RANGE
            freq_low, freq_high = TRAJECTORY_FREQUENCY_RANGE
            amplitude = amp_low + (amp_high - amp_low) * float(
                torch.rand((), generator=generator).item()
            )
            frequency = freq_low + (freq_high - freq_low) * float(
                torch.rand((), generator=generator).item()
            )
            phase = 2.0 * math.pi * float(torch.rand((), generator=generator).item())
            new_phase = 2.0 * math.pi * float(
                torch.rand((), generator=generator).item()
            )
            trajectory = _sinusoidal_trajectory(
                base=base,
                amplitude=amplitude,
                frequency=frequency,
                phase=phase,
                new_phase=new_phase,
                steps=steps,
                change_step=change_step,
                has_change=has_change,
            )
            item_metadata = {
                "amplitude": amplitude,
                "base": base,
                "frequency": frequency,
                "has_change_point": has_change,
                "mode": mode,
                "new_phase": new_phase,
                "phase": phase,
            }
        values.append(trajectory)
        effective_change_step = change_step if has_change else steps + 1
        change_steps.append(effective_change_step)
        item_metadata["change_step"] = change_step
        metadata.append(item_metadata)
        for step in range(steps):
            position = 1 + 2 * step
            trajectory_type[example, position] = 0 if mode == "linear" else 1
            change_case[example, position] = int(has_change)
    return _assemble_dynamics_batch(
        task="trajectory",
        batch_size=batch_size,
        length=length,
        seed=seed,
        split=split,
        values=values,
        example_ids=example_ids,
        metadata=metadata,
        extra_strata={
            "change_case": change_case,
            "trajectory_type": trajectory_type,
        },
        change_steps=change_steps,
    )


__all__ = [
    "DRIFT_SLOPE_RANGE",
    "DYNAMICS_SCHEMA_VERSION",
    "DYNAMICS_TOKENS_PER_STEP",
    "TRAJECTORY_AMPLITUDE_RANGE",
    "TRAJECTORY_FREQUENCY_RANGE",
    "generate_drift_reversal",
    "generate_trajectory",
]
