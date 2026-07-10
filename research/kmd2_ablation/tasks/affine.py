"""Direct-factor affine associative regression bias control."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


AFFINE_SCHEMA_VERSION = "1.0.0"
AFFINE_DEFAULT_INPUT_DIM = 3
AFFINE_DEFAULT_OUTPUT_DIM = 2
AFFINE_INTERCEPT_RANGE = (0.4, 1.2)
AFFINE_QUERY_RANGE = (-1.5, 1.5)


def generate_affine(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"input_dim", "output_dim"}
    if unknown:
        raise ValueError(f"unknown affine params: {', '.join(sorted(unknown))}")
    input_dim = params.get("input_dim", AFFINE_DEFAULT_INPUT_DIM)
    output_dim = params.get("output_dim", AFFINE_DEFAULT_OUTPUT_DIM)
    if type(input_dim) is not int or not 1 <= input_dim <= 32:
        raise ValueError("affine input_dim must be an int in [1,32]")
    if type(output_dim) is not int or not 1 <= output_dim <= 32:
        raise ValueError("affine output_dim must be an int in [1,32]")
    pair_count = _operation_count(length, split)
    if pair_count < input_dim:
        raise ValueError("affine logical length must provide at least input_dim write pairs")
    time = 2 * pair_count + 1

    q = torch.zeros(batch_size, time, 1, 1, input_dim, dtype=torch.float64)
    k = torch.zeros(batch_size, time, 1, 1, input_dim, dtype=torch.float64)
    v = torch.zeros(batch_size, time, 1, 1, output_dim, dtype=torch.float64)
    decay = torch.ones(batch_size, time, 1, input_dim, dtype=torch.float64)
    beta_e = torch.zeros(batch_size, time, 1, 1, dtype=torch.float64)
    beta_w = torch.zeros(batch_size, time, 1, 1, dtype=torch.float64)
    out_mix = torch.ones(batch_size, time, 1, 1, dtype=torch.float64)
    write_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    query_role = torch.zeros(batch_size, time, dtype=torch.bool)
    targets = torch.zeros(batch_size, time, output_dim, dtype=torch.float64)
    valid = torch.ones(batch_size, time, dtype=torch.bool)
    query_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries[:, 0] = True
    source_spans = torch.full((batch_size, time, 2), -1, dtype=torch.int64)
    intercept_control = torch.full((batch_size, time), -1, dtype=torch.int64)
    role = torch.zeros(batch_size, time, dtype=torch.int64)
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    query_position = time - 1
    query_scale = float(_operation_count(1, split))
    for example in range(batch_size):
        identity, generator = _example_identity(
            "affine_associative_regression",
            AFFINE_SCHEMA_VERSION,
            {"input_dim": input_dim, "output_dim": output_dim},
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        matrix = torch.randn(
            output_dim, input_dim, generator=generator, dtype=torch.float64
        ) / math.sqrt(input_dim)
        zero_intercept = example % 2 == 0
        if zero_intercept:
            intercept = torch.zeros(output_dim, dtype=torch.float64)
        else:
            low, high = AFFINE_INTERCEPT_RANGE
            magnitude = low + (high - low) * torch.rand(
                output_dim, generator=generator, dtype=torch.float64
            )
            signs = 2.0 * torch.randint(
                0, 2, (output_dim,), generator=generator, dtype=torch.int64
            ).to(torch.float64) - 1.0
            intercept = magnitude * signs
        pair_inputs = 2.0 * torch.rand(
            pair_count, input_dim, generator=generator, dtype=torch.float64
        ) - 1.0
        for pair_index in range(pair_count):
            positive_position = 2 * pair_index
            negative_position = positive_position + 1
            x = pair_inputs[pair_index]
            y_positive = matrix @ x + intercept
            y_negative = matrix @ (-x) + intercept
            k[example, positive_position, 0, 0] = x
            k[example, negative_position, 0, 0] = -x
            v[example, positive_position, 0, 0] = y_positive
            v[example, negative_position, 0, 0] = y_negative
            beta_e[example, positive_position : negative_position + 1] = 1.0
            beta_w[example, positive_position : negative_position + 1] = 1.0
            write_mask[example, positive_position : negative_position + 1] = True
            role[example, positive_position : negative_position + 1] = 1
        query_low, query_high = AFFINE_QUERY_RANGE
        query_x = query_scale * (
            query_low
            + (query_high - query_low)
            * torch.rand(input_dim, generator=generator, dtype=torch.float64)
        )
        q[example, query_position, 0, 0] = query_x
        targets[example, query_position] = matrix @ query_x + intercept
        query_role[example, query_position] = True
        query_mask[example, query_position] = True
        loss_mask[example, query_position] = True
        source_spans[example, query_position] = torch.tensor([0, query_position])
        intercept_control[example, query_position] = int(zero_intercept)
        role[example, query_position] = 2
        metadata.append(
            {
                "actual_length": time,
                "input_dim": input_dim,
                "intercept": intercept.tolist(),
                "logical_length": length,
                "matrix": matrix.tolist(),
                "operation_count": pair_count,
                "output_dim": output_dim,
                "qk_bias": False,
                "query_range_scale": query_scale,
                "schema_version": AFFINE_SCHEMA_VERSION,
                "split_multiplier": pair_count // length,
                "zero_intercept_control": zero_intercept,
            }
        )
    return EpisodeBatch(
        task="affine_associative_regression",
        split=split,
        seed=seed,
        example_ids=tuple(example_ids),
        input_ids=None,
        continuous_inputs=None,
        direct_factors={
            "beta_e": beta_e,
            "beta_w": beta_w,
            "decay": decay,
            "k": k,
            "out_mix": out_mix,
            "q": q,
            "query_role": query_role,
            "v": v,
            "write_mask": write_mask,
        },
        targets=targets,
        valid=valid,
        positions=_segment_positions(valid, boundaries),
        loss_mask=loss_mask,
        query_mask=query_mask,
        boundaries=boundaries,
        source_spans=source_spans,
        strata={"intercept_control": intercept_control, "role": role},
        metadata=tuple(metadata),
    )


__all__ = [
    "AFFINE_DEFAULT_INPUT_DIM",
    "AFFINE_DEFAULT_OUTPUT_DIM",
    "AFFINE_INTERCEPT_RANGE",
    "AFFINE_QUERY_RANGE",
    "AFFINE_SCHEMA_VERSION",
    "generate_affine",
]
