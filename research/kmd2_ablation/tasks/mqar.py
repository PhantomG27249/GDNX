"""Deterministic multi-query associative recall capacity task."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


MQAR_SCHEMA_VERSION = "1.0.0"
MQAR_DEFAULT_WIDTH = 8
MQAR_LOAD_FACTORS = (0.5, 1.0, 1.5)
MQAR_OVERWRITE_FRACTION = 0.25
MQAR_TOKENS = MappingProxyType(
    {"PAD": 0, "QUERY": 1, "FILLER": 2, "KEY_BASE": 100, "VALUE_BASE": 1000}
)


def _distance_bin(distance: int, width: int) -> int:
    if distance < width:
        return 0
    if distance <= 2 * width:
        return 1
    return 2


def generate_mqar(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"width", "overwrite_fraction"}
    if unknown:
        raise ValueError(f"unknown mqar params: {', '.join(sorted(unknown))}")
    width = params.get("width", MQAR_DEFAULT_WIDTH)
    if type(width) is not int or not 4 <= width <= 256:
        raise ValueError("mqar width must be an int in [4,256]")
    overwrite_fraction = params.get("overwrite_fraction", MQAR_OVERWRITE_FRACTION)
    if type(overwrite_fraction) not in (int, float) or not 0.0 <= overwrite_fraction <= 1.0:
        raise ValueError("mqar overwrite_fraction must be in [0,1]")
    overwrite_fraction = float(overwrite_fraction)
    queries = _operation_count(length, split)

    rows: list[dict[str, Any]] = []
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    for example in range(batch_size):
        identity, generator = _example_identity(
            "mqar",
            MQAR_SCHEMA_VERSION,
            {"overwrite_fraction": overwrite_fraction, "width": width},
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        load_bin = example % len(MQAR_LOAD_FACTORS)
        load = max(2, int(round(width * MQAR_LOAD_FACTORS[load_bin])))
        key_offset = int(torch.randint(0, max(1, 800 - load), (), generator=generator))
        keys = [MQAR_TOKENS["KEY_BASE"] + key_offset + index for index in range(load)]
        permutation = torch.randperm(load, generator=generator).tolist()
        keys = [keys[index] for index in permutation]
        values = [
            MQAR_TOKENS["VALUE_BASE"]
            + int(torch.randint(0, 900, (), generator=generator).item())
            for _ in range(load)
        ]
        tokens: list[int] = []
        targets: list[int] = []
        query_mask: list[bool] = []
        spans: list[tuple[int, int]] = []
        latest: dict[int, tuple[int, int]] = {}
        overwritten: set[int] = set()

        def append(token: int) -> int:
            position = len(tokens)
            tokens.append(token)
            targets.append(-100)
            query_mask.append(False)
            spans.append((-1, -1))
            return position

        for key, value in zip(keys, values):
            append(key)
            value_position = append(value)
            latest[key] = (value, value_position)
        overwrite_count = min(load, int(round(load * overwrite_fraction)))
        if overwrite_fraction > 0 and overwrite_count == 0:
            overwrite_count = 1
        for index in range(overwrite_count):
            key = keys[index]
            old_value = latest[key][0]
            new_value = MQAR_TOKENS["VALUE_BASE"] + (
                (old_value - MQAR_TOKENS["VALUE_BASE"] + 401 + index) % 900
            )
            append(key)
            value_position = append(new_value)
            latest[key] = (new_value, value_position)
            overwritten.add(key)

        query_keys = keys[:]
        if overwrite_count and load > overwrite_count:
            query_keys = [keys[0], keys[overwrite_count], *keys[1:overwrite_count], *keys[overwrite_count + 1 :]]
        while len(query_keys) < queries:
            query_keys.extend(keys)
        query_keys = query_keys[:queries]
        query_details: list[tuple[int, int, int, int]] = []
        for key in query_keys:
            append(key)
            query_position = append(MQAR_TOKENS["QUERY"])
            value, source_position = latest[key]
            targets[query_position] = value
            query_mask[query_position] = True
            spans[query_position] = (source_position, source_position + 1)
            distance = query_position - source_position
            query_details.append(
                (query_position, _distance_bin(distance, width), int(key in overwritten), distance)
            )
        rows.append(
            {
                "details": query_details,
                "load_bin": load_bin,
                "query_mask": query_mask,
                "spans": spans,
                "targets": targets,
                "tokens": tokens,
            }
        )
        metadata.append(
            {
                "actual_length": len(tokens),
                "load": load,
                "logical_length": length,
                "operation_count": queries,
                "overwrite_count": overwrite_count,
                "schema_version": MQAR_SCHEMA_VERSION,
                "split_multiplier": queries // length,
                "width": width,
            }
        )

    time = max(len(row["tokens"]) for row in rows)
    input_ids = torch.zeros(batch_size, time, dtype=torch.int64)
    targets_tensor = torch.full((batch_size, time), -100, dtype=torch.int64)
    valid = torch.zeros(batch_size, time, dtype=torch.bool)
    query_mask_tensor = torch.zeros(batch_size, time, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries = torch.zeros(batch_size, time, dtype=torch.bool)
    source_spans = torch.full((batch_size, time, 2), -1, dtype=torch.int64)
    distance_bin_tensor = torch.full((batch_size, time), -1, dtype=torch.int64)
    load_bin_tensor = torch.full((batch_size, time), -1, dtype=torch.int64)
    overwrite_tensor = torch.full((batch_size, time), -1, dtype=torch.int64)
    source_distance = torch.full((batch_size, time), -1, dtype=torch.int64)
    for example, row in enumerate(rows):
        count = len(row["tokens"])
        input_ids[example, :count] = torch.tensor(row["tokens"])
        targets_tensor[example, :count] = torch.tensor(row["targets"])
        valid[example, :count] = True
        query_mask_tensor[example, :count] = torch.tensor(row["query_mask"])
        loss_mask[example, :count] = torch.tensor(row["query_mask"])
        boundaries[example, 0] = True
        source_spans[example, :count] = torch.tensor(row["spans"])
        for position, distance_cell, is_overwrite, distance in row["details"]:
            distance_bin_tensor[example, position] = distance_cell
            load_bin_tensor[example, position] = row["load_bin"]
            overwrite_tensor[example, position] = is_overwrite
            source_distance[example, position] = distance

    return EpisodeBatch(
        task="mqar",
        split=split,
        seed=seed,
        example_ids=tuple(example_ids),
        input_ids=input_ids,
        continuous_inputs=None,
        direct_factors=None,
        targets=targets_tensor,
        valid=valid,
        positions=_segment_positions(valid, boundaries),
        loss_mask=loss_mask,
        query_mask=query_mask_tensor,
        boundaries=boundaries,
        source_spans=source_spans,
        strata={
            "distance_bin": distance_bin_tensor,
            "load_bin": load_bin_tensor,
            "overwrite": overwrite_tensor,
            "source_distance": source_distance,
        },
        metadata=tuple(metadata),
    )


__all__ = [
    "MQAR_DEFAULT_WIDTH",
    "MQAR_LOAD_FACTORS",
    "MQAR_OVERWRITE_FRACTION",
    "MQAR_SCHEMA_VERSION",
    "MQAR_TOKENS",
    "generate_mqar",
]
