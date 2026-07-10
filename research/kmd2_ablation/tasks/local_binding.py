"""Local order/binding controls for convolution ablations."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


LOCAL_BINDING_SCHEMA_VERSION = "1.0.0"
LOCAL_BINDING_VOCAB_VERSION = "1.0.0"
LOCAL_BINDING_DEFAULT_WIDTH = 8
LOCAL_BINDING_MODES = ("adjacent", "separated", "motif", "delayed_copy")
LOCAL_BINDING_TOKENS = MappingProxyType(
    {
        "PAD": 0,
        "QUERY": 1,
        "COPY": 2,
        "FILLER": 3,
        "KEY_BASE": 100,
        "VALUE_BASE": 1000,
        "MOTIF_BASE": 2000,
    }
)


def _width_cells(width: int) -> tuple[int, int, int]:
    return (max(1, width // 2), width, width + max(2, width // 2))


def _distance_targets(width: int) -> tuple[int, int, int]:
    return (max(1, width // 2), width, 2 * width + 1)


def _distance_bin(distance: int, width: int) -> int:
    if distance < width:
        return 0
    if distance <= 2 * width:
        return 1
    return 2


def generate_local_binding(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"width"}
    if unknown:
        raise ValueError(f"unknown local_binding params: {', '.join(sorted(unknown))}")
    width = params.get("width", LOCAL_BINDING_DEFAULT_WIDTH)
    if type(width) is not int or not 4 <= width <= 256:
        raise ValueError("local_binding width must be an int in [4,256]")
    queries = _operation_count(length, split)
    load_cells = _width_cells(width)
    distance_targets = _distance_targets(width)

    token_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    query_rows: list[list[bool]] = []
    boundary_rows: list[list[bool]] = []
    span_rows: list[list[tuple[int, int]]] = []
    strata_rows: list[dict[str, list[int]]] = []
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    for example in range(batch_size):
        identity, generator = _example_identity(
            "local_binding",
            LOCAL_BINDING_SCHEMA_VERSION,
            {"width": width},
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        tokens: list[int] = []
        targets: list[int] = []
        queries_mask: list[bool] = []
        boundaries: list[bool] = []
        spans: list[tuple[int, int]] = []
        strata = {
            "distance_bin": [],
            "load_bin": [],
            "mode": [],
            "overwrite": [],
            "source_distance": [],
            "write_load": [],
        }

        def append(token: int, *, boundary: bool = False) -> int:
            position = len(tokens)
            tokens.append(token)
            targets.append(-100)
            queries_mask.append(False)
            boundaries.append(boundary)
            spans.append((-1, -1))
            for values in strata.values():
                values.append(-1)
            return position

        for query_index in range(queries):
            item_start = len(tokens)
            mode_index = (query_index + example) % len(LOCAL_BINDING_MODES)
            load_bin = (query_index + example) % 3
            distance_bin = (query_index + 2 * example) % 3
            overwrite = mode_index in (0, 1) and query_index % 2 == 1
            target_writes = 2 if overwrite else 1
            write_load = max(target_writes, load_cells[load_bin])
            distractors = write_load - target_writes
            for distractor in range(distractors):
                distractor_key = (
                    LOCAL_BINDING_TOKENS["KEY_BASE"]
                    + (query_index * 31 + distractor + 11) % 700
                )
                distractor_value = (
                    LOCAL_BINDING_TOKENS["VALUE_BASE"]
                    + (query_index * 47 + distractor + 23) % 900
                )
                append(distractor_key, boundary=(len(tokens) == item_start))
                append(distractor_value)

            target_key = (
                LOCAL_BINDING_TOKENS["KEY_BASE"]
                + (query_index * 17 + example * 73 + 503) % 700
            )
            target_value = LOCAL_BINDING_TOKENS["VALUE_BASE"] + int(
                torch.randint(0, 900, (), generator=generator).item()
            )
            old_value = LOCAL_BINDING_TOKENS["VALUE_BASE"] + (
                (target_value - LOCAL_BINDING_TOKENS["VALUE_BASE"] + 451) % 900
            )
            motif = tuple(
                LOCAL_BINDING_TOKENS["MOTIF_BASE"]
                + (query_index * 13 + example * 29 + offset) % 500
                for offset in range(3)
            )
            first_target_token = len(tokens) == item_start
            if mode_index == 0:
                if overwrite:
                    append(target_key, boundary=first_target_token)
                    append(old_value)
                    first_target_token = False
                append(target_key, boundary=first_target_token)
                source_position = append(target_value)
                query_prefix = (target_key,)
            elif mode_index == 1:
                if overwrite:
                    append(target_key, boundary=first_target_token)
                    append(LOCAL_BINDING_TOKENS["FILLER"])
                    append(old_value)
                    first_target_token = False
                append(target_key, boundary=first_target_token)
                append(LOCAL_BINDING_TOKENS["FILLER"])
                append(LOCAL_BINDING_TOKENS["FILLER"])
                append(LOCAL_BINDING_TOKENS["FILLER"])
                source_position = append(target_value)
                query_prefix = (target_key,)
            elif mode_index == 2:
                for offset, token in enumerate(motif):
                    append(token, boundary=first_target_token and offset == 0)
                source_position = append(target_value)
                query_prefix = motif
            else:
                source_position = append(target_value, boundary=first_target_token)
                query_prefix = ()

            desired_distance = distance_targets[distance_bin]
            minimum_distance = len(tokens) + len(query_prefix) - source_position
            for _ in range(max(0, desired_distance - minimum_distance)):
                append(LOCAL_BINDING_TOKENS["FILLER"])
            for token in query_prefix:
                append(token)
            query_token = (
                LOCAL_BINDING_TOKENS["COPY"]
                if mode_index == 3
                else LOCAL_BINDING_TOKENS["QUERY"]
            )
            query_position = append(query_token)
            targets[query_position] = target_value
            queries_mask[query_position] = True
            spans[query_position] = (source_position, source_position + 1)
            measured_distance = query_position - source_position
            strata["distance_bin"][query_position] = _distance_bin(
                measured_distance, width
            )
            strata["load_bin"][query_position] = load_bin
            strata["mode"][query_position] = mode_index
            strata["overwrite"][query_position] = int(overwrite)
            strata["source_distance"][query_position] = measured_distance
            strata["write_load"][query_position] = write_load

        token_rows.append(tokens)
        target_rows.append(targets)
        query_rows.append(queries_mask)
        boundary_rows.append(boundaries)
        span_rows.append(spans)
        strata_rows.append(strata)
        metadata.append(
            {
                "actual_length": len(tokens),
                "logical_length": length,
                "operation_count": queries,
                "schema_version": LOCAL_BINDING_SCHEMA_VERSION,
                "split_multiplier": queries // length,
                "vocab_version": LOCAL_BINDING_VOCAB_VERSION,
                "width": width,
            }
        )

    time = max(len(row) for row in token_rows)
    input_ids = torch.zeros(batch_size, time, dtype=torch.int64)
    targets_tensor = torch.full((batch_size, time), -100, dtype=torch.int64)
    valid = torch.zeros(batch_size, time, dtype=torch.bool)
    query_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries_tensor = torch.zeros(batch_size, time, dtype=torch.bool)
    source_spans = torch.full((batch_size, time, 2), -1, dtype=torch.int64)
    strata_tensors = {
        name: torch.full((batch_size, time), -1, dtype=torch.int64)
        for name in strata_rows[0]
    }
    for example, tokens in enumerate(token_rows):
        count = len(tokens)
        input_ids[example, :count] = torch.tensor(tokens)
        targets_tensor[example, :count] = torch.tensor(target_rows[example])
        valid[example, :count] = True
        query_mask[example, :count] = torch.tensor(query_rows[example])
        loss_mask[example, :count] = torch.tensor(query_rows[example])
        boundaries_tensor[example, :count] = torch.tensor(boundary_rows[example])
        source_spans[example, :count] = torch.tensor(span_rows[example])
        for name, tensor in strata_tensors.items():
            tensor[example, :count] = torch.tensor(strata_rows[example][name])

    return EpisodeBatch(
        task="local_binding",
        split=split,
        seed=seed,
        example_ids=tuple(example_ids),
        input_ids=input_ids,
        continuous_inputs=None,
        direct_factors=None,
        targets=targets_tensor,
        valid=valid,
        positions=_segment_positions(valid, boundaries_tensor),
        loss_mask=loss_mask,
        query_mask=query_mask,
        boundaries=boundaries_tensor,
        source_spans=source_spans,
        strata=strata_tensors,
        metadata=tuple(metadata),
    )


__all__ = [
    "LOCAL_BINDING_DEFAULT_WIDTH",
    "LOCAL_BINDING_MODES",
    "LOCAL_BINDING_SCHEMA_VERSION",
    "LOCAL_BINDING_TOKENS",
    "LOCAL_BINDING_VOCAB_VERSION",
    "generate_local_binding",
]
