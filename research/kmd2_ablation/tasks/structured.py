"""Compressible rules with one-shot episodic exceptions."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


STRUCTURED_SCHEMA_VERSION = "1.0.0"
STRUCTURED_RULE_MODULUS = 97
STRUCTURED_EXCEPTION_FRACTION = 0.25
STRUCTURED_TOKENS = MappingProxyType(
    {"PAD": 0, "QUERY": 1, "KEY_BASE": 100, "VALUE_BASE": 1000}
)


def generate_structured_exceptions(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"exception_fraction"}
    if unknown:
        raise ValueError(
            f"unknown structured_exceptions params: {', '.join(sorted(unknown))}"
        )
    fraction = params.get("exception_fraction", STRUCTURED_EXCEPTION_FRACTION)
    if type(fraction) not in (int, float) or not 0.0 < fraction < 0.5:
        raise ValueError("structured_exceptions exception_fraction must be in (0,0.5)")
    fraction = float(fraction)
    queries = _operation_count(length, split)
    load = max(8, queries)
    rows: list[dict[str, Any]] = []
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    for example in range(batch_size):
        identity, generator = _example_identity(
            "structured_exceptions",
            STRUCTURED_SCHEMA_VERSION,
            {"exception_fraction": fraction},
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        factor = 2 * int(torch.randint(1, 24, (), generator=generator).item()) + 1
        offset = int(
            torch.randint(0, STRUCTURED_RULE_MODULUS, (), generator=generator).item()
        )
        domain_offset = int(torch.randint(0, 128, (), generator=generator).item())
        keys = [STRUCTURED_TOKENS["KEY_BASE"] + domain_offset + i for i in range(load)]
        exception_count = max(1, int(round(load * fraction)))
        exception_count = min(exception_count, load - 1)
        exception_keys_ordered = keys[:exception_count]
        exception_keys = set(exception_keys_ordered)
        tokens: list[int] = []
        targets: list[int] = []
        query_mask: list[bool] = []
        spans: list[tuple[int, int]] = []
        source_by_key: dict[int, int] = {}
        value_by_key: dict[int, int] = {}

        def append(token: int) -> int:
            position = len(tokens)
            tokens.append(token)
            targets.append(-100)
            query_mask.append(False)
            spans.append((-1, -1))
            return position

        for key in keys:
            raw_key = key - STRUCTURED_TOKENS["KEY_BASE"]
            rule_value = STRUCTURED_TOKENS["VALUE_BASE"] + (
                factor * raw_key + offset
            ) % STRUCTURED_RULE_MODULUS
            if key in exception_keys:
                shift = 1 + int(
                    torch.randint(
                        0,
                        STRUCTURED_RULE_MODULUS - 1,
                        (),
                        generator=generator,
                    ).item()
                )
                value = STRUCTURED_TOKENS["VALUE_BASE"] + (
                    rule_value - STRUCTURED_TOKENS["VALUE_BASE"] + shift
                ) % STRUCTURED_RULE_MODULUS
            else:
                value = rule_value
            append(key)
            source = append(value)
            source_by_key[key] = source
            value_by_key[key] = value
        rule_keys = [key for key in keys if key not in exception_keys]
        episodic_keys = exception_keys_ordered
        query_details: list[tuple[int, int]] = []
        for query_index in range(queries):
            item_type = query_index % 2
            candidates = episodic_keys if item_type else rule_keys
            key = candidates[(query_index // 2) % len(candidates)]
            append(key)
            query_position = append(STRUCTURED_TOKENS["QUERY"])
            targets[query_position] = value_by_key[key]
            query_mask[query_position] = True
            source = source_by_key[key]
            spans[query_position] = (source, source + 1)
            query_details.append((query_position, item_type))
        rows.append(
            {
                "details": query_details,
                "query_mask": query_mask,
                "spans": spans,
                "targets": targets,
                "tokens": tokens,
            }
        )
        metadata.append(
            {
                "actual_length": len(tokens),
                "exception_count": exception_count,
                "exception_fraction": fraction,
                "load": load,
                "logical_length": length,
                "operation_count": queries,
                "rule_factor": factor,
                "rule_offset": offset,
                "schema_version": STRUCTURED_SCHEMA_VERSION,
                "split_multiplier": queries // length,
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
    item_type = torch.full((batch_size, time), -1, dtype=torch.int64)
    for example, row in enumerate(rows):
        count = len(row["tokens"])
        input_ids[example, :count] = torch.tensor(row["tokens"])
        targets_tensor[example, :count] = torch.tensor(row["targets"])
        valid[example, :count] = True
        query_mask_tensor[example, :count] = torch.tensor(row["query_mask"])
        loss_mask[example, :count] = torch.tensor(row["query_mask"])
        boundaries[example, 0] = True
        source_spans[example, :count] = torch.tensor(row["spans"])
        for position, value in row["details"]:
            item_type[example, position] = value
    return EpisodeBatch(
        task="structured_exceptions",
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
        strata={"item_type": item_type},
        metadata=tuple(metadata),
    )


__all__ = [
    "STRUCTURED_EXCEPTION_FRACTION",
    "STRUCTURED_RULE_MODULUS",
    "STRUCTURED_SCHEMA_VERSION",
    "STRUCTURED_TOKENS",
    "generate_structured_exceptions",
]
