"""Discrete key-version freshness and stale-score adversary task."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


FRESHNESS_SCHEMA_VERSION = "1.0.0"
FRESHNESS_DEFAULT_WIDTH = 8
FRESHNESS_STALE_SCORE = 10.0
FRESHNESS_TOKENS = MappingProxyType(
    {
        "PAD": 0,
        "QUERY": 1,
        "FILLER": 2,
        "LATEST": 3,
        "HISTORY_BASE": 10,
        "KEY_BASE": 100,
        "VALUE_BASE": 1000,
    }
)


def generate_freshness(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"width"}
    if unknown:
        raise ValueError(f"unknown freshness params: {', '.join(sorted(unknown))}")
    width = params.get("width", FRESHNESS_DEFAULT_WIDTH)
    if type(width) is not int or not 1 <= width <= 128:
        raise ValueError("freshness width must be an int in [1,128]")
    groups = _operation_count(length, split)
    rows: list[dict[str, Any]] = []
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    for example in range(batch_size):
        identity, generator = _example_identity("freshness", seed, split, length, example)
        example_ids.append(identity)
        tokens: list[int] = []
        targets: list[int] = []
        query_mask: list[bool] = []
        boundaries: list[bool] = []
        spans: list[tuple[int, int]] = []
        scores: list[float] = []
        query_details: list[tuple[int, int, int, int, int]] = []

        def append(token: int, *, boundary: bool = False, score: float = 0.0) -> int:
            position = len(tokens)
            tokens.append(token)
            targets.append(-100)
            query_mask.append(False)
            boundaries.append(boundary)
            spans.append((-1, -1))
            scores.append(score)
            return position

        for group in range(groups):
            key = FRESHNESS_TOKENS["KEY_BASE"] + (example * 131 + group) % 350
            update_count = 3 + group % 3
            gap = 1 + group % 3
            versions: list[tuple[int, int]] = []
            for version in range(update_count):
                value = FRESHNESS_TOKENS["VALUE_BASE"] + int(
                    torch.randint(0, 900, (), generator=generator).item()
                )
                append(key, boundary=(version == 0))
                score = FRESHNESS_STALE_SCORE if version == 0 else 1.0 / (version + 1)
                source_position = append(value, score=score)
                versions.append((value, source_position))
                for _ in range(gap):
                    append(FRESHNESS_TOKENS["FILLER"])
                if version == 0:
                    for distractor in range(width + 1):
                        distractor_key = (
                            FRESHNESS_TOKENS["KEY_BASE"]
                            + 450
                            + (group * (width + 1) + distractor) % 400
                        )
                        distractor_value = (
                            FRESHNESS_TOKENS["VALUE_BASE"]
                            + (group * 37 + distractor) % 900
                        )
                        append(distractor_key)
                        append(distractor_value, score=2.0 + 0.01 * distractor)

            latest_value, latest_source = versions[-1]
            append(key)
            append(FRESHNESS_TOKENS["LATEST"])
            latest_query = append(FRESHNESS_TOKENS["QUERY"])
            targets[latest_query] = latest_value
            query_mask[latest_query] = True
            spans[latest_query] = (latest_source, latest_source + 1)
            query_details.append(
                (latest_query, 0, update_count, latest_query - latest_source, update_count - 1)
            )

            history_index = group % (update_count - 1)
            historical_value, historical_source = versions[history_index]
            append(key)
            append(FRESHNESS_TOKENS["HISTORY_BASE"] + history_index)
            historical_query = append(FRESHNESS_TOKENS["QUERY"])
            targets[historical_query] = historical_value
            query_mask[historical_query] = True
            spans[historical_query] = (historical_source, historical_source + 1)
            query_details.append(
                (
                    historical_query,
                    1,
                    update_count,
                    historical_query - historical_source,
                    history_index,
                )
            )
        rows.append(
            {
                "boundaries": boundaries,
                "details": query_details,
                "query_mask": query_mask,
                "scores": scores,
                "spans": spans,
                "targets": targets,
                "tokens": tokens,
            }
        )
        metadata.append(
            {
                "actual_length": len(tokens),
                "group_count": groups,
                "logical_length": length,
                "operation_count": groups,
                "schema_version": FRESHNESS_SCHEMA_VERSION,
                "split_multiplier": groups // length,
                "width": width,
            }
        )

    time = max(len(row["tokens"]) for row in rows)
    input_ids = torch.zeros(batch_size, time, dtype=torch.int64)
    targets_tensor = torch.full((batch_size, time), -100, dtype=torch.int64)
    valid = torch.zeros(batch_size, time, dtype=torch.bool)
    query_mask_tensor = torch.zeros(batch_size, time, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, time, dtype=torch.bool)
    boundaries_tensor = torch.zeros(batch_size, time, dtype=torch.bool)
    source_spans = torch.full((batch_size, time, 2), -1, dtype=torch.int64)
    admission_score = torch.zeros(batch_size, time, dtype=torch.float64)
    query_type = torch.full((batch_size, time), -1, dtype=torch.int64)
    update_count_tensor = torch.full((batch_size, time), -1, dtype=torch.int64)
    query_lag = torch.full((batch_size, time), -1, dtype=torch.int64)
    version_index = torch.full((batch_size, time), -1, dtype=torch.int64)
    stale_score_wins = torch.full((batch_size, time), -1, dtype=torch.int64)
    for example, row in enumerate(rows):
        count = len(row["tokens"])
        input_ids[example, :count] = torch.tensor(row["tokens"])
        targets_tensor[example, :count] = torch.tensor(row["targets"])
        valid[example, :count] = True
        query_mask_tensor[example, :count] = torch.tensor(row["query_mask"])
        loss_mask[example, :count] = torch.tensor(row["query_mask"])
        boundaries_tensor[example, :count] = torch.tensor(row["boundaries"])
        source_spans[example, :count] = torch.tensor(row["spans"])
        admission_score[example, :count] = torch.tensor(row["scores"])
        for position, kind, updates, lag, version in row["details"]:
            query_type[example, position] = kind
            update_count_tensor[example, position] = updates
            query_lag[example, position] = lag
            version_index[example, position] = version
            stale_score_wins[example, position] = 1
    return EpisodeBatch(
        task="freshness",
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
        query_mask=query_mask_tensor,
        boundaries=boundaries_tensor,
        source_spans=source_spans,
        strata={
            "admission_score": admission_score,
            "query_lag": query_lag,
            "query_type": query_type,
            "stale_score_wins": stale_score_wins,
            "update_count": update_count_tensor,
            "version_index": version_index,
        },
        metadata=tuple(metadata),
    )


__all__ = [
    "FRESHNESS_DEFAULT_WIDTH",
    "FRESHNESS_SCHEMA_VERSION",
    "FRESHNESS_STALE_SCORE",
    "FRESHNESS_TOKENS",
    "generate_freshness",
]
