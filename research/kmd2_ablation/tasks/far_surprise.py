"""Selector replay task with early relevant facts and high-score distractors."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from . import EpisodeBatch, _example_identity, _operation_count, _segment_positions


FAR_SURPRISE_SCHEMA_VERSION = "1.0.0"
FAR_SURPRISE_DEFAULT_WIDTH = 8
FAR_SURPRISE_DISTRACTOR_MARGIN = 8
FAR_SURPRISE_TOKENS = MappingProxyType(
    {"PAD": 0, "QUERY": 1, "KEY_BASE": 100, "VALUE_BASE": 1000}
)


def generate_far_surprise(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"width"}
    if unknown:
        raise ValueError(f"unknown far_surprise params: {', '.join(sorted(unknown))}")
    width = params.get("width", FAR_SURPRISE_DEFAULT_WIDTH)
    if type(width) is not int or not 1 <= width <= 256:
        raise ValueError("far_surprise width must be an int in [1,256]")
    queries = _operation_count(length, split)
    distractor_count = width + FAR_SURPRISE_DISTRACTOR_MARGIN
    rows: list[dict[str, Any]] = []
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    for example in range(batch_size):
        identity, generator = _example_identity(
            "far_surprise", seed, split, length, example
        )
        example_ids.append(identity)
        key_offset = int(torch.randint(0, 128, (), generator=generator).item())
        relevant_keys = [FAR_SURPRISE_TOKENS["KEY_BASE"] + key_offset + i for i in range(queries)]
        relevant_values = [
            FAR_SURPRISE_TOKENS["VALUE_BASE"]
            + int(torch.randint(0, 700, (), generator=generator).item())
            for _ in range(queries)
        ]
        tokens: list[int] = []
        targets: list[int] = []
        query_mask: list[bool] = []
        spans: list[tuple[int, int]] = []
        score: list[float] = []
        distractor: list[int] = []
        distractor_key: list[int] = []

        def append(
            token: int,
            *,
            native_score: float = 0.0,
            is_distractor: int = 0,
            is_distractor_key: int = 0,
        ) -> int:
            position = len(tokens)
            tokens.append(token)
            targets.append(-100)
            query_mask.append(False)
            spans.append((-1, -1))
            score.append(native_score)
            distractor.append(is_distractor)
            distractor_key.append(is_distractor_key)
            return position

        sources: list[int] = []
        for key, value in zip(relevant_keys, relevant_values):
            append(key)
            sources.append(append(value, native_score=1.0))
        distractor_start = len(tokens)
        distractor_keys = [
            FAR_SURPRISE_TOKENS["KEY_BASE"] + 500 + example * 17 + index
            for index in range(distractor_count)
        ]
        for index, key in enumerate(distractor_keys):
            append(key, is_distractor_key=1)
            value = FAR_SURPRISE_TOKENS["VALUE_BASE"] + 700 + index
            append(value, native_score=10.0 + index, is_distractor=1)
        query_details: list[int] = []
        for index, (key, value, source) in enumerate(
            zip(relevant_keys, relevant_values, sources)
        ):
            append(key)
            query_position = append(FAR_SURPRISE_TOKENS["QUERY"])
            targets[query_position] = value
            query_mask[query_position] = True
            spans[query_position] = (source, source + 1)
            query_details.append(query_position)
        rows.append(
            {
                "details": query_details,
                "distractor": distractor,
                "distractor_key": distractor_key,
                "query_mask": query_mask,
                "score": score,
                "spans": spans,
                "targets": targets,
                "tokens": tokens,
            }
        )
        metadata.append(
            {
                "actual_length": len(tokens),
                "distractor_count": distractor_count,
                "distractor_start": distractor_start,
                "logical_length": length,
                "operation_count": queries,
                "schema_version": FAR_SURPRISE_SCHEMA_VERSION,
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
    native_score = torch.zeros(batch_size, time, dtype=torch.float64)
    distractor_tensor = torch.zeros(batch_size, time, dtype=torch.int64)
    distractor_key_tensor = torch.zeros(batch_size, time, dtype=torch.int64)
    load_over_width = torch.full((batch_size, time), -1, dtype=torch.int64)
    relevant_age = torch.full((batch_size, time), -1, dtype=torch.int64)
    for example, row in enumerate(rows):
        count = len(row["tokens"])
        input_ids[example, :count] = torch.tensor(row["tokens"])
        targets_tensor[example, :count] = torch.tensor(row["targets"])
        valid[example, :count] = True
        query_mask_tensor[example, :count] = torch.tensor(row["query_mask"])
        loss_mask[example, :count] = torch.tensor(row["query_mask"])
        boundaries[example, 0] = True
        source_spans[example, :count] = torch.tensor(row["spans"])
        native_score[example, :count] = torch.tensor(row["score"])
        distractor_tensor[example, :count] = torch.tensor(row["distractor"])
        distractor_key_tensor[example, :count] = torch.tensor(row["distractor_key"])
        for position in row["details"]:
            load_over_width[example, position] = 1
            relevant_age[example, position] = position - int(source_spans[example, position, 0])
    return EpisodeBatch(
        task="far_surprise",
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
            "distractor": distractor_tensor,
            "distractor_key": distractor_key_tensor,
            "load_over_width": load_over_width,
            "native_score": native_score,
            "relevant_age": relevant_age,
        },
        metadata=tuple(metadata),
    )


__all__ = [
    "FAR_SURPRISE_DEFAULT_WIDTH",
    "FAR_SURPRISE_DISTRACTOR_MARGIN",
    "FAR_SURPRISE_SCHEMA_VERSION",
    "FAR_SURPRISE_TOKENS",
    "generate_far_surprise",
]
