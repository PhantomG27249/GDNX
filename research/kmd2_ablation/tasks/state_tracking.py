"""Exact cyclic-state tasks for rotation ablations."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from . import (
    EpisodeBatch,
    _example_identity,
    _operation_count,
    _segment_positions,
)


STATE_TRACKING_SCHEMA_VERSION = "1.0.0"
STATE_TRACKING_TOKENS_PER_QUERY = 3
STATE_TRACKING_QUERY_DENSITY = 1.0 / STATE_TRACKING_TOKENS_PER_QUERY

PARITY_TOKENS = MappingProxyType({"PAD": 0, "HOLD": 1, "ONE": 2, "QUERY": 3})
MODULAR_TOKENS = MappingProxyType(
    {"PAD": 0, "HOLD": 1, "QUERY": 2, "RESET": 3, "ADD_BASE": 16}
)
TOGGLE_TOKENS = MappingProxyType(
    {"PAD": 0, "SET0": 1, "SET1": 2, "TOGGLE": 3, "NOOP": 4, "QUERY": 5}
)


def _allocate(batch_size: int, queries: int) -> dict[str, torch.Tensor]:
    time = queries * STATE_TRACKING_TOKENS_PER_QUERY
    return {
        "input_ids": torch.zeros(batch_size, time, dtype=torch.int64),
        "targets": torch.full((batch_size, time), -100, dtype=torch.int64),
        "valid": torch.ones(batch_size, time, dtype=torch.bool),
        "loss_mask": torch.zeros(batch_size, time, dtype=torch.bool),
        "query_mask": torch.zeros(batch_size, time, dtype=torch.bool),
        "boundaries": torch.zeros(batch_size, time, dtype=torch.bool),
        "source_spans": torch.full((batch_size, time, 2), -1, dtype=torch.int64),
        "label": torch.full((batch_size, time), -1, dtype=torch.int64),
        "operation": torch.zeros(batch_size, time, dtype=torch.int64),
    }


def _finalize(
    task: str,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    arrays: Mapping[str, torch.Tensor],
    example_ids: list[str],
    extra_metadata: list[dict[str, Any]],
) -> EpisodeBatch:
    queries = _operation_count(length, split)
    time = queries * STATE_TRACKING_TOKENS_PER_QUERY
    metadata = tuple(
        {
            "actual_length": time,
            "logical_length": length,
            "operation_count": queries,
            "query_density": STATE_TRACKING_QUERY_DENSITY,
            "schema_version": STATE_TRACKING_SCHEMA_VERSION,
            "split_multiplier": queries // length,
            **extra,
        }
        for extra in extra_metadata
    )
    return EpisodeBatch(
        task=task,
        split=split,
        seed=seed,
        example_ids=tuple(example_ids),
        input_ids=arrays["input_ids"],
        continuous_inputs=None,
        direct_factors=None,
        targets=arrays["targets"],
        valid=arrays["valid"],
        positions=_segment_positions(arrays["valid"], arrays["boundaries"]),
        loss_mask=arrays["loss_mask"],
        query_mask=arrays["query_mask"],
        boundaries=arrays["boundaries"],
        source_spans=arrays["source_spans"],
        strata={"label": arrays["label"], "operation": arrays["operation"]},
        metadata=metadata,
    )


def _boundary_queries(queries: int) -> frozenset[int]:
    if queries == 1:
        return frozenset({0})
    return frozenset({0, queries // 2})


def _counterbalance_phase(
    task: str,
    params: Mapping[str, Any],
    seed: int,
    split: str,
    length: int,
    classes: int,
) -> int:
    _, generator = _example_identity(
        task,
        STATE_TRACKING_SCHEMA_VERSION,
        params,
        seed,
        split,
        length,
        -1,
    )
    return int(torch.randint(0, classes, (), generator=generator).item())


def generate_parity(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    if params:
        raise ValueError("parity params must be empty")
    queries = _operation_count(length, split)
    arrays = _allocate(batch_size, queries)
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    boundaries = _boundary_queries(queries)
    phase = _counterbalance_phase("parity", {}, seed, split, length, 2)
    for example in range(batch_size):
        identity, _ = _example_identity(
            "parity",
            STATE_TRACKING_SCHEMA_VERSION,
            {},
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        state = 0
        segment_start = 0
        for query_index in range(queries):
            base = query_index * STATE_TRACKING_TOKENS_PER_QUERY
            if query_index in boundaries:
                arrays["boundaries"][example, base] = True
                state = 0
                segment_start = base
            desired = (example * queries + query_index + phase) % 2
            operation = (
                PARITY_TOKENS["ONE"] if state != desired else PARITY_TOKENS["HOLD"]
            )
            arrays["input_ids"][example, base] = operation
            arrays["input_ids"][example, base + 1] = PARITY_TOKENS["HOLD"]
            arrays["input_ids"][example, base + 2] = PARITY_TOKENS["QUERY"]
            if operation == PARITY_TOKENS["ONE"]:
                state ^= 1
            query_position = base + 2
            arrays["targets"][example, query_position] = state
            arrays["loss_mask"][example, query_position] = True
            arrays["query_mask"][example, query_position] = True
            arrays["source_spans"][example, query_position] = torch.tensor(
                [segment_start, query_position], dtype=torch.int64
            )
            arrays["label"][example, query_position] = state
            arrays["operation"][example, base : base + 3] = torch.tensor(
                [operation, PARITY_TOKENS["HOLD"], PARITY_TOKENS["QUERY"]]
            )
        metadata.append({"phase": phase})
    return _finalize(
        "parity", batch_size, length, seed, split, arrays, example_ids, metadata
    )


def generate_modular_counter(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    unknown = set(params) - {"modulus"}
    if unknown:
        raise ValueError(f"unknown modular_counter params: {', '.join(sorted(unknown))}")
    modulus = params.get("modulus", 5)
    if type(modulus) is not int or modulus < 2 or modulus > 32:
        raise ValueError("modular_counter modulus must be an int in [2,32]")
    queries = _operation_count(length, split)
    arrays = _allocate(batch_size, queries)
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    boundaries = _boundary_queries(queries)
    identity_params = {"modulus": modulus}
    phase = _counterbalance_phase(
        "modular_counter", identity_params, seed, split, length, modulus
    )
    for example in range(batch_size):
        identity, _ = _example_identity(
            "modular_counter",
            STATE_TRACKING_SCHEMA_VERSION,
            identity_params,
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        state = 0
        segment_start = 0
        for query_index in range(queries):
            base = query_index * STATE_TRACKING_TOKENS_PER_QUERY
            is_boundary = query_index in boundaries
            if is_boundary:
                arrays["boundaries"][example, base] = True
                state = 0
                segment_start = base
                first = MODULAR_TOKENS["RESET"]
            else:
                desired = (example * queries + query_index + phase) % modulus
                delta = (desired - state) % modulus
                first = (
                    MODULAR_TOKENS["HOLD"]
                    if delta == 0
                    else MODULAR_TOKENS["ADD_BASE"] + delta
                )
                state = desired
            if is_boundary:
                desired = (example * queries + query_index + phase) % modulus
                delta = desired
                second = (
                    MODULAR_TOKENS["HOLD"]
                    if delta == 0
                    else MODULAR_TOKENS["ADD_BASE"] + delta
                )
                state = desired
            else:
                second = MODULAR_TOKENS["HOLD"]
            query_position = base + 2
            arrays["input_ids"][example, base : base + 3] = torch.tensor(
                [first, second, MODULAR_TOKENS["QUERY"]]
            )
            arrays["targets"][example, query_position] = state
            arrays["loss_mask"][example, query_position] = True
            arrays["query_mask"][example, query_position] = True
            arrays["source_spans"][example, query_position] = torch.tensor(
                [segment_start, query_position]
            )
            arrays["label"][example, query_position] = state
            arrays["operation"][example, base : base + 3] = torch.tensor(
                [first, second, MODULAR_TOKENS["QUERY"]]
            )
        metadata.append({"modulus": modulus, "phase": phase})
    return _finalize(
        "modular_counter",
        batch_size,
        length,
        seed,
        split,
        arrays,
        example_ids,
        metadata,
    )


def generate_toggle_fsm(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    if params:
        raise ValueError("toggle_fsm params must be empty")
    queries = _operation_count(length, split)
    arrays = _allocate(batch_size, queries)
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    boundaries = _boundary_queries(queries)
    phase = _counterbalance_phase("toggle_fsm", {}, seed, split, length, 2)
    for example in range(batch_size):
        identity, _ = _example_identity(
            "toggle_fsm",
            STATE_TRACKING_SCHEMA_VERSION,
            {},
            seed,
            split,
            length,
            example,
        )
        example_ids.append(identity)
        state = 0
        segment_start = 0
        for query_index in range(queries):
            base = query_index * STATE_TRACKING_TOKENS_PER_QUERY
            if query_index in boundaries:
                arrays["boundaries"][example, base] = True
                state = 0
                segment_start = base
            desired = (example * queries + query_index + phase) % 2
            if query_index % 3 == 0:
                operation = TOGGLE_TOKENS["SET1" if desired else "SET0"]
                state = desired
            elif state != desired:
                operation = TOGGLE_TOKENS["TOGGLE"]
                state ^= 1
            else:
                operation = TOGGLE_TOKENS["NOOP"]
            query_position = base + 2
            arrays["input_ids"][example, base : base + 3] = torch.tensor(
                [operation, TOGGLE_TOKENS["NOOP"], TOGGLE_TOKENS["QUERY"]]
            )
            arrays["targets"][example, query_position] = state
            arrays["loss_mask"][example, query_position] = True
            arrays["query_mask"][example, query_position] = True
            arrays["source_spans"][example, query_position] = torch.tensor(
                [segment_start, query_position]
            )
            arrays["label"][example, query_position] = state
            arrays["operation"][example, base : base + 3] = torch.tensor(
                [operation, TOGGLE_TOKENS["NOOP"], TOGGLE_TOKENS["QUERY"]]
            )
        metadata.append({"phase": phase})
    return _finalize(
        "toggle_fsm",
        batch_size,
        length,
        seed,
        split,
        arrays,
        example_ids,
        metadata,
    )


def generate_state_tracking(
    *,
    batch_size: int,
    length: int,
    seed: int,
    split: str,
    params: Mapping[str, Any],
) -> EpisodeBatch:
    kind = params.get("kind")
    if kind not in {"parity", "modular_counter", "toggle_fsm"}:
        raise ValueError(
            "state_tracking params.kind must be one of: parity, modular_counter, "
            "toggle_fsm"
        )
    forwarded = {key: value for key, value in params.items() if key != "kind"}
    generators = {
        "parity": generate_parity,
        "modular_counter": generate_modular_counter,
        "toggle_fsm": generate_toggle_fsm,
    }
    return generators[kind](
        batch_size=batch_size,
        length=length,
        seed=seed,
        split=split,
        params=forwarded,
    )


__all__ = [
    "MODULAR_TOKENS",
    "PARITY_TOKENS",
    "STATE_TRACKING_QUERY_DENSITY",
    "STATE_TRACKING_SCHEMA_VERSION",
    "STATE_TRACKING_TOKENS_PER_QUERY",
    "TOGGLE_TOKENS",
    "generate_modular_counter",
    "generate_parity",
    "generate_state_tracking",
    "generate_toggle_fsm",
]
