"""Assemble a GDN-X heal data bundle (qwen_windows.pt) for the flagship campaign.

Train windows come from the attn_data heal band (50% long-digit synth stress /
50% csa_v1 filler, repacked at SEQ_LEN); eval windows are canonical RULER
episodes built by the suite's own `tasks.ruler.build_ruler_episode`, so every
episode_id / span / target_digest is consistent with the runtime validator.
The default is the preregistered 18-cell matrix: 64 teacher-forced episodes and
an eight-episode deterministic free-generation subset per cell and seed.

The bundle format matches research.kmd2_ablation.qwen_training._default_load_data:
    {"train": [ {example_id, input_ids}, ... 64 windows ... ],
     "eval_by_seed": {
         "11": [ {example_id, input_ids, ruler_metadata}, ... ],
         "29": [...], "47": [...]
     }}

Token IDs are stored as compact int32 tensors.  Labels and dense RULER
annotations are losslessly reconstructed by the production loader, avoiding
duplicate token arrays and O(sequence) sparse-annotation padding on disk.

RULER eval metadata carries `seed`, which the runtime requires to equal the
job seed.  The campaign binds every seed to one immutable data asset, so that
asset contains one evaluation partition per campaign seed and the runtime
selects the matching partition.  Eval scope is set here (not by the config):
only a complete three-seed canonical matrix is marked promotion-eligible.

Usage:
    python -m research.kmd2_ablation.scripts.build_data_bundle \
        --band /home/dev/attn_data/data/csa_heal_4096 \
        --tokenizer /home/dev/csa_retrofit/models/qwen3_5_0_8b \
        --out /home/dev/attn_data/data/gdnx_heal_bundle \
        --seeds 11 29 47

Use --eval-grid only for an explicitly feasibility-scoped diagnostic bundle,
for example ``--eval-grid 512:1,4x4 --free-generation-subset 0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

TRAIN_WINDOWS = 1536
TRAIN_SEQ_LEN = 4096
CANONICAL_CONTEXT_LENGTHS = (512, 2048, 4096, 8192, 16384, 32768)
CANONICAL_QUERY_COUNTS = (1, 4, 8)
CANONICAL_EPISODES_PER_CELL = 64
CANONICAL_FREE_GENERATION_SUBSET = 8
CANONICAL_EVAL_GRID = tuple(
    (context_length, CANONICAL_QUERY_COUNTS, CANONICAL_EPISODES_PER_CELL)
    for context_length in CANONICAL_CONTEXT_LENGTHS
)


def _load_band(band_dir: Path, train_windows: int) -> torch.Tensor:
    shards = sorted(band_dir.glob("shard_main_*.pt"))
    if not shards:
        raise SystemExit(f"no shard_main_*.pt under {band_dir}")
    blocks = torch.cat([torch.load(p, map_location="cpu", weights_only=True) for p in shards])
    if blocks.ndim != 2 or blocks.shape[1] != TRAIN_SEQ_LEN:
        raise SystemExit(f"band must be [N,{TRAIN_SEQ_LEN}], got {tuple(blocks.shape)}")
    if blocks.shape[0] < train_windows:
        raise SystemExit(f"band has {blocks.shape[0]} blocks, need >= {train_windows}")
    return blocks


def build_train(band_dir: Path, train_windows: int = TRAIN_WINDOWS) -> list[dict]:
    blocks = _load_band(band_dir, train_windows)[:train_windows].to(torch.int32)
    return [
        {
            "example_id": f"heal-window-{index:04d}",
            "input_ids": blocks[index],
        }
        for index in range(train_windows)
    ]


def _episode_record(
    tokenizer,
    *,
    cell,
    seed: int,
    example_index: int,
    evidence_scope: str,
) -> tuple[object, dict]:
    from research.kmd2_ablation.tasks.ruler import build_ruler_episode

    episode = build_ruler_episode(
        tokenizer, cell=cell, seed=seed, example_index=example_index
    )
    target_digest = hashlib.sha256(
        json.dumps(
            [list(values) for values in episode.answer_token_ids],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    example_id = f"ruler-{cell.cell_id}-{example_index:03d}"
    metadata = {
        "cell_id": cell.cell_id,
        "context_length": cell.context_length,
        "needles": cell.needles,
        "queries": cell.queries,
        "depth_stratum": episode.depth_strata[0],
        "example_id": example_id,
        "episode_id": episode.episode_id,
        "evaluation_mode": "teacher_forced",
        "evidence_scope": evidence_scope,
        "seed": seed,
        "example_index": example_index,
        "prompt_end": episode.prompt_end,
        "answers": list(episode.answers),
        "answer_token_ids": [list(values) for values in episode.answer_token_ids],
        "answer_spans": [list(span) for span in episode.answer_spans],
        "source_spans": [list(span) for span in episode.source_spans],
        "depth_strata": list(episode.depth_strata),
        "query_keys": list(episode.query_keys),
        "target_digest": target_digest,
        "paired_interval": {
            "kind": "paired_seed_interval",
            "status": (
                "promotion_eligible"
                if evidence_scope == "promotion"
                else "feasibility_only"
            ),
        },
    }
    record = {
        "example_id": example_id,
        "input_ids": torch.tensor(episode.input_ids, dtype=torch.int32),
        "ruler_metadata": metadata,
    }
    return episode, record


def _free_generation_record(record: dict) -> dict:
    """Reuse one teacher episode's token storage with a distinct result identity."""

    metadata = dict(record["ruler_metadata"])
    example_id = f"{record['example_id']}-free"
    metadata["example_id"] = example_id
    metadata["evaluation_mode"] = "free_generation"
    return {
        "example_id": example_id,
        # Deliberately share tensor storage; torch.save preserves this alias.
        "input_ids": record["input_ids"],
        "ruler_metadata": metadata,
    }


def build_eval(
    tokenizer,
    *,
    seed: int,
    grid: list[tuple[int, tuple[int, ...], int]],
    free_generation_subset: int = 0,
    evidence_scope: str = "feasibility",
) -> list[dict]:
    from research.kmd2_ablation.tasks.ruler import (
        RulerCell,
        select_free_generation_subset,
    )

    if type(free_generation_subset) is not int or free_generation_subset < 0:
        raise ValueError("free_generation_subset must be a nonnegative integer")
    if evidence_scope not in {"feasibility", "promotion"}:
        raise ValueError("evidence_scope must be feasibility or promotion")

    records: list[dict] = []
    example_index = 0
    for context_length, query_counts, episodes in grid:
        for queries in query_counts:
            cell = RulerCell(context_length=context_length, needles=16, queries=queries)
            cell_episodes: list[object] = []
            cell_records: list[dict] = []
            for _ in range(episodes):
                episode, record = _episode_record(
                    tokenizer,
                    cell=cell,
                    seed=seed,
                    example_index=example_index,
                    evidence_scope=evidence_scope,
                )
                cell_episodes.append(episode)
                cell_records.append(record)
                example_index += 1
            records.extend(cell_records)
            if free_generation_subset:
                if free_generation_subset > len(cell_episodes):
                    raise ValueError(
                        "free-generation subset exceeds episodes in " + cell.cell_id
                    )
                by_episode_id = {
                    episode.episode_id: record
                    for episode, record in zip(cell_episodes, cell_records)
                }
                selected = select_free_generation_subset(
                    cell_episodes, count=free_generation_subset, seed=seed
                )
                records.extend(
                    _free_generation_record(by_episode_id[episode.episode_id])
                    for episode in selected
                )
    return records


def build_seeded_eval(
    tokenizer,
    *,
    seeds: tuple[int, ...],
    grid: list[tuple[int, tuple[int, ...], int]],
    free_generation_subset: int = 0,
    evidence_scope: str = "feasibility",
) -> dict[str, list[dict]]:
    return {
        str(seed): build_eval(
            tokenizer,
            seed=seed,
            grid=grid,
            free_generation_subset=free_generation_subset,
            evidence_scope=evidence_scope,
        )
        for seed in seeds
    }


def _parse_grid(specs: list[str]) -> list[tuple[int, tuple[int, ...], int]]:
    grid: list[tuple[int, tuple[int, ...], int]] = []
    seen: set[tuple[int, int]] = set()
    for spec in specs:
        try:
            length_part, rest = spec.split(":")
            queries_part, episodes_part = rest.split("x")
            context_length = int(length_part)
            query_counts = tuple(int(q) for q in queries_part.split(","))
            episodes = int(episodes_part)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid eval-grid item {spec!r}; expected CONTEXT:Q1,Q2xEPISODES"
            ) from error
        if context_length not in CANONICAL_CONTEXT_LENGTHS:
            raise ValueError(f"unsupported RULER context length {context_length}")
        if not query_counts or any(q not in CANONICAL_QUERY_COUNTS for q in query_counts):
            raise ValueError("RULER query counts must be selected from 1, 4, and 8")
        if len(set(query_counts)) != len(query_counts) or episodes < 1:
            raise ValueError("eval-grid cells must be unique and episodes positive")
        for queries in query_counts:
            key = (context_length, queries)
            if key in seen:
                raise ValueError(f"duplicate RULER cell {context_length}:{queries}")
            seen.add(key)
        grid.append((context_length, query_counts, episodes))
    return grid


def _promotion_grade_grid(
    grid: list[tuple[int, tuple[int, ...], int]], *, free_generation_subset: int
) -> bool:
    actual = {
        (context_length, queries): episodes
        for context_length, query_counts, episodes in grid
        for queries in query_counts
    }
    expected = {
        (context_length, queries)
        for context_length in CANONICAL_CONTEXT_LENGTHS
        for queries in CANONICAL_QUERY_COUNTS
    }
    return (
        set(actual) == expected
        and min(actual.values()) >= CANONICAL_EPISODES_PER_CELL
        and free_generation_subset >= CANONICAL_FREE_GENERATION_SUBSET
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seed", type=int,
        help="single-job compatibility form; campaign bundles should use --seeds",
    )
    seed_group.add_argument(
        "--seeds", type=int, nargs="+",
        help="ordered campaign job seeds included in this immutable bundle",
    )
    parser.add_argument(
        "--eval-grid",
        nargs="+",
        default=[
            f"{context_length}:1,4,8x{CANONICAL_EPISODES_PER_CELL}"
            for context_length in CANONICAL_CONTEXT_LENGTHS
        ],
        help="cells as CONTEXT:Q1,Q2xEPISODES; overriding the canonical grid is feasibility-only",
    )
    parser.add_argument(
        "--free-generation-subset",
        type=int,
        default=CANONICAL_FREE_GENERATION_SUBSET,
        help="deterministic free-generation episodes per cell (canonical: 8)",
    )
    parser.add_argument(
        "--train-windows",
        type=int,
        default=TRAIN_WINDOWS,
        help="heal windows to include (non-canonical values are smoke-only)",
    )
    options = parser.parse_args()
    seeds = tuple(options.seeds) if options.seeds is not None else (options.seed,)
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must be unique nonnegative integers")
    if options.train_windows < 1:
        raise SystemExit("train-windows must be positive")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(options.tokenizer))

    train = build_train(options.band, options.train_windows)
    grid = _parse_grid(options.eval_grid)
    evidence_scope = (
        "promotion"
        if len(seeds) >= 3
        and options.train_windows == TRAIN_WINDOWS
        and _promotion_grade_grid(
            grid, free_generation_subset=options.free_generation_subset
        )
        else "feasibility"
    )
    evaluation = build_seeded_eval(
        tokenizer,
        seeds=seeds,
        grid=grid,
        free_generation_subset=options.free_generation_subset,
        evidence_scope=evidence_scope,
    )

    train_tokens = sum(int(record["input_ids"].numel()) for record in train)
    for seed, records in evaluation.items():
        eval_ids = [record["example_id"] for record in records]
        if len(eval_ids) != len(set(eval_ids)):
            raise SystemExit(f"eval example_ids are not unique for seed {seed}")

    options.out.mkdir(parents=True, exist_ok=True)
    bundle_path = options.out / "qwen_windows.pt"
    tmp = options.out / ".qwen_windows.pt.tmp"
    torch.save({
        "schema_version": "2.0.0",
        "train": train,
        "eval_by_seed": evaluation,
    }, tmp)
    tmp.replace(bundle_path)

    identity = {
        "seeds": list(seeds),
        "train_windows": len(train),
        "train_tokens": train_tokens,
        "eval_episodes_by_seed": {
            seed: len(records) for seed, records in evaluation.items()
        },
        "teacher_forced_episodes_by_seed": {
            seed: sum(
                record["ruler_metadata"]["evaluation_mode"] == "teacher_forced"
                for record in records
            )
            for seed, records in evaluation.items()
        },
        "free_generation_episodes_by_seed": {
            seed: sum(
                record["ruler_metadata"]["evaluation_mode"] == "free_generation"
                for record in records
            )
            for seed, records in evaluation.items()
        },
        "evidence_scope": evidence_scope,
        "compact_tensor_storage": True,
        "eval_grid": [{"context_length": c, "query_counts": list(q), "episodes": e} for c, q, e in grid],
        "band": str(options.band),
        "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "bytes": bundle_path.stat().st_size,
    }
    (options.out / "bundle_identity.json").write_text(json.dumps(identity, indent=1) + "\n")
    print(json.dumps(identity, indent=1))


if __name__ == "__main__":
    main()
