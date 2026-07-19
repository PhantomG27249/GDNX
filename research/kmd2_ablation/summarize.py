"""Deterministic ledgers, tabular summaries, and delegated scientific decisions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import (
    FactorialBootstrapResult,
    MetricSample,
    Option3Decision,
    Option3Evidence,
    Option3Thresholds,
    ProtectedEffect,
    BootstrapInterval,
    classify_addition,
    classify_reliance,
    decide_option3,
    metric_direction,
    paired_bootstrap,
    paired_factorial_bootstrap,
)
from .results import canonical_json_bytes
from .tasks.ruler import (
    RULER_ARMS,
    RULER_CONTEXT_LENGTHS,
    RULER_DEPTH_STRATA,
    RULER_HEAL_SEED_COUNT,
    RULER_LONG_CELLS,
    RULER_MIN_EPISODES_PER_CELL,
    RULER_QUERY_COUNTS,
    RulerCell,
    ruler_evidence_scope,
)


SUMMARY_SCHEMA_VERSION = "1.0.0"
_EXECUTION_STATUSES = {"completed", "failed"}
_EVALUATION_MODES = {"teacher_forced", "free_generation"}
_EXPECTED_JOB_IDENTITY_FIELDS = (
    "experiment_id",
    "arm_id",
    "seed",
    "backend",
    "stage",
    "pairing_id",
)
_SUMMARY_PAYLOAD_NAMES = ("ledger.jsonl", "results.json", "results.csv")
_CURRENT_POINTER_NAME = "current.json"
_CSV_FIELDS = (
    "row_type",
    "job_id",
    "experiment_id",
    "arm_id",
    "seed",
    "execution_status",
    "scientific_label",
    "task",
    "cell_id",
    "example_id",
    "episode_id",
    "evaluation_mode",
    "evidence_scope",
    "numerator",
    "denominator",
    "episode_exact",
    "source_spans",
    "cache_diagnostics",
    "paired_interval",
    "metrics",
    "error_code",
)


class SummaryValidationError(ValueError):
    """Raised when execution records cannot support an honest summary."""


@dataclass(frozen=True)
class SummaryArtifacts:
    ledger_jsonl: bytes
    results_json: bytes
    results_csv: bytes


@dataclass(frozen=True)
class ClassifiedFactorial:
    label: str
    statistics: FactorialBootstrapResult


@dataclass(frozen=True)
class ClassifiedReliance:
    label: str
    interval: BootstrapInterval


def _plain_json(value: object, *, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise SummaryValidationError(f"{label} must be finite canonical JSON") from error


def _required_string(mapping: Mapping[str, Any], field: str, *, label: str) -> str:
    value = mapping.get(field)
    if type(value) is not str or not value:
        raise SummaryValidationError(f"{label}.{field} must be a nonempty string")
    return value


def _required_int(mapping: Mapping[str, Any], field: str, *, label: str) -> int:
    value = mapping.get(field)
    if type(value) is not int:
        raise SummaryValidationError(f"{label}.{field} must be an int")
    return value


def _normalize_evaluation(
    raw: object, *, record: Mapping[str, Any], index: int
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SummaryValidationError(f"evaluations[{index}] must be a mapping")
    evaluation = _plain_json(raw, label=f"evaluations[{index}]")
    task = evaluation.get("task")
    if task is None:
        return evaluation
    if type(task) is not str or not task:
        raise SummaryValidationError(
            f"evaluations[{index}].task must be a nonempty string when present"
        )
    if task != "ruler":
        return evaluation
    label = f"RULER evaluation {record['job_id']}[{index}]"
    for field in ("cell_id", "example_id", "episode_id", "evaluation_mode", "evidence_scope"):
        _required_string(evaluation, field, label=label)
    if evaluation["evaluation_mode"] not in _EVALUATION_MODES:
        raise SummaryValidationError(f"{label} has an invalid evaluation mode")
    if evaluation["evidence_scope"] not in {"feasibility", "promotion"}:
        raise SummaryValidationError(f"{label} has an invalid evidence scope")
    cell_id = evaluation["cell_id"]
    context_length = _required_int(evaluation, "context_length", label=label)
    needles = _required_int(evaluation, "needles", label=label)
    queries = _required_int(evaluation, "queries", label=label)
    if context_length not in RULER_CONTEXT_LENGTHS:
        raise SummaryValidationError(f"{label} must use the pinned RULER context grid")
    if needles != 16:
        raise SummaryValidationError(f"{label} must use exactly 16 needles")
    if queries not in RULER_QUERY_COUNTS:
        raise SummaryValidationError(f"{label} queries must be 1, 4, or 8")
    expected_cell = RulerCell(context_length, needles, queries).cell_id
    if cell_id != expected_cell:
        raise SummaryValidationError(f"{label} cell identity does not match context/query metadata")
    depth = _required_string(evaluation, "depth_stratum", label=label)
    if depth not in RULER_DEPTH_STRATA:
        raise SummaryValidationError(f"{label} has an invalid depth stratum")
    numerator = _required_int(evaluation, "numerator", label=label)
    denominator = _required_int(evaluation, "denominator", label=label)
    if denominator < 1 or not 0 <= numerator <= denominator:
        raise SummaryValidationError(f"{label} has invalid metric totals")
    if type(evaluation.get("episode_exact")) is not bool:
        raise SummaryValidationError(f"{label}.episode_exact must be bool")
    if evaluation["episode_exact"] != (numerator == denominator):
        raise SummaryValidationError(f"{label} episode_exact conflicts with metric totals")
    spans = evaluation.get("source_spans")
    if not isinstance(spans, list) or len(spans) != queries:
        raise SummaryValidationError(f"{label} must retain one exact source span per query")
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= context_length
        ):
            raise SummaryValidationError(f"{label} contains an invalid source span")
    _required_string(evaluation, "target_digest", label=label)
    for field in ("cache_diagnostics", "paired_interval"):
        if field not in evaluation or not isinstance(evaluation[field], Mapping):
            raise SummaryValidationError(f"{label} must retain {field}")
    if evaluation.get("seed", record["seed"]) != record["seed"]:
        raise SummaryValidationError(f"{label} seed conflicts with its execution record")
    if evaluation.get("arm_id", record["arm_id"]) != record["arm_id"]:
        raise SummaryValidationError(f"{label} arm conflicts with its execution record")
    return evaluation


def _normalize_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes, bytearray)):
        raise TypeError("records must be an iterable of mappings")
    normalized: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        record = _plain_json(raw, label=f"records[{index}]")
        label = f"records[{index}]"
        job_id = _required_string(record, "job_id", label=label)
        if job_id in seen_jobs:
            raise SummaryValidationError(f"duplicate job record: {job_id}")
        seen_jobs.add(job_id)
        _required_string(record, "experiment_id", label=label)
        _required_string(record, "arm_id", label=label)
        _required_int(record, "seed", label=label)
        status = _required_string(record, "status", label=label)
        if status not in _EXECUTION_STATUSES:
            raise SummaryValidationError(
                "persisted execution status must be completed or failed; missing is derived"
            )
        evaluations = record.get("evaluations", [])
        if isinstance(evaluations, (str, bytes, bytearray)) or not isinstance(evaluations, Sequence):
            raise SummaryValidationError(f"{label}.evaluations must be a sequence")
        if status == "failed" and evaluations:
            raise SummaryValidationError("failed execution records cannot contain completed evaluations")
        record["evaluations"] = [
            _normalize_evaluation(value, record=record, index=evaluation_index)
            for evaluation_index, value in enumerate(evaluations)
        ]
        scientific = record.get("scientific_label")
        if scientific is not None and (type(scientific) is not str or not scientific):
            raise SummaryValidationError(f"{label}.scientific_label must be null or nonempty")
        normalized.append(record)
    return normalized


def _normalize_expected_jobs(expected_jobs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(expected_jobs):
        if not isinstance(raw, Mapping):
            raise TypeError(f"expected_jobs[{index}] must be a mapping")
        job = _plain_json(raw, label=f"expected_jobs[{index}]")
        label = f"expected_jobs[{index}]"
        job_id = _required_string(job, "job_id", label=label)
        if job_id in seen:
            raise SummaryValidationError(f"duplicate expected job: {job_id}")
        seen.add(job_id)
        _required_string(job, "experiment_id", label=label)
        _required_string(job, "arm_id", label=label)
        _required_int(job, "seed", label=label)
        for field in ("backend", "stage", "pairing_id"):
            if field in job:
                _required_string(job, field, label=label)
        jobs.append(job)
    return jobs


def _validate_expected_job_identities(
    records: Sequence[Mapping[str, Any]], expected_jobs: Sequence[Mapping[str, Any]]
) -> None:
    if not expected_jobs:
        return
    by_job = {record["job_id"]: record for record in records}
    for job in expected_jobs:
        record = by_job.get(job["job_id"])
        if record is None:
            continue
        for field in _EXPECTED_JOB_IDENTITY_FIELDS:
            if field in job and record.get(field) != job[field]:
                raise SummaryValidationError(
                    f"expected job identity mismatch for {field}: {job['job_id']}"
                )


def _evaluation_key(record: Mapping[str, Any], evaluation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["seed"],
        evaluation["example_id"],
        evaluation["cell_id"],
        evaluation["evaluation_mode"],
    )


def _evaluation_signature(evaluation: Mapping[str, Any]) -> bytes:
    identity = {
        "episode_id": evaluation["episode_id"],
        "context_length": evaluation["context_length"],
        "needles": evaluation["needles"],
        "queries": evaluation["queries"],
        "depth_stratum": evaluation["depth_stratum"],
        "source_spans": evaluation["source_spans"],
        "target_digest": evaluation["target_digest"],
    }
    return canonical_json_bytes(identity)


def _index_ruler_evaluations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[tuple[Any, ...], Mapping[str, Any]]]:
    """Index every completed RULER row and reject duplicates for all summaries."""

    by_arm: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for record in records:
        arm = record["arm_id"]
        if record["status"] != "completed":
            continue
        for evaluation in record["evaluations"]:
            if evaluation.get("task") != "ruler":
                continue
            key = _evaluation_key(record, evaluation)
            arm_rows = by_arm.setdefault(arm, {})
            if key in arm_rows:
                raise SummaryValidationError(
                    f"duplicate RULER seed/example/cell record for arm {arm}"
                )
            arm_rows[key] = evaluation
    return by_arm


def _signatures_match(
    by_arm: Mapping[str, Mapping[tuple[Any, ...], Mapping[str, Any]]],
    reference_keys: set[tuple[Any, ...]],
) -> bool:
    return all(
        len({_evaluation_signature(by_arm[arm][key]) for arm in RULER_ARMS}) == 1
        for key in reference_keys
    )


def _validate_promotion(
    indexed: Mapping[str, Mapping[tuple[Any, ...], Mapping[str, Any]]],
) -> None:
    by_arm = {arm: indexed.get(arm, {}) for arm in RULER_ARMS}
    if any(not values for values in by_arm.values()):
        raise SummaryValidationError("promotion requires native, recency, and surprise arms")
    if any(
        evaluation["evidence_scope"] != "promotion"
        for values in by_arm.values()
        for evaluation in values.values()
    ):
        raise SummaryValidationError(
            "feasibility RULER data cannot be used as promotion evidence"
        )
    reference_keys = set(by_arm["native"])
    if any(set(by_arm[arm]) != reference_keys for arm in RULER_ARMS[1:]):
        raise SummaryValidationError(
            "promotion arms must have exactly matched seed/example/cell identities"
        )
    if not _signatures_match(by_arm, reference_keys):
        raise SummaryValidationError("RULER episode identity must match across arms")
    teacher_keys = [key for key in reference_keys if key[-1] == "teacher_forced"]
    seeds = {int(key[0]) for key in teacher_keys}
    if len(seeds) < RULER_HEAL_SEED_COUNT:
        raise SummaryValidationError("promotion requires three paired RULER heal seeds")
    for seed in seeds:
        cells = {str(key[2]) for key in teacher_keys if key[0] == seed}
        if not set(RULER_LONG_CELLS) <= cells:
            raise SummaryValidationError("promotion is missing required RULER cells")
        for cell in RULER_LONG_CELLS:
            count = sum(key[0] == seed and key[2] == cell for key in teacher_keys)
            if count < RULER_MIN_EPISODES_PER_CELL:
                raise SummaryValidationError(
                    "promotion requires at least 64 matched episodes per RULER cell"
                )


def _ledger_rows(
    records: Sequence[Mapping[str, Any]], expected_jobs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_job = {record["job_id"]: record for record in records}
    if expected_jobs:
        expected_ids = {job["job_id"] for job in expected_jobs}
        unexpected = set(by_job) - expected_ids
        if unexpected:
            raise SummaryValidationError(
                "records contain jobs absent from expected_jobs: " + ", ".join(sorted(unexpected))
            )
        sources = list(expected_jobs)
    else:
        sources = [
            {
                "job_id": record["job_id"],
                "experiment_id": record["experiment_id"],
                "arm_id": record["arm_id"],
                "seed": record["seed"],
            }
            for record in records
        ]
    rows: list[dict[str, Any]] = []
    for job in sources:
        record = by_job.get(job["job_id"])
        base = {
            "row_type": "run",
            "job_id": job["job_id"],
            "experiment_id": job["experiment_id"],
            "arm_id": job["arm_id"],
            "seed": job["seed"],
            "execution_status": record["status"] if record else "missing",
            "scientific_label": record.get("scientific_label") if record else None,
        }
        if record is None:
            rows.append(base)
            continue
        if record["status"] == "failed":
            error = record.get("error")
            if error is not None:
                base["error"] = error
            rows.append(base)
            continue
        evaluations = record["evaluations"]
        if not evaluations:
            if "metrics" in record:
                base["metrics"] = record["metrics"]
            rows.append(base)
            continue
        for evaluation in evaluations:
            row = dict(base)
            row["row_type"] = "evaluation"
            row["evaluation"] = evaluation
            if "metrics" in record:
                row["metrics"] = record["metrics"]
            rows.append(row)
    return sorted(rows, key=canonical_json_bytes)


def _json_cell(value: object) -> str:
    if value is None:
        return ""
    return canonical_json_bytes(value).decode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        evaluation = row.get("evaluation", {})
        error = row.get("error", {})
        writer.writerow(
            {
                "row_type": row["row_type"],
                "job_id": row["job_id"],
                "experiment_id": row["experiment_id"],
                "arm_id": row["arm_id"],
                "seed": row["seed"],
                "execution_status": row["execution_status"],
                "scientific_label": row.get("scientific_label") or "",
                "task": evaluation.get("task", ""),
                "cell_id": evaluation.get("cell_id", ""),
                "example_id": evaluation.get("example_id", ""),
                "episode_id": evaluation.get("episode_id", ""),
                "evaluation_mode": evaluation.get("evaluation_mode", ""),
                "evidence_scope": evaluation.get("evidence_scope", ""),
                "numerator": evaluation.get("numerator", ""),
                "denominator": evaluation.get("denominator", ""),
                "episode_exact": evaluation.get("episode_exact", ""),
                "source_spans": _json_cell(evaluation.get("source_spans")),
                "cache_diagnostics": _json_cell(evaluation.get("cache_diagnostics")),
                "paired_interval": _json_cell(evaluation.get("paired_interval")),
                "metrics": _json_cell(row.get("metrics")),
                "error_code": error.get("code", "") if isinstance(error, Mapping) else "",
            }
        )
    return stream.getvalue().encode("utf-8")


def build_summary_artifacts(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_jobs: Iterable[Mapping[str, Any]] = (),
    promotion: bool = False,
) -> SummaryArtifacts:
    """Build canonical summary bytes independent of input record ordering."""

    if type(promotion) is not bool:
        raise TypeError("promotion must be bool")
    normalized = _normalize_records(records)
    jobs = _normalize_expected_jobs(expected_jobs)
    _validate_expected_job_identities(normalized, jobs)
    ruler_index = _index_ruler_evaluations(normalized)
    if promotion:
        _validate_promotion(ruler_index)
    rows = _ledger_rows(normalized, jobs)
    ledger = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    execution_counts = Counter(record["status"] for record in normalized)
    if jobs:
        execution_counts["missing"] = len(jobs) - len(normalized)
    for status in ("completed", "failed", "missing"):
        execution_counts.setdefault(status, 0)
    ruler_rows = [
        (record, evaluation)
        for record in normalized
        for evaluation in record["evaluations"]
        if evaluation.get("task") == "ruler"
    ]
    ruler_evaluations = [evaluation for _record, evaluation in ruler_rows]
    scope: str | None = None
    if ruler_evaluations:
        if any(item["evidence_scope"] == "feasibility" for item in ruler_evaluations):
            scope = "feasibility"
        else:
            scope = ruler_evidence_scope(
                identities={
                    arm: tuple(ruler_index.get(arm, {})) for arm in RULER_ARMS
                }
            )
            if scope == "promotion":
                required = {arm: ruler_index[arm] for arm in RULER_ARMS}
                reference_keys = set(required["native"])
                if not _signatures_match(required, reference_keys):
                    scope = "feasibility"
    scientific_counts = Counter(
        record["scientific_label"]
        for record in normalized
        if record["status"] == "completed" and record.get("scientific_label") is not None
    )
    hybrid_seed_rows = _derive_hybrid_promotion_rows(normalized)
    hybrid_campaign = any(
        str(item.get("arm_id", "")).startswith("gdn2-mimo-r4-braid-")
        for item in (*normalized, *jobs)
    )
    hybrid_arms = {
        str(item.get("arm_id")) for item in (*normalized, *jobs)
        if str(item.get("arm_id", "")).startswith("gdn2-mimo-r4-braid-")
    }
    hybrid_promotion = (
        {arm: aggregate_hybrid_promotion(hybrid_seed_rows.get(arm, ())) for arm in sorted(hybrid_arms)}
        if hybrid_campaign else None
    )
    results = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "execution_counts": {
            status: execution_counts[status] for status in ("completed", "failed", "missing")
        },
        "scientific_counts": dict(sorted(scientific_counts.items())),
        "ledger_rows": len(rows),
        "ledger_sha256": hashlib.sha256(ledger).hexdigest(),
        "promotion_requested": promotion,
        "ruler": {
            "arms": sorted({record["arm_id"] for record, _evaluation in ruler_rows}),
            "cells": sorted({item["cell_id"] for item in ruler_evaluations}),
            "evaluation_modes": sorted(
                {item["evaluation_mode"] for item in ruler_evaluations}
            ),
            "evidence_scope": scope,
            "row_count": len(ruler_evaluations),
            "seeds": sorted({record["seed"] for record, _evaluation in ruler_rows}),
        },
    }
    if hybrid_promotion is not None:
        results["hybrid_promotion"] = hybrid_promotion
    return SummaryArtifacts(
        ledger_jsonl=ledger,
        results_json=canonical_json_bytes(results) + b"\n",
        results_csv=_csv_bytes(rows),
    )


def _derive_hybrid_promotion_rows(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Join candidate observations only to byte-identical matched baseline runs."""
    observations = [
        record["hybrid_promotion_observation"] for record in records
        if record.get("status") == "completed"
        and isinstance(record.get("hybrid_promotion_observation"), Mapping)
    ]
    rows: dict[str, list[dict[str, Any]]] = {}
    for candidate in observations:
        arm = str(candidate.get("arm_id", ""))
        if not arm.startswith("gdn2-mimo-r4-braid-"):
            continue
        baseline_arm = candidate.get("baseline_arm_id")
        matches = [baseline for baseline in observations if (
            baseline.get("arm_id") == baseline_arm
            and baseline.get("seed") == candidate.get("seed")
            and baseline.get("pairing_id") == candidate.get("pairing_id")
            and baseline.get("checkpoint_sha256") == candidate.get("checkpoint_sha256")
            and baseline.get("data_sha256") == candidate.get("data_sha256")
            and baseline.get("cells") == candidate.get("cells")
        )]
        if len(matches) != 1:
            continue
        baseline = matches[0]
        rows.setdefault(arm, []).append({
            "seed": candidate["seed"], "baseline": baseline["recall"],
            "candidate": candidate["recall"],
            "free_generation": candidate["free_generation"],
            "lm_loss": candidate["lm_loss"], "baseline_lm_loss": baseline["lm_loss"],
            "nonfinite_count": candidate["nonfinite_count"],
            "capacity_confounded": candidate["capacity_confounded"],
            "effective_rank": candidate.get("effective_rank", 0.0),
            "decay_horizon_ratios": candidate.get("decay_horizon_ratios", [1.0, 16.0, 64.0, 256.0]),
            "thresholds": candidate.get("thresholds", {}),
        })
    return rows


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publication-stage hook; rollback intentionally bypasses this wrapper."""

    _replace_bytes(path, payload)


def _artifact_payloads(artifacts: SummaryArtifacts) -> dict[str, bytes]:
    return {
        "ledger.jsonl": artifacts.ledger_jsonl,
        "results.json": artifacts.results_json,
        "results.csv": artifacts.results_csv,
    }


def _generation_manifest(payloads: Mapping[str, bytes]) -> tuple[str, bytes]:
    files = {
        name: {
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "size": len(payloads[name]),
        }
        for name in _SUMMARY_PAYLOAD_NAMES
    }
    generation_id = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generation_id": generation_id,
        "generation_path": f".generations/{generation_id}",
        "files": files,
    }
    return generation_id, canonical_json_bytes(manifest) + b"\n"


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _prepare_generation(
    output: Path, payloads: Mapping[str, bytes], generation_id: str, manifest: bytes
) -> None:
    generations = output / ".generations"
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations / generation_id
    if destination.exists():
        expected = dict(payloads)
        expected["generation.json"] = manifest
        if any(
            not (destination / name).is_file()
            or (destination / name).read_bytes() != payload
            for name, payload in expected.items()
        ):
            raise SummaryValidationError(
                f"summary generation {generation_id} exists with conflicting content"
            )
        return
    temporary = generations / f".{generation_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name in _SUMMARY_PAYLOAD_NAMES:
            _write_new_file(temporary / name, payloads[name])
        _write_new_file(temporary / "generation.json", manifest)
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _snapshot(paths: Sequence[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def _restore_snapshot(snapshot: Mapping[Path, bytes | None]) -> None:
    for path, payload in snapshot.items():
        if payload is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        else:
            _replace_bytes(path, payload)


def write_summary_artifacts(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | os.PathLike[str],
    *,
    expected_jobs: Iterable[Mapping[str, Any]] = (),
    promotion: bool = False,
) -> SummaryArtifacts:
    """Atomically publish deterministic ledger, JSON, and CSV artifacts."""

    artifacts = build_summary_artifacts(
        records, expected_jobs=expected_jobs, promotion=promotion
    )
    output = Path(output_dir)
    payloads = _artifact_payloads(artifacts)
    generation_id, manifest = _generation_manifest(payloads)
    _prepare_generation(output, payloads, generation_id, manifest)
    public_paths = [output / name for name in _SUMMARY_PAYLOAD_NAMES]
    pointer_path = output / _CURRENT_POINTER_NAME
    snapshot = _snapshot((*public_paths, pointer_path))
    try:
        for path in public_paths:
            _atomic_write(path, payloads[path.name])
        _atomic_write(pointer_path, manifest)
    except BaseException:
        _restore_snapshot(snapshot)
        raise
    return artifacts


def _read_cli_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SummaryValidationError(f"cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise SummaryValidationError(f"{label} must be a JSON object")
    return value


def cli_handler(arguments: Any) -> dict[str, Any]:
    """Production CLI adapter over immutable jobs and authoritative run records."""

    from .results import (
        ResultStore,
        validate_completed_run,
        validate_failed_run,
    )

    root = Path(arguments.out).expanduser().resolve()
    manifest = _read_cli_json(root / "manifest.json", label="manifest")
    jobs_document = _read_cli_json(root / "jobs.json", label="jobs document")
    jobs = jobs_document.get("jobs")
    if (
        not isinstance(jobs, Sequence)
        or isinstance(jobs, (str, bytes, bytearray))
        or any(not isinstance(job, Mapping) for job in jobs)
    ):
        raise SummaryValidationError("jobs.json.jobs must be a sequence of objects")
    by_job = {job.get("job_id"): job for job in jobs}
    if (
        len(by_job) != len(jobs)
        or any(type(job_id) is not str or not job_id for job_id in by_job)
    ):
        raise SummaryValidationError("jobs.json contains invalid or duplicate job IDs")
    provenance_fields = (
        "schema_version",
        "suite_version",
        "source_hashes",
        "config_hash",
        "asset_hashes",
        "git",
        "environment",
    )
    try:
        provenance = {field: manifest[field] for field in provenance_fields}
    except KeyError as error:
        raise SummaryValidationError(
            f"manifest is missing provenance field: {error.args[0]}"
        ) from error
    store = ResultStore(root, provenance=provenance, job_index=0, num_jobs=1)
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    runs_root = root / "runs"
    for path in sorted(runs_root.rglob("*.json")) if runs_root.is_dir() else ():
        record = _read_cli_json(path, label="run record")
        job_id = record.get("job_id")
        if type(job_id) is not str or job_id not in by_job:
            raise SummaryValidationError(f"run record has an unknown job ID: {path}")
        if job_id in seen:
            raise SummaryValidationError(f"duplicate authoritative run record: {job_id}")
        job = by_job[job_id]
        if path.resolve() != store.run_path(job).resolve():
            raise SummaryValidationError(f"run record is outside its canonical path: {path}")
        status = record.get("status")
        try:
            if status == "completed":
                validate_completed_run(record, job, provenance)
            elif status == "failed":
                validate_failed_run(record, job, provenance)
            else:
                raise SummaryValidationError(
                    f"run record has an invalid execution status: {path}"
                )
        except (TypeError, ValueError) as error:
            raise SummaryValidationError(f"run record failed validation: {path}") from error
        seen.add(job_id)
        records.append(record)
    canonical = manifest.get("canonical_config")
    task = canonical.get("task", {}) if isinstance(canonical, Mapping) else {}
    params = task.get("params", {}) if isinstance(task, Mapping) else {}
    promotion = bool(
        isinstance(canonical, Mapping)
        and len(records) == len(jobs)
        and all(record.get("status") == "completed" for record in records)
        and task.get("name") == "ruler"
        and canonical.get("required_stage") == "qwen_heal"
        and isinstance(params, Mapping)
        and params.get("episodes_per_cell", 0) >= RULER_MIN_EPISODES_PER_CELL
    )
    artifacts = write_summary_artifacts(
        records,
        root / "summary",
        expected_jobs=jobs,
        promotion=promotion,
    )
    results = json.loads(artifacts.results_json)
    return {
        "ok": True,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "codes": [],
        "warnings": [],
        "records": len(records),
        "execution_counts": results["execution_counts"],
        "ledger_sha256": results["ledger_sha256"],
        "summary_path": str((root / "summary").resolve()),
    }


def decide_option3_summary(
    surprise: Option3Evidence,
    recency: Option3Evidence,
    thresholds: Option3Thresholds,
    *,
    evidence_scope: str = "promotion",
) -> Option3Decision:
    """Apply the Task 8 Option-3 gates without copying any thresholds."""

    if evidence_scope != "promotion":
        raise SummaryValidationError("feasibility evidence cannot drive promotion")
    return decide_option3(surprise, recency, thresholds)


def classify_factorial_addition(
    cells: Mapping[str, Sequence[MetricSample]],
    *,
    metric: str,
    protected: Iterable[ProtectedEffect],
    valid: bool,
    min_useful: float,
    harm_threshold: float,
    min_synergy: float,
    random_seed: int,
    resamples: int,
) -> ClassifiedFactorial:
    """Bootstrap all four matched cells, then delegate the ordered addition label."""

    direction = metric_direction(metric)
    if direction is None:
        raise ValueError("diagnostic metrics cannot drive factorial classification")
    statistics = paired_factorial_bootstrap(
        cells, direction=direction, random_seed=random_seed, resamples=resamples
    )
    label = classify_addition(
        metric=metric,
        primary=statistics.current_effect,
        protected=protected,
        valid=valid,
        min_useful=min_useful,
        harm_threshold=harm_threshold,
        min_synergy=min_synergy,
        interaction=statistics.interaction,
        existing_feature_off=statistics.feature_off_effect,
    )
    return ClassifiedFactorial(label, statistics)


def classify_paired_reliance(
    *,
    current: Sequence[MetricSample],
    ablated: Sequence[MetricSample],
    metric: str,
    valid: bool,
    min_reliance: float,
    equivalence: float,
    harm_threshold: float,
    random_seed: int,
    resamples: int,
) -> ClassifiedReliance:
    """Bootstrap matched current-minus-ablated evidence and delegate its label."""

    direction = metric_direction(metric)
    if direction is None:
        raise ValueError("diagnostic metrics cannot drive reliance classification")
    interval = paired_bootstrap(
        current,
        ablated,
        direction=direction,
        random_seed=random_seed,
        resamples=resamples,
    )
    label = classify_reliance(
        metric=metric,
        effect=interval,
        valid=valid,
        min_reliance=min_reliance,
        equivalence=equivalence,
        harm_threshold=harm_threshold,
    )
    return ClassifiedReliance(label, interval)


def aggregate_hybrid_promotion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Make an explicit three-seed hybrid decision; never infer promotion."""
    values = tuple(rows)
    if len(values) != 3 or any(not isinstance(row, Mapping) for row in values):
        return {"decision": "no_promote", "reasons": ("missing_seed_runs",),
                "seed_count": len(values)}
    seeds = tuple(row.get("seed") for row in values)
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 3:
        return {"decision": "no_promote", "reasons": ("unmatched_seeds",),
                "seed_count": len(set(seed for seed in seeds if type(seed) is int))}
    required = {
        "seed", "baseline", "candidate", "free_generation", "lm_loss",
        "nonfinite_count", "capacity_confounded",
    }
    optional = {"baseline_lm_loss", "effective_rank", "decay_horizon_ratios", "thresholds"}
    if any(not required <= set(row) or set(row) - required - optional for row in values):
        raise SummaryValidationError("hybrid promotion seed fields must be exact")
    for row in values:
        for field in ("baseline", "candidate", "free_generation", "lm_loss"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SummaryValidationError(f"hybrid promotion {field} must be finite")
        if type(row["nonfinite_count"]) is not int or row["nonfinite_count"] < 0:
            raise SummaryValidationError("hybrid promotion nonfinite_count must be nonnegative")
        if type(row["capacity_confounded"]) is not bool:
            raise SummaryValidationError("hybrid promotion capacity_confounded must be bool")
    reasons: list[str] = []
    threshold_rows = [row.get("thresholds", {}) for row in values]
    if any(not isinstance(item, Mapping) for item in threshold_rows) or any(
        dict(item) != dict(threshold_rows[0]) for item in threshold_rows[1:]
    ):
        raise SummaryValidationError("hybrid promotion thresholds must be matched")
    thresholds = {"min_gain": 0.0, "min_free_generation": 0.5,
                  "max_lm_loss": float("inf"), "max_lm_loss_increase": 0.0,
                  "min_effective_rank": 1.5,
                  "bootstrap_resamples": 2000, **dict(threshold_rows[0])}
    if any(row["capacity_confounded"] for row in values):
        reasons.append("capacity_confounded")
    if any(row["nonfinite_count"] for row in values):
        reasons.append("nonfinite")
    gains = [float(row["candidate"]) - float(row["baseline"]) for row in values]
    if any(gain <= float(thresholds["min_gain"]) for gain in gains):
        reasons.append("no_paired_improvement")
    from .metrics import MetricSample, paired_bootstrap
    candidate_samples = [MetricSample(int(row["seed"]), str(row["seed"]), "hybrid", float(row["candidate"]), 1.0) for row in values]
    baseline_samples = [MetricSample(int(row["seed"]), str(row["seed"]), "hybrid", float(row["baseline"]), 1.0) for row in values]
    interval = paired_bootstrap(candidate_samples, baseline_samples, direction=1,
                                random_seed=0, resamples=int(thresholds["bootstrap_resamples"]))
    if interval.lower <= 0:
        reasons.append("paired_ci_not_positive")
    if min(float(row["free_generation"]) for row in values) < float(thresholds["min_free_generation"]):
        reasons.append("free_generation_below_threshold")
    if max(float(row["lm_loss"]) for row in values) > float(thresholds["max_lm_loss"]):
        reasons.append("lm_loss_above_threshold")
    if any("baseline_lm_loss" in row and float(row["lm_loss"]) - float(row["baseline_lm_loss"])
           > float(thresholds["max_lm_loss_increase"]) for row in values):
        reasons.append("lm_loss_regression")
    if any(float(row.get("effective_rank", 4.0)) < float(thresholds["min_effective_rank"]) for row in values):
        reasons.append("rank_collapse")
    if any(tuple(round(float(x), 3) for x in row.get("decay_horizon_ratios", ()))
           != (1.0, 16.0, 64.0, 256.0) for row in values):
        reasons.append("decay_timescale_ratio_mismatch")
    return {
        "decision": "no_promote" if reasons else "promote",
        "reasons": tuple(reasons),
        "seed_count": 3,
    }


__all__ = [
    "SUMMARY_SCHEMA_VERSION",
    "ClassifiedFactorial",
    "ClassifiedReliance",
    "SummaryArtifacts",
    "SummaryValidationError",
    "build_summary_artifacts",
    "aggregate_hybrid_promotion",
    "classify_factorial_addition",
    "classify_paired_reliance",
    "cli_handler",
    "decide_option3_summary",
    "write_summary_artifacts",
]
