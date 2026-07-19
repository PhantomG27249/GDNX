"""Deterministic sharded execution for portable KMD-2 experiments."""

from __future__ import annotations

import ast
import copy
import functools
import importlib
import importlib.metadata
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


_MAXIMUM_REQUIRED_ASSETS = ("model", "tokenizer", "checkpoint", "data", "teacher_model")


def select_maximum_package_controls(
    control_ids: Sequence[str], package: str
) -> tuple[str, ...]:
    """Select package-local controls plus shared controls in stable order."""
    normalized = package.casefold()
    if normalized not in {"a", "b", "both"}:
        raise ValueError("package must be A, B, or both")
    opposite = "package-b-" if normalized == "a" else "package-a-"
    return tuple(
        control_id
        for control_id in control_ids
        if normalized == "both" or not control_id.startswith(opposite)
    )


def _maximum_asset_identities(raw: Mapping[str, Any]) -> dict[str, str]:
    assets = raw.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("maximum campaign requires full asset identities")
    identities: dict[str, str] = {}
    for name in _MAXIMUM_REQUIRED_ASSETS:
        item = assets.get(name)
        if not isinstance(item, Mapping):
            raise ValueError("maximum campaign requires full asset identities")
        identity = item.get("tree_sha256") or item.get("sha256")
        path = item.get("path")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(identity, str)
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise ValueError("maximum campaign requires full asset identities")
        identities[name] = identity
    return identities


def materialize_maximum_hybrid_campaign(
    campaign_path: Path | str,
    assets_manifest: Mapping[str, Any] | Path | str,
    destination: Path | str,
    *,
    smoke: bool = False,
) -> tuple[Path, ...]:
    """Materialize all full control documents with immutable asset identities."""
    campaign_file = Path(campaign_path)
    campaign = json.loads(campaign_file.read_text(encoding="utf-8"))
    if isinstance(assets_manifest, (str, Path)):
        asset_document = json.loads(Path(assets_manifest).read_text(encoding="utf-8"))
    else:
        asset_document = dict(assets_manifest)
    identities = _maximum_asset_identities(asset_document)
    controls = campaign.get("controls")
    if not isinstance(controls, list) or len(controls) != 11:
        raise ValueError("maximum campaign requires exactly eleven controls")
    specialization = campaign.get("specialization")
    if (
        not isinstance(specialization, Mapping)
        or specialization.get("lambda_spec") != 0.001
        or specialization.get("lambda_gate") != 0.001
        or specialization.get("specialization_updates") != 8
    ):
        raise ValueError("maximum campaign specialization contract is missing or stale")
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    base_path = campaign_file.parents[3] / campaign.get(
        "base_config", "research/kmd2_ablation/configs/qwen_exact_cache.json"
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    written: list[Path] = []
    for control in controls:
        source = campaign_file.parents[3] / control["config"]
        overlay = json.loads(source.read_text(encoding="utf-8"))
        document = copy.deepcopy(base)
        control_id = overlay["control_id"]
        from .qwen_variants import maximum_control_contract
        contract = maximum_control_contract(control_id)
        mechanism_variant = {
            "gdn2-r1": ("gdn2_decoupled", "channelwise_erase_write"),
            "gdn2-mimo-r2": ("true_mimo", "true_mimo_sweep"),
            "gdn2-mimo-r4": ("true_mimo", "true_mimo_sweep"),
            "package-a-native-decay": ("maximum_hybrid", "braid_shared_hola_w64"),
            "package-a-braid-no-cache": ("maximum_hybrid", "braid_shared_hola_w64"),
            "package-a-recency-w64": ("maximum_hybrid", "braid_shared_hola_w64"),
            "package-a-hola-w64": ("maximum_hybrid", "braid_shared_hola_w64"),
            "package-b-recency-w64": ("maximum_hybrid", "braid_four_state_hola_w64"),
            "package-b-hola-w64": ("maximum_hybrid", "braid_four_state_hola_w64"),
            "shared-query-widening": ("state_size", "state_size_sweep"),
            "stock-qwen": ("native", "native"),
        }[control_id]
        document["mechanism"], document["variant"] = mechanism_variant
        document["task"]["params"].update({
            "maximum_control": control_id,
            "maximum_features": asdict(contract),
            "maximum_contract": asdict(contract),
            "maximum_contract_sha256": contract.identity_sha256,
            "capacity_label": overlay["classification"],
            "asset_identities": identities,
            "lambda_spec": specialization["lambda_spec"],
            "lambda_gate": specialization["lambda_gate"],
            "specialization_updates": specialization["specialization_updates"],
        })
        architecture_ids = {
            "gdn2-r1": "gdn2-channel-r1", "gdn2-mimo-r2": "mimo-r2",
            "gdn2-mimo-r4": "mimo-r4", "package-a-native-decay": "gdn2-mimo-r4-braid-shared-hola-w64",
            "package-a-braid-no-cache": "gdn2-mimo-r4-braid-shared-hola-w64",
            "package-a-recency-w64": "gdn2-mimo-r4-braid-shared-hola-w64",
            "package-a-hola-w64": "gdn2-mimo-r4-braid-shared-hola-w64",
            "package-b-recency-w64": "gdn2-mimo-r4-braid-four-state-hola-w64",
            "package-b-hola-w64": "gdn2-mimo-r4-braid-four-state-hola-w64",
            "shared-query-widening": "rout-4",
        }
        if control_id in architecture_ids:
            from .architecture import architecture_record, registry_sha256
            record = architecture_record(architecture_ids[control_id])
            document["architecture"] = {
                "arm_id": record.arm_id, "registry_sha256": registry_sha256(),
                "model_tree_sha256": identities["model"],
                "ordered_examples_sha256": identities["data"],
                "pretraining_checkpoint_sha256": identities["checkpoint"],
            }
            document["task"]["params"].update({
                "target_layers": list(record.target_layers), "output_width": record.output_width,
                "mimo_rank": record.mimo_rank, "gate_mode": record.gate_mode,
            })
            if record.arm_id.startswith("gdn2-mimo-r4-braid-"):
                document["task"]["params"].update({
                    "timescales": list(record.timescales), "state_topology": record.state_topology,
                    "cache_scope": record.cache_scope, "update_paths": 4, "convolution_on": True,
                    "rotation_mode": "mamba3_complex_input_dependent_cumulative",
                    "phase_topology": record.phase_topology, "input_rank": 4, "output_rank": 4,
                    "read_topology": record.read_topology, "read_paths": record.read_paths,
                    "state_input_mode": "trapezoid", "lookahead": record.lookahead,
                    "qk_contract": record.qk_contract,
                })
                document["cache"]["read_init"] = record.cache.read_init
        if control_id == "stock-qwen":
            document["qwen"]["run_mode"] = "reliance"
            document["required_stage"] = "qwen_reliance"
        if smoke:
            params = document["task"]["params"]
            params["objective"] = "synthetic_only"
            params["example_ids"] = params["example_ids"][:1]
            params["training_window_example_counts"] = [1]
            params["training_window_token_counts"] = [64]
            params["episodes_per_cell"] = 1
            params["free_generation_subset_per_cell"] = 1
            document["budget"] = {"tokens": 64, "updates": 1}
            document["lengths"] = {"curriculum": [64], "extrapolation": [64]}
            if "architecture" not in document:
                document["cache"]["width"] = 4
                document["cache"]["block_size"] = 16
        target = output / Path(control["config"]).name
        target.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        written.append(target)
    return tuple(written)


def run_campaign(
    campaign: Mapping[str, Any], *,
    preflight: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run a campaign only after every control has passed preflight."""
    controls = campaign.get("controls")
    if not isinstance(controls, list) or len(controls) != 11:
        raise ValueError("maximum campaign requires exactly eleven controls")
    preflights = [dict(preflight(control)) for control in controls]
    failed = [
        controls[index]["control_id"]
        for index, report in enumerate(preflights)
        if report.get("ok") is not True
    ]
    if failed:
        return {"ok": False, "phase": "preflight", "failed_controls": failed, "runs": []}
    runs = [dict(execute(control)) for control in controls]
    return {"ok": all(run.get("ok") is True for run in runs), "preflights": preflights, "runs": runs}


def promotion_decision(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a visible, fail-closed decision for the immutable three-seed screen."""
    required = (11, 29, 47)
    by_seed = {item.get("seed"): item for item in seed_results}
    missing = [seed for seed in required if seed not in by_seed]
    if missing:
        return {"promoted": False, "reason": "missing_required_seeds", "missing_seeds": missing}
    failed = [seed for seed in required if by_seed[seed].get("passed") is not True]
    if failed:
        return {"promoted": False, "reason": "seed_threshold_failed", "failed_seeds": failed}
    return {"promoted": True, "reason": "all_required_seeds_passed"}

from .config import SUITE_VERSION, ExperimentConfig
from .results import (
    RESULT_SCHEMA_VERSION,
    ResultStore,
    RunRecordError,
    assign_shard,
    build_job,
    build_jobs_document,
    build_manifest,
    canonical_json_bytes,
    select_shard,
    validate_completed_run,
    validate_failed_run,
)


BackendDispatcher = Callable[[Mapping[str, Any]], Mapping[str, Any]]
_MAX_TRACEBACK_CHARS = 8192
_RESERVED_FIELDS = {
    "schema_version",
    "suite_version",
    "status",
    "job_id",
    "experiment_id",
    "seed",
    "stage",
    "backend",
    "arm_id",
    "pairing_id",
    "shard",
    "provenance",
    "canonical_config",
    "command",
    "error",
    "scientific_classification",
    "scientific_label",
}


class JobFailure(RuntimeError):
    """An expected typed failure that must become an atomic failed record."""

    code = "execution_error"

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if type(message) is not str or not message:
            raise TypeError("failure message must be a nonempty str")
        if type(phase) is not str or not phase:
            raise TypeError("failure phase must be a nonempty str")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("failure context must be a mapping")
        super().__init__(message)
        self.phase = phase
        self.context = {} if context is None else dict(context)


class ForcedOOM(JobFailure):
    code = "oom"


class NonFiniteLoss(JobFailure):
    code = "nonfinite_loss"


class NonFiniteGradient(JobFailure):
    code = "nonfinite_gradient"


class MalformedInput(JobFailure):
    code = "malformed_input"


class BackendUnavailable(JobFailure):
    code = "backend_unavailable"


class PreflightCheckError(ValueError):
    """One preflight check failed with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("preflight error code must be a nonempty str")
        if type(message) is not str or not message:
            raise TypeError("preflight error message must be a nonempty str")
        self.code = code
        super().__init__(message)


def _dependency_versions(backend: str) -> dict[str, str | None]:
    names = ("transformers", "triton") if backend == "qwen" else ()
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "triton" and platform.system() == "Windows":
                try:
                    versions[name] = importlib.metadata.version("triton-windows")
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            else:
                versions[name] = None
    return versions


@functools.lru_cache(maxsize=2)
def _dependency_capabilities(backend: str) -> dict[str, bool]:
    """Verify optional imports in an isolated process, never in this runner."""

    if backend != "qwen":
        return {}
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from transformers.models.qwen3_5.modeling_qwen3_5 "
                    "import Qwen3_5RMSNormGated as C; "
                    "assert C.__name__ == 'Qwen3_5RMSNormGated'"
                ),
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        available = completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        available = False
    return {"transformers_qwen3_5": available}


def _device_ordinal(device: str) -> int | None:
    if device == "cpu":
        return None
    if not device.startswith("cuda:"):
        return -1
    raw = device.partition(":")[2]
    if not raw.isdigit():
        return -1
    return int(raw)


def probe_environment(
    *,
    backend: str,
    device_preferences: Sequence[str],
    dtype_preferences: Sequence[str],
    student_device: str | None = None,
    teacher_device: str | None = None,
    requested_dtype: str | None = None,
    python_version: Sequence[int] | None = None,
    python_implementation: str | None = None,
    torch_module: Any | None = None,
    dependency_versions: Mapping[str, str | None] | None = None,
    dependency_capabilities: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Inspect Python, PyTorch, optional packages, devices, and dtype support.

    Package versions are read from distribution metadata. The Qwen symbol is
    imported once in an isolated child process, never into this runner process;
    Triton is not imported. ``torch_module`` is injectable for tests.
    """

    if backend not in {"tiny", "qwen"}:
        raise ValueError("backend must be tiny or qwen")
    version = tuple(sys.version_info[:3] if python_version is None else python_version)
    if len(version) < 3 or any(type(item) is not int for item in version[:3]):
        raise TypeError("python_version must contain three integers")
    implementation = python_implementation or platform.python_implementation()
    codes: list[str] = []
    warnings: list[str] = []
    if version[:2] < (3, 10):
        codes.append("python_version_unsupported")

    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError):
            torch_module = None
            codes.append("dependency_missing:pytorch")

    dependencies = dict(
        _dependency_versions(backend)
        if dependency_versions is None
        else dependency_versions
    )
    capabilities = dict(
        _dependency_capabilities(backend)
        if dependency_capabilities is None
        else dependency_capabilities
    )
    if any(type(name) is not str or type(value) is not bool for name, value in capabilities.items()):
        raise TypeError("dependency_capabilities must map strings to bools")
    if backend == "qwen" and dependencies.get("transformers") is None:
        codes.append("dependency_missing:transformers")
    elif backend == "qwen" and capabilities.get("transformers_qwen3_5") is not True:
        codes.append("transformers_qwen3_5_unavailable")
    if backend == "qwen" and dependencies.get("triton") is None:
        codes.append("dependency_missing:triton")

    torch_version = (
        "unavailable"
        if torch_module is None
        else str(getattr(torch_module, "__version__", "unknown"))
    )
    cuda_version = None
    cuda_available = False
    device_count = 0
    gpu_names: list[str] = []
    if torch_module is not None:
        cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None:
            try:
                cuda_available = bool(cuda.is_available())
                device_count = int(cuda.device_count()) if cuda_available else 0
                gpu_names = [str(cuda.get_device_name(index)) for index in range(device_count)]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                cuda_available = False
                device_count = 0
                gpu_names = []

    preferences = tuple(device_preferences)
    if student_device is None:
        student_device = (
            "cuda:0"
            if "cuda" in preferences and cuda_available and device_count
            else "cpu"
        )
    selected_dtype = requested_dtype or (
        str(dtype_preferences[0]) if dtype_preferences else "float32"
    )
    requested_devices = [student_device]
    if teacher_device is not None:
        requested_devices.append(teacher_device)
    for device in requested_devices:
        ordinal = _device_ordinal(device)
        if ordinal == -1:
            codes.append(f"device_invalid:{device}")
        elif ordinal is not None and (
            not cuda_available or ordinal >= device_count
        ):
            codes.append(f"cuda_device_unavailable:{device}")

    if selected_dtype not in {"bfloat16", "float32"}:
        codes.append(f"dtype_unsupported:{selected_dtype}")
    elif selected_dtype == "bfloat16" and any(
        device.startswith("cuda:") for device in requested_devices
    ):
        bf16_supported = False
        if torch_module is not None and getattr(torch_module, "cuda", None) is not None:
            try:
                bf16_supported = bool(torch_module.cuda.is_bf16_supported())
            except (AttributeError, RuntimeError):
                bf16_supported = False
        if not bf16_supported:
            codes.append("dtype_unsupported:bfloat16")

    codes = list(dict.fromkeys(codes))
    warnings = list(dict.fromkeys(warnings))
    return {
        "ok": not codes,
        "codes": codes,
        "warnings": warnings,
        "environment": {
            "python": f"{implementation} {version[0]}.{version[1]}.{version[2]}",
            "pytorch": torch_version,
            "cuda": None if cuda_version is None else str(cuda_version),
            "gpu": gpu_names,
            "dependencies": dict(sorted(dependencies.items())),
            "capabilities": dict(sorted(capabilities.items())),
        },
        "resources": {
            "student_device": student_device,
            "teacher_device": teacher_device,
            "dtype": selected_dtype,
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
        },
    }


def validate_output_writable(output: str | os.PathLike[str]) -> Path:
    """Create the result root and prove write/flush/remove access without residue."""

    path = Path(output).expanduser().resolve()
    probe = path / f".preflight-write-{uuid.uuid4().hex}"
    try:
        path.mkdir(parents=True, exist_ok=True)
        with probe.open("xb") as stream:
            stream.write(b"kmd2-preflight\n")
            stream.flush()
            os.fsync(stream.fileno())
        probe.unlink()
    except OSError as error:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        raise PreflightCheckError(
            "output_not_writable", f"output directory is not writable: {path}"
        ) from error
    return path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise PreflightCheckError("asset_missing", f"external asset is missing: {resolved}")
    if resolved.is_symlink():
        raise PreflightCheckError(
            "asset_symlink_unsupported", f"external asset is a symlink: {resolved}"
        )
    if resolved.is_file():
        size = resolved.stat().st_size
        digest = _hash_file(resolved)
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": size,
            "sha256": digest,
            "tree_manifest": [],
        }
    if not resolved.is_dir():
        raise PreflightCheckError(
            "asset_kind_unsupported", f"external asset is not a file or directory: {resolved}"
        )
    entries: list[dict[str, Any]] = []
    for child in sorted(
        resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()
    ):
        if child.is_symlink():
            raise PreflightCheckError(
                "asset_symlink_unsupported",
                f"directory asset contains a symlink: {child}",
            )
        if child.is_file():
            entries.append(
                {
                    "path": child.relative_to(resolved).as_posix(),
                    "size_bytes": child.stat().st_size,
                    "sha256": _hash_file(child),
                }
            )
    encoded = json.dumps(
        [
            [item["path"], item["size_bytes"], item["sha256"]]
            for item in entries
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": str(resolved),
        "kind": "directory",
        "size_bytes": sum(item["size_bytes"] for item in entries),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "tree_manifest": entries,
    }


def _expected_tree(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise PreflightCheckError(
            "asset_manifest_invalid", "asset files manifest must be a mapping"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for relative, identity in value.items():
        if type(relative) is not str or not isinstance(identity, Mapping):
            raise PreflightCheckError(
                "asset_manifest_invalid", "asset tree entries must be mappings"
            )
        normalized[relative] = {
            "size_bytes": identity.get("size_bytes"),
            "sha256": identity.get("sha256"),
        }
    return normalized


def inspect_external_assets(
    assets: Mapping[str, str | os.PathLike[str]],
    *,
    expected: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hash external files/trees and enforce optional root and per-file identities."""

    if not isinstance(assets, Mapping):
        raise TypeError("assets must be a mapping")
    expected_map = {} if expected is None else dict(expected)
    if set(expected_map) - set(assets):
        raise PreflightCheckError(
            "asset_manifest_mismatch", "asset manifest declares unknown assets"
        )
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(assets):
        if type(name) is not str or not name:
            raise TypeError("asset names must be nonempty strings")
        record = _asset_record(Path(assets[name]))
        declaration = expected_map.get(name)
        if declaration is not None:
            if not isinstance(declaration, Mapping):
                raise PreflightCheckError(
                    "asset_manifest_invalid", f"asset {name} identity must be a mapping"
                )
            if "kind" in declaration and declaration["kind"] != record["kind"]:
                raise PreflightCheckError(
                    "asset_kind_mismatch", f"external asset {name} kind does not match"
                )
            if (
                "size_bytes" in declaration
                and declaration["size_bytes"] != record["size_bytes"]
            ):
                raise PreflightCheckError(
                    "asset_size_mismatch", f"external asset {name} size does not match"
                )
            if "sha256" in declaration and declaration["sha256"] != record["sha256"]:
                raise PreflightCheckError(
                    "asset_hash_mismatch", f"external asset {name} SHA-256 does not match"
                )
            if "files" in declaration:
                actual_tree = {
                    item["path"]: {
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in record["tree_manifest"]
                }
                if _expected_tree(declaration["files"]) != actual_tree:
                    raise PreflightCheckError(
                        "asset_tree_mismatch",
                        f"external asset {name} tree manifest does not match",
                    )
        result[name] = record
    return result


def _mapping_at(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _is_exact_three_seed_matrix(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 3
        and all(type(seed) is int and seed >= 0 for seed in value)
        and len(set(value)) == 3
    )


def validate_raw_scientific_config(
    raw: Mapping[str, Any], *, backend: str, mode: str | None
) -> list[str]:
    """Return named scientific failures that must precede schema construction."""

    if not isinstance(raw, Mapping):
        return ["config_not_mapping"]
    codes: list[str] = []
    cache = _mapping_at(raw, "cache")
    lengths = _mapping_at(raw, "lengths")
    model = _mapping_at(raw, "model")
    qwen = _mapping_at(raw, "qwen")
    task = _mapping_at(raw, "task")
    params = _mapping_at(task, "params")
    mechanism = raw.get("mechanism")
    variant = raw.get("variant")
    width = cache.get("width")
    block_size = cache.get("block_size")
    curriculum = lengths.get("curriculum")

    cache_variants = {
        "top_surprise",
        "coupled_surprise",
        "residual_only",
        "write_value_only",
        "recency",
        "reservoir",
        "future_query_oracle",
        "unbounded_oracle",
        "per_slot_read",
        "cache_rotation_factorial",
        "cache_r_out_factorial",
    }
    if mechanism == "exact_cache" and variant in cache_variants:
        if type(width) is not int or width < 1:
            codes.append("cache_width_invalid")
        if (
            type(block_size) is int
            and block_size > 0
            and isinstance(curriculum, Sequence)
            and not isinstance(curriculum, (str, bytes, bytearray))
            and curriculum
            and all(type(item) is int for item in curriculum)
        ):
            maximum = max(curriculum)
            if maximum < 2 * block_size:
                codes.append("cache_requires_two_blocks")
            if type(width) is int and width >= maximum:
                codes.append("cache_eviction_impossible")

    ffn_dim = model.get("ffn_dim")
    ffn_lower = model.get("ffn_match_lower")
    ffn_upper = model.get("ffn_match_upper")
    if (
        type(ffn_dim) is not int
        or type(ffn_lower) is not int
        or type(ffn_upper) is not int
        or not ffn_lower <= ffn_dim <= ffn_upper
        or any(value % 8 for value in (ffn_dim, ffn_lower, ffn_upper))
    ):
        codes.append("ffn_match_invalid")

    if variant in {"cache_rotation_factorial", "cache_r_out_factorial"}:
        cells = params.get("four_cells")
        if (
            not isinstance(cells, Sequence)
            or isinstance(cells, (str, bytes, bytearray))
            or set(cells) != {"M00", "M10", "M01", "M11"}
        ):
            codes.append("four_cell_incomplete")

    option3 = params.get("option3_inputs")
    if option3 is not None:
        required_option3 = {"surprise", "recency"}
        if not isinstance(option3, Mapping):
            codes.append("option3_inputs_incomplete")
        else:
            option3_keys = set(option3)
            if not required_option3 <= option3_keys:
                codes.append("option3_inputs_incomplete")
            if option3_keys - required_option3:
                codes.append("option3_inputs_unknown")

    if backend == "qwen":
        restrictions = (
            ("streaming", False, "qwen_streaming_unsupported"),
            ("decode", False, "qwen_decode_unsupported"),
            ("packing", False, "qwen_packing_unsupported"),
            ("padding", "none", "qwen_padding_unsupported"),
        )
        for field, allowed, code in restrictions:
            if qwen.get(field) != allowed:
                codes.append(code)
        if qwen.get("attention_mask") not in {"none", "all_ones"}:
            codes.append("qwen_attention_mask_unsupported")
        if mode is not None and qwen.get("run_mode") != mode:
            codes.append("qwen_mode_mismatch")
        if raw.get("required_stage") == "qwen_heal" and mechanism == "exact_cache":
            if not _is_exact_three_seed_matrix(raw.get("seeds")):
                codes.append("qwen_seed_matrix_invalid")
            if "synthetic_only" in params:
                codes.append("qwen_synthetic_only_declaration_invalid")
            example_ids = params.get("example_ids")
            if (
                not isinstance(example_ids, Sequence)
                or isinstance(example_ids, (str, bytes, bytearray))
                or not example_ids
                or any(type(item) is not str or not item for item in example_ids)
                or len(set(example_ids)) != len(example_ids)
            ):
                codes.append("qwen_example_ids_invalid")
            if type(params.get("native_r_out")) is not int or params.get(
                "native_r_out"
            ) < 1:
                codes.append("qwen_r_out_unpinned")
            if params.get("score_scan") != (
                "gdn3.kmd2_fast_scan.scan_with_update_norm"
            ):
                codes.append("qwen_fast_scan_unpinned")

            budget = _mapping_at(raw, "budget")
            updates = budget.get("updates")
            tokens = budget.get("tokens")
            accumulation = params.get("accumulation_steps", 1)
            expected_windows = (
                updates * accumulation
                if type(updates) is int
                and updates > 0
                and type(accumulation) is int
                and accumulation > 0
                else None
            )
            token_counts = params.get("training_window_token_counts")
            example_counts = params.get("training_window_example_counts")
            window_contract_valid = (
                expected_windows is not None
                and isinstance(token_counts, Sequence)
                and not isinstance(token_counts, (str, bytes, bytearray))
                and len(token_counts) == expected_windows
                and all(type(item) is int and item > 0 for item in token_counts)
                and type(tokens) is int
                and tokens > 0
                and sum(token_counts) == tokens
                and isinstance(example_counts, Sequence)
                and not isinstance(example_counts, (str, bytes, bytearray))
                and len(example_counts) == expected_windows
                and all(type(item) is int and item > 0 for item in example_counts)
                and isinstance(example_ids, Sequence)
                and not isinstance(example_ids, (str, bytes, bytearray))
                and sum(example_counts) == len(example_ids)
            )
            if not window_contract_valid:
                codes.append("qwen_data_window_contract_invalid")
    return list(dict.fromkeys(codes))


def _resolve_variant(config: ExperimentConfig) -> Any:
    from .variants import all_variants

    matches = tuple(
        spec
        for spec in all_variants()
        if spec.mechanism == config.mechanism and spec.variant == config.variant
    )
    if len(matches) != 1:
        raise PreflightCheckError(
            "variant_unregistered",
            "configuration does not resolve to exactly one registered arm",
        )
    return matches[0]


def validate_scientific_preflight(
    config: ExperimentConfig,
    *,
    compatibility_validator: Callable[..., Any] | None = None,
    scientific_evidence: Mapping[str, Any] | None = None,
    gate_evaluator: Callable[[ExperimentConfig, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate compatibility and static identity/active-effect declarations."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    from .variants import (
        VariantCompatibilityError,
        validate_cache_compatibility,
        validate_variant_compatibility,
    )

    preliminary = _resolve_variant(config)
    validator = compatibility_validator or validate_variant_compatibility
    codes: list[str] = []
    try:
        spec = validator(
            preliminary.arm_id,
            backend=config.backend,
            task=config.task.name,
            stage=config.required_stage,
            experiment_kind=preliminary.experiment_kind,
        )
    except VariantCompatibilityError as error:
        spec = preliminary
        codes.extend(f"variant_incompatible:{field}" for field in error.violations)
    except (KeyError, TypeError, ValueError) as error:
        spec = preliminary
        codes.append("variant_compatibility_invalid")

    _ = scientific_evidence  # Caller-authored dictionaries are not measured evidence.
    if spec.evidence_kind == "addition":
        if gate_evaluator is None:
            codes.append("gate_evaluator_unavailable")
        else:
            try:
                evidence = gate_evaluator(config, spec)
            except Exception:
                evidence = None
                codes.append("gate_evaluator_failed")
            if not isinstance(evidence, Mapping) or evidence.get("available") is not True:
                codes.append("gate_evaluator_unavailable")
            else:
                if spec.mechanism == "maximum_hybrid":
                    probe = evidence.get("probe")
                    required_real_probe = {
                        "kind": "real_converted_kmd2_hybrid_module",
                        "native_base_class": "KMD2NativeAttn",
                        "forward_executed": True,
                        "scan_executed": True,
                        "neutral_repeat_deterministic": True,
                        "finite_connected_gradients": True,
                        "lane_specialization_gradients": True,
                        "braid_staging_passed": True,
                    }
                    # Native output parity is required exactly where the probe
                    # declares it applies: Package A (shared) must match the
                    # source layer; Package B's identity contract is conversion
                    # determinism because braided horizons, CMS clocks, and the
                    # 0.5-initialized trapezoid intentionally move its output
                    # ("Option B", 2026-07-14 - see bible.txt).
                    parity_consistent = (
                        isinstance(probe, Mapping)
                        and type(probe.get("native_parity_expected")) is bool
                        and type(probe.get("native_parity_passed")) is bool
                        and (probe["native_parity_passed"]
                             or not probe["native_parity_expected"])
                    )
                    if (not isinstance(probe, Mapping)
                            or any(probe.get(name) != value
                                   for name, value in required_real_probe.items())
                            or not parity_consistent
                            or probe.get("cache_schema") not in {"state", "states"}
                            or type(probe.get("hola_admissions")) is not int
                            or probe.get("hola_admissions", 0) < 1):
                        codes.append("real_module_gate_required")
                if spec.native_warm_start:
                    if "identity_passed" not in evidence:
                        codes.append("identity_evidence_missing")
                    elif evidence["identity_passed"] is not True:
                        codes.append("identity_gate_failed")
                if "active_effect_passed" not in evidence:
                    codes.append("active_effect_evidence_missing")
                elif evidence["active_effect_passed"] is not True:
                    codes.append("active_effect_missing")
                for field, prefix in (
                    ("missing_parameters", "gate_parameter_missing"),
                    ("disconnected_parameters", "gate_parameter_disconnected"),
                    ("frozen_zero_gates", "frozen_zero_gate"),
                ):
                    names = evidence.get(field, ())
                    if isinstance(names, Sequence) and not isinstance(
                        names, (str, bytes, bytearray)
                    ):
                        for name in names:
                            if type(name) is str and name:
                                codes.append(f"{prefix}:{name}")
                    else:
                        codes.append(f"gate_evidence_invalid:{field}")
                if spec.arm_id.startswith("exact_cache."):
                    if "native_feature_present" not in evidence:
                        codes.append("native_feature_evidence_missing")
                    elif type(evidence["native_feature_present"]) is not bool:
                        codes.append("native_feature_evidence_invalid")
                    else:
                        try:
                            validate_cache_compatibility(
                                spec.arm_id,
                                width=config.cache.width,
                                block_size=config.cache.block_size,
                                max_sequence_length=max(config.lengths.curriculum),
                                claimed_evidence_kind=spec.evidence_kind,
                                disabled_identity=(
                                    evidence.get("identity_passed") is True
                                ),
                                active_output_changed=(
                                    evidence.get("active_effect_passed") is True
                                ),
                                native_feature_present=evidence[
                                    "native_feature_present"
                                ],
                            )
                        except (TypeError, ValueError):
                            codes.append("cache_compatibility_failed")
    return {
        "ok": not codes,
        "codes": list(dict.fromkeys(codes)),
        "arm_id": spec.arm_id,
        "variant": {
            "mechanism": spec.mechanism,
            "variant": spec.variant,
            "evidence_kind": spec.evidence_kind,
            "comparison": spec.comparison,
            "experiment_kind": spec.experiment_kind,
            "changed_parameters": list(spec.changed_parameters),
            "changed_state": list(spec.changed_state),
        },
    }


def probe_backend_dispatch(
    backend: str,
    *,
    source_loader: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Check that a production backend exports an executable job entry point."""

    module_name = _BACKEND_MODULES.get(backend)
    if module_name is None:
        return {
            "ok": False,
            "codes": [f"backend_dispatch_unavailable:{backend}"],
            "backend": backend,
        }
    source_path = Path(__file__).with_name(module_name.rpartition(".")[2] + ".py")
    try:
        source = (
            source_path.read_text(encoding="utf-8")
            if source_loader is None
            else source_loader(module_name)
        )
        if type(source) is not str:
            raise TypeError("backend source loader must return str")
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, SyntaxError, TypeError, UnicodeDecodeError):
        tree = None
    entrypoints = (
        {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if tree is not None
        else set()
    )
    ready = bool(
        entrypoints & {"build_job_dispatcher", "run_job", "execute_job"}
    )
    return {
        "ok": ready,
        "codes": [] if ready else [f"backend_dispatch_unavailable:{backend}"],
        "backend": backend,
    }


def _nonnegative_resource_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _normalize_ffn_match(value: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, Mapping):
        return {}, ["ffn_match_evidence_missing"]
    required = {
        "matched",
        "target_parameters",
        "matched_parameters",
        "selected_d_ff",
        "residual_mismatch",
        "tolerance",
    }
    if not required <= set(value):
        return {}, ["ffn_match_evidence_missing"]
    normalized = {field: value[field] for field in required}
    counts_valid = all(
        _nonnegative_resource_int(normalized[field])
        for field in ("target_parameters", "matched_parameters")
    )
    selected_d_ff = normalized["selected_d_ff"]
    residual = normalized["residual_mismatch"]
    tolerance = normalized["tolerance"]
    numeric_valid = (
        type(selected_d_ff) is int
        and selected_d_ff >= 8
        and selected_d_ff % 8 == 0
        and type(residual) is int
        and type(tolerance) in (int, float)
        and math.isfinite(float(tolerance))
        and tolerance >= 0
    )
    relationship_valid = (
        counts_valid
        and type(residual) is int
        and normalized["matched_parameters"]
        - normalized["target_parameters"]
        == residual
    )
    if not counts_valid or not numeric_valid or not relationship_valid:
        return normalized, ["resource_accounting_invalid"]
    if (
        normalized["matched"] is not True
        or abs(residual) > float(tolerance)
    ):
        return normalized, ["ffn_match_failed"]
    return normalized, []


def _tiny_model_config(config: ExperimentConfig, spec: Any) -> Any:
    from .tiny_backend import TinyKMD2Config
    import torch

    params = config.task.params
    vocab_size = params.get("vocab_size", 256)
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("tiny exact accounting requires task.params.vocab_size")
    rotation_mode = {
        "rotation_off": "none",
        "constant_rate_rotation": "constant_rate",
        "non_cumulative_rotation": "non_cumulative",
        "fixed_rope": "fixed_rope",
        "moving_frame_oracle": "moving_frame",
    }.get(config.variant, "current")
    cache_enabled = (
        spec.mechanism == "exact_cache"
        and config.cache.width > 0
        and config.variant != "cache_off"
    )
    return TinyKMD2Config(
        d_model=config.model.hidden_size,
        heads=config.model.num_heads,
        dk=config.model.state_key_dim,
        dv=config.model.state_value_dim,
        layers=config.model.num_layers,
        vocab_size=vocab_size,
        d_ff=config.model.ffn_dim,
        r_out=params.get("r_out", 1),
        mimo_rank=params.get("mimo_rank", 1),
        continuous_input_dim=params.get("continuous_input_dim"),
        output_dim=params.get("output_dim"),
        dtype=torch.float32,
        rotation_mode=rotation_mode,
        trapezoid=config.mechanism == "trapezoid",
        corrected_momentum=config.mechanism == "corrected_momentum",
        causal_lookahead=config.mechanism == "causal_lookahead",
        bc_bias_mode=(
            config.variant
            if config.mechanism == "bc_bias"
            else "none"
        ),
        cache=config.cache if cache_enabled else None,
    )


def _default_tiny_resource_evaluator(
    config: ExperimentConfig, spec: Any
) -> Mapping[str, Any]:
    """Use Task 9 instantiated accounting for tensor-small tiny preflight."""

    if config.backend != "tiny":
        return {"available": False}
    from .tiny_backend import TinyKMD2Model
    from .variants import (
        construct_equal_state_byte_control,
        match_tiny_parameter_count,
    )

    arm = _tiny_model_config(config, spec)
    comparison = (
        "state_size"
        if config.variant == "state_size_sweep"
        else "mimo_rank"
        if config.variant == "true_mimo_sweep"
        else None
    )
    if comparison is not None:
        target_declaration = config.task.params.get("parameter_match_target")
        if not isinstance(target_declaration, Mapping):
            return {"available": False}
        target = replace(
            arm,
            cache=None,
            dk=target_declaration.get("state_key_dim", arm.dk),
            dv=target_declaration.get("state_value_dim", arm.dv),
            mimo_rank=target_declaration.get("mimo_rank", 1),
        )
        arm_without_cache = replace(arm, cache=None)
        try:
            match = match_tiny_parameter_count(
                target,
                arm_without_cache,
                comparison=comparison,
                d_ff_match_min=config.model.ffn_match_lower,
                d_ff_match_max=config.model.ffn_match_upper,
            )
        except ValueError as error:
            raise PreflightCheckError(
                "ffn_match_failed", "no exact tiny FFN parameter match"
            ) from error
        return {
            "available": True,
            "exact": True,
            "parameter_match": match,
            "total_parameters": match.matched.trainable_parameters,
            "cache_persistent_bytes": 0,
            "cache_block_bytes": 0,
            "cache_storage_dtype": config.cache.storage_dtype,
            "cache_compute_dtype": config.cache.compute_dtype,
        }

    model = TinyKMD2Model(arm, init_seed=0)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if arm.cache is None:
        recurrent_state_elements = arm.layers * arm.heads * arm.dk * arm.dv
        if arm.trapezoid:
            recurrent_state_elements += (
                arm.layers * arm.heads * (arm.dk + arm.dv)
            )
        if arm.corrected_momentum:
            recurrent_state_elements += arm.layers * arm.heads * arm.dk * arm.dv
        if arm.causal_lookahead:
            recurrent_state_elements += arm.layers * arm.heads * arm.dv
        if arm.rotation_mode == "moving_frame":
            recurrent_state_elements += arm.layers * arm.heads * (arm.dk // 2)
        tolerance = max(0.005 * trainable_parameters, 1024.0)
        return {
            "available": True,
            "exact": True,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "recurrent_state_elements": recurrent_state_elements,
            "recurrent_state_bytes": 4 * recurrent_state_elements,
            "cache_persistent_bytes": 0,
            "cache_block_bytes": 0,
            "cache_storage_dtype": config.cache.storage_dtype,
            "cache_compute_dtype": config.cache.compute_dtype,
            "ffn_match": {
                "matched": True,
                "target_parameters": trainable_parameters,
                "matched_parameters": trainable_parameters,
                "selected_d_ff": arm.d_ff,
                "residual_mismatch": 0,
                "tolerance": tolerance,
            },
        }

    base = replace(arm, cache=None)
    state_control = construct_equal_state_byte_control(
        base,
        cache_width=config.cache.width,
        storage_dtype=config.cache.storage_dtype,
    )
    storage_bytes = 2 if config.cache.storage_dtype == "bf16" else 4
    cache_block_bytes = (
        arm.layers
        * arm.heads
        * config.cache.block_size
        * ((arm.dk + arm.dv) * storage_bytes + 4 + 8 + 1)
    )
    tolerance = max(0.005 * trainable_parameters, 1024.0)
    return {
        "available": True,
        "exact": True,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "recurrent_state_elements": state_control.base.recurrent_state_elements,
        "recurrent_state_bytes": state_control.base.recurrent_state_bytes,
        "state_byte_control": state_control,
        "cache_block_bytes": cache_block_bytes,
        "cache_storage_dtype": config.cache.storage_dtype,
        "cache_compute_dtype": config.cache.compute_dtype,
        "ffn_match": {
            "matched": True,
            "target_parameters": trainable_parameters,
            "matched_parameters": trainable_parameters,
            "selected_d_ff": arm.d_ff,
            "residual_mismatch": 0,
            "tolerance": tolerance,
        },
    }


def evaluate_exact_resources(
    config: ExperimentConfig,
    spec: Any,
    *,
    resource_evaluator: Callable[[ExperimentConfig, Any], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Normalize exact instantiated accounting and refuse estimates or guesses."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if resource_evaluator is None:
        resource_evaluator = _default_tiny_resource_evaluator
    try:
        evidence = resource_evaluator(config, spec)
    except PreflightCheckError as error:
        return {
            "ok": False,
            "codes": [error.code],
            "resources": {},
        }
    except Exception:
        return {
            "ok": False,
            "codes": ["resource_evaluator_failed"],
            "resources": {},
        }
    if not isinstance(evidence, Mapping) or evidence.get("available") is not True:
        return {
            "ok": False,
            "codes": ["resource_evaluator_unavailable"],
            "resources": {},
        }
    codes: list[str] = []
    conservative_hybrid = (
        evidence.get("exact") is False
        and evidence.get("preflight_safe") is True
        and isinstance(evidence.get("accounting"), Mapping)
        and evidence["accounting"].get("activation") in {
            "conservative_shape_upper_bound",
            "phase_separated_fused_path_upper_bound",
        }
    )
    if evidence.get("exact") is not True and not conservative_hybrid:
        codes.append("resource_accounting_not_exact")

    from .variants import EqualStateByteControl, ParameterMatchResult

    parameter_match = evidence.get("parameter_match")
    if isinstance(parameter_match, ParameterMatchResult):
        matched = parameter_match.matched
        resources: dict[str, Any] = {
            "trainable_parameters": matched.trainable_parameters,
            "recurrent_state_elements": matched.recurrent_state_elements,
            "recurrent_state_tensor_elements": matched.recurrent_state_elements,
            "recurrent_state_auxiliary_bytes": 0,
            "recurrent_state_bytes": matched.recurrent_state_bytes,
        }
        ffn_value: Any = {
            "matched": True,
            "target_parameters": parameter_match.target.trainable_parameters,
            "matched_parameters": matched.trainable_parameters,
            "selected_d_ff": matched.d_ff,
            "residual_mismatch": parameter_match.residual_mismatch,
            "tolerance": parameter_match.tolerance,
        }
    elif parameter_match is not None:
        resources = {}
        ffn_value = None
        codes.append("resource_accounting_invalid")
    else:
        resources = {
            field: evidence.get(field)
            for field in (
                "trainable_parameters",
                "recurrent_state_elements",
                "recurrent_state_bytes",
            )
        }
        resources["recurrent_state_tensor_elements"] = evidence.get(
            "recurrent_state_tensor_elements", resources["recurrent_state_elements"]
        )
        resources["recurrent_state_auxiliary_bytes"] = evidence.get(
            "recurrent_state_auxiliary_bytes", 0
        )
        ffn_value = evidence.get("ffn_match")

    ffn_match, ffn_codes = _normalize_ffn_match(ffn_value)
    codes.extend(ffn_codes)
    resources["ffn_match"] = ffn_match
    resources["total_parameters"] = evidence.get("total_parameters")
    state_control = evidence.get("state_byte_control")
    if isinstance(state_control, EqualStateByteControl):
        resources["cache_persistent_bytes"] = state_control.cache_persistent_bytes
        resources["state_byte_control"] = {
            "recurrent_increase_bytes": state_control.recurrent_increase_bytes,
            "byte_mismatch": state_control.byte_mismatch,
            "absolute_byte_mismatch": state_control.absolute_byte_mismatch,
        }
    else:
        resources["cache_persistent_bytes"] = evidence.get(
            "cache_persistent_bytes"
        )
    resources.update(
        {
            "cache_block_bytes": evidence.get("cache_block_bytes"),
            "cache_storage_dtype": evidence.get("cache_storage_dtype"),
            "cache_compute_dtype": evidence.get("cache_compute_dtype"),
        }
    )
    for field in (
        "parameter_metadata_kind",
        "parameter_metadata_tensors",
        "parameter_scope",
        "total_base_parameters",
        "native_addition_parameters",
        "native_r_out",
        "cache_parameter_count",
        "arm_trainable_parameters",
        "arm_total_parameters",
        "declared_cache_parameter_count",
        "qwen_execution",
    ):
        if field in evidence:
            resources[field] = evidence[field]
    integer_fields = (
        "trainable_parameters",
        "total_parameters",
        "recurrent_state_elements",
        "recurrent_state_tensor_elements",
        "recurrent_state_auxiliary_bytes",
        "recurrent_state_bytes",
        "cache_persistent_bytes",
        "cache_block_bytes",
    )
    if any(not _nonnegative_resource_int(resources.get(field)) for field in integer_fields):
        codes.append("resource_accounting_invalid")
    elif (
        resources["total_parameters"] < resources["trainable_parameters"]
        or resources["recurrent_state_bytes"]
        != (4 * resources["recurrent_state_tensor_elements"]
            + resources["recurrent_state_auxiliary_bytes"])
        or ffn_match.get("matched_parameters") != resources["trainable_parameters"]
    ):
        codes.append("resource_accounting_invalid")
    if (
        resources.get("cache_storage_dtype") != config.cache.storage_dtype
        or resources.get("cache_compute_dtype") != config.cache.compute_dtype
    ):
        codes.append("resource_accounting_invalid")
    codes = list(dict.fromkeys(codes))
    return {"ok": not codes, "codes": codes, "resources": resources}


def _pairing_id(config: ExperimentConfig, seed: int) -> str:
    semantics = config.semantic_dict()
    basis = {
        "task": semantics["task"],
        "seed": seed,
        "budget": semantics["budget"],
        "lengths": semantics["lengths"],
        "evaluation": semantics["evaluation"],
        "stage": config.required_stage,
        "backend": config.backend,
    }
    return hashlib.sha256(canonical_json_bytes(basis)).hexdigest()


def _qwen_pairing_inputs(
    config: ExperimentConfig,
    asset_hashes: Mapping[str, str] | None,
) -> tuple[tuple[str, ...], str, str]:
    """Validate immutable inputs shared by paired Qwen execution modes."""
    if not _is_exact_three_seed_matrix(config.seeds):
        raise PreflightCheckError(
            "qwen_seed_matrix_invalid",
            "Qwen heal requires exactly three unique nonnegative seeds",
        )
    if not isinstance(asset_hashes, Mapping):
        raise ValueError("Qwen heal expansion requires measured asset_hashes")
    checkpoint_digest = asset_hashes.get("checkpoint")
    if (
        type(checkpoint_digest) is not str
        or len(checkpoint_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checkpoint_digest
        )
    ):
        raise ValueError(
            "Qwen heal expansion requires a measured checkpoint SHA-256"
        )
    data_digest = asset_hashes.get("data")
    if (
        type(data_digest) is not str
        or len(data_digest) != 64
        or any(character not in "0123456789abcdef" for character in data_digest)
    ):
        raise ValueError("Qwen heal expansion requires a measured data SHA-256")
    example_ids = config.task.params.get("example_ids")
    if (
        not isinstance(example_ids, (list, tuple))
        or not example_ids
        or any(type(item) is not str or not item for item in example_ids)
        or len(set(example_ids)) != len(example_ids)
    ):
        raise ValueError(
            "Qwen heal expansion requires ordered unique task.params.example_ids"
        )
    return tuple(example_ids), checkpoint_digest, data_digest


def _expand_jobs(
    config: ExperimentConfig,
    arm_id: str,
    *,
    asset_hashes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    maximum_control = config.task.params.get("maximum_control")
    architecture = getattr(config, "architecture", None)
    if (
        config.backend == "qwen"
        and isinstance(maximum_control, str)
        and maximum_control
        and (architecture is not None or maximum_control == "stock-qwen")
    ):
        from .qwen_training import derive_three_arm_pairing

        example_ids, checkpoint_digest, data_digest = _qwen_pairing_inputs(
            config, asset_hashes
        )
        execution_arm_id = architecture.arm_id if architecture is not None else arm_id
        jobs: list[dict[str, Any]] = []
        for seed in config.seeds:
            provisional = build_job(
                config,
                seed=seed,
                stage=config.required_stage,
                backend=config.backend,
                arm_id=execution_arm_id,
            )
            pairing = derive_three_arm_pairing(
                provisional,
                example_ids=example_ids,
                pre_replacement_checkpoint_sha256=checkpoint_digest,
                data_sha256=data_digest,
            )
            jobs.append(
                build_job(
                    config,
                    seed=seed,
                    stage=config.required_stage,
                    backend=config.backend,
                    arm_id=execution_arm_id,
                    pairing_id=pairing.pairing_id,
                )
            )
        return build_jobs_document(jobs)["jobs"]
    if (
        config.backend == "qwen"
        and config.required_stage == "qwen_heal"
        and config.mechanism == "exact_cache"
    ):
        from .qwen_training import derive_three_arm_pairing

        example_ids, checkpoint_digest, data_digest = _qwen_pairing_inputs(
            config, asset_hashes
        )
        # Keep the generic runner's source closure backend-neutral so Tiny
        # bundles never pull a Qwen-named module into their payload.  The
        # Qwen release contract independently fail-closes this same trio.
        release_cache_arm_ids = ("native", "recency", "surprise")

        jobs: list[dict[str, Any]] = []
        for seed in config.seeds:
            provisional = build_job(
                config,
                seed=seed,
                stage=config.required_stage,
                backend=config.backend,
                arm_id="surprise",
            )
            pairing = derive_three_arm_pairing(
                provisional,
                example_ids=example_ids,
                pre_replacement_checkpoint_sha256=checkpoint_digest,
                data_sha256=data_digest,
            )
            jobs.extend(
                build_job(
                    config,
                    seed=seed,
                    stage=config.required_stage,
                    backend=config.backend,
                    arm_id=paired_arm,
                    pairing_id=pairing.pairing_id,
                )
                for paired_arm in release_cache_arm_ids
            )
        return build_jobs_document(jobs)["jobs"]
    from .variants import get_variant

    spec = get_variant(arm_id)
    if spec.evidence_kind == "baseline":
        paired_arm_ids = (arm_id,)
    elif spec.evidence_kind == "reliance":
        paired_arm_ids = {
            "rotation": ("rotation.current", "rotation.off"),
            "convolution": ("convolution.on", "convolution.off"),
        }.get(spec.mechanism, ("native", arm_id))
    elif arm_id in {
        "exact_cache.rotation_factorial",
        "exact_cache.r_out_factorial",
    }:
        paired_arm_ids = tuple(f"{arm_id}.{cell}" for cell in ("M00", "M10", "M01", "M11"))
    else:
        # Every non-baseline Tiny screen is a matched comparison against the
        # complete current native arm.  The job-level arm is intentionally the
        # only difference: task examples, seed, budget, and canonical config
        # remain byte-identical within the pair.
        paired_arm_ids = ("native", arm_id)
    paired_arm_ids = tuple(dict.fromkeys(paired_arm_ids))
    jobs = [
        build_job(
            config,
            seed=seed,
            stage=config.required_stage,
            backend=config.backend,
            arm_id=paired_arm_id,
            pairing_id=_pairing_id(config, seed),
        )
        for seed in config.seeds
        for paired_arm_id in paired_arm_ids
    ]
    return build_jobs_document(jobs)["jobs"]


def _command_for(options: Any, command: str) -> list[str]:
    result = [
        sys.executable,
        "-m",
        "research.kmd2_ablation.run_ablation",
        command,
        "--backend",
        str(options.backend),
        "--config",
        str(Path(options.config)),
        "--out",
        str(Path(options.out)),
        "--job-index",
        str(options.job_index),
        "--num-jobs",
        str(options.num_jobs),
        "--resume" if options.resume else "--no-resume",
    ]
    optional = (
        ("mode", "--mode"),
        ("model", "--model"),
        ("tokenizer", "--tokenizer"),
        ("checkpoint", "--checkpoint"),
        ("data", "--data"),
        ("teacher_model", "--teacher-model"),
        ("student_device", "--student-device"),
        ("teacher_device", "--teacher-device"),
        ("dtype", "--dtype"),
        ("checkpoint_every", "--checkpoint-every"),
        ("model_sha256", "--model-sha256"),
        ("tokenizer_sha256", "--tokenizer-sha256"),
        ("checkpoint_sha256", "--checkpoint-sha256"),
        ("data_sha256", "--data-sha256"),
        ("teacher_model_sha256", "--teacher-model-sha256"),
        ("assets_manifest", "--assets-manifest"),
        ("repo_root", "--repo-root"),
    )
    for field, flag in optional:
        value = getattr(options, field, None)
        if value is not None:
            result.extend((flag, str(value)))
    if command == "preflight" and getattr(options, "dry_run", False):
        result.append("--dry-run")
    return result


def build_reproduction_commands(options: Any) -> dict[str, list[str]]:
    """Return exact copy/paste commands including every supplied identity flag."""

    return {
        command: _command_for(options, command)
        for command in ("preflight", "run", "summarize", "bundle")
    }


def _canonical_manifest_command(config: ExperimentConfig) -> list[str]:
    """Return a runtime-path-free command identity for the canonical manifest."""

    return [
        "python",
        "-m",
        "research.kmd2_ablation.run_ablation",
        "run",
        "--backend",
        config.backend,
    ]


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=15,
        )
        return completed.stdout if completed.returncode == 0 else b""

    revision = run("rev-parse", "HEAD").decode("utf-8", "replace").strip()
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD") + b"\0" + status
    return {
        "revision": revision or "unavailable",
        "diff_hash": hashlib.sha256(diff).hexdigest(),
        "dirty": bool(status),
    }


def _preflight_report(
    *,
    ok: bool,
    codes: Sequence[str],
    warnings: Sequence[str] = (),
    inventory: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
    assets: Mapping[str, Any] | None = None,
    jobs: Sequence[Mapping[str, Any]] = (),
    commands: Mapping[str, Any] | None = None,
    manifest_path: str = "",
    exit_code: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": ok,
        "schema_version": RESULT_SCHEMA_VERSION,
        "codes": list(codes),
        "warnings": list(warnings),
        "inventory": {} if inventory is None else dict(inventory),
        "resources": {} if resources is None else dict(resources),
        "assets": {} if assets is None else dict(assets),
        "jobs": [dict(job) for job in jobs],
        "commands": {} if commands is None else dict(commands),
        "manifest_path": manifest_path,
    }
    if exit_code is not None:
        report["_exit_code"] = exit_code
    return report


def _load_asset_expectations(options: Any) -> dict[str, dict[str, Any]]:
    path = getattr(options, "assets_manifest", None)
    expected: dict[str, dict[str, Any]] = {}
    if path is not None:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PreflightCheckError(
                "asset_manifest_invalid", f"cannot read asset manifest: {path}"
            ) from error
        if isinstance(raw, Mapping) and isinstance(raw.get("assets"), Mapping):
            raw = raw["assets"]
        if not isinstance(raw, Mapping):
            raise PreflightCheckError(
                "asset_manifest_invalid", "asset manifest must be a mapping"
            )
        for name, identity in raw.items():
            if type(name) is not str or not isinstance(identity, Mapping):
                raise PreflightCheckError(
                    "asset_manifest_invalid", "asset identities must be mappings"
                )
            expected[name] = dict(identity)
            tree_digest = expected[name].pop("tree_sha256", None)
            if tree_digest is not None:
                if "sha256" in expected[name] and expected[name]["sha256"] != tree_digest:
                    raise PreflightCheckError(
                        "asset_manifest_invalid", f"asset {name} has conflicting tree identities"
                    )
                expected[name]["sha256"] = tree_digest
    for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model"):
        digest = getattr(options, f"{name}_sha256", None)
        if digest is not None:
            expected.setdefault(name, {})["sha256"] = digest
    return expected


def _external_asset_paths(options: Any, config: ExperimentConfig) -> dict[str, Path]:
    if config.backend != "qwen":
        return {}
    paths = {
        name: getattr(options, name, None)
        for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model")
    }
    missing = [name for name in ("model", "checkpoint", "data") if paths[name] is None]
    synthetic_only = config.task.params.get("objective") == "synthetic_only"
    if config.qwen.run_mode == "heal" and not synthetic_only and paths["teacher_model"] is None:
        missing.append("teacher_model")
    if missing:
        raise PreflightCheckError(
            "asset_missing",
            "Qwen preflight requires explicit assets: " + ", ".join(missing),
        )
    return {name: Path(path) for name, path in paths.items() if path is not None}


def preflight(
    options: Any,
    *,
    environment_probe: Callable[..., Mapping[str, Any]] | None = None,
    inventory_builder: Callable[[Path], Mapping[str, Any]] | None = None,
    inventory_verifier: Callable[[Mapping[str, Any], Path], None] | None = None,
    asset_inspector: Callable[..., Mapping[str, Any]] | None = None,
    compatibility_validator: Callable[..., Any] | None = None,
    gate_evaluator: Callable[[ExperimentConfig, Any], Mapping[str, Any]] | None = None,
    resource_evaluator: Callable[[ExperimentConfig, Any], Mapping[str, Any]] | None = None,
    backend_probe: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run complete tensor-free preflight and publish immutable run documents."""

    commands = build_reproduction_commands(options)
    manifest_path = str((Path(options.out).expanduser().resolve() / "manifest.json"))
    try:
        raw = json.loads(Path(options.config).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _preflight_report(
            ok=False,
            codes=["config_unreadable"],
            commands=commands,
            manifest_path=manifest_path,
            exit_code=2,
        )
    raw_codes = validate_raw_scientific_config(
        raw, backend=options.backend, mode=getattr(options, "mode", None)
    )
    if raw_codes:
        return _preflight_report(
            ok=False,
            codes=raw_codes,
            commands=commands,
            manifest_path=manifest_path,
        )
    try:
        config = ExperimentConfig.from_dict(raw)
        if options.backend == "qwen":
            from .qwen_variants import validate_maximum_control_config
            validate_maximum_control_config(raw)
    except (TypeError, ValueError):
        return _preflight_report(
            ok=False,
            codes=["config_invalid"],
            commands=commands,
            manifest_path=manifest_path,
            exit_code=2,
        )
    if config.backend != options.backend:
        return _preflight_report(
            ok=False,
            codes=["backend_config_mismatch"],
            commands=commands,
            manifest_path=manifest_path,
            exit_code=2,
        )
    if getattr(options, "mode", None) not in {None, config.qwen.run_mode}:
        return _preflight_report(
            ok=False,
            codes=["qwen_mode_mismatch"],
            commands=commands,
            manifest_path=manifest_path,
            exit_code=2,
        )
    maximum_control = config.task.params.get("maximum_control")
    requested_control = getattr(options, "maximum_control", None)
    if requested_control is not None and requested_control != maximum_control:
        return _preflight_report(
            ok=False,
            codes=["maximum_control_mismatch"],
            commands=commands,
            manifest_path=manifest_path,
            exit_code=2,
        )
    if maximum_control is not None and requested_control is None:
        return _preflight_report(
            ok=False,
            codes=["maximum_control_required"],
            commands=commands,
            manifest_path=manifest_path,
            exit_code=2,
        )

    root = Path(getattr(options, "repo_root", None) or Path(__file__).resolve().parents[2])
    try:
        output = validate_output_writable(options.out)
    except PreflightCheckError as error:
        return _preflight_report(
            ok=False,
            codes=[error.code],
            commands=commands,
            manifest_path=manifest_path,
        )

    probe = environment_probe or probe_environment
    environment_report = probe(
        backend=config.backend,
        device_preferences=config.device_preferences,
        dtype_preferences=config.dtype_preferences,
        student_device=getattr(options, "student_device", None),
        teacher_device=getattr(options, "teacher_device", None),
        requested_dtype=getattr(options, "dtype", None),
    )
    if not environment_report.get("ok"):
        return _preflight_report(
            ok=False,
            codes=environment_report.get("codes", ["environment_invalid"]),
            warnings=environment_report.get("warnings", ()),
            resources=environment_report.get("resources", {}),
            commands=commands,
            manifest_path=manifest_path,
        )

    if inventory_builder is None or inventory_verifier is None:
        from .inventory import build_inventory, verify_inventory_sources

        inventory_builder = inventory_builder or build_inventory
        inventory_verifier = inventory_verifier or verify_inventory_sources
    try:
        inventory = inventory_builder(root)
        inventory_verifier(inventory, root)
    except (OSError, TypeError, ValueError):
        return _preflight_report(
            ok=False,
            codes=["source_hash_stale"],
            warnings=environment_report.get("warnings", ()),
            resources=environment_report.get("resources", {}),
            commands=commands,
            manifest_path=manifest_path,
        )

    try:
        asset_paths = _external_asset_paths(options, config)
        inspector = asset_inspector or inspect_external_assets
        assets = inspector(asset_paths, expected=_load_asset_expectations(options))
    except PreflightCheckError as error:
        return _preflight_report(
            ok=False,
            codes=[error.code],
            warnings=environment_report.get("warnings", ()),
            inventory=inventory,
            resources=environment_report.get("resources", {}),
            commands=commands,
            manifest_path=manifest_path,
        )

    if gate_evaluator is None:
        from .gate_probes import measure_scientific_gates

        gate_evaluator = measure_scientific_gates
    scientific = validate_scientific_preflight(
        config,
        compatibility_validator=compatibility_validator,
        gate_evaluator=gate_evaluator,
    )
    if not scientific["ok"]:
        return _preflight_report(
            ok=False,
            codes=scientific["codes"],
            warnings=environment_report.get("warnings", ()),
            inventory=inventory,
            resources=environment_report.get("resources", {}),
            assets=assets,
            commands=commands,
            manifest_path=manifest_path,
        )
    readiness = (backend_probe or probe_backend_dispatch)(config.backend)
    if not readiness.get("ok"):
        return _preflight_report(
            ok=False,
            codes=readiness.get("codes", ["backend_dispatch_unavailable"]),
            warnings=environment_report.get("warnings", ()),
            inventory=inventory,
            resources=environment_report.get("resources", {}),
            assets=assets,
            commands=commands,
            manifest_path=manifest_path,
        )

    selected_resource_evaluator = resource_evaluator
    if selected_resource_evaluator is None and config.backend == "qwen":
        from .resource_probes import measure_qwen_resources

        selected_resource_evaluator = lambda candidate, candidate_spec: (
            measure_qwen_resources(candidate, candidate_spec, assets=assets)
        )
    exact_resources = evaluate_exact_resources(
        config,
        _resolve_variant(config),
        resource_evaluator=selected_resource_evaluator,
    )
    if not exact_resources["ok"]:
        return _preflight_report(
            ok=False,
            codes=exact_resources["codes"],
            warnings=environment_report.get("warnings", ()),
            inventory=inventory,
            resources=environment_report.get("resources", {}),
            assets=assets,
            commands=commands,
            manifest_path=manifest_path,
        )

    resources = dict(environment_report["resources"])
    resources.update(exact_resources["resources"])
    source_hashes = dict(inventory.get("source_files", {}))
    asset_hashes = {name: record["sha256"] for name, record in assets.items()}
    jobs = _expand_jobs(
        config,
        scientific["arm_id"],
        asset_hashes=asset_hashes,
    )
    canonical_config = config.semantic_dict()
    provenance_environment = dict(environment_report["environment"])
    if config.backend == "qwen" and isinstance(
        resources.get("qwen_execution"), Mapping
    ):
        provenance_environment["qwen_execution"] = resources["qwen_execution"]
    provenance = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "source_hashes": source_hashes,
        "config_hash": hashlib.sha256(canonical_json_bytes(canonical_config)).hexdigest(),
        "asset_hashes": asset_hashes,
        "git": _git_provenance(root),
        "environment": provenance_environment,
    }
    try:
        manifest = build_manifest(
            canonical_config=canonical_config,
            jobs=jobs,
            provenance=provenance,
            command=_canonical_manifest_command(config),
        )
        store = ResultStore(
            output,
            provenance=provenance,
            job_index=options.job_index,
            num_jobs=options.num_jobs,
        )
        store.initialize(manifest=manifest, jobs=jobs)
    except (FileExistsError, RunRecordError, TypeError, ValueError):
        return _preflight_report(
            ok=False,
            codes=["immutable_manifest_conflict"],
            warnings=environment_report.get("warnings", ()),
            inventory=inventory,
            resources=resources,
            assets=assets,
            jobs=jobs,
            commands=commands,
            manifest_path=manifest_path,
        )
    return _preflight_report(
        ok=True,
        codes=[],
        warnings=environment_report.get("warnings", ()),
        inventory=inventory,
        resources=resources,
        assets=assets,
        jobs=jobs,
        commands=commands,
        manifest_path=manifest_path,
    )


def preflight_command(options: Any) -> dict[str, Any]:
    """Production CLI adapter for :func:`preflight`."""

    return preflight(options)


def _manifest_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: manifest[field]
        for field in (
            "schema_version",
            "suite_version",
            "source_hashes",
            "config_hash",
            "asset_hashes",
            "git",
            "environment",
        )
    }


def build_runtime_dispatchers(
    options: Any,
    *,
    manifest: Mapping[str, Any],
    module_loader: Callable[[str], Any] | None = None,
) -> dict[str, BackendDispatcher]:
    """Bind runtime-only CLI inputs into a backend closure outside job identity."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    backend = options.backend
    module_name = _BACKEND_MODULES.get(backend)
    if module_name is None:
        raise BackendUnavailable(
            f"unsupported backend: {backend}",
            phase="backend_bind",
            context={"backend": backend},
        )
    if backend == "qwen":
        try:
            canonical = manifest.get("canonical_config")
            environment = manifest.get("environment")
            if not isinstance(canonical, Mapping) or not isinstance(
                environment, Mapping
            ):
                raise ValueError("manifest has no Qwen execution provenance")
            recorded = environment.get("qwen_execution")
            if not isinstance(recorded, Mapping):
                raise ValueError("manifest has no Qwen fast-scan contract")
            from .resource_probes import verify_qwen_execution_contract

            measured = verify_qwen_execution_contract(canonical)
            stable_fields = {
                "fast_scan",
                "native_r_out",
                "native_scan",
                "score_scan",
            }
            if {field: recorded.get(field) for field in stable_fields} != {
                field: measured.get(field) for field in stable_fields
            }:
                raise ValueError("runtime Qwen fast-scan contract changed")
        except (PreflightCheckError, TypeError, ValueError) as error:
            raise BackendUnavailable(
                f"Qwen fast scan execution contract failed: {error}",
                phase="backend_bind",
                context={"backend": backend},
            ) from error
    loader = module_loader or importlib.import_module
    try:
        module = loader(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailable(
            f"backend module is unavailable: {module_name}",
            phase="backend_bind",
            context={"backend": backend, "module": module_name},
        ) from error

    dtype = getattr(options, "dtype", None)
    if dtype is None:
        canonical = manifest.get("canonical_config")
        preferences = (
            canonical.get("dtype_preferences")
            if isinstance(canonical, Mapping)
            else None
        )
        dtype = (
            preferences[0]
            if isinstance(preferences, Sequence)
            and not isinstance(preferences, (str, bytes, bytearray))
            and preferences
            else "float32"
        )
    asset_hashes = manifest.get("asset_hashes", {})
    if not isinstance(asset_hashes, Mapping):
        raise BackendUnavailable(
            "manifest asset_hashes must be a mapping",
            phase="backend_bind",
            context={"backend": backend},
        )
    runtime: dict[str, Any] = {
        "output": Path(options.out),
        "dtype": dtype,
        "asset_hashes": dict(asset_hashes),
        "resume": options.resume,
    }
    if backend == "qwen":
        for field in (
            "model",
            "tokenizer",
            "checkpoint",
            "data",
            "teacher_model",
            "student_device",
            "teacher_device",
            "checkpoint_every",
        ):
            value = getattr(options, field, None)
            if value is not None:
                runtime[field] = value

    builder = getattr(module, "build_job_dispatcher", None)
    if callable(builder):
        dispatcher = (
            builder(runtime, dependencies=None)
            if backend == "qwen"
            else builder(runtime)
        )
    else:
        dispatcher = next(
            (
                candidate
                for name in ("run_job", "execute_job")
                if callable(candidate := getattr(module, name, None))
            ),
            None,
        )
    if not callable(dispatcher):
        raise BackendUnavailable(
            f"backend module has no bound job dispatcher: {module_name}",
            phase="backend_bind",
            context={"backend": backend, "module": module_name},
        )
    return {backend: dispatcher}


def run(
    options: Any,
    *,
    execute_fn: Callable[..., list[dict[str, Any]]] | None = None,
    dispatchers: Mapping[str, BackendDispatcher] | None = None,
    dispatcher_builder: Callable[..., Mapping[str, BackendDispatcher]] | None = None,
) -> dict[str, Any]:
    """Execute exactly one deterministic shard from immutable preflight files."""

    output = Path(options.out).expanduser().resolve()
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        jobs_document = json.loads((output / "jobs.json").read_text(encoding="utf-8"))
        jobs = jobs_document["jobs"]
        provenance = _manifest_provenance(manifest)
        if any(job.get("backend") != options.backend for job in jobs):
            raise ValueError("requested backend does not match jobs.json")
        store = ResultStore(
            output,
            provenance=provenance,
            job_index=options.job_index,
            num_jobs=options.num_jobs,
        )
        command = _command_for(options, "run")
        executor = execute_fn or execute_jobs
        bound_dispatchers = (
            dispatchers
            if dispatchers is not None
            else (dispatcher_builder or build_runtime_dispatchers)(
                options, manifest=manifest
            )
        )
        outcomes = executor(
            jobs,
            store=store,
            command=command,
            dispatchers=bound_dispatchers,
            resume=options.resume,
        )
    except (
        BackendUnavailable,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RunRecordError,
    ) as error:
        return {
            "ok": False,
            "schema_version": RESULT_SCHEMA_VERSION,
            "codes": ["execution_setup_failed"],
            "warnings": [],
            "error": str(error),
            "outcomes": [],
        }
    failed = [outcome for outcome in outcomes if outcome.get("status") == "failed"]
    return {
        "ok": not failed,
        "schema_version": RESULT_SCHEMA_VERSION,
        "codes": [] if not failed else ["execution_jobs_failed"],
        "warnings": [],
        "outcomes": outcomes,
        "job_index": options.job_index,
        "num_jobs": options.num_jobs,
    }


def run_command(options: Any) -> dict[str, Any]:
    """Production CLI adapter for deterministic sharded execution."""

    return run(options)


def _json_copy(value: Any, *, code: str, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise RunRecordError(code, f"{label} is not finite JSON") from error


def _command(command: Sequence[str]) -> list[str]:
    if (
        isinstance(command, (str, bytes, bytearray))
        or not isinstance(command, Sequence)
        or not command
        or any(type(item) is not str or not item for item in command)
    ):
        raise RunRecordError(
            "missing_diagnostics", "command must be a nonempty string sequence"
        )
    return list(command)


def _common_record(
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    status: str,
    shard_index: int,
    num_jobs: int,
    command: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(job, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError("job and provenance must be mappings")
    job_id = job.get("job_id")
    if type(job_id) is not str or not job_id:
        raise RunRecordError("job_identity", "job_id must be a nonempty str")
    if type(num_jobs) is not int or isinstance(num_jobs, bool) or num_jobs < 1:
        raise RunRecordError("shard_mismatch", "num_jobs must be a positive int")
    if type(shard_index) is not int or isinstance(shard_index, bool):
        raise RunRecordError("shard_mismatch", "shard_index must be an int")
    if not 0 <= shard_index < num_jobs or assign_shard(job_id, num_jobs) != shard_index:
        raise RunRecordError("shard_mismatch", "worker does not own this job")
    required = {
        "experiment_id",
        "seed",
        "stage",
        "backend",
        "arm_id",
        "canonical_config",
    }
    missing = required - set(job)
    if missing:
        raise RunRecordError(
            "job_identity", "job is missing: " + ", ".join(sorted(missing))
        )
    record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "status": status,
        "job_id": job_id,
        "experiment_id": job["experiment_id"],
        "seed": job["seed"],
        "stage": job["stage"],
        "backend": job["backend"],
        "arm_id": job["arm_id"],
        "shard": {"index": shard_index, "count": num_jobs},
        "provenance": _json_copy(
            provenance, code="provenance_mismatch", label="provenance"
        ),
        "canonical_config": _json_copy(
            job["canonical_config"], code="config_mismatch", label="canonical_config"
        ),
        "command": _command(command),
    }
    if "pairing_id" in job:
        record["pairing_id"] = job["pairing_id"]
    return record


def build_completed_record(
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    shard_index: int,
    num_jobs: int,
    command: Sequence[str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and fully validate one completed execution record."""

    if not isinstance(payload, Mapping):
        raise RunRecordError("invalid_diagnostics", "backend payload must be a mapping")
    conflicts = _RESERVED_FIELDS & set(payload)
    if conflicts:
        raise RunRecordError(
            "invalid_diagnostics",
            "backend payload overrides reserved fields: " + ", ".join(sorted(conflicts)),
        )
    required = {
        "metrics",
        "loss_curves",
        "counts",
        "parameters",
        "recurrent_state",
        "performance",
        "identities",
    }
    missing = required - set(payload)
    if missing:
        raise RunRecordError(
            "missing_diagnostics",
            "backend payload is missing: " + ", ".join(sorted(missing)),
        )
    record = _common_record(
        job,
        provenance,
        status="completed",
        shard_index=shard_index,
        num_jobs=num_jobs,
        command=command,
    )
    copied = _json_copy(payload, code="invalid_diagnostics", label="backend payload")
    record.update(copied)
    validate_completed_run(record, job, provenance)
    return record


def build_failed_record(
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    shard_index: int,
    num_jobs: int,
    command: Sequence[str],
    error: BaseException,
    traceback_text: str,
) -> dict[str, Any]:
    """Build and fully validate one bounded typed failed execution record."""

    if isinstance(error, JobFailure):
        failure = error
    else:
        failure = JobFailure(str(error) or type(error).__name__, phase="execution")
    if type(traceback_text) is not str:
        raise TypeError("traceback_text must be a str")
    record = _common_record(
        job,
        provenance,
        status="failed",
        shard_index=shard_index,
        num_jobs=num_jobs,
        command=command,
    )
    record["error"] = {
        "code": failure.code,
        "message": str(failure),
        "phase": failure.phase,
        "context": _json_copy(
            failure.context, code="invalid_failure", label="failure context"
        ),
        "traceback": traceback_text[-_MAX_TRACEBACK_CHARS:],
    }
    validate_failed_run(record, job, provenance)
    return record


_BACKEND_MODULES = {
    "tiny": "research.kmd2_ablation.tiny_training",
    "qwen": "research.kmd2_ablation.qwen_training",
}


def load_backend_dispatcher(backend: str) -> BackendDispatcher:
    """Lazily import only the backend requested by the selected job."""

    if backend not in _BACKEND_MODULES:
        raise ValueError(f"unsupported backend: {backend!r}")
    module_name = _BACKEND_MODULES[backend]
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailable(
            f"backend module is unavailable: {module_name}",
            phase="backend_import",
            context={"backend": backend, "module": module_name},
        ) from error
    for name in ("run_job", "execute_job"):
        dispatcher = getattr(module, name, None)
        if callable(dispatcher):
            return dispatcher
    raise BackendUnavailable(
        f"backend module has no run_job entry point: {module_name}",
        phase="backend_import",
        context={"backend": backend, "module": module_name},
    )


def _lookup(mapping: Mapping[str, Any] | None, *path: str, default: Any) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _oom_context_from_job(job: Mapping[str, Any]) -> dict[str, Any]:
    config = job.get("canonical_config")
    if not isinstance(config, Mapping):
        config = {}
    sequence_length = _lookup(config, "task", "params", "sequence_length", default=None)
    if type(sequence_length) is not int or sequence_length < 1:
        sequence_length = _lookup(config, "task", "params", "length", default=1)
    if type(sequence_length) is not int or sequence_length < 1:
        sequence_length = 1
    batch_size = _lookup(config, "task", "params", "batch_size", default=1)
    if type(batch_size) is not int or batch_size < 1:
        batch_size = 1
    dtype_preferences = config.get("dtype_preferences", [])
    dtype = (
        dtype_preferences[0]
        if isinstance(dtype_preferences, Sequence)
        and not isinstance(dtype_preferences, (str, bytes, bytearray))
        and dtype_preferences
        and type(dtype_preferences[0]) is str
        else "unknown"
    )
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_heads": _lookup(config, "model", "num_heads", default=0),
        "state_key_dim": _lookup(config, "model", "state_key_dim", default=0),
        "state_value_dim": _lookup(config, "model", "state_value_dim", default=0),
        "cache_width": _lookup(config, "cache", "width", default=0),
        "block_size": _lookup(config, "cache", "block_size", default=1),
        "dtype": dtype,
        "device": "unknown",
        "estimated_bytes": 0,
        "peak_vram_bytes": 0,
    }


def execute_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    store: ResultStore,
    command: Sequence[str],
    dispatchers: Mapping[str, BackendDispatcher] | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Execute this store's deterministic shard and continue after typed failures."""

    if not isinstance(store, ResultStore):
        raise TypeError("store must be a ResultStore")
    if type(resume) is not bool:
        raise TypeError("resume must be a bool")
    if dispatchers is not None and not isinstance(dispatchers, Mapping):
        raise TypeError("dispatchers must be a mapping")
    command_list = _command(command)
    selected = select_shard(jobs, store.job_index, store.num_jobs)
    outcomes: list[dict[str, Any]] = []
    for job in selected:
        job_id = job["job_id"]
        if resume and not store.should_run(job):
            outcome = {"job_id": job_id, "status": "skipped"}
            store.append_event(outcome)
            outcomes.append(outcome)
            continue
        store.append_event({"job_id": job_id, "status": "started"})
        try:
            dispatcher = (
                dispatchers.get(job["backend"])
                if dispatchers is not None
                else None
            )
            if dispatcher is None:
                dispatcher = load_backend_dispatcher(job["backend"])
            if not callable(dispatcher):
                raise MalformedInput(
                    "backend dispatcher is not callable", phase="dispatch"
                )
            payload = dispatcher(job)
            record = build_completed_record(
                job,
                store.provenance,
                shard_index=store.job_index,
                num_jobs=store.num_jobs,
                command=command_list,
                payload=payload,
            )
            outcome = {"job_id": job_id, "status": "completed"}
        except Exception as caught:
            if isinstance(caught, JobFailure):
                failure = caught
            elif isinstance(caught, RunRecordError):
                failure = MalformedInput(
                    f"backend result failed validation: {caught}",
                    phase="record",
                    context={"validation_code": caught.code},
                )
            elif isinstance(caught, MemoryError) or type(caught).__name__ in {
                "OutOfMemoryError",
                "CUDAOutOfMemoryError",
            }:
                failure = ForcedOOM(
                    str(caught) or "out of memory",
                    phase="execution",
                    context=_oom_context_from_job(job),
                )
            else:
                failure = JobFailure(
                    str(caught) or type(caught).__name__,
                    phase="execution",
                    context={"exception_type": type(caught).__name__},
                )
            failed_record = build_failed_record(
                job,
                store.provenance,
                shard_index=store.job_index,
                num_jobs=store.num_jobs,
                command=command_list,
                error=failure,
                traceback_text=traceback.format_exc(),
            )
            record = failed_record
            outcome = {
                "job_id": job_id,
                "status": "failed",
                "code": failed_record["error"]["code"],
            }
        store.persist(job, record)
        store.append_event(outcome)
        outcomes.append(outcome)
    return outcomes


__all__ = [
    "BackendUnavailable",
    "ForcedOOM",
    "JobFailure",
    "MalformedInput",
    "NonFiniteGradient",
    "NonFiniteLoss",
    "PreflightCheckError",
    "build_completed_record",
    "build_failed_record",
    "build_reproduction_commands",
    "build_runtime_dispatchers",
    "evaluate_exact_resources",
    "execute_jobs",
    "inspect_external_assets",
    "materialize_maximum_hybrid_campaign",
    "load_backend_dispatcher",
    "preflight",
    "preflight_command",
    "promotion_decision",
    "probe_backend_dispatch",
    "probe_environment",
    "run",
    "run_command",
    "run_campaign",
    "validate_raw_scientific_config",
    "validate_scientific_preflight",
    "validate_output_writable",
]
