"""Strict, injectable Qwen loading for paired KMD-2 heal experiments.

Importing this module never imports Transformers or loads external assets.  The
heavy model loader and the production upgrade manager are resolved only when a
real execution calls :func:`load_qwen_arm`; tests can inject small fakes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import torch


_ARMS = ("native", "recency", "surprise")
_NATIVE_ENV_LOCK = threading.RLock()


class AssetIdentityError(ValueError):
    """An external asset does not match its preregistered identity."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class NativeCheckpointError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ExternalAssetIdentity:
    """Expected identity for one external file or directory tree."""

    name: str
    path: Path | str | os.PathLike[str]
    kind: str
    size_bytes: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("asset name must be a nonempty string")
        if self.kind not in {"file", "directory"}:
            raise ValueError("asset kind must be 'file' or 'directory'")
        try:
            path = Path(self.path)
        except TypeError as error:
            raise TypeError("asset path must be path-like") from error
        object.__setattr__(self, "path", path)
        if self.size_bytes is not None and (
            type(self.size_bytes) is not int or self.size_bytes < 0
        ):
            raise ValueError("asset size_bytes must be a nonnegative integer or None")
        if self.sha256 is not None:
            if (
                type(self.sha256) is not str
                or len(self.sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.sha256)
            ):
                raise ValueError("asset sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class ValidatedAssetIdentity:
    """Resolved measured identity passed to manifests and checkpoints."""

    name: str
    path: Path
    kind: str
    size_bytes: int
    sha256: str


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_identity(path: Path) -> tuple[int, str]:
    entries: list[tuple[str, int, str]] = []
    total = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise AssetIdentityError(
                "asset_symlink_unsupported",
                f"directory asset {path} contains symlink {child}",
            )
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        file_digest = _hash_file(child)
        entries.append((relative, size, file_digest))
        total += size
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return total, hashlib.sha256(encoded).hexdigest()


def _read_frozen_model_config(
    path: Path, expected_fields: set[str] | Mapping[str, object]
) -> dict[str, object]:
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file() or config_path.is_symlink():
        raise NativeCheckpointError("model_config_missing", "model asset must contain a regular config.json")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeCheckpointError("model_config_invalid", "model config.json is not valid UTF-8 JSON") from error
    if not isinstance(raw, Mapping):
        raise NativeCheckpointError("model_config_invalid", "model config.json must be an object")
    expected_values = (
        dict(expected_fields) if isinstance(expected_fields, Mapping) else None
    )
    fields = set(expected_fields)
    nested = raw.get("text_config")
    if nested is None:
        text_config = raw
    elif isinstance(nested, Mapping):
        text_config = nested
    else:
        raise NativeCheckpointError(
            "model_config_invalid", "model config text_config must be an object"
        )
    normalized = dict(text_config)
    if "tie_word_embeddings" not in normalized and "tie_word_embeddings" in raw:
        normalized["tie_word_embeddings"] = raw["tie_word_embeddings"]
    dtype = text_config.get("dtype", text_config.get("torch_dtype"))
    dtype_aliases = {
        "float32": "torch.float32", "fp32": "torch.float32", "torch.float32": "torch.float32",
        "bfloat16": "torch.bfloat16", "bf16": "torch.bfloat16", "torch.bfloat16": "torch.bfloat16",
        "float16": "torch.float16", "fp16": "torch.float16", "torch.float16": "torch.float16",
    }
    if isinstance(dtype, str) and dtype.lower() in dtype_aliases:
        normalized["dtype"] = dtype_aliases[dtype.lower()]
    elif "dtype" in fields:
        raise NativeCheckpointError("model_config_invalid", "model config dtype is missing or unsupported")
    if "rms_norm_type" not in normalized:
        model_type = str(text_config.get("model_type", raw.get("model_type", ""))).lower()
        architectures = raw.get("architectures", [])
        qwen_architecture = (isinstance(architectures, list)
            and any(isinstance(name, str) and "qwen" in name.lower() for name in architectures))
        if model_type.startswith("qwen") or qwen_architecture:
            normalized["rms_norm_type"] = "RMSNorm"
        elif "rms_norm_type" in fields:
            raise NativeCheckpointError("model_config_invalid", "model config rms_norm_type cannot be derived")
    rope_parameters = normalized.get("rope_parameters")
    if isinstance(rope_parameters, Mapping):
        canonical_rope = dict(rope_parameters)
        for field in ("rope_theta", "partial_rotary_factor"):
            if field in canonical_rope and field not in normalized:
                normalized[field] = canonical_rope.pop(field)
            else:
                canonical_rope.pop(field, None)
        normalized.setdefault("rope_scaling", {
            key: canonical_rope[key] for key in sorted(canonical_rope)
        })
    elif rope_parameters is not None:
        raise NativeCheckpointError(
            "model_config_invalid", "rope_parameters must be null or an object"
        )
    rope_scaling = normalized.get("rope_scaling")
    if isinstance(rope_scaling, Mapping):
        rope_scaling = dict(rope_scaling)
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.pop("type")
        normalized["rope_scaling"] = {
            key: rope_scaling[key] for key in sorted(rope_scaling)
        }
    elif rope_scaling is not None:
        raise NativeCheckpointError("model_config_invalid", "rope_scaling must be null or an object")
    # ``use_cache=False`` is a frozen execution rule, not a mutation of the
    # published asset.  Validate that the asset field is boolean, then
    # normalize it to the checkpoint's runtime identity.
    if expected_values is not None and expected_values.get("use_cache") is False:
        if type(normalized.get("use_cache")) is not bool:
            raise NativeCheckpointError(
                "model_config_invalid", "model config use_cache must be boolean"
            )
        normalized["use_cache"] = False
    missing = fields - set(normalized)
    if missing:
        raise NativeCheckpointError("model_config_invalid", "model config is missing: " + ", ".join(sorted(missing)))
    return {name: normalized[name] for name in sorted(fields)}


def validate_external_assets(
    assets: Sequence[ExternalAssetIdentity],
) -> tuple[ValidatedAssetIdentity, ...]:
    """Validate all identities before any model loader is allowed to execute."""
    if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence):
        raise TypeError("assets must be a sequence of ExternalAssetIdentity records")
    seen: set[str] = set()
    validated: list[ValidatedAssetIdentity] = []
    for asset in assets:
        if not isinstance(asset, ExternalAssetIdentity):
            raise TypeError("assets must contain ExternalAssetIdentity records")
        if asset.name in seen:
            raise ValueError(f"duplicate external asset name: {asset.name}")
        seen.add(asset.name)
        path = asset.path.expanduser().resolve()
        if not path.exists():
            raise AssetIdentityError(
                "asset_missing", f"external asset {asset.name!r} is missing: {path}"
            )
        if asset.kind == "file":
            if not path.is_file():
                raise AssetIdentityError(
                    "asset_kind_mismatch",
                    f"external asset {asset.name!r} must be a file: {path}",
                )
            size = path.stat().st_size
            digest = _hash_file(path)
        else:
            if not path.is_dir():
                raise AssetIdentityError(
                    "asset_kind_mismatch",
                    f"external asset {asset.name!r} must be a directory: {path}",
                )
            size, digest = _directory_identity(path)
        if asset.size_bytes is not None and size != asset.size_bytes:
            raise AssetIdentityError(
                "asset_size_mismatch",
                f"external asset {asset.name!r} expected {asset.size_bytes} bytes, got {size}",
            )
        if asset.sha256 is not None and digest != asset.sha256:
            raise AssetIdentityError(
                "asset_hash_mismatch",
                f"external asset {asset.name!r} SHA-256 does not match",
            )
        validated.append(
            ValidatedAssetIdentity(
                name=asset.name,
                path=path,
                kind=asset.kind,
                size_bytes=size,
                sha256=digest,
            )
        )
    return tuple(sorted(validated, key=lambda item: item.name))


@dataclass(frozen=True)
class QwenArmLoadSpec:
    """Execution-only inputs needed to construct one paired Qwen arm."""

    arm: str
    job_id: str
    model_asset: ExternalAssetIdentity
    native_checkpoint: ExternalAssetIdentity | None
    data_asset: ExternalAssetIdentity
    cache_resume: ExternalAssetIdentity | None
    trainable_names: tuple[str, ...]
    pre_replacement_checkpoint_sha256: str
    model_loader_kwargs: Mapping[str, object] = MappingProxyType({})
    architecture_arm_id: str | None = None
    architecture_registry_sha256: str | None = None
    diagnostic_training: bool = False
    teacher_asset: ExternalAssetIdentity | None = None
    architecture_implementation_sha256: str | None = None
    source_checkpoint_sha256: str | None = None
    frozen_qwen_config: Mapping[str, object] | None = None
    architecture_cache_policy: Mapping[str, object] | None = None
    maximum_control_id: str | None = None

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"arm must be one of: {', '.join(_ARMS)}")
        if type(self.job_id) is not str or not self.job_id:
            raise ValueError("job_id must be a nonempty string")
        for name in ("model_asset", "data_asset"):
            if not isinstance(getattr(self, name), ExternalAssetIdentity):
                raise TypeError(f"{name} must be an ExternalAssetIdentity")
        for name in ("native_checkpoint", "cache_resume"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ExternalAssetIdentity):
                raise TypeError(f"{name} must be an ExternalAssetIdentity or None")
        if self.teacher_asset is not None and not isinstance(self.teacher_asset, ExternalAssetIdentity):
            raise TypeError("teacher_asset must be an ExternalAssetIdentity or None")
        if self.maximum_control_id is not None:
            from .qwen_variants import maximum_control_contract
            maximum_control_contract(self.maximum_control_id)
        for name in ("architecture_implementation_sha256", "source_checkpoint_sha256"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError(f"{name} must be lowercase SHA-256 or None")
        for name in ("frozen_qwen_config", "architecture_cache_policy"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, Mapping) or not value:
                    raise ValueError(f"{name} must be a nonempty mapping or None")
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        if self.native_checkpoint is None:
            raise ValueError(
                "native_checkpoint_required: Qwen heal arms require a native checkpoint"
            )
        if self.arm == "native" and self.cache_resume is not None:
            raise ValueError("native continuation cannot load a cache resume")
        if type(self.trainable_names) is not tuple or (
            not self.trainable_names and self.architecture_arm_id is None
        ):
            raise ValueError("trainable_names must be a nonempty tuple unless the architecture is frozen")
        if any(type(name) is not str or not name for name in self.trainable_names):
            raise ValueError("trainable_names must contain nonempty strings")
        if len(set(self.trainable_names)) != len(self.trainable_names):
            raise ValueError("trainable_names must not contain duplicates")
        digest = self.pre_replacement_checkpoint_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "pre_replacement_checkpoint_sha256 must be lowercase SHA-256"
            )
        if not isinstance(self.model_loader_kwargs, Mapping):
            raise TypeError("model_loader_kwargs must be a mapping")
        frozen_kwargs = MappingProxyType(dict(self.model_loader_kwargs))
        if any(type(key) is not str or not key for key in frozen_kwargs):
            raise ValueError("model_loader_kwargs keys must be nonempty strings")
        object.__setattr__(self, "model_loader_kwargs", frozen_kwargs)
        identity = (self.architecture_arm_id, self.architecture_registry_sha256)
        if type(self.diagnostic_training) is not bool:
            raise TypeError("diagnostic_training must be boolean")
        if (identity[0] is None) != (identity[1] is None):
            raise ValueError("architecture_identity_incomplete")
        if identity[0] is not None:
            if self.arm != "native":
                raise ValueError("legacy_architecture_arm_mismatch")
            if identity[0] not in {
                "gdn2-channel-r1", "rout-4", "mimo-r2", "mimo-r4", "rot-off", "rot-constant",
                "rot-noncumulative", "rot-fixed-rope", "rot-moving-frame-oracle",
                "trapezoid", "lookahead", "qk-bc-additive", "qk-diagonal",
                "gdn2-mimo-r4-braid-shared-hola-w64",
                "gdn2-mimo-r4-braid-four-state-hola-w64",
            }:
                raise ValueError("architecture_not_implemented")
            from .architecture import registry_sha256
            if identity[1] != registry_sha256():
                raise ValueError("architecture_registry_hash_mismatch")
            if self.diagnostic_training and identity[0] != "rot-moving-frame-oracle":
                raise ValueError("diagnostic_training_arm_mismatch")
        elif self.diagnostic_training:
            raise ValueError("diagnostic_training requires architecture identity")


@dataclass(frozen=True)
class LoadedQwenArm:
    """A constructed arm plus the exact identity and trainability record."""

    model: torch.nn.Module
    arm: str
    job_id: str
    upgraded_indices: tuple[int, ...]
    trainable_names: tuple[str, ...]
    assets: tuple[ValidatedAssetIdentity, ...]
    architecture_arm_id: str | None = None
    architecture_registry_sha256: str | None = None
    architecture_classification: str | None = None
    architecture_identity_passed: bool = False
    architecture_implementation: str | None = None
    architecture_tensor_manifest: Mapping[str, object] | None = None


class PairingContractError(ValueError):
    """The three Qwen heal arms are not a mechanically paired comparison."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ArchitectureManifestError(ValueError):
    """Typed failure for incomplete or heterogeneous conversion evidence."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _normalize_architecture_tensor_manifest(manifest: object) -> dict[str, tuple]:
    if not isinstance(manifest, Mapping) or set(manifest) != {"copied", "transformed", "new"}:
        raise ArchitectureManifestError("architecture_tensor_manifest_invalid")
    copied, transformed, new = (
        manifest["copied"], manifest["transformed"], manifest["new"]
    )
    if any(type(items) not in (tuple, list) for items in (copied, transformed, new)):
        raise ArchitectureManifestError("architecture_tensor_manifest_invalid")
    if any(type(name) is not str or not name for name in (*copied, *new)):
        raise ArchitectureManifestError("architecture_tensor_manifest_invalid")
    normalized_transformed = []
    for item in transformed:
        if (
            type(item) not in (tuple, list)
            or len(item) != 3
            or any(type(value) is not str or not value for value in item)
        ):
            raise ArchitectureManifestError("architecture_tensor_manifest_invalid")
        normalized_transformed.append(tuple(item))
    return {
        "copied": tuple(copied),
        "transformed": tuple(normalized_transformed),
        "new": tuple(new),
    }


def _aggregate_architecture_tensor_manifest(
    model: torch.nn.Module, indices: Sequence[int]
) -> Mapping[str, object]:
    canonical = None
    canonical_rank = None
    aggregated: dict[str, list[object]] = {"copied": [], "transformed": [], "new": []}
    for index in indices:
        module = model.model.layers[index].linear_attn
        builder = getattr(module, "architecture_tensor_manifest", None)
        if not callable(builder):
            builder = getattr(module, "transformation_manifest", None)
        if not callable(builder):
            raise ArchitectureManifestError("architecture_tensor_manifest_missing")
        manifest = builder()
        normalized = _normalize_architecture_tensor_manifest(manifest)
        rank = getattr(module, "rank", 1)
        if type(rank) is not int or rank not in (1, 2, 4):
            raise ArchitectureManifestError("architecture_tensor_manifest_invalid_rank")
        if canonical is None:
            canonical = normalized
            canonical_rank = rank
        elif normalized != canonical:
            raise ArchitectureManifestError("architecture_tensor_manifest_heterogeneous")
        elif rank != canonical_rank:
            raise ArchitectureManifestError("architecture_tensor_manifest_heterogeneous_rank")
        prefix = f"model.layers.{index}.linear_attn."
        aggregated["copied"].extend(prefix + name for name in normalized["copied"])
        aggregated["new"].extend(prefix + name for name in normalized["new"])
        aggregated["transformed"].extend(
            (prefix + source, prefix + target, operation)
            for source, target, operation in normalized["transformed"]
        )
    if canonical is None:
        raise ArchitectureManifestError("architecture_tensor_manifest_missing")
    return MappingProxyType({
        **{name: tuple(values) for name, values in aggregated.items()},
        "mimo_rank": canonical_rank,
        "layer_count": len(indices),
    })


def _freeze_json(value: object, context: str) -> object:
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} must not contain nonfinite values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if type(key) is not str or not key:
                raise ValueError(f"{context} keys must be nonempty strings")
            frozen[key] = _freeze_json(value[key], f"{context}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{context} must contain only JSON-compatible values")


@dataclass(frozen=True)
class QwenHealArmContract:
    """All scientific identity fields for one arm of a paired Qwen heal."""

    arm: str
    job_id: str
    seed: int
    pre_replacement_checkpoint_sha256: str
    data_sha256: str
    example_ids: tuple[str, ...]
    token_budget: int
    update_budget: int
    curriculum: tuple[int, ...]
    optimizer: Mapping[str, object]
    schedule: Mapping[str, object]
    stopping: Mapping[str, object]
    eval_cells: tuple[str, ...]
    cache_match: Mapping[str, object] | None
    selection_policy: str | None

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"arm must be one of: {', '.join(_ARMS)}")
        if type(self.job_id) is not str or not self.job_id:
            raise ValueError("job_id must be a nonempty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        digest = self.pre_replacement_checkpoint_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "pre_replacement_checkpoint_sha256 must be lowercase SHA-256"
            )
        data_digest = self.data_sha256
        if (
            type(data_digest) is not str
            or len(data_digest) != 64
            or any(character not in "0123456789abcdef" for character in data_digest)
        ):
            raise ValueError("data_sha256 must be lowercase SHA-256")
        for field_name in ("example_ids", "eval_cells"):
            value = getattr(self, field_name)
            if type(value) is not tuple or not value:
                raise ValueError(f"{field_name} must be a nonempty tuple")
            if any(type(item) is not str or not item for item in value):
                raise ValueError(f"{field_name} must contain nonempty strings")
            if len(set(value)) != len(value):
                raise ValueError(f"{field_name} must not contain duplicates")
        for field_name in ("token_budget", "update_budget"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            type(self.curriculum) is not tuple
            or not self.curriculum
            or any(type(length) is not int or length < 1 for length in self.curriculum)
            or tuple(sorted(set(self.curriculum))) != self.curriculum
        ):
            raise ValueError("curriculum must be a strictly increasing tuple of lengths")
        for field_name in ("optimizer", "schedule", "stopping"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{field_name} must be a nonempty mapping")
            object.__setattr__(self, field_name, _freeze_json(value, field_name))

        if self.arm == "native":
            if self.cache_match is not None or self.selection_policy is not None:
                raise ValueError("native arm cannot declare a cache policy")
            return
        if not isinstance(self.cache_match, Mapping) or not self.cache_match:
            raise ValueError("cache arms require a nonempty cache_match mapping")
        frozen_cache = _freeze_json(self.cache_match, "cache_match")
        assert isinstance(frozen_cache, Mapping)
        required = {
            "width",
            "block_size",
            "read",
            "read_init",
            "storage_dtype",
            "lr_cache",
        }
        missing = sorted(required - set(frozen_cache))
        if missing:
            raise ValueError(
                "cache_match is missing matched settings: " + ", ".join(missing)
            )
        object.__setattr__(self, "cache_match", frozen_cache)
        if self.arm == "recency" and self.selection_policy != "recency":
            raise ValueError("recency arm requires selection_policy='recency'")
        if self.arm == "surprise" and (
            type(self.selection_policy) is not str
            or not self.selection_policy
            or self.selection_policy == "recency"
        ):
            raise ValueError("surprise arm requires a non-recency selection policy")


@dataclass(frozen=True)
class PairedQwenHealContract:
    """Validated native/recency/surprise comparison with a shared hash ID."""

    pairing_id: str
    arms: tuple[QwenHealArmContract, ...]
    canonical_bytes: bytes
    example_ids: tuple[str, ...]


def validate_three_arm_pairing(
    arms: Sequence[QwenHealArmContract],
) -> PairedQwenHealContract:
    """Require exact byte/checkpoint/data/budget pairing across all three arms."""
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
        raise TypeError("arms must be a sequence of QwenHealArmContract records")
    if len(arms) != 3 or any(not isinstance(arm, QwenHealArmContract) for arm in arms):
        raise PairingContractError(
            "pairing_arm_set", "pairing requires exactly three Qwen arm records"
        )
    by_arm = {arm.arm: arm for arm in arms}
    if len(by_arm) != 3 or set(by_arm) != set(_ARMS):
        raise PairingContractError(
            "pairing_arm_set", "pairing requires native, recency, and surprise once each"
        )
    ordered = tuple(by_arm[name] for name in _ARMS)
    native, recency, surprise = ordered
    shared_fields = (
        "seed",
        "pre_replacement_checkpoint_sha256",
        "data_sha256",
        "example_ids",
        "token_budget",
        "update_budget",
        "curriculum",
        "optimizer",
        "schedule",
        "stopping",
        "eval_cells",
    )
    for field_name in shared_fields:
        expected = getattr(native, field_name)
        mismatched = [
            arm.arm for arm in ordered[1:] if getattr(arm, field_name) != expected
        ]
        if mismatched:
            raise PairingContractError(
                "pairing_mismatch",
                f"{field_name} differs for arm(s): {', '.join(mismatched)}",
            )
    if recency.cache_match != surprise.cache_match:
        raise PairingContractError(
            "cache_match_mismatch",
            "capacity/read/gate/cache optimizer settings differ between recency and surprise",
        )
    payload = {
        "cache_match": dict(recency.cache_match or {}),
        "curriculum": list(native.curriculum),
        "eval_cells": list(native.eval_cells),
        "example_ids": list(native.example_ids),
        "optimizer": dict(native.optimizer),
        "policies": {
            arm.arm: arm.selection_policy for arm in ordered
        },
        "pre_replacement_checkpoint_sha256": native.pre_replacement_checkpoint_sha256,
        "data_sha256": native.data_sha256,
        "schedule": dict(native.schedule),
        "seed": native.seed,
        "stopping": dict(native.stopping),
        "token_budget": native.token_budget,
        "update_budget": native.update_budget,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return PairedQwenHealContract(
        pairing_id=hashlib.sha256(canonical).hexdigest(),
        arms=ordered,
        canonical_bytes=canonical,
        example_ids=native.example_ids,
    )


def _default_base_model_loader(path: Path, **kwargs: object) -> torch.nn.Module:
    """Load the language model from either text-only or official Qwen assets.

    Qwen3.5-0.8B is published as a multimodal wrapper whose causal language
    model lives at ``model.language_model``.  The experiment is text-only and
    all of its frozen parameter/checkpoint names use the ordinary
    ``model.layers`` causal-LM layout.  Load the published wrapper once, then
    move the already-loaded language backbone and tied LM head into a
    meta-initialized causal-LM shell.  This avoids allocating a second 0.8B
    model and releases the unused vision tower before the model is moved to
    the training GPU.
    """
    from transformers import (  # type: ignore[import-not-found]
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
    )

    config = AutoConfig.from_pretrained(str(path))
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return AutoModelForCausalLM.from_pretrained(str(path), **kwargs)

    wrapper = AutoModelForMultimodalLM.from_pretrained(str(path), **kwargs)
    language_model = getattr(getattr(wrapper, "model", None), "language_model", None)
    lm_head = getattr(wrapper, "lm_head", None)
    if not isinstance(language_model, torch.nn.Module) or not isinstance(
        lm_head, torch.nn.Module
    ):
        raise TypeError(
            "multimodal Qwen asset must expose model.language_model and lm_head"
        )
    with torch.device("meta"):
        causal_model = AutoModelForCausalLM.from_config(text_config)
    if not hasattr(causal_model, "model") or not hasattr(causal_model, "lm_head"):
        raise TypeError("Qwen text config did not construct a causal-LM shell")
    causal_model.model = language_model
    causal_model.lm_head = lm_head
    return causal_model


def _default_manager_factory(model: torch.nn.Module, model_config: object) -> object:
    from gdn3.gdn3_upgrade import GDN3UpgradeManager

    return GDN3UpgradeManager(model, model_config)


def _intended_model_dtype(
    model: torch.nn.Module, loader_kwargs: Mapping[str, object]
) -> torch.dtype:
    requested = loader_kwargs.get("torch_dtype")
    candidates = [requested, getattr(model, "dtype", None)]
    candidates.extend(
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    )
    for candidate in candidates:
        if isinstance(candidate, torch.dtype) and torch.empty(
            (), dtype=candidate
        ).is_floating_point():
            return candidate
    raise TypeError("loaded Qwen model must expose an intended floating-point dtype")


def _load_tensor_mapping(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if checkpoint is None:
        return {}
    loaded = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and isinstance(loaded.get("state_dict"), Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping):
        raise TypeError("native checkpoint must contain a tensor mapping")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        if type(name) is not str or not name:
            raise TypeError("native checkpoint names must be nonempty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"native checkpoint value {name!r} is not a tensor")
        result[name] = tensor
    return result


def _validate_indices(model: object, raw: object) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("upgrade manager must return a sequence of layer indices")
    indices = tuple(raw)
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("upgrade manager returned an invalid layer index")
    if len(set(indices)) != len(indices) or not indices:
        raise ValueError("upgrade manager must return unique upgraded layer indices")
    try:
        layer_count = len(model.model.layers)
    except (AttributeError, TypeError) as error:
        raise TypeError("model must expose model.layers") from error
    if any(index >= layer_count for index in indices):
        raise ValueError("upgrade manager returned an out-of-range layer index")
    return indices


def _default_native_installer(
    *,
    model: torch.nn.Module,
    manager: object,
    model_config: object,
    cache_config: object,
    native_checkpoint: Path | None,
    cache_resume: Path | None,
    expected_job_id: str,
    target_dtype: torch.dtype,
    native_r_out: int | None = None,
    strict_architecture_checkpoint: bool = False,
) -> tuple[int, ...]:
    del model_config, cache_config, expected_job_id
    if cache_resume is not None:
        raise ValueError("native continuation cannot load a cache resume")
    _NATIVE_ENV_LOCK.acquire()
    prior = os.environ.get("GDN3_KMD2_NATIVE")
    prior_rout = os.environ.get("GDN3_KMD2_ROUT")
    os.environ["GDN3_KMD2_NATIVE"] = "1"
    if native_r_out is not None:
        os.environ["GDN3_KMD2_ROUT"] = str(native_r_out)
    try:
        apply_upgrade = getattr(manager, "apply_upgrade", None)
        if not callable(apply_upgrade):
            raise TypeError("manager must expose apply_upgrade()")
        indices = _validate_indices(model, apply_upgrade())
        from gdn3.kmd2_native import KMD2NativeAttn

        prefixes: list[str] = []
        named_modules = dict(model.named_modules())
        for index in indices:
            module = model.model.layers[index].linear_attn
            if type(module) is not KMD2NativeAttn:
                raise TypeError(f"upgraded layer {index} is not an actual KMD2NativeAttn")
            if native_r_out is not None and (
                module.r_out != native_r_out
                or hasattr(module, "q_slot_scale")
                or hasattr(module, "out_mix")
            ):
                raise ValueError("canonical native R1 construction invariant failed")
            module.to(dtype=target_dtype)
            names = [name for name, candidate in named_modules.items() if candidate is module]
            if len(names) != 1:
                raise ValueError(f"upgraded layer {index} has no unique model name")
            prefixes.append(names[0] + ".")

        checkpoint = _load_tensor_mapping(native_checkpoint)
        state = model.state_dict()
        if strict_architecture_checkpoint:
            from .architecture import TARGET_LAYERS
            if tuple(indices) != TARGET_LAYERS:
                raise NativeCheckpointError(
                    "native_checkpoint_target_invalid", "upgraded target layers are not canonical"
                )
            suffixes = (
                "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
                "in_proj_a.weight", "conv1d.weight", "dt_bias", "A_log",
                "norm.weight", "out_proj.weight", "rot_proj.weight",
                "rot_proj.bias", "decay_chan", "bw_off",
            )
            expected = {
                f"model.layers.{index}.linear_attn.{suffix}"
                for index in TARGET_LAYERS for suffix in suffixes
            }
            wrong_layer = []
            for name in checkpoint:
                if not name.startswith("model.layers."):
                    continue
                parts = name.split(".")
                if len(parts) < 5 or not parts[2].isdigit() or int(parts[2]) not in TARGET_LAYERS:
                    wrong_layer.append(name)
            if wrong_layer:
                raise NativeCheckpointError("native_checkpoint_target_invalid", wrong_layer[0])
            missing = sorted(expected - set(checkpoint))
            if missing:
                raise NativeCheckpointError("native_checkpoint_tensor_missing", missing[0])
            unexpected = sorted(set(checkpoint) - expected)
            if unexpected:
                raise NativeCheckpointError("native_checkpoint_tensor_unexpected", unexpected[0])
        targets: dict[str, torch.Tensor] = {}
        for name, tensor in checkpoint.items():
            if not name.startswith(tuple(prefixes)) or name not in state:
                raise KeyError(
                    f"native checkpoint key {name!r} does not target an upgraded layer"
                )
            target = state[name]
            if target.shape != tensor.shape:
                if strict_architecture_checkpoint:
                    raise NativeCheckpointError("native_checkpoint_shape_mismatch", name)
                raise ValueError(f"native checkpoint tensor {name!r} shape does not match")
            if target.dtype != tensor.dtype:
                if strict_architecture_checkpoint:
                    raise NativeCheckpointError("native_checkpoint_dtype_mismatch", name)
                raise ValueError(f"native checkpoint tensor {name!r} dtype does not match")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"native checkpoint tensor {name!r} is nonfinite")
            targets[name] = target
        with torch.no_grad():
            for name, target in targets.items():
                target.copy_(checkpoint[name].to(device=target.device))
        return indices
    finally:
        if prior is None:
            os.environ.pop("GDN3_KMD2_NATIVE", None)
        else:
            os.environ["GDN3_KMD2_NATIVE"] = prior
        if prior_rout is None:
            os.environ.pop("GDN3_KMD2_ROUT", None)
        else:
            os.environ["GDN3_KMD2_ROUT"] = prior_rout
        _NATIVE_ENV_LOCK.release()


_RECENCY_CACHE_TYPE: type[torch.nn.Module] | None = None


def _recency_cache_type() -> type[torch.nn.Module]:
    global _RECENCY_CACHE_TYPE
    if _RECENCY_CACHE_TYPE is not None:
        return _RECENCY_CACHE_TYPE
    from .qwen_exact_cache import KMD2ExactCacheAttn

    class KMD2RecencyCacheAttn(KMD2ExactCacheAttn):
        """Exact-cache read whose persistent admission order is pure recency."""

        def _native_state_and_scores(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta_e: torch.Tensor,
            beta_w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state, _ = KMD2ExactCacheAttn._native_state_and_scores(
                self, q, k, v, g, beta_e, beta_w
            )
            scores = torch.arange(
                1,
                k.shape[1] + 1,
                device=k.device,
                dtype=torch.float32,
            ).view(1, -1, 1)
            return state, scores.expand(k.shape[0], -1, k.shape[2])

    KMD2RecencyCacheAttn.__name__ = "KMD2RecencyCacheAttn"
    KMD2RecencyCacheAttn.__qualname__ = "KMD2RecencyCacheAttn"
    KMD2RecencyCacheAttn.__module__ = __name__
    _RECENCY_CACHE_TYPE = KMD2RecencyCacheAttn
    return _RECENCY_CACHE_TYPE


def _default_cache_installer(
    *,
    arm: str,
    model: torch.nn.Module,
    manager: object,
    model_config: object,
    cache_config: object,
    native_checkpoint: Path | None,
    cache_resume: Path | None,
    expected_job_id: str,
    target_dtype: torch.dtype,
) -> tuple[int, ...]:
    from .config import CacheConfig
    from .qwen_exact_cache import KMD2ExactCacheAttn, load_native_then_install

    if not isinstance(cache_config, CacheConfig):
        raise TypeError("cache_config must be a CacheConfig for cache arms")
    if arm == "surprise":
        if cache_config.score != "exact_outer":
            raise ValueError("the initial Qwen surprise arm requires cache.score=exact_outer")
        return load_native_then_install(
            model,
            manager,
            model_config,
            cache_config,
            native_checkpoint,
            cache_resume,
            expected_job_id=expected_job_id,
            target_dtype=target_dtype,
        )
    if arm != "recency":
        raise ValueError("cache installer supports only recency and surprise arms")
    if cache_config.score != "recency":
        raise ValueError("the recency arm requires cache.score=recency")
    exact_config = replace(cache_config, score="exact_outer")
    indices = load_native_then_install(
        model,
        manager,
        model_config,
        exact_config,
        native_checkpoint,
        cache_resume,
        expected_job_id=expected_job_id,
        target_dtype=target_dtype,
    )
    recency_type = _recency_cache_type()
    for index in indices:
        layer = model.model.layers[index].linear_attn
        if type(layer) is not KMD2ExactCacheAttn:
            raise TypeError("recency conversion expected an exact-cache installation")
        layer.__class__ = recency_type
        layer.cache_config = cache_config
    return indices


def _configure_trainables(
    model: torch.nn.Module, declared: tuple[str, ...]
) -> tuple[str, ...]:
    named = dict(model.named_parameters())
    missing = sorted(set(declared) - set(named))
    if missing:
        raise KeyError("declared trainable parameters are missing: " + ", ".join(missing))
    original = {name: parameter.requires_grad for name, parameter in named.items()}
    try:
        selected = set(declared)
        for name, parameter in named.items():
            parameter.requires_grad_(name in selected)
        actual = tuple(sorted(name for name, parameter in named.items() if parameter.requires_grad))
        expected = tuple(sorted(declared))
        if actual != expected:
            raise RuntimeError("trainable parameter set does not match the declaration")
        return actual
    except Exception:
        for name, parameter in named.items():
            parameter.requires_grad_(original[name])
        raise


def _verify_architecture_module(
    module: torch.nn.Module,
    expected_type: type[torch.nn.Module],
    expected_rank: int,
    *,
    expected_output_width: int | None = None,
) -> None:
    if type(module) is not expected_type:
        raise TypeError("architecture_type_mismatch")
    if expected_rank > 1 and getattr(module, "rank", None) != expected_rank:
        raise ValueError("architecture_rank_mismatch")
    if expected_output_width is not None:
        width = getattr(module, "output_width", None)
        r_out = getattr(module, "r_out", None)
        if (type(width) is not int or type(r_out) is not int
                or width != expected_output_width or r_out != expected_output_width):
            raise ValueError("architecture_output_width_invalid")


def _verify_architecture_modules(
    model: torch.nn.Module,
    indices: tuple[int, ...],
    expected_type: type[torch.nn.Module],
    expected_rank: int,
    *,
    expected_output_width: int | None = None,
) -> None:
    modules = tuple(model.model.layers[index].linear_attn for index in indices)
    if expected_output_width is not None:
        widths = tuple(getattr(module, "output_width", None) for module in modules)
        r_outs = tuple(getattr(module, "r_out", None) for module in modules)
        if (all(type(value) is int for value in (*widths, *r_outs))
                and (len(set(widths)) > 1 or len(set(r_outs)) > 1)):
            raise ValueError("architecture_output_width_heterogeneous")
    for module in modules:
        _verify_architecture_module(
            module, expected_type, expected_rank,
            expected_output_width=expected_output_width,
        )


def load_qwen_arm(
    spec: QwenArmLoadSpec,
    *,
    model_config: object,
    cache_config: object | None,
    base_model_loader: Callable[..., torch.nn.Module] | None = None,
    manager_factory: Callable[[torch.nn.Module, object], object] | None = None,
    native_installer: Callable[..., Sequence[int]] | None = None,
    cache_installer: Callable[..., Sequence[int]] | None = None,
    architecture_factory: Callable[..., torch.nn.Module] | None = None,
    architecture_verifier: Callable[[torch.nn.Module, tuple[int, ...]], object] | None = None,
    architecture_expected_type: type[torch.nn.Module] | None = None,
    event: Callable[[str], object] | None = None,
) -> LoadedQwenArm:
    """Validate assets, construct one arm, then freeze exactly as declared."""
    if not isinstance(spec, QwenArmLoadSpec):
        raise TypeError("spec must be a QwenArmLoadSpec")
    assets = [spec.model_asset, spec.data_asset]
    if spec.native_checkpoint is not None:
        assets.append(spec.native_checkpoint)
    if spec.cache_resume is not None:
        assets.append(spec.cache_resume)
    if spec.teacher_asset is not None:
        assets.append(spec.teacher_asset)
    if event: event("validate_assets")
    validated = validate_external_assets(assets)
    paths = {asset.name: asset.path for asset in validated}
    measured = {asset.name: asset for asset in validated}
    assert spec.native_checkpoint is not None
    checkpoint_identity = measured[spec.native_checkpoint.name]
    if checkpoint_identity.sha256 != spec.pre_replacement_checkpoint_sha256:
        raise AssetIdentityError(
            "checkpoint_identity_mismatch",
            "measured native checkpoint identity does not match "
            "pre_replacement_checkpoint_sha256",
        )
    model_path = paths[spec.model_asset.name]
    checkpoint_path = paths[spec.native_checkpoint.name]
    resume_path = (
        None if spec.cache_resume is None else paths[spec.cache_resume.name]
    )

    # Architecture-conversion checkpoints are self-describing and must be
    # rejected before the expensive/base model constructor is entered.
    architecture_checkpoint: Mapping[str, object] | None = None
    if spec.architecture_arm_id is not None:
        try:
            candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            candidate = None
        if isinstance(candidate, Mapping) and candidate.get("kind") == "qwen_hybrid_architecture_conversion":
            required = {"schema_version", "kind", "architecture_arm_id", "identity",
                        "model_state", "tensor_manifest", "conversion_manifest", "optimizer_state"}
            if set(candidate) != required or candidate.get("schema_version") != 1:
                raise NativeCheckpointError("architecture_checkpoint_fields", "hybrid conversion fields are incomplete or stale")
            if candidate.get("architecture_arm_id") != spec.architecture_arm_id:
                raise NativeCheckpointError("architecture_identity_mismatch", "hybrid conversion arm differs")
            if candidate.get("optimizer_state") is not None:
                raise NativeCheckpointError("source_optimizer_forbidden", "conversion checkpoints may not import optimizer state")
            from .qwen_checkpoint import QwenHybridCheckpointIdentity, expected_hybrid_tensor_contract
            raw_identity = candidate.get("identity")
            try:
                if not isinstance(raw_identity, Mapping):
                    raise TypeError("identity must be a mapping")
                checkpoint_identity = QwenHybridCheckpointIdentity(
                    **{**raw_identity,
                       "target_module_names": tuple(raw_identity.get("target_module_names", ())),
                       "trainable_manifest": tuple(raw_identity.get("trainable_manifest", ()))},
                )
            except (TypeError, ValueError) as error:
                raise NativeCheckpointError("architecture_identity_invalid", str(error)) from error
            if checkpoint_identity.architecture_registry_sha256 != spec.architecture_registry_sha256:
                raise NativeCheckpointError("architecture_registry_mismatch", "hybrid registry identity differs")
            required_runtime = (spec.architecture_implementation_sha256,
                                spec.source_checkpoint_sha256, spec.teacher_asset,
                                spec.frozen_qwen_config, spec.architecture_cache_policy)
            if any(value is None for value in required_runtime):
                raise NativeCheckpointError("architecture_runtime_identity_missing", "hybrid runtime identity is incomplete")
            if checkpoint_identity.implementation_sha256 != spec.architecture_implementation_sha256:
                raise NativeCheckpointError("architecture_implementation_mismatch", "hybrid implementation identity differs")
            if checkpoint_identity.pre_replacement_checkpoint_sha256 != spec.source_checkpoint_sha256:
                raise NativeCheckpointError("pre_replacement_checkpoint_mismatch", "hybrid source checkpoint identity differs")
            assert spec.teacher_asset is not None
            if checkpoint_identity.teacher_sha256 != measured[spec.teacher_asset.name].sha256:
                raise NativeCheckpointError("teacher_identity_mismatch", "hybrid teacher identity differs")
            if dict(checkpoint_identity.frozen_qwen_config) != dict(spec.frozen_qwen_config):
                raise NativeCheckpointError("frozen_qwen_config_mismatch", "hybrid frozen Qwen config differs")
            asset_config = _read_frozen_model_config(
                model_path, checkpoint_identity.frozen_qwen_config
            )
            if asset_config != dict(checkpoint_identity.frozen_qwen_config):
                raise NativeCheckpointError("frozen_qwen_config_mismatch", "model asset config differs from checkpoint")
            if dict(checkpoint_identity.cache_policy) != dict(spec.architecture_cache_policy):
                raise NativeCheckpointError("cache_policy_mismatch", "hybrid cache policy differs")
            expected_contract = expected_hybrid_tensor_contract(checkpoint_identity, spec.architecture_arm_id)
            from .architecture import TARGET_LAYERS
            expected_targets = tuple(sorted(
                f"model.layers.{index}.linear_attn" for index in TARGET_LAYERS
            ))
            if checkpoint_identity.target_module_names != expected_targets:
                raise NativeCheckpointError("architecture_target_indices_mismatch", "hybrid recurrent targets are not canonical")
            if int(checkpoint_identity.frozen_qwen_config["num_hidden_layers"]) <= max(TARGET_LAYERS):
                raise NativeCheckpointError("frozen_qwen_config_mismatch", "frozen layer count cannot contain recurrent targets")
            expected_trainables = tuple(sorted(
                ({"name": f"{target}.{name}", "shape": tuple(shape), "dtype": dtype}
                 for target in checkpoint_identity.target_module_names
                 for name, (shape, dtype) in expected_contract.items()),
                key=lambda row: row["name"],
            ))
            if tuple(dict(row) for row in checkpoint_identity.trainable_manifest) != expected_trainables:
                raise NativeCheckpointError("trainable_manifest_mismatch", "hybrid trainable tensor contract differs")
            if tuple(sorted(spec.trainable_names)) != tuple(row["name"] for row in expected_trainables):
                raise NativeCheckpointError("trainable_manifest_mismatch", "runtime hybrid trainable names differ")
            if checkpoint_identity.model_tree_sha256 != measured[spec.model_asset.name].sha256:
                raise NativeCheckpointError("model_tree_mismatch", "hybrid model-tree identity differs")
            if checkpoint_identity.ordered_examples_sha256 != measured[spec.data_asset.name].sha256:
                raise NativeCheckpointError("ordered_examples_mismatch", "hybrid ordered-example identity differs")
            state = candidate.get("model_state")
            manifest = candidate.get("tensor_manifest")
            if not isinstance(state, Mapping) or not isinstance(manifest, list):
                raise NativeCheckpointError("architecture_checkpoint_partial", "hybrid state or manifest is missing")
            actual_manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                               for name, tensor in state.items() if isinstance(tensor, torch.Tensor)]
            if len(actual_manifest) != len(state) or manifest != actual_manifest:
                raise NativeCheckpointError("architecture_checkpoint_stale", "hybrid tensor manifest differs")
            conversion = candidate.get("conversion_manifest")
            layers = conversion.get("layers") if isinstance(conversion, Mapping) else None
            if not isinstance(layers, Mapping) or tuple(layers) != checkpoint_identity.target_module_names:
                raise NativeCheckpointError("architecture_checkpoint_partial", "per-layer conversion manifest differs")
            aggregate_rows = []
            for target, layer_manifest in layers.items():
                if not isinstance(layer_manifest, Mapping) or set(layer_manifest) != {
                    "source_sha256", "source_tensors", "target_tensors", "transformation"
                }:
                    raise NativeCheckpointError("architecture_checkpoint_partial", f"layer {target} manifest is incomplete")
                source_sha = layer_manifest["source_sha256"]
                if (type(source_sha) is not str or len(source_sha) != 64
                        or any(character not in "0123456789abcdef" for character in source_sha)):
                    raise NativeCheckpointError("pre_replacement_checkpoint_mismatch", "layer source hash is invalid")
                aggregate_rows.append((target, source_sha))
                prefix = target + "."
                target_state = {name[len(prefix):]: tensor for name, tensor in state.items() if name.startswith(prefix)}
                target_manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                                   for name, tensor in target_state.items()]
                if not target_state or layer_manifest["target_tensors"] != target_manifest:
                    raise NativeCheckpointError("architecture_checkpoint_stale", f"layer {target} target contract differs")
                actual_contract = {name: (tuple(tensor.shape), str(tensor.dtype))
                                   for name, tensor in target_state.items()}
                if actual_contract != expected_contract:
                    raise NativeCheckpointError("architecture_checkpoint_incompatible",
                                                f"layer {target} tensors violate the derived target contract")
            aggregate_sha = hashlib.sha256(json.dumps(
                aggregate_rows, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            if aggregate_sha != checkpoint_identity.pre_replacement_checkpoint_sha256:
                raise NativeCheckpointError("pre_replacement_checkpoint_mismatch",
                                            "aggregate conversion source identity differs")
            architecture_checkpoint = candidate

    loader = base_model_loader or _default_base_model_loader
    manager_builder = manager_factory or _default_manager_factory
    if event: event("load_model")
    model = loader(model_path, **dict(spec.model_loader_kwargs))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("base model loader must return a torch.nn.Module")
    resolved_model_config = (
        getattr(model, "config", None) if model_config is None else model_config
    )
    if resolved_model_config is None:
        raise TypeError(
            "model_config must be supplied or exposed by the loaded model"
        )
    target_dtype = _intended_model_dtype(model, spec.model_loader_kwargs)
    manager = manager_builder(model, resolved_model_config)
    common = {
        "model": model,
        "manager": manager,
        "model_config": resolved_model_config,
        "cache_config": cache_config,
        "native_checkpoint": None if architecture_checkpoint is not None else checkpoint_path,
        "cache_resume": resume_path,
        "expected_job_id": spec.job_id,
        "target_dtype": target_dtype,
    }
    if spec.maximum_control_id == "stock-qwen":
        # Reliance evaluation of the untouched source model: no GDN3 upgrade,
        # no checkpoint overlay.  execute_job fails closed if any layer was
        # replaced, so an empty index tuple is the contract here.
        raw_indices: tuple[int, ...] = ()
    elif spec.arm == "native":
        installer = native_installer or _default_native_installer
        if spec.architecture_arm_id is None:
            raw_indices = installer(**common)
        else:
            if event: event("native_install_r1")
            raw_indices = installer(
                **common, native_r_out=1,
                strict_architecture_checkpoint=architecture_checkpoint is None,
            )
            if event: event("checkpoint_overlay_complete")
    else:
        installer = cache_installer
        if installer is None:
            raw_indices = _default_cache_installer(arm=spec.arm, **common)
        else:
            raw_indices = installer(**common)
    indices = tuple(raw_indices)
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("installer returned invalid upgraded indices")
    if len(set(indices)) != len(indices):
        raise ValueError("installer returned duplicate upgraded indices")
    if spec.architecture_arm_id is None:
        trainable_names = _configure_trainables(model, spec.trainable_names)
    else:
        from .architecture import architecture_record
        from .qwen_architecture import (
            KMD2ChannelwiseGDN2Attn,
            KMD2RotationControlAttn,
            KMD2SharedQueryWideningAttn,
            KMD2TrueMIMOAttn,
            QwenArchitectureConfig,
            install_qwen_architecture,
            _INCREMENTAL_TYPES,
        )
        assert spec.architecture_registry_sha256 is not None
        architecture_record_value = architecture_record(spec.architecture_arm_id)
        if spec.architecture_arm_id == "gdn2-mimo-r4-braid-shared-hola-w64":
            from .qwen_hybrid_shared import QwenSharedBraidHybrid
            expected_type = QwenSharedBraidHybrid
        elif spec.architecture_arm_id == "gdn2-mimo-r4-braid-four-state-hola-w64":
            from .qwen_hybrid_four_state import QwenFourStateHybrid
            expected_type = QwenFourStateHybrid
        else:
            expected_type = (_INCREMENTAL_TYPES[spec.architecture_arm_id]
                         if spec.architecture_arm_id in _INCREMENTAL_TYPES else
                         KMD2RotationControlAttn if spec.architecture_arm_id.startswith("rot-") else
                         KMD2SharedQueryWideningAttn if spec.architecture_arm_id == "rout-4" else
                         KMD2TrueMIMOAttn if architecture_record_value.mimo_rank > 1 else
                         KMD2ChannelwiseGDN2Attn)
        if architecture_factory is None:
            incremental_suffixes = {
                "trapezoid": ("rho_head", "rho_proj.weight"),
                "lookahead": ("lookahead_rho", "lookahead_projection.weight"),
                "qk-bc-additive": ("bc_q_amplitude", "bc_k_amplitude", "bc_q_bias", "bc_k_bias"),
                "qk-diagonal": ("bc_q_amplitude", "bc_k_amplitude", "bc_q_scale", "bc_k_scale"),
            }
            hybrid_ids = {"gdn2-mimo-r4-braid-shared-hola-w64",
                          "gdn2-mimo-r4-braid-four-state-hola-w64"}
            if spec.architecture_arm_id in hybrid_ids:
                if spec.maximum_control_id is not None:
                    from .qwen_architecture import build_maximum_control_architecture
                    prototype = build_maximum_control_architecture(
                        model.model.layers[indices[0]].linear_attn, spec.maximum_control_id
                    )
                    suffixes = tuple(name for name, parameter in prototype.named_parameters() if parameter.requires_grad)
                    architecture_factory = lambda native, _config: build_maximum_control_architecture(native, spec.maximum_control_id)
                else:
                    prototype = expected_type.from_native(model.model.layers[indices[0]].linear_attn)
                    suffixes = tuple(name for name, _ in prototype.named_parameters())
                del prototype
            else:
                suffixes = (incremental_suffixes[spec.architecture_arm_id]
                        if spec.architecture_arm_id in incremental_suffixes else
                        {"rot-constant": ("rotation_rate",),
                         "rot-noncumulative": ("rot_proj.weight", "rot_proj.bias"),
                         "rot-moving-frame-oracle": (("rot_proj.weight", "rot_proj.bias")
                                                     if spec.diagnostic_training else ())}.get(
                             spec.architecture_arm_id, ())
                        if spec.architecture_arm_id.startswith("rot-") else
                ("q_slot_scale", "out_mix") if spec.architecture_arm_id == "rout-4" else
                ("mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out")
                if architecture_record_value.mimo_rank > 1
                else ("erase_proj.weight", "write_proj.weight", "write_offset"))
            required_architecture_trainables = tuple(sorted(
                f"model.layers.{index}.linear_attn.{suffix}"
                for index in indices
                for suffix in suffixes
            ))
            if tuple(sorted(spec.trainable_names)) != required_architecture_trainables:
                raise ValueError("architecture_trainable_manifest_mismatch")
        architecture_config = QwenArchitectureConfig(
            spec.architecture_arm_id,
            spec.architecture_registry_sha256,
            architecture_record_value,
            diagnostic_training=spec.diagnostic_training,
        )
        def verify_architecture(model_value, indices_value):
            _verify_architecture_modules(
                model_value, indices_value, architecture_expected_type or expected_type,
                architecture_record_value.mimo_rank,
                expected_output_width=(4 if spec.architecture_arm_id == "rout-4" else None),
            )
            if architecture_verifier is not None:
                architecture_verifier(model_value, indices_value)
        install_qwen_architecture(
            model, indices, architecture_config, factory=architecture_factory,
            configure_trainables=_configure_trainables,
            declared_trainables=spec.trainable_names,
            verify_conversion=verify_architecture,
            event=event,
            expected_type=architecture_expected_type or expected_type,
            swap_verifier=(
                lambda _model, _index, module: _verify_architecture_module(
                    module, architecture_expected_type or expected_type,
                    architecture_record_value.mimo_rank, expected_output_width=4,
                )
                if spec.architecture_arm_id == "rout-4" else None
            ),
        )
        if architecture_checkpoint is not None:
            checkpoint_state = architecture_checkpoint["model_state"]
            assert isinstance(checkpoint_state, Mapping)
            for index in indices:
                module = model.model.layers[index].linear_attn
                target = f"model.layers.{index}.linear_attn."
                layer_state = {name[len(target):]: tensor for name, tensor in checkpoint_state.items()
                               if name.startswith(target)}
                try:
                    if not layer_state:
                        raise ValueError("target layer is absent")
                    module.load_state_dict(layer_state, strict=True)
                except (RuntimeError, TypeError, ValueError) as error:
                    raise NativeCheckpointError(
                        "architecture_checkpoint_incompatible",
                        f"hybrid conversion tensors do not match layer {index}",
                    ) from error
        trainable_names = tuple(sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ))
    architecture_manifest = None
    architecture_classification = None
    architecture_implementation = None
    if spec.architecture_arm_id is not None:
        from .architecture import architecture_record
        architecture_manifest = _aggregate_architecture_tensor_manifest(model, indices)
        architecture_classification = architecture_record(
            spec.architecture_arm_id
        ).classification
        if spec.architecture_arm_id in {
            "gdn2-mimo-r4-braid-shared-hola-w64",
            "gdn2-mimo-r4-braid-four-state-hola-w64",
        }:
            from .qwen_hybrid_math import REFERENCE_IMPLEMENTATION
        architecture_implementation = (
            REFERENCE_IMPLEMENTATION
            if spec.architecture_arm_id in {"gdn2-mimo-r4-braid-shared-hola-w64", "gdn2-mimo-r4-braid-four-state-hola-w64"} else
            _INCREMENTAL_TYPES[spec.architecture_arm_id].implementation_reference
            if spec.architecture_arm_id in _INCREMENTAL_TYPES else
            "qwen_architecture.KMD2RotationControlAttn.reference_fp32"
            if spec.architecture_arm_id.startswith("rot-") else
            "qwen_architecture.KMD2SharedQueryWideningAttn.reference_fp32"
            if spec.architecture_arm_id == "rout-4" else
            "qwen_architecture.KMD2TrueMIMOAttn.reference_fp32"
            if architecture_record(spec.architecture_arm_id).mimo_rank > 1
            else "qwen_architecture.KMD2ChannelwiseGDN2Attn.reference_fp32"
        )
    return LoadedQwenArm(
        model=model,
        arm=spec.arm,
        job_id=spec.job_id,
        upgraded_indices=indices,
        trainable_names=trainable_names,
        assets=validated,
        architecture_arm_id=spec.architecture_arm_id,
        architecture_registry_sha256=spec.architecture_registry_sha256,
        architecture_classification=architecture_classification,
        architecture_identity_passed=(
            spec.architecture_arm_id is not None
            and architecture_record(spec.architecture_arm_id).mimo_rank == 1
        ),
        architecture_implementation=architecture_implementation,
        architecture_tensor_manifest=architecture_manifest,
    )


__all__ = [
    "ArchitectureManifestError",
    "AssetIdentityError",
    "ExternalAssetIdentity",
    "LoadedQwenArm",
    "NativeCheckpointError",
    "PairedQwenHealContract",
    "PairingContractError",
    "QwenArmLoadSpec",
    "QwenHealArmContract",
    "ValidatedAssetIdentity",
    "load_qwen_arm",
    "validate_three_arm_pairing",
    "validate_external_assets",
]
