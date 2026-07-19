"""Canonical, atomic result storage for the portable KMD-2 suite.

This module intentionally depends only on the Python standard library.  It is
used by preflight, workers, resume, and the standalone bundle verifier, so all
identities and files are defined in terms of canonical JSON bytes rather than
Python object hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import SUITE_VERSION


RESULT_SCHEMA_VERSION = "1.0.0"
_MAX_JSON_DEPTH = 128
_LOCK_VERSION = 1
_LOCK_RELEASE_TIMEOUT = 1.0
_LOCK_RETRY_DELAY = 0.001
_MALFORMED_LOCK_STALE_SECONDS = 60.0
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SUMMARY_ONLY_FIELDS = frozenset(
    {"scientific_classification", "scientific_label"}
)


class RunRecordError(ValueError):
    """A malformed, stale, or conflicting authoritative run record."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _plain_json(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    active: set[int] | None = None,
) -> Any:
    """Return a detached JSON value while rejecting ambiguous inputs."""

    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{path} exceeds the maximum JSON depth")
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return 0.0 if value == 0.0 else value

    is_mapping = isinstance(value, Mapping)
    is_sequence = isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )
    if not is_mapping and not is_sequence:
        raise TypeError(
            f"{path} contains non-JSON value of type {value_type.__name__}"
        )

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{path} contains a JSON cycle")
    active.add(identity)
    try:
        if is_mapping:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} JSON object keys must be strings")
                result[key] = _plain_json(
                    item,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    active=active,
                )
            return result
        return [
            _plain_json(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                active=active,
            )
            for index, item in enumerate(value)
        ]
    finally:
        active.remove(identity)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value to its stable UTF-8 identity bytes."""

    plain = _plain_json(value)
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_string(name: str, value: Any) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty str")
    return value


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _semantic_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "semantic_dict") and callable(config.semantic_dict):
        config = config.semantic_dict()
    if not isinstance(config, Mapping):
        raise TypeError("canonical_config must be a mapping")
    return _plain_json(config, path="canonical_config")


def semantic_job_id(
    canonical_config: Any,
    *,
    backend: str,
    arm_id: str,
    seed: int,
    stage: str,
    pairing_id: str | None = None,
) -> str:
    """Hash every explicit semantic dimension of one execution job."""

    _require_string("backend", backend)
    _require_string("arm_id", arm_id)
    _require_int("seed", seed)
    _require_string("stage", stage)
    if pairing_id is not None:
        _require_string("pairing_id", pairing_id)
    identity = {
        "canonical_config": _semantic_config(canonical_config),
        "backend": backend,
        "arm_id": arm_id,
        "seed": seed,
        "stage": stage,
        "pairing_id": pairing_id,
    }
    return _sha256(identity)


def build_job(
    canonical_config: Any,
    *,
    seed: int,
    stage: str,
    backend: str,
    arm_id: str,
    experiment_id: str | None = None,
    pairing_id: str | None = None,
) -> dict[str, Any]:
    """Build one canonical immutable execution job."""

    config = _semantic_config(canonical_config)
    _require_int("seed", seed)
    _require_string("stage", stage)
    _require_string("backend", backend)
    _require_string("arm_id", arm_id)
    computed_experiment_id = _sha256(config)
    if experiment_id is not None:
        _require_string("experiment_id", experiment_id)
        if experiment_id != computed_experiment_id:
            raise ValueError("experiment_id does not match canonical_config")
    record: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "job_id": semantic_job_id(
            config,
            backend=backend,
            arm_id=arm_id,
            seed=seed,
            stage=stage,
            pairing_id=pairing_id,
        ),
        "experiment_id": computed_experiment_id,
        "seed": seed,
        "stage": stage,
        "backend": backend,
        "arm_id": arm_id,
        "canonical_config": config,
    }
    if pairing_id is not None:
        record["pairing_id"] = _require_string("pairing_id", pairing_id)
    architecture = config.get("architecture")
    if (
        isinstance(architecture, Mapping)
        and architecture.get("arm_id") == arm_id
    ):
        record["architecture_registry_sha256"] = _require_string(
            "architecture_registry_sha256", architecture.get("registry_sha256")
        )
        if arm_id.startswith("rot-"):
            diagnostic = architecture.get("diagnostic_training", False)
            if type(diagnostic) is not bool:
                raise TypeError("architecture.diagnostic_training must be a bool")
            record["architecture_diagnostic_training"] = diagnostic
    return record


def _validated_jobs(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(jobs, (str, bytes, bytearray)) or not isinstance(jobs, Sequence):
        raise TypeError("jobs must be a sequence of mappings")
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            raise TypeError(f"jobs[{index}] must be a mapping")
        item = _plain_json(job, path=f"jobs[{index}]")
        required = {
            "schema_version",
            "suite_version",
            "job_id",
            "experiment_id",
            "seed",
            "stage",
            "backend",
            "arm_id",
            "canonical_config",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"jobs[{index}] is missing: " + ", ".join(sorted(missing))
            )
        if item["schema_version"] != RESULT_SCHEMA_VERSION:
            raise ValueError(f"jobs[{index}].schema_version does not match")
        if item["suite_version"] != SUITE_VERSION:
            raise ValueError(f"jobs[{index}].suite_version does not match")
        config = item["canonical_config"]
        if not isinstance(config, Mapping):
            raise TypeError(f"jobs[{index}].canonical_config must be a mapping")
        _require_int(f"jobs[{index}].seed", item["seed"])
        _require_string(f"jobs[{index}].stage", item["stage"])
        _require_string(f"jobs[{index}].backend", item["backend"])
        _require_string(f"jobs[{index}].arm_id", item["arm_id"])
        expected_experiment = _sha256(config)
        if item["experiment_id"] != expected_experiment:
            raise ValueError(f"jobs[{index}].experiment_id does not match")
        expected_job = semantic_job_id(
            config,
            backend=item["backend"],
            arm_id=item["arm_id"],
            seed=item["seed"],
            stage=item["stage"],
            pairing_id=item.get("pairing_id"),
        )
        if item["job_id"] != expected_job:
            raise ValueError(f"jobs[{index}].job_id does not match")
        job_id = _require_string(f"jobs[{index}].job_id", item.get("job_id"))
        if job_id in identifiers:
            raise ValueError(f"duplicate job_id: {job_id}")
        identifiers.add(job_id)
        normalized.append(item)
    return sorted(normalized, key=lambda item: item["job_id"])


def build_jobs_document(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the canonical job-table document in stable job-id order."""

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "jobs": _validated_jobs(jobs),
    }


def expanded_jobs_digest(jobs: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical immutable job table."""

    return _sha256(build_jobs_document(jobs))


def _validate_digest_mapping(name: str, value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        _require_string(f"{name} key", key)
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name}[{key!r}] must be a lowercase SHA-256 digest")
        result[key] = digest
    return dict(sorted(result.items()))


def _validate_provenance_structure(provenance: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "suite_version",
        "source_hashes",
        "config_hash",
        "asset_hashes",
        "git",
        "environment",
    }
    missing = required - set(provenance)
    if missing:
        raise ValueError(
            "provenance is missing required fields: " + ", ".join(sorted(missing))
        )
    if provenance["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("provenance schema_version does not match")
    if provenance["suite_version"] != SUITE_VERSION:
        raise ValueError("provenance suite_version does not match")
    source_hashes = _validate_digest_mapping(
        "provenance.source_hashes", provenance["source_hashes"]
    )
    asset_hashes = _validate_digest_mapping(
        "provenance.asset_hashes", provenance["asset_hashes"]
    )
    config_hash = provenance["config_hash"]
    _validate_digest_mapping("provenance", {"config_hash": config_hash})

    git = _plain_json(provenance["git"], path="provenance.git")
    if not isinstance(git, dict):
        raise TypeError("provenance.git must be a mapping")
    git_required = {"revision", "diff_hash", "dirty"}
    git_missing = git_required - set(git)
    if git_missing:
        raise ValueError(
            "provenance.git is missing: " + ", ".join(sorted(git_missing))
        )
    _require_string("provenance.git.revision", git["revision"])
    _validate_digest_mapping(
        "provenance.git", {"diff_hash": git["diff_hash"]}
    )
    if type(git["dirty"]) is not bool:
        raise TypeError("provenance.git.dirty must be a bool")

    environment = _plain_json(
        provenance["environment"], path="provenance.environment"
    )
    if not isinstance(environment, dict):
        raise TypeError("provenance.environment must be a mapping")
    environment_required = {
        "python",
        "pytorch",
        "cuda",
        "gpu",
        "dependencies",
    }
    environment_missing = environment_required - set(environment)
    if environment_missing:
        raise ValueError(
            "provenance.environment is missing: "
            + ", ".join(sorted(environment_missing))
        )
    _require_string("provenance.environment.python", environment["python"])
    _require_string("provenance.environment.pytorch", environment["pytorch"])
    if environment["cuda"] is not None:
        _require_string("provenance.environment.cuda", environment["cuda"])
    if not isinstance(environment["dependencies"], Mapping):
        raise TypeError("provenance.environment.dependencies must be a mapping")
    for dependency, version in environment["dependencies"].items():
        _require_string("dependency name", dependency)
        if version is not None:
            _require_string(f"dependency {dependency}", version)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "source_hashes": source_hashes,
        "config_hash": config_hash,
        "asset_hashes": asset_hashes,
        "git": git,
        "environment": environment,
    }


def build_manifest(
    *,
    canonical_config: Any,
    jobs: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    """Build the complete canonical suite manifest."""

    config = _semantic_config(canonical_config)
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    normalized_provenance = _validate_provenance_structure(provenance)
    config_hash = _sha256(config)
    if normalized_provenance["config_hash"] != config_hash:
        raise ValueError("provenance config_hash does not match canonical_config")
    if isinstance(command, (str, bytes, bytearray)) or not isinstance(
        command, Sequence
    ):
        raise TypeError("command must be a sequence of strings")
    command_list = [_require_string("command entry", item) for item in command]
    if not command_list:
        raise ValueError("command must not be empty")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "canonical_config": config,
        "source_hashes": normalized_provenance["source_hashes"],
        "config_hash": config_hash,
        "asset_hashes": normalized_provenance["asset_hashes"],
        "git": normalized_provenance["git"],
        "environment": normalized_provenance["environment"],
        "command": command_list,
        "expanded_jobs_digest": expanded_jobs_digest(jobs),
    }


def assign_shard(job_id: str, num_jobs: int) -> int:
    """Return uint64_be(sha256(job_id)[:8]) modulo ``num_jobs``."""

    _require_string("job_id", job_id)
    _require_int("num_jobs", num_jobs, minimum=1)
    prefix = hashlib.sha256(job_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(prefix, "big", signed=False) % num_jobs


def select_shard(
    jobs: Sequence[Mapping[str, Any]], job_index: int, num_jobs: int
) -> list[dict[str, Any]]:
    """Select one deterministic, stable-order shard from a job table."""

    _require_int("num_jobs", num_jobs, minimum=1)
    _require_int("job_index", job_index, minimum=0)
    if job_index >= num_jobs:
        raise ValueError("job_index must be less than num_jobs")
    return [
        job
        for job in _validated_jobs(jobs)
        if assign_shard(job["job_id"], num_jobs) == job_index
    ]


@contextmanager
def _creation_lock(path: Path, *, timeout: float = 10.0) -> Iterator[None]:
    """Portable token-owned lock whose published owner is always complete."""

    if type(timeout) not in (int, float) or not math.isfinite(float(timeout)):
        raise TypeError("lock timeout must be finite")
    if timeout <= 0:
        raise ValueError("lock timeout must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    owner = {
        "version": _LOCK_VERSION,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "token": uuid.uuid4().hex,
    }
    payload = canonical_json_bytes(owner) + b"\n"
    deadline = time.monotonic() + float(timeout)
    candidate = _prepare_lock_payload(lock_path, payload, kind="owner")
    acquired = False
    candidate_removed = False
    try:
        while True:
            try:
                os.link(candidate, lock_path)
                _fsync_directory(lock_path.parent)
                acquired = True
                break
            except (FileExistsError, PermissionError) as error:
                if isinstance(error, PermissionError) and not lock_path.exists():
                    # Windows can report a sharing violation while the prior
                    # lock is delete-pending, then make the path disappear.
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(_LOCK_RETRY_DELAY)
                    continue
                raw = _read_path_bytes(lock_path)
                observed = _parse_lock_owner(raw)
                if observed is not None and _lock_owner_confirmed_dead(observed):
                    _claim_and_remove_dead_lock(
                        lock_path, observed, owner, raw=raw
                    )
                elif (
                    raw is not None
                    and observed is None
                    and _path_is_stale(lock_path)
                ):
                    _claim_and_remove_malformed_lock(
                        lock_path, raw, owner
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring result lock: {lock_path}"
                    )
                time.sleep(0.01)
        candidate_removed = _remove_unique_artifact(candidate)
    finally:
        if not acquired:
            _remove_unique_artifact(candidate)
    try:
        yield
    finally:
        _release_owned_lock(lock_path, owner)
        if not candidate_removed:
            _remove_unique_artifact(candidate)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if type(written) is not int or written <= 0:
            raise OSError("lock payload write made no progress")
        offset += written


def _prepare_lock_payload(path: Path, payload: bytes, *, kind: str) -> Path:
    candidate = path.with_name(
        f"{path.name}.{kind}-{os.getpid()}-{threading.get_ident()}-"
        f"{uuid.uuid4().hex}.tmp"
    )
    try:
        descriptor = os.open(
            candidate,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return candidate
    except BaseException:
        _remove_unique_artifact(candidate)
        raise


def _remove_unique_artifact(path: Path) -> bool:
    deadline = time.monotonic() + _LOCK_RELEASE_TIMEOUT
    while True:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(_LOCK_RETRY_DELAY)


def _read_path_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (FileNotFoundError, PermissionError):
        return None


def _parse_lock_owner(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {
        "version",
        "host",
        "pid",
        "thread_id",
        "token",
    }:
        return None
    if value["version"] != _LOCK_VERSION:
        return None
    if type(value["host"]) is not str or not value["host"]:
        return None
    if type(value["pid"]) is not int or value["pid"] < 1:
        return None
    if type(value["thread_id"]) is not int or value["thread_id"] < 1:
        return None
    if type(value["token"]) is not str or not value["token"]:
        return None
    return value


def _read_lock_owner(lock_path: Path) -> dict[str, Any] | None:
    return _parse_lock_owner(_read_path_bytes(lock_path))


def _parse_claim_owner(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {
        "version",
        "host",
        "pid",
        "thread_id",
        "token",
        "target_digest",
    }:
        return None
    owner = {key: value[key] for key in (
        "version",
        "host",
        "pid",
        "thread_id",
        "token",
    )}
    if _parse_lock_owner(canonical_json_bytes(owner)) is None:
        return None
    digest = value["target_digest"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    return value


def _path_is_stale(path: Path) -> bool:
    try:
        modified = path.stat().st_mtime
    except (FileNotFoundError, PermissionError):
        return False
    return time.time() - modified >= _MALFORMED_LOCK_STALE_SECONDS


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if process:
                ctypes.windll.kernel32.CloseHandle(process)
                return True
            return ctypes.windll.kernel32.GetLastError() == 5
        except (AttributeError, OSError):
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _lock_owner_confirmed_dead(owner: Mapping[str, Any]) -> bool:
    if owner["host"] != socket.gethostname():
        return False
    pid = owner["pid"]
    if pid != os.getpid():
        return not _pid_alive(pid)
    thread_id = owner["thread_id"]
    return not any(
        thread.ident == thread_id and thread.is_alive()
        for thread in threading.enumerate()
    )


def _same_lock_owner(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in (
        "version",
        "host",
        "pid",
        "thread_id",
        "token",
    ))


def _claim_and_remove_dead_lock(
    lock_path: Path,
    observed: Mapping[str, Any],
    claimant: Mapping[str, Any],
    *,
    raw: bytes | None = None,
) -> bool:
    target_raw = _read_path_bytes(lock_path) if raw is None else raw
    if target_raw is None:
        return False
    claim_digest = hashlib.sha256(observed["token"].encode("utf-8")).hexdigest()
    claim = lock_path.with_name(f"{lock_path.name}.claim-{claim_digest}")
    return _claim_and_remove_lock(
        lock_path,
        target_raw,
        claim,
        claimant,
        require_dead_owner=observed,
        require_stale=False,
    )


def _claim_and_remove_malformed_lock(
    lock_path: Path,
    raw: bytes,
    claimant: Mapping[str, Any],
) -> bool:
    digest = hashlib.sha256(raw).hexdigest()
    claim = lock_path.with_name(
        f"{lock_path.name}.claim-malformed-{digest}"
    )
    return _claim_and_remove_lock(
        lock_path,
        raw,
        claim,
        claimant,
        require_dead_owner=None,
        require_stale=True,
    )


def _publish_exclusive_payload(path: Path, payload: bytes) -> bool:
    candidate = _prepare_lock_payload(path, payload, kind="claim")
    linked = False
    try:
        try:
            os.link(candidate, path)
            linked = True
            _fsync_directory(path.parent)
            return True
        except (FileExistsError, PermissionError):
            return False
    finally:
        removed = _remove_unique_artifact(candidate)
        if linked and not removed:
            # The alias cannot wedge acquisition; retry after its published
            # claim has been consumed by the caller.
            _remove_unique_artifact(candidate)


def _remove_path_if_unchanged(path: Path, expected: bytes) -> bool:
    deadline = time.monotonic() + _LOCK_RELEASE_TIMEOUT
    while True:
        current = _read_path_bytes(path)
        if current is None and path.exists():
            if time.monotonic() >= deadline:
                return False
            time.sleep(_LOCK_RETRY_DELAY)
            continue
        if current != expected:
            return False
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except PermissionError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(_LOCK_RETRY_DELAY)


def _recover_abandoned_claim(claim: Path, target_digest: str) -> None:
    raw = _read_path_bytes(claim)
    if raw is None:
        return
    owner = _parse_claim_owner(raw)
    if owner is not None:
        if owner["target_digest"] != target_digest:
            return
        if not _lock_owner_confirmed_dead(owner):
            return
    elif not _path_is_stale(claim):
        return
    _remove_path_if_unchanged(claim, raw)


def _claim_and_remove_lock(
    lock_path: Path,
    target_raw: bytes,
    claim: Path,
    claimant: Mapping[str, Any],
    *,
    require_dead_owner: Mapping[str, Any] | None,
    require_stale: bool,
) -> bool:
    target_digest = hashlib.sha256(target_raw).hexdigest()
    claim_owner = dict(claimant)
    claim_owner["target_digest"] = target_digest
    claim_payload = canonical_json_bytes(claim_owner) + b"\n"
    if not _publish_exclusive_payload(claim, claim_payload):
        _recover_abandoned_claim(claim, target_digest)
        return False
    try:
        if _read_path_bytes(lock_path) != target_raw:
            return False
        if require_stale and not _path_is_stale(lock_path):
            return False
        if require_dead_owner is not None:
            current = _read_lock_owner(lock_path)
            if (
                current is None
                or not _same_lock_owner(current, require_dead_owner)
                or not _lock_owner_confirmed_dead(current)
            ):
                return False
        return _remove_path_if_unchanged(lock_path, target_raw)
    finally:
        _remove_path_if_unchanged(claim, claim_payload)


def _release_owned_lock(lock_path: Path, owner: Mapping[str, Any]) -> None:
    deadline = time.monotonic() + _LOCK_RELEASE_TIMEOUT
    while True:
        current = _read_lock_owner(lock_path)
        if current is None or not _same_lock_owner(current, owner):
            return
        try:
            lock_path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            # Windows denies deletion while a contender has the metadata file
            # open for reading. Recheck the owner token before every retry so a
            # prior owner can never unlink a successor's lock.
            if time.monotonic() >= deadline:
                raise
            time.sleep(_LOCK_RETRY_DELAY)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], record: Any) -> str:
    """Fsync canonical JSON to a unique sibling and atomically replace ``path``."""

    destination = Path(path)
    canonical = canonical_json_bytes(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _creation_lock(destination):
        _atomic_write_bytes(destination, canonical + b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _safe_component(name: str, value: Any) -> str:
    text = _require_string(name, value)
    if not _SAFE_COMPONENT.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"{name} is not a safe path component")
    return text


def quarantine_file(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
    job_id: str,
    reason: str,
) -> Path:
    """Move invalid bytes to their deterministic content-addressed quarantine."""

    source = Path(path)
    root_path = Path(root)
    safe_job_id = _safe_component("job_id", job_id)
    safe_reason = _safe_component("reason", reason)
    try:
        source.resolve(strict=True).relative_to(root_path.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("quarantine source must exist beneath the result root") from error
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    target = (
        root_path
        / "quarantine"
        / safe_job_id
        / f"{safe_reason}-{digest}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with _creation_lock(target):
        if target.exists():
            if target.read_bytes() != raw:
                raise RunRecordError(
                    "quarantine_conflict",
                    f"quarantine digest collision at {target}",
                )
        else:
            _atomic_write_bytes(target, raw)
        try:
            source.unlink()
        except FileNotFoundError:
            pass
    return target


def _validate_common_run(
    record: Mapping[str, Any],
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    status: str,
) -> None:
    if not isinstance(record, Mapping):
        raise RunRecordError("malformed_record", "execution record must be a mapping")
    if not isinstance(job, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError("job and provenance must be mappings")
    summary_only = _SUMMARY_ONLY_FIELDS & set(record)
    if summary_only:
        raise RunRecordError(
            "summary_only_field",
            "scientific classifications belong only in summaries: "
            + ", ".join(sorted(summary_only)),
        )
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise RunRecordError("schema_mismatch", "record schema_version does not match")
    if record.get("suite_version") != SUITE_VERSION:
        raise RunRecordError("suite_mismatch", "record suite_version does not match")
    if record.get("status") != status:
        code = "not_completed" if status == "completed" else "not_failed"
        raise RunRecordError(code, f"record status must be {status}")
    identity_fields = (
        "job_id",
        "experiment_id",
        "seed",
        "stage",
        "backend",
        "arm_id",
    )
    for field in identity_fields:
        if record.get(field) != job.get(field):
            raise RunRecordError(
                "job_identity", f"record {field} does not match its immutable job"
            )
    if "pairing_id" in job and record.get("pairing_id") != job.get("pairing_id"):
        raise RunRecordError(
            "job_identity", "record pairing_id does not match its immutable job"
        )
    try:
        same_config = canonical_json_bytes(record.get("canonical_config")) == canonical_json_bytes(
            job.get("canonical_config")
        )
    except (TypeError, ValueError) as error:
        raise RunRecordError("config_mismatch", "record config is malformed") from error
    if not same_config:
        raise RunRecordError(
            "config_mismatch", "record canonical_config does not match its job"
        )
    try:
        same_provenance = canonical_json_bytes(record.get("provenance")) == canonical_json_bytes(
            provenance
        )
    except (TypeError, ValueError) as error:
        raise RunRecordError(
            "provenance_mismatch", "record provenance is malformed"
        ) from error
    if not same_provenance:
        raise RunRecordError(
            "provenance_mismatch", "record provenance does not match this run"
        )
    try:
        _validate_provenance_structure(provenance)
    except (TypeError, ValueError) as error:
        raise RunRecordError(
            "provenance_mismatch", f"run provenance is incomplete: {error}"
        ) from error
    shard = record.get("shard")
    if not isinstance(shard, Mapping):
        raise RunRecordError("shard_mismatch", "record shard must be a mapping")
    try:
        count = _require_int("record.shard.count", shard.get("count"), minimum=1)
        index = _require_int("record.shard.index", shard.get("index"), minimum=0)
    except (TypeError, ValueError) as error:
        raise RunRecordError("shard_mismatch", str(error)) from error
    job_id = job.get("job_id")
    if type(job_id) is not str or index >= count or assign_shard(job_id, count) != index:
        raise RunRecordError(
            "shard_mismatch", "record shard assignment does not match its job_id"
        )


def _finite_real(name: str, value: Any, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RunRecordError("invalid_diagnostics", f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise RunRecordError(
            "invalid_diagnostics", f"{name} must be at least {minimum}"
        )
    return result


def _nonnegative_int(name: str, value: Any, *, code: str = "invalid_diagnostics") -> int:
    if type(value) is not int or value < 0:
        raise RunRecordError(code, f"{name} must be a nonnegative int")
    return value


def _required_mapping(
    record: Mapping[str, Any],
    name: str,
    required: set[str],
    *,
    missing_code: str = "missing_diagnostics",
) -> Mapping[str, Any]:
    value = record.get(name)
    if not isinstance(value, Mapping):
        raise RunRecordError(missing_code, f"record requires {name}")
    missing = required - set(value)
    if missing:
        raise RunRecordError(
            missing_code,
            f"{name} is missing: {', '.join(sorted(missing))}",
        )
    return value


def _validate_command(record: Mapping[str, Any]) -> None:
    command = record.get("command")
    if (
        isinstance(command, (str, bytes, bytearray))
        or not isinstance(command, Sequence)
        or not command
        or any(type(item) is not str or not item for item in command)
    ):
        raise RunRecordError(
            "missing_diagnostics", "record command must be a nonempty string sequence"
        )


def _requires_exact_cache(job: Mapping[str, Any]) -> bool:
    config = job.get("canonical_config")
    mechanism = config.get("mechanism") if isinstance(config, Mapping) else None
    arm_id = job.get("arm_id")
    if arm_id == "native":
        return False
    if (
        job.get("backend") == "qwen"
        and job.get("stage") == "qwen_heal"
        and arm_id in {"native", "recency", "surprise"}
    ):
        return arm_id in {"recency", "surprise"}
    return mechanism in {"exact_cache", "current_block_only"} or (
        type(arm_id) is str and arm_id.startswith("exact_cache.")
    )


_EXACT_CACHE_FIELDS = {
    "width",
    "block_size",
    "score_definition",
    "compute_dtype",
    "storage_dtype",
    "coordinate_frame",
    "inclusive_causality",
    "tie_policy",
    "amplitude_initial",
    "amplitude_final",
    "selected_index_digest",
    "selected_index_sample",
    "score_digest",
    "score_statistics",
    "retention_count",
    "eviction_count",
    "persistent_hit_rate",
    "conditional_read_accuracy",
    "sink_mass",
    "attention_entropy",
    "top1_mass",
    "stale_occupancy",
    "stale_error",
    "cache_output_norm",
    "state_output_norm",
    "persistent_bytes",
    "block_bytes",
    "implementation_paths",
}


def _validate_sha256(name: str, value: Any, *, code: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunRecordError(code, f"{name} must be a lowercase SHA-256 digest")


def _validate_exact_cache(record: Mapping[str, Any]) -> None:
    diagnostics = _required_mapping(
        record,
        "exact_cache",
        _EXACT_CACHE_FIELDS,
        missing_code="missing_exact_cache_diagnostics",
    )
    code = "invalid_exact_cache_diagnostics"
    try:
        _nonnegative_int("exact_cache.width", diagnostics["width"], code=code)
        _nonnegative_int("exact_cache.block_size", diagnostics["block_size"], code=code)
        if diagnostics["block_size"] < 1:
            raise RunRecordError(code, "exact_cache.block_size must be positive")
        if diagnostics["compute_dtype"] != "fp32":
            raise RunRecordError(code, "exact-cache compute_dtype must be fp32")
        if diagnostics["storage_dtype"] not in {"fp32", "bf16"}:
            raise RunRecordError(code, "invalid exact-cache storage_dtype")
        if diagnostics["coordinate_frame"] not in {
            "rotated_recurrence",
            "pre_rotation",
        }:
            raise RunRecordError(code, "invalid exact-cache coordinate_frame")
        if diagnostics["inclusive_causality"] is not True:
            raise RunRecordError(code, "exact-cache causality must be inclusive")
        if diagnostics["tie_policy"] != "score_desc_position_desc":
            raise RunRecordError(code, "invalid exact-cache tie_policy")
        if type(diagnostics["score_definition"]) is not str or not diagnostics[
            "score_definition"
        ]:
            raise RunRecordError(code, "score_definition must be nonempty")
        for field in ("amplitude_initial", "amplitude_final"):
            values = diagnostics[field]
            if (
                isinstance(values, (str, bytes, bytearray))
                or not isinstance(values, Sequence)
                or not values
            ):
                raise RunRecordError(code, f"exact_cache.{field} must be nonempty")
            for value in values:
                amplitude = _finite_real(f"exact_cache.{field}", value)
                if not 0.0 <= amplitude <= 1.0:
                    raise RunRecordError(code, f"exact_cache.{field} is out of range")
        _validate_sha256(
            "selected_index_digest", diagnostics["selected_index_digest"], code=code
        )
        _validate_sha256("score_digest", diagnostics["score_digest"], code=code)
        sample = diagnostics["selected_index_sample"]
        if isinstance(sample, (str, bytes, bytearray)) or not isinstance(sample, Sequence):
            raise RunRecordError(code, "selected_index_sample must be a sequence")
        if any(type(index) is not int for index in sample):
            raise RunRecordError(code, "selected_index_sample must contain ints")
        statistics = diagnostics["score_statistics"]
        if not isinstance(statistics, Mapping) or not {
            "count",
            "min",
            "max",
            "mean",
        } <= set(statistics):
            raise RunRecordError(code, "score_statistics is incomplete")
        _nonnegative_int("score_statistics.count", statistics["count"], code=code)
        score_min = _finite_real("score_statistics.min", statistics["min"], minimum=0.0)
        score_max = _finite_real("score_statistics.max", statistics["max"], minimum=0.0)
        score_mean = _finite_real("score_statistics.mean", statistics["mean"], minimum=0.0)
        if not score_min <= score_mean <= score_max:
            raise RunRecordError(code, "score statistics are not ordered")
        for field in (
            "retention_count",
            "eviction_count",
            "persistent_bytes",
            "block_bytes",
        ):
            _nonnegative_int(f"exact_cache.{field}", diagnostics[field], code=code)
        for field in (
            "persistent_hit_rate",
            "conditional_read_accuracy",
            "sink_mass",
            "top1_mass",
            "stale_occupancy",
            "stale_error",
        ):
            value = _finite_real(f"exact_cache.{field}", diagnostics[field])
            if not 0.0 <= value <= 1.0:
                raise RunRecordError(code, f"exact_cache.{field} must lie in [0,1]")
        for field in ("attention_entropy", "cache_output_norm", "state_output_norm"):
            _finite_real(f"exact_cache.{field}", diagnostics[field], minimum=0.0)
        paths = diagnostics["implementation_paths"]
        if not isinstance(paths, Mapping) or not {
            "scan",
            "score",
            "selection",
            "read",
        } <= set(paths):
            raise RunRecordError(code, "implementation_paths is incomplete")
        if any(type(value) is not str or not value for value in paths.values()):
            raise RunRecordError(code, "implementation paths must be nonempty strings")
    except RunRecordError as error:
        if error.code == "invalid_diagnostics":
            raise RunRecordError(code, str(error)) from error
        raise


def _validate_completed_diagnostics(
    record: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    required_groups = {
        "metrics",
        "loss_curves",
        "counts",
        "parameters",
        "recurrent_state",
        "performance",
        "identities",
    }
    missing = required_groups - set(record)
    if missing:
        raise RunRecordError(
            "missing_diagnostics",
            "completed record is missing: " + ", ".join(sorted(missing)),
        )
    metrics = record["metrics"]
    if not isinstance(metrics, Mapping) or not metrics:
        raise RunRecordError("invalid_diagnostics", "metrics must be nonempty")
    try:
        canonical_json_bytes(metrics)
    except (TypeError, ValueError) as error:
        raise RunRecordError("invalid_diagnostics", "metrics are not finite JSON") from error
    curves = record["loss_curves"]
    if not isinstance(curves, Mapping) or not curves:
        raise RunRecordError("invalid_diagnostics", "loss_curves must be nonempty")
    for name, values in curves.items():
        if type(name) is not str or isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise RunRecordError("invalid_diagnostics", "loss curves must be named sequences")
        for value in values:
            _finite_real(f"loss_curves.{name}", value)
    counts = _required_mapping(
        record,
        "counts",
        {"nonfinite_loss", "nonfinite_gradient", "skipped_steps"},
    )
    for field in ("nonfinite_loss", "nonfinite_gradient", "skipped_steps"):
        _nonnegative_int(f"counts.{field}", counts[field])
    parameters = _required_mapping(record, "parameters", {"trainable", "total"})
    trainable = _nonnegative_int("parameters.trainable", parameters["trainable"])
    total = _nonnegative_int("parameters.total", parameters["total"])
    if trainable > total:
        raise RunRecordError("invalid_diagnostics", "trainable parameters exceed total")
    state = _required_mapping(record, "recurrent_state", {"elements", "bytes"})
    _nonnegative_int("recurrent_state.elements", state["elements"])
    _nonnegative_int("recurrent_state.bytes", state["bytes"])
    performance = _required_mapping(
        record,
        "performance",
        {
            "wall_time_seconds",
            "examples_per_second",
            "tokens_per_second",
            "peak_vram_bytes",
        },
    )
    for field in ("wall_time_seconds", "examples_per_second", "tokens_per_second"):
        _finite_real(f"performance.{field}", performance[field], minimum=0.0)
    _nonnegative_int("performance.peak_vram_bytes", performance["peak_vram_bytes"])
    identities = _required_mapping(record, "identities", {"checkpoint", "data"})
    if identities["checkpoint"] is None or identities["data"] is None:
        raise RunRecordError("invalid_diagnostics", "identities must be explicit")
    try:
        canonical_json_bytes(identities)
    except (TypeError, ValueError) as error:
        raise RunRecordError("invalid_diagnostics", "identities are invalid JSON") from error
    _validate_command(record)
    if _requires_exact_cache(job):
        _validate_exact_cache(record)
    if str(job.get("arm_id", "")).startswith("gdn2-mimo-r4-braid-"):
        diagnostics = record.get("hybrid_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise RunRecordError(
                "missing_hybrid_diagnostics", "completed hybrid run requires measured hybrid_diagnostics"
            )
        validate_hybrid_diagnostics(diagnostics)


def validate_completed_run(
    record: Mapping[str, Any],
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Validate an authoritative completed execution record in full."""

    _validate_common_run(record, job, provenance, status="completed")
    _validate_completed_diagnostics(record, job)


_HYBRID_DIAGNOSTIC_FIELDS = {
    "rank_update_norms", "rank_similarity", "effective_rank", "phase_magnitude",
    "phase_frequency", "effective_decay_horizons", "decay_horizon_ratios",
    "all_lanes_update_each_token", "state_router_active", "cache_admission_rate",
    "cache_occupancy", "cache_mean_age", "cache_read_entropy", "cache_gate_mean", "state_norm",
    "gate_mean", "nonfinite_count", "flops_per_token", "tokens_per_second",
    "peak_memory_bytes", "capacity_confounded",
    "cache_admissions", "cache_opportunities", "cache_schema", "layer_count",
}


def validate_hybrid_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the additive Package-A/B diagnostic payload without changing v1 records."""
    if not isinstance(record, Mapping) or set(record) != _HYBRID_DIAGNOSTIC_FIELDS:
        raise RunRecordError("invalid_hybrid_diagnostics", "hybrid diagnostic fields must be exact")
    result = dict(record)
    ranks = result["rank_update_norms"]
    if (isinstance(ranks, (str, bytes, bytearray)) or not isinstance(ranks, Sequence)
            or len(ranks) != 4):
        raise RunRecordError("invalid_hybrid_diagnostics", "rank_update_norms must contain four ranks")
    for index, value in enumerate(ranks):
        _finite_real(f"rank_update_norms[{index}]", value, minimum=0.0)
    for field in (
        "rank_similarity", "effective_rank", "phase_magnitude", "phase_frequency",
        "cache_mean_age", "cache_read_entropy", "state_norm",
        "flops_per_token", "tokens_per_second",
    ):
        _finite_real(field, result[field], minimum=0.0)
    for field in ("cache_admission_rate", "cache_occupancy", "cache_gate_mean", "gate_mean"):
        value = _finite_real(field, result[field])
        if not 0.0 <= value <= 1.0:
            raise RunRecordError("invalid_hybrid_diagnostics", f"{field} must lie in [0,1]")
    for field in ("effective_decay_horizons", "decay_horizon_ratios"):
        values = result[field]
        if (isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence)
                or len(values) != 4):
            raise RunRecordError("invalid_hybrid_diagnostics", f"{field} must contain four lanes")
        [_finite_real(f"{field}[{index}]", value, minimum=0.0)
         for index, value in enumerate(values)]
    expected_ratios = (1.0, 16.0, 64.0, 256.0)
    if any(not math.isclose(float(value), expected, rel_tol=1e-4, abs_tol=1e-4)
           for value, expected in zip(result["decay_horizon_ratios"], expected_ratios)):
        raise RunRecordError("invalid_hybrid_diagnostics", "decay horizon ratios are not configured timescales")
    if result["all_lanes_update_each_token"] is not False or result["state_router_active"] is not False:
        raise RunRecordError("invalid_hybrid_diagnostics", "time braid must use clocked lane updates and remain router-free")
    for field in ("cache_admissions", "cache_opportunities", "layer_count"):
        if _nonnegative_int(field, result[field]) < 1:
            raise RunRecordError("invalid_hybrid_diagnostics", f"{field} must be positive")
    if result["cache_admissions"] > result["cache_opportunities"]:
        raise RunRecordError("invalid_hybrid_diagnostics", "cache admissions exceed opportunities")
    if result["cache_schema"] not in {"state", "states"}:
        raise RunRecordError("invalid_hybrid_diagnostics", "cache_schema must be state or states")
    _nonnegative_int("nonfinite_count", result["nonfinite_count"])
    _nonnegative_int("peak_memory_bytes", result["peak_memory_bytes"])
    if type(result["capacity_confounded"]) is not bool:
        raise RunRecordError("invalid_hybrid_diagnostics", "capacity_confounded must be bool")
    canonical_json_bytes(result)
    return result


_OOM_CONTEXT_FIELDS = {
    "batch_size",
    "sequence_length",
    "num_heads",
    "state_key_dim",
    "state_value_dim",
    "cache_width",
    "block_size",
    "dtype",
    "device",
    "estimated_bytes",
    "peak_vram_bytes",
}


def validate_failed_run(
    record: Mapping[str, Any],
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Validate a typed atomic failed execution record."""

    _validate_common_run(record, job, provenance, status="failed")
    _validate_command(record)
    error = _required_mapping(
        record,
        "error",
        {"code", "message", "phase", "context", "traceback"},
        missing_code="invalid_failure",
    )
    allowed = {
        "oom",
        "nonfinite_loss",
        "nonfinite_gradient",
        "malformed_input",
        "execution_error",
        "backend_unavailable",
    }
    if error["code"] not in allowed:
        raise RunRecordError("invalid_failure", "unknown failure code")
    for field in ("message", "phase"):
        if type(error[field]) is not str or not error[field]:
            raise RunRecordError("invalid_failure", f"error.{field} must be nonempty")
    if not isinstance(error["context"], Mapping):
        raise RunRecordError("invalid_failure", "error.context must be a mapping")
    traceback_text = error["traceback"]
    if type(traceback_text) is not str or len(traceback_text) > 8192:
        raise RunRecordError("invalid_failure", "error.traceback is not bounded")
    try:
        canonical_json_bytes(error["context"])
    except (TypeError, ValueError) as caught:
        raise RunRecordError("invalid_failure", "error.context is invalid JSON") from caught
    if error["code"] == "oom":
        context = error["context"]
        missing = _OOM_CONTEXT_FIELDS - set(context)
        if missing:
            raise RunRecordError(
                "invalid_oom_context",
                "OOM context is missing: " + ", ".join(sorted(missing)),
            )
        for field in (
            "batch_size",
            "sequence_length",
            "num_heads",
            "state_key_dim",
            "state_value_dim",
            "cache_width",
            "block_size",
            "estimated_bytes",
            "peak_vram_bytes",
        ):
            _nonnegative_int(f"OOM context.{field}", context[field], code="invalid_oom_context")
        if context["batch_size"] < 1 or context["sequence_length"] < 1:
            raise RunRecordError("invalid_oom_context", "OOM B/T must be positive")
        for field in ("dtype", "device"):
            if type(context[field]) is not str or not context[field]:
                raise RunRecordError("invalid_oom_context", f"OOM {field} is invalid")


def _quarantine_reason(error: RunRecordError) -> str:
    return {
        "provenance_mismatch": "stale-provenance",
        "config_mismatch": "stale-config",
        "job_identity": "conflicting-identity",
        "shard_mismatch": "stale-shard",
        "schema_mismatch": "stale-schema",
        "suite_mismatch": "stale-suite",
        "not_completed": "invalid-status",
    }.get(error.code, "malformed-record")


class ResultStore:
    """Authoritative per-run records plus immutable manifests and resume logic."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        provenance: Mapping[str, Any],
        job_index: int,
        num_jobs: int,
    ) -> None:
        _require_int("num_jobs", num_jobs, minimum=1)
        _require_int("job_index", job_index, minimum=0)
        if job_index >= num_jobs:
            raise ValueError("job_index must be less than num_jobs")
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        self.root = Path(root)
        self.provenance = _plain_json(provenance, path="provenance")
        self.job_index = job_index
        self.num_jobs = num_jobs

    def _validate_assignment(self, job: Mapping[str, Any]) -> None:
        if not isinstance(job, Mapping):
            raise TypeError("job must be a mapping")
        job_id = _require_string("job.job_id", job.get("job_id"))
        expected = assign_shard(job_id, self.num_jobs)
        if expected != self.job_index:
            raise ValueError(
                f"job {job_id} belongs to shard {expected}, not {self.job_index}"
            )

    def initialize(
        self,
        *,
        manifest: Mapping[str, Any],
        jobs: Sequence[Mapping[str, Any]],
    ) -> None:
        """Create immutable preflight documents without ever rewriting them."""

        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        normalized = _plain_json(manifest, path="manifest")
        manifest_fields = {
            "schema_version",
            "suite_version",
            "canonical_config",
            "source_hashes",
            "config_hash",
            "asset_hashes",
            "git",
            "environment",
            "command",
            "expanded_jobs_digest",
        }
        missing = manifest_fields - set(normalized)
        unexpected = set(normalized) - manifest_fields
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unexpected:
                details.append("unexpected " + ", ".join(sorted(unexpected)))
            raise ValueError("manifest fields do not match schema: " + "; ".join(details))
        manifest_provenance = {
            field: normalized[field]
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
        if canonical_json_bytes(manifest_provenance) != canonical_json_bytes(
            self.provenance
        ):
            raise ValueError("manifest provenance does not equal store provenance")
        expected = build_manifest(
            canonical_config=normalized["canonical_config"],
            jobs=jobs,
            provenance=self.provenance,
            command=normalized["command"],
        )
        if canonical_json_bytes(normalized) != canonical_json_bytes(expected):
            raise ValueError(
                "manifest identity or provenance does not match this result store"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        write_immutable_json(self.root / "manifest.json", normalized)
        write_immutable_json(self.root / "jobs.json", build_jobs_document(jobs))

    def run_path(self, job: Mapping[str, Any]) -> Path:
        experiment_id = _safe_component(
            "job.experiment_id", job.get("experiment_id")
        )
        stage = _safe_component("job.stage", job.get("stage"))
        job_id = _safe_component("job.job_id", job.get("job_id"))
        seed = _require_int("job.seed", job.get("seed"))
        return self.root / "runs" / experiment_id / stage / f"{seed}-{job_id}.json"

    @property
    def event_path(self) -> Path:
        return (
            self.root
            / "events"
            / f"worker-{self.job_index}-of-{self.num_jobs}.jsonl"
        )

    def append_event(self, event: Mapping[str, Any]) -> None:
        """Append diagnostics to this shard's non-authoritative event stream."""

        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        payload = canonical_json_bytes(event) + b"\n"
        path = self.event_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with _creation_lock(path):
            created = not path.exists()
            with path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if created:
                _fsync_directory(path.parent)

    def _quarantine_interrupted(self, job: Mapping[str, Any], path: Path) -> None:
        pattern = f".{path.name}.*.tmp"
        for temporary in sorted(path.parent.glob(pattern), key=lambda item: item.name):
            quarantine_file(
                temporary,
                root=self.root,
                job_id=str(job["job_id"]),
                reason="interrupted-temp",
            )

    def _validate_for_store(
        self, record: Mapping[str, Any], job: Mapping[str, Any]
    ) -> None:
        status = record.get("status")
        if status not in {"completed", "failed"}:
            raise RunRecordError(
                "invalid_status", "execution status must be completed or failed"
            )
        if status == "completed":
            validate_completed_run(record, job, self.provenance)
        else:
            validate_failed_run(record, job, self.provenance)
        shard = record["shard"]
        if (
            shard.get("count") != self.num_jobs
            or shard.get("index") != self.job_index
        ):
            raise RunRecordError(
                "shard_mismatch", "record does not match this worker assignment"
            )

    def should_run(self, job: Mapping[str, Any]) -> bool:
        """Return whether this job lacks a valid authoritative completed record."""

        self._validate_assignment(job)
        path = self.run_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _creation_lock(path):
            self._quarantine_interrupted(job, path)
            if not path.exists():
                return True
            raw = path.read_bytes()
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                quarantine_file(
                    path,
                    root=self.root,
                    job_id=str(job["job_id"]),
                    reason="truncated",
                )
                return True
            try:
                canonical = canonical_json_bytes(record) + b"\n"
            except (TypeError, ValueError):
                quarantine_file(
                    path,
                    root=self.root,
                    job_id=str(job["job_id"]),
                    reason="malformed-record",
                )
                return True
            if raw != canonical:
                quarantine_file(
                    path,
                    root=self.root,
                    job_id=str(job["job_id"]),
                    reason="noncanonical",
                )
                return True
            if not isinstance(record, Mapping):
                quarantine_file(
                    path,
                    root=self.root,
                    job_id=str(job["job_id"]),
                    reason="malformed-record",
                )
                return True
            try:
                self._validate_for_store(record, job)
            except RunRecordError as error:
                quarantine_file(
                    path,
                    root=self.root,
                    job_id=str(job["job_id"]),
                    reason=_quarantine_reason(error),
                )
                return True
            return record["status"] != "completed"

    def _quarantine_incoming(
        self,
        job: Mapping[str, Any],
        payload: bytes,
        reason: str,
    ) -> Path:
        incoming = self.run_path(job).with_name(
            f".incoming-{uuid.uuid4().hex}.json"
        )
        _atomic_write_bytes(incoming, payload)
        return quarantine_file(
            incoming,
            root=self.root,
            job_id=str(job["job_id"]),
            reason=reason,
        )

    def persist(self, job: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
        """Publish one execution record, detecting duplicate-writer conflicts."""

        self._validate_assignment(job)
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        self._validate_for_store(record, job)
        path = self.run_path(job)
        payload = canonical_json_bytes(record) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _creation_lock(path):
            if not path.exists():
                _atomic_write_bytes(path, payload)
                return True
            existing = path.read_bytes()
            if existing == payload:
                return False
            try:
                existing_record = json.loads(existing)
                existing_status = (
                    existing_record.get("status")
                    if isinstance(existing_record, Mapping)
                    else None
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                existing_status = None

            if existing_status == "completed" and record.get("status") == "failed":
                self._quarantine_incoming(job, payload, "completed-won")
                return False
            if existing_status == "completed" and record.get("status") == "completed":
                quarantine_file(
                    path,
                    root=self.root,
                    job_id=str(job["job_id"]),
                    reason="conflicting-completed",
                )
                self._quarantine_incoming(job, payload, "conflicting-completed")
                raise RunRecordError(
                    "conflicting_completed",
                    "different completed records were produced for one job",
                )

            quarantine_file(
                path,
                root=self.root,
                job_id=str(job["job_id"]),
                reason="retry-replaced",
            )
            _atomic_write_bytes(path, payload)
            return True


def write_immutable_json(path: str | os.PathLike[str], record: Any) -> bool:
    """Write canonical JSON once; identical repeats are no-ops, conflicts fail."""

    destination = Path(path)
    payload = canonical_json_bytes(record) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _creation_lock(destination):
        if destination.exists():
            if destination.read_bytes() == payload:
                return False
            raise FileExistsError(f"immutable JSON conflict: {destination}")
        _atomic_write_bytes(destination, payload)
        return True


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "ResultStore",
    "RunRecordError",
    "assign_shard",
    "atomic_write_json",
    "build_job",
    "build_jobs_document",
    "build_manifest",
    "canonical_json_bytes",
    "expanded_jobs_digest",
    "quarantine_file",
    "select_shard",
    "semantic_job_id",
    "validate_completed_run",
    "validate_failed_run",
    "validate_hybrid_diagnostics",
    "write_immutable_json",
]
