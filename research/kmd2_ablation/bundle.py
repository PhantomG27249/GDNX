"""Deterministic, fail-closed upload bundles for the KMD-2 ablation suite."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
BUNDLE_SCHEMA_VERSION = "1.0.0"
MAX_ARCHIVE_MEMBERS = 512
MAX_MANIFEST_COMPRESSED_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
VERIFIER_SIDECAR_NAME = "verify_bundle.py"
_MANIFEST_CONVENTION = (
    "MANIFEST.json is not self-hashed; every other exact member is hashed; "
    "all archive members including MANIFEST.json are lexicographically sorted"
)
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
    | {f"com{digit}" for digit in "¹²³"}
    | {f"lpt{digit}" for digit in "¹²³"}
)
_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"|?*')
_REQUIREMENT_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?:\[[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:\s*,\s*[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)*\])?"
    r"(?:\s*(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]*"
    r"(?:\s*,\s*(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]*)*)?"
    r"(?:\s*;\s*(?:python_version|python_full_version|platform_python_implementation|"
    r"platform_release|platform_system|platform_version|os_name|sys_platform|"
    r"platform_machine|implementation_name|implementation_version|extra)"
    r"\s*(?:==|!=|~=|<=|>=|<|>|not\s+in|in)\s*(?:\"[A-Za-z0-9_.+ -]+\"|'[A-Za-z0-9_.+ -]+')"
    r"(?:\s+(?:and|or)\s+(?:python_version|python_full_version|platform_python_implementation|"
    r"platform_release|platform_system|platform_version|os_name|sys_platform|"
    r"platform_machine|implementation_name|implementation_version|extra)"
    r"\s*(?:==|!=|~=|<=|>=|<|>|not\s+in|in)\s*(?:\"[A-Za-z0-9_.+ -]+\"|'[A-Za-z0-9_.+ -]+'))*)?$"
)
_TINY_ALLOWED_DEPENDENCIES = frozenset({"torch"})
_QWEN_GDN3_FILES = (
    "gdn3/_reference_recurrence.py",
    "gdn3/gdn3_upgrade.py",
    "gdn3/kmd2_fast_scan.py",
    "gdn3/kmd2_native.py",
)
_COMMON_SOURCE_SEEDS = (
    "research/kmd2_ablation/__init__.py",
    "research/kmd2_ablation/bundle.py",
    "research/kmd2_ablation/run_ablation.py",
    "research/kmd2_ablation/runner.py",
    "research/kmd2_ablation/summarize.py",
)
_TASK_SOURCE_SEEDS = (
    "research/kmd2_ablation/tasks/__init__.py",
    "research/kmd2_ablation/tasks/affine.py",
    "research/kmd2_ablation/tasks/dynamics.py",
    "research/kmd2_ablation/tasks/far_surprise.py",
    "research/kmd2_ablation/tasks/freshness.py",
    "research/kmd2_ablation/tasks/integration.py",
    "research/kmd2_ablation/tasks/local_binding.py",
    "research/kmd2_ablation/tasks/mqar.py",
    "research/kmd2_ablation/tasks/state_tracking.py",
    "research/kmd2_ablation/tasks/structured.py",
)
_KIND_SOURCE_SEEDS = {
    "tiny": (
        "research/kmd2_ablation/tiny_backend.py",
        "research/kmd2_ablation/tiny_training.py",
    ),
    "qwen": (
        "research/kmd2_ablation/qwen_backend.py",
        "research/kmd2_ablation/qwen_cache_campaign.py",
        "research/kmd2_ablation/qwen_checkpoint.py",
        "research/kmd2_ablation/qwen_exact_cache.py",
        "research/kmd2_ablation/qwen_fused_loss.py",
        "research/kmd2_ablation/qwen_hybrid_chunkwise.py",
        "research/kmd2_ablation/qwen_hybrid_liger_chunked.py",
        "research/kmd2_ablation/qwen_hybrid_triton.py",
        "research/kmd2_ablation/qwen_training.py",
        "research/kmd2_ablation/qwen_variants.py",
        "research/kmd2_ablation/tiny_backend.py",
        "research/kmd2_ablation/tiny_training.py",
        "research/kmd2_ablation/tasks/ruler.py",
        "gdn3/__init__.py",
        *_QWEN_GDN3_FILES,
    ),
}
_KIND_TEST_SEEDS = {
    "tiny": (
        "tests/ablation/__init__.py",
        "tests/ablation/test_tiny_backend.py",
        "tests/ablation/test_fast_scan_api.py",
    ),
    "qwen": (
        "tests/ablation/__init__.py",
        "tests/ablation/test_tiny_backend.py",
        "tests/ablation/test_fast_scan_api.py",
        "tests/ablation/test_qwen_backend.py",
        "tests/ablation/test_qwen_fused_loss.py",
    ),
}
_KIND_STATIC_ARTIFACTS = {
    "tiny": ("research/kmd2_ablation/scripts/run_remote_tiny.sh",),
    "qwen": (
        "research/kmd2_ablation/scripts/run_remote_qwen.sh",
        "research/kmd2_ablation/scripts/run_remote_qwen_campaign.sh",
        "research/kmd2_ablation/campaigns/qwen08b-runnable-v1.json",
        "research/kmd2_ablation/campaigns/configs/stock.json",
        "research/kmd2_ablation/campaigns/configs/kmd2-r1.json",
        "research/kmd2_ablation/campaigns/configs/rot-off.json",
        "research/kmd2_ablation/campaigns/configs/rot-constant.json",
        "research/kmd2_ablation/campaigns/configs/rot-noncumulative.json",
        "research/kmd2_ablation/campaigns/configs/rot-fixed-rope.json",
        "research/kmd2_ablation/campaigns/configs/rot-moving-frame-oracle.json",
        "research/kmd2_ablation/campaigns/configs/rout-4.json",
        "research/kmd2_ablation/campaigns/configs/mimo-r2.json",
        "research/kmd2_ablation/campaigns/configs/mimo-r4.json",
        "research/kmd2_ablation/campaigns/configs/gdn2-channel-r1.json",
        "research/kmd2_ablation/campaigns/configs/trapezoid.json",
        "research/kmd2_ablation/campaigns/configs/lookahead.json",
        "research/kmd2_ablation/campaigns/configs/qk-bc-additive.json",
        "research/kmd2_ablation/campaigns/configs/qk-diagonal.json",
        "research/kmd2_ablation/campaigns/configs/cache-recency-w64.json",
        "research/kmd2_ablation/campaigns/configs/cache-surprise-w64.json",
    ),
}
_TINY_EVIDENCE_FILES = _QWEN_GDN3_FILES
_LOCAL_MODULE_PREFIXES = ("research.kmd2_ablation", "tests.ablation", "gdn3")
_TINY_OPTIONAL_LOCAL_IMPORTS = frozenset(
    {
        (
            "research.kmd2_ablation.runner",
            "research.kmd2_ablation.qwen_training",
        ),
        (
            "research.kmd2_ablation.runner",
            "research.kmd2_ablation.resource_probes",
        ),
        (
            "research.kmd2_ablation.runner",
            "research.kmd2_ablation.qwen_variants",
        ),
        (
            "research.kmd2_ablation.gate_probes",
            "research.kmd2_ablation.qwen_training",
        ),
        (
            "research.kmd2_ablation.gate_probes",
            "research.kmd2_ablation.qwen_hybrid_four_state",
        ),
        (
            "research.kmd2_ablation.gate_probes",
            "research.kmd2_ablation.qwen_hybrid_shared",
        ),
        (
            "research.kmd2_ablation.bundle",
            "research.kmd2_ablation.qwen_hybrid_math",
        ),
        (
            "gdn3.gdn3_upgrade",
            "gdn3.triton_kernels",
        ),
    }
)
_FORBIDDEN_SOURCE_SEGMENTS = frozenset(
    {
        ".git",
        ".worktrees",
        "__pycache__",
        "secrets",
        "cache",
        "models",
        "data",
        "checkpoints",
        "runs",
        "outputs",
        "artifacts",
    }
)
_FORBIDDEN_SOURCE_SUFFIXES = (
    ".bin",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".zip",
)
_ASSET_NAMES = ("model", "tokenizer", "checkpoint", "data", "teacher_model")
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_FILES = 256
_MAX_CONFIG_DEPTH = 4
_MAXIMUM_KINDS = frozenset({"maximum-hybrid-a", "maximum-hybrid-b"})
_MAXIMUM_FLAGSHIPS = {
    "maximum-hybrid-a": "research/kmd2_ablation/campaigns/maximum_hybrid/07-package-a-hola-w64.json",
    "maximum-hybrid-b": "research/kmd2_ablation/campaigns/maximum_hybrid/09-package-b-hola-w64.json",
}
_MAXIMUM_READMES = {
    "maximum-hybrid-a": "research/kmd2_ablation/package_readmes/MAXIMUM-HYBRID-A.md",
    "maximum-hybrid-b": "research/kmd2_ablation/package_readmes/MAXIMUM-HYBRID-B.md",
}
_MAXIMUM_DEFAULT_CONTROLS = {
    "maximum-hybrid-a": (
        "gdn2-r1", "gdn2-mimo-r2", "gdn2-mimo-r4", "package-a-native-decay",
        "package-a-braid-no-cache", "package-a-recency-w64", "package-a-hola-w64",
        "shared-query-widening", "stock-qwen",
    ),
    "maximum-hybrid-b": (
        "gdn2-r1", "gdn2-mimo-r2", "gdn2-mimo-r4", "package-b-recency-w64",
        "package-b-hola-w64", "shared-query-widening", "stock-qwen",
    ),
}


VERIFY_BUNDLE_SOURCE = r'''#!/usr/bin/env python3
"""Verify and optionally extract one deterministic KMD-2 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
import zipfile


ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
BUNDLE_SCHEMA_VERSION = "1.0.0"
MAX_ARCHIVE_MEMBERS = 512
MAX_MANIFEST_COMPRESSED_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {"com{}".format(index) for index in range(1, 10)}
    | {"lpt{}".format(index) for index in range(1, 10)}
    | {"com{}".format(digit) for digit in "¹²³"}
    | {"lpt{}".format(digit) for digit in "¹²³"}
)
_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"|?*')
_HEX = frozenset("0123456789abcdef")
_MANIFEST_NAME = "MANIFEST.json"
_MANIFEST_CONVENTION = (
    "MANIFEST.json is not self-hashed; every other exact member is hashed; "
    "all archive members including MANIFEST.json are lexicographically sorted"
)
_MANIFEST_FIELDS = {
    "schema_version",
    "suite_version",
    "kind",
    "git",
    "config",
    "config_sha256",
    "production_source_sha256",
    "entries",
    "expected_members",
    "provenance",
    "smoke",
    "manifest_convention",
}


def _canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _append_code(codes, code):
    if code not in codes:
        codes.append(code)


def _report(ok, codes, sha256, member_count, extracted_to=None):
    return {
        "ok": bool(ok),
        "codes": list(codes),
        "sha256": sha256,
        "member_count": member_count,
        "extracted_to": None if extracted_to is None else str(extracted_to),
    }


def _safe_member_name(name):
    if (
        type(name) is not str
        or not name
        or "\\" in name
        or any(character in _WINDOWS_FORBIDDEN_CHARS for character in name)
        or _DRIVE_PREFIX.match(name)
        or PurePosixPath(name).is_absolute()
        or any(ord(character) < 32 for character in name)
    ):
        return False
    parts = name.split("/")
    return not any(
        part in {"", ".", ".."}
        or part.endswith((".", " "))
        or part.partition(".")[0].casefold() in _WINDOWS_RESERVED_NAMES
        for part in parts
    )


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError("non-finite JSON number: " + value)


def _load_manifest(data):
    document = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if type(document) is not dict:
        raise ValueError("manifest root must be an object")
    return document


def _valid_sha256(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _expected_smoke(kind, config):
    command = [
        "python",
        "-m",
        "research.kmd2_ablation.run_ablation",
    ]
    if kind == "tiny":
        return command + [
            "run",
            "--backend",
            "tiny",
            "--config",
            config,
            "--out",
            "results",
            "--job-index",
            "0",
            "--num-jobs",
            "1",
        ]
    return command + [
        "preflight",
        "--backend",
        "qwen",
        "--config",
        config,
        "--out",
        "results",
        "--dry-run",
    ]


def _manifest_metadata(manifest, names, codes):
    if set(manifest) != _MANIFEST_FIELDS:
        _append_code(codes, "manifest_schema_invalid")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        _append_code(codes, "manifest_schema_invalid")
    if type(manifest.get("suite_version")) is not str or not manifest.get("suite_version"):
        _append_code(codes, "manifest_schema_invalid")
    if manifest.get("kind") not in {"tiny", "qwen", "maximum-hybrid-a", "maximum-hybrid-b"}:
        _append_code(codes, "manifest_schema_invalid")
    if manifest.get("manifest_convention") != _MANIFEST_CONVENTION:
        _append_code(codes, "manifest_schema_invalid")

    expected = manifest.get("expected_members")
    if (
        type(expected) is not list
        or any(type(name) is not str for name in expected)
        or expected != sorted(names)
    ):
        _append_code(codes, "member_set_mismatch")

    entries = manifest.get("entries")
    expected_hashed = set(names) - {_MANIFEST_NAME}
    if type(entries) is not dict or set(entries) != expected_hashed:
        _append_code(codes, "member_set_mismatch")
        return {}

    for name, metadata in entries.items():
        if (
            type(metadata) is not dict
            or set(metadata) != {"mode", "sha256", "size"}
            or type(metadata.get("mode")) is not int
            or metadata["mode"] not in {0o644, 0o755}
            or type(metadata.get("size")) is not int
            or metadata["size"] < 0
            or not _valid_sha256(metadata.get("sha256"))
        ):
            _append_code(codes, "manifest_schema_invalid")

    config = manifest.get("config")
    config_sha256 = manifest.get("config_sha256")
    if (
        type(config) is not str
        or not _safe_member_name(config)
        or config not in entries
        or not _valid_sha256(config_sha256)
        or type(entries.get(config)) is not dict
        or entries[config].get("sha256") != config_sha256
    ):
        _append_code(codes, "manifest_schema_invalid")

    if not _valid_sha256(manifest.get("production_source_sha256")):
        _append_code(codes, "manifest_schema_invalid")

    git = manifest.get("git")
    if (
        type(git) is not dict
        or set(git) != {"revision", "dirty", "diff_sha256"}
        or type(git.get("revision")) is not str
        or len(git.get("revision", "")) != 40
        or any(character not in _HEX for character in git.get("revision", ""))
        or type(git.get("dirty")) is not bool
        or not _valid_sha256(git.get("diff_sha256"))
    ):
        _append_code(codes, "manifest_schema_invalid")

    provenance = manifest.get("provenance")
    smoke = manifest.get("smoke")
    expected_smoke = (
        _expected_smoke(manifest.get("kind"), config)
        if type(config) is str and manifest.get("kind") in {"tiny", "qwen", "maximum-hybrid-a", "maximum-hybrid-b"}
        else None
    )
    if (
        type(provenance) is not dict
        or set(provenance) != {"build_command", "smoke_command"}
        or provenance.get("build_command")
        != "python -m research.kmd2_ablation.run_ablation bundle"
        or provenance.get("smoke_command") != expected_smoke
        or type(smoke) is not dict
        or set(smoke) != {"command"}
        or smoke.get("command") != expected_smoke
    ):
        _append_code(codes, "manifest_schema_invalid")

    if "verify_bundle.py" not in entries:
        _append_code(codes, "member_set_mismatch")
    requirements = "research/kmd2_ablation/requirements-{}.txt".format(
        "qwen" if manifest.get("kind") in {"maximum-hybrid-a", "maximum-hybrid-b"} else manifest.get("kind")
    )
    if requirements not in entries:
        _append_code(codes, "member_set_mismatch")
    launcher = (
        "research/kmd2_ablation/scripts/run_remote_qwen_maximum_hybrids.sh"
        if manifest.get("kind") in {"maximum-hybrid-a", "maximum-hybrid-b"}
        else "research/kmd2_ablation/scripts/run_remote_{}.sh".format(manifest.get("kind"))
    )
    if launcher not in entries:
        _append_code(codes, "member_set_mismatch")
    if manifest.get("kind") == "qwen" and "external-assets.json" not in entries:
        _append_code(codes, "member_set_mismatch")
    if manifest.get("kind") in {"maximum-hybrid-a", "maximum-hybrid-b"}:
        if "PACKAGE.json" not in entries:
            _append_code(codes, "member_set_mismatch")
        if "research/kmd2_ablation/requirements-qwen.lock" not in entries:
            _append_code(codes, "member_set_mismatch")
        if "research/kmd2_ablation/assets/qwen08b-maximum-assets.template.json" not in entries:
            _append_code(codes, "member_set_mismatch")
    return entries


def _verify_zip_metadata(archive, infos, codes):
    if archive.comment:
        _append_code(codes, "noncanonical_zip_metadata")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        _append_code(codes, "duplicate_member")
    if names != sorted(names):
        _append_code(codes, "noncanonical_member_order")
    collision_keys = {}
    for info in infos:
        name = info.filename
        if not _safe_member_name(name):
            _append_code(codes, "unsafe_member_name")
        collision_key = unicodedata.normalize("NFC", name).casefold()
        if collision_key in collision_keys:
            _append_code(codes, "member_name_collision")
        else:
            collision_keys[collision_key] = name
        if info.flag_bits & 1:
            _append_code(codes, "encrypted_member")
        expected_flags = 0 if name.isascii() else 0x800
        if info.flag_bits != expected_flags:
            _append_code(codes, "noncanonical_zip_metadata")
        if info.compress_type != zipfile.ZIP_DEFLATED:
            _append_code(codes, "unsupported_compression")
        raw_mode = info.external_attr >> 16
        if stat.S_IFMT(raw_mode) != stat.S_IFREG:
            _append_code(codes, "special_mode")
        elif stat.S_IMODE(raw_mode) not in {0o644, 0o755}:
            _append_code(codes, "noncanonical_zip_metadata")
        if (
            info.date_time != ZIP_EPOCH
            or info.create_system != 3
            or info.internal_attr != 0
            or info.external_attr & 0xFFFF
            or info.extra
            or info.comment
        ):
            _append_code(codes, "noncanonical_zip_metadata")


def _verify_archive_limits(infos, codes):
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        _append_code(codes, "member_count_limit")
    total = 0
    for info in infos:
        size = info.file_size
        compressed = info.compress_size
        total += size
        if info.filename == _MANIFEST_NAME and (
            size > MAX_MANIFEST_BYTES or compressed > MAX_MANIFEST_COMPRESSED_BYTES
        ):
            _append_code(codes, "manifest_size_limit")
        if size > MAX_MEMBER_BYTES:
            _append_code(codes, "member_size_limit")
        if size and (compressed <= 0 or size > compressed * MAX_COMPRESSION_RATIO):
            _append_code(codes, "compression_ratio_limit")
    if total > MAX_TOTAL_BYTES:
        _append_code(codes, "total_size_limit")


def _extract_contents(contents, modes, destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        return "extraction_destination_exists"
    parent = destination.parent
    staging = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix="." + (destination.name or "extraction") + ".",
                suffix=".tmp",
                dir=str(parent),
            )
        )
        for name in sorted(contents):
            target = staging.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(contents[name])
            os.chmod(target, modes[name])
        os.replace(staging, destination)
        staging = None
    except (OSError, ValueError):
        return "extraction_failed"
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return None


def verify_archive(archive_path, extract_to=None):
    archive_path = Path(archive_path)
    try:
        outer_sha256 = _sha256_file(archive_path)
    except OSError:
        return _report(False, ["archive_unreadable"], "", 0)

    codes = []
    member_count = 0
    contents = {}
    modes = {}
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            infos = archive.infolist()
            member_count = len(infos)
            _verify_zip_metadata(archive, infos, codes)
            _verify_archive_limits(infos, codes)
            if codes:
                return _report(False, codes, outer_sha256, member_count)

            names = [info.filename for info in infos]
            if _MANIFEST_NAME not in names:
                return _report(
                    False,
                    ["manifest_missing"],
                    outer_sha256,
                    member_count,
                )
            manifest_bytes = archive.read(_MANIFEST_NAME)
            try:
                manifest = _load_manifest(manifest_bytes)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                return _report(
                    False,
                    ["manifest_invalid"],
                    outer_sha256,
                    member_count,
                )
            try:
                canonical_manifest = _canonical_json_bytes(manifest)
            except (TypeError, ValueError):
                canonical_manifest = b""
            if manifest_bytes != canonical_manifest:
                _append_code(codes, "manifest_not_canonical")
            entries = _manifest_metadata(manifest, names, codes)

            by_name = {info.filename: info for info in infos}
            manifest_info = by_name[_MANIFEST_NAME]
            if stat.S_IMODE(manifest_info.external_attr >> 16) != 0o644:
                _append_code(codes, "mode_mismatch")
            config_member = manifest.get("config")
            for name, info in by_name.items():
                digest = hashlib.sha256()
                size = 0
                retained = bytearray() if extract_to is not None or name in {_MANIFEST_NAME, config_member} else None
                with archive.open(info, "r") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_MEMBER_BYTES:
                            _append_code(codes, "member_size_limit")
                            break
                        digest.update(chunk)
                        if retained is not None:
                            retained.extend(chunk)
                if retained is not None:
                    contents[name] = bytes(retained)
                modes[name] = stat.S_IMODE(info.external_attr >> 16)
                if name == _MANIFEST_NAME:
                    continue
                metadata = entries.get(name)
                if type(metadata) is not dict:
                    continue
                if info.file_size != metadata.get("size") or size != metadata.get("size"):
                    _append_code(codes, "size_mismatch")
                if digest.hexdigest() != metadata.get("sha256"):
                    _append_code(codes, "hash_mismatch")
                if modes[name] != metadata.get("mode"):
                    _append_code(codes, "mode_mismatch")

            if type(entries) is dict:
                config_name = manifest.get("config")
                try:
                    config_document = _load_manifest(contents[config_name])
                except (
                    KeyError,
                    TypeError,
                    UnicodeDecodeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    config_document = {}
                    _append_code(codes, "manifest_schema_invalid")
                maximum_kind = manifest.get("kind") in {"maximum-hybrid-a", "maximum-hybrid-b"}
                if (
                    config_document.get("schema_version") != BUNDLE_SCHEMA_VERSION
                    or (
                        not maximum_kind
                        and (
                            config_document.get("suite_version") != manifest.get("suite_version")
                            or config_document.get("backend") != manifest.get("kind")
                        )
                    )
                ):
                    _append_code(codes, "manifest_schema_invalid")
                source_hashes = {
                    name: metadata["sha256"]
                    for name, metadata in entries.items()
                    if type(metadata) is dict
                    and name.endswith(".py")
                    and not name.startswith("tests/")
                    and name != "verify_bundle.py"
                    and _valid_sha256(metadata.get("sha256"))
                }
                production_hash = hashlib.sha256(
                    _canonical_json_bytes(source_hashes)
                ).hexdigest()
                if manifest.get("production_source_sha256") != production_hash:
                    _append_code(codes, "manifest_schema_invalid")
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        _append_code(codes, "invalid_zip")

    if codes:
        return _report(False, codes, outer_sha256, member_count)
    if extract_to is not None:
        extraction_code = _extract_contents(contents, modes, extract_to)
        if extraction_code is not None:
            return _report(
                False,
                [extraction_code],
                outer_sha256,
                member_count,
            )
        return _report(True, [], outer_sha256, member_count, extract_to)
    return _report(True, [], outer_sha256, member_count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    options = parser.parse_args(argv)
    report = verify_archive(options.archive, options.extract_to)
    sys.stdout.write(_canonical_json_bytes(report).decode("utf-8"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("utf-8")


class BundleError(ValueError):
    """A bundle cannot be planned, written, or verified safely."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class BundleEntry:
    """One regular archive member with repository-relative POSIX identity."""

    name: str
    data: bytes
    mode: int = 0o644

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("bundle member name must be a str")
        if type(self.data) is not bytes:
            raise TypeError("bundle member data must be bytes")
        if type(self.mode) is not int:
            raise TypeError("bundle member mode must be an int")
        if self.mode not in {0o644, 0o755}:
            raise BundleError(
                "member_mode_invalid",
                "member mode must be canonical regular-file permissions",
            )


@dataclass(frozen=True, slots=True)
class BundlePlan:
    """Complete immutable archive plan including its canonical manifest."""

    kind: str
    entries: tuple[BundleEntry, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BundleResult:
    """Identity of a completed deterministic archive."""

    kind: str
    path: Path
    sha256: str
    member_count: int
    verifier_path: Path
    verifier_sha256: str


@dataclass(frozen=True, slots=True)
class BundleVerificationResult:
    """Machine-readable result from reopening and checking one archive."""

    ok: bool
    codes: tuple[str, ...]
    sha256: str
    member_count: int
    extracted_to: Path | None


_VERIFIER_GLOBALS: dict[str, Any] = {"__name__": "_kmd2_embedded_verifier"}
exec(
    compile(VERIFY_BUNDLE_SOURCE, "verify_bundle.py", "exec"),
    _VERIFIER_GLOBALS,
)
_VERIFY_ARCHIVE = _VERIFIER_GLOBALS["verify_archive"]


def _validated_member_name(name: str) -> str:
    if (
        not name
        or "\\" in name
        or any(character in _WINDOWS_FORBIDDEN_CHARS for character in name)
        or _DRIVE_PREFIX.match(name)
        or PurePosixPath(name).is_absolute()
        or any(ord(character) < 32 for character in name)
    ):
        raise BundleError("unsafe_member_name", f"unsafe archive member: {name!r}")
    parts = name.split("/")
    if any(
        part in {"", ".", ".."}
        or part.endswith((".", " "))
        or part.partition(".")[0].casefold() in _WINDOWS_RESERVED_NAMES
        for part in parts
    ):
        raise BundleError("unsafe_member_name", f"unsafe archive member: {name!r}")
    return name


def _validated_entries(entries: Iterable[BundleEntry]) -> tuple[BundleEntry, ...]:
    materialized = tuple(entries)
    collision_keys: dict[str, str] = {}
    for entry in materialized:
        if not isinstance(entry, BundleEntry):
            raise TypeError("entries must contain BundleEntry records")
        name = _validated_member_name(entry.name)
        collision_key = unicodedata.normalize("NFC", name).casefold()
        previous = collision_keys.get(collision_key)
        if previous is not None:
            raise BundleError(
                "member_name_collision",
                f"archive member collision: {previous!r} and {name!r}",
            )
        collision_keys[collision_key] = name
    return tuple(sorted(materialized, key=lambda entry: entry.name))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_deterministic_zip(
    entries: Iterable[BundleEntry], destination: Path | str
) -> str:
    """Write sorted regular members with fixed ZIP metadata and return SHA-256."""

    ordered = _validated_entries(entries)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for entry in ordered:
            info = zipfile.ZipInfo(entry.name, date_time=ZIP_EPOCH)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | entry.mode) << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                entry.data,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return _sha256_file(output)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _run_git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            encoding="utf-8" if text else None,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BundleError("git_provenance_unavailable", "cannot inspect Git provenance") from error
    return completed.stdout


def _repo_relative(root: Path, path: Path) -> tuple[Path, str]:
    try:
        unresolved = path if path.is_absolute() else root / path
        relative_unresolved = unresolved.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise BundleError("unsafe_source_path", "source path escapes repository root") from error
    cursor = root
    for part in relative_unresolved.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BundleError("symlink_forbidden", f"symlink source is forbidden: {relative_unresolved.as_posix()}")
    try:
        resolved = unresolved.resolve(strict=True)
        relative = resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise BundleError(
            "required_artifact_missing",
            f"required artifact missing: {relative_unresolved.as_posix()}",
        ) from error
    return resolved, relative.as_posix()


def _git_modes(root: Path) -> dict[str, int]:
    raw = _run_git(root, "ls-files", "-s", "-z", text=False)
    assert isinstance(raw, bytes)
    modes: dict[str, int] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_name = record.partition(b"\t")
        if not separator:
            raise BundleError("git_index_invalid", "malformed Git index record")
        fields = metadata.split()
        if len(fields) < 3:
            raise BundleError("git_index_invalid", "malformed Git index metadata")
        name = encoded_name.decode("utf-8")
        mode = int(fields[0], 8)
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise BundleError("symlink_forbidden", f"Git symlink is forbidden: {name}")
        modes[name] = 0o755 if mode & 0o111 else 0o644
    return modes


def _entry_from_repo(root: Path, path: Path, git_modes: Mapping[str, int]) -> BundleEntry:
    resolved, name = _repo_relative(root, path)
    _validated_source_name(name)
    if not resolved.is_file():
        raise BundleError("unsafe_source_type", f"source is not a regular file: {name}")
    size = resolved.stat().st_size
    if size > _MAX_SOURCE_BYTES:
        raise BundleError("source_too_large", f"source exceeds size limit: {name}")
    return BundleEntry(name=name, data=resolved.read_bytes(), mode=git_modes.get(name, 0o644))


def _validated_source_name(name: str) -> str:
    _validated_member_name(name)
    parts = PurePosixPath(name).parts
    folded = tuple(part.casefold() for part in parts)
    basename = folded[-1]
    if (
        any(part in _FORBIDDEN_SOURCE_SEGMENTS for part in folded)
        or basename == ".env"
        or basename.startswith(".env.")
        or basename.endswith(_FORBIDDEN_SOURCE_SUFFIXES)
    ):
        raise BundleError("forbidden_source_path", f"forbidden source path: {name}")
    return name


def _unique_config_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_config_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _workspace_kind_configs(root: Path, kind: str) -> tuple[Path, ...]:
    """Collect strict configs from the one explicit bounded workspace root.

    Bundle construction is intentionally useful before a development branch is
    committed.  Git provenance still records every untracked path, while this
    collector admits only complete schema-valid JSON files below the canonical
    config directory and never scans the rest of the worktree.
    """

    from .config import ExperimentConfig

    prefix = "research/kmd2_ablation/configs/"
    config_root, _ = _repo_relative(root, root / PurePosixPath(prefix))
    if not config_root.is_dir():
        raise BundleError(
            "required_artifact_missing", f"config root is not a directory: {prefix}"
        )
    pending: list[tuple[Path, int]] = [(config_root, 0)]
    workspace_names: list[str] = []
    while pending:
        directory, depth = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise BundleError(
                "config_invalid", "cannot enumerate the canonical config root"
            ) from error
        for candidate in children:
            if candidate.is_symlink():
                _repo_relative(root, candidate)
            if candidate.is_dir():
                if depth >= _MAX_CONFIG_DEPTH:
                    raise BundleError(
                        "config_invalid", "config directory nesting exceeds the limit"
                    )
                pending.append((candidate, depth + 1))
                continue
            if candidate.suffix != ".json":
                continue
            resolved, normalized = _repo_relative(root, candidate)
            _validated_source_name(normalized)
            if not resolved.is_file():
                raise BundleError(
                    "config_invalid", f"config is not a regular file: {normalized}"
                )
            workspace_names.append(normalized)
            if len(workspace_names) > _MAX_CONFIG_FILES:
                raise BundleError(
                    "config_invalid", "config file count exceeds the bounded limit"
                )

    names: list[str] = []
    for name in sorted(workspace_names):
        resolved, normalized = _repo_relative(root, root / PurePosixPath(name))
        _validated_source_name(normalized)
        try:
            document = json.loads(
                resolved.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_config_object,
                parse_constant=_reject_config_constant,
            )
            parsed = ExperimentConfig.from_dict(document)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise BundleError(
                "config_invalid",
                f"workspace config is not complete canonical JSON: {name}",
            ) from error
        if parsed.backend == kind:
            names.append(normalized)
    if not names:
        raise BundleError(
            "required_artifact_missing",
            f"no validated {kind} configs under {prefix}",
        )
    return tuple(root / PurePosixPath(name) for name in names)


def _required_paths(root: Path, kind: str, config_path: Path) -> tuple[Path, ...]:
    requirements_name = f"requirements-{kind}.txt"
    static = [
        root / "LICENSE",
        root / "README.md",
        root / "research/kmd2_ablation/README.md",
        root / "research/kmd2_ablation/config.schema.json",
        root / "research/kmd2_ablation" / requirements_name,
        config_path,
    ]
    evidence_names = _TINY_EVIDENCE_FILES if kind == "tiny" else ()
    return tuple(
        [*static]
        + [root / PurePosixPath(name) for name in _KIND_STATIC_ARTIFACTS[kind]]
        + [root / PurePosixPath(name) for name in evidence_names]
    )


def _requirement_roots(data: bytes, *, kind: str) -> frozenset[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleError("requirements_invalid", "requirements must be UTF-8") from error
    roots: set[str] = set()
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        match = _REQUIREMENT_LINE.fullmatch(stripped)
        if match is None:
            raise BundleError("requirements_invalid", f"unsupported requirement: {stripped}")
        roots.add(re.sub(r"[-_.]+", "_", match.group("name")).casefold())
    forbidden = roots - _TINY_ALLOWED_DEPENDENCIES if kind == "tiny" else set()
    if forbidden:
        raise BundleError(
            "forbidden_dependency",
            "tiny requirements contain forbidden dependencies: " + ", ".join(sorted(forbidden)),
        )
    return frozenset(roots)


def validate_qwen_lock(data: bytes) -> bool:
    """Validate the complete, exact Qwen lock used by clean extractions."""

    roots = _requirement_roots(data, kind="qwen")
    required = {"torch", "transformers", "safetensors", "triton", "fla_core"}
    if roots != required:
        raise BundleError("requirements_invalid", "Qwen lock has an incomplete dependency set")
    text = data.decode("utf-8")
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line and "==" not in line.partition(";")[0]:
            raise BundleError("requirements_invalid", f"Qwen lock is not exact: {line}")
    return True


def _collect_maximum_bundle_files(root: Path, kind: str) -> tuple[BundleEntry, ...]:
    """Collect the complete tracked production campaign without runtime assets."""

    git_modes = _git_modes(root)
    candidates: set[Path] = {
        root / "LICENSE",
        root / "README.md",
        root / "research/kmd2_ablation/README.md",
        root / "research/kmd2_ablation/config.schema.json",
        root / "research/kmd2_ablation/requirements-qwen.txt",
        root / "research/kmd2_ablation/requirements-qwen.lock",
        root / "research/kmd2_ablation/assets/qwen08b-maximum-assets.template.json",
        root / "research/kmd2_ablation/campaigns/qwen08b-maximum-hybrid-v1.json",
        root / "research/kmd2_ablation/scripts/run_remote_qwen_maximum_hybrids.sh",
        root / PurePosixPath(_MAXIMUM_READMES[kind]),
    }
    for directory, patterns in (
        (root / "research/kmd2_ablation", ("*.py",)),
        (root / "gdn3", ("*.py",)),
        (root / "tests/ablation", ("*.py",)),
        (root / "research/kmd2_ablation/configs", ("*.json",)),
        (root / "research/kmd2_ablation/campaigns", ("*.json",)),
    ):
        for pattern in patterns:
            candidates.update(path for path in directory.rglob(pattern) if path.is_file() or path.is_symlink())
    entries = list(_validated_entries(_entry_from_repo(root, path, git_modes) for path in candidates))
    if kind == "maximum-hybrid-b":
        package_root = root / "research/kmd2_ablation/package_root/package-b"
        entries = [entry for entry in entries if entry.name != "README.md"]
        entries.extend(
            (
                BundleEntry("README.md", (package_root / "README.md").read_bytes()),
                BundleEntry("requirements.txt", (root / "research/kmd2_ablation/requirements-qwen.txt").read_bytes()),
                BundleEntry("config.toml", (package_root / "config.toml").read_bytes()),
                BundleEntry("run.sh", (package_root / "run.sh").read_bytes(), 0o755),
            )
        )
    entries = _validated_entries(entries)
    lock = next(entry for entry in entries if entry.name.endswith("requirements-qwen.lock"))
    validate_qwen_lock(lock.data)
    requirements = next(entry for entry in entries if entry.name.endswith("requirements-qwen.txt"))
    if _requirement_roots(requirements.data, kind="qwen") != _requirement_roots(lock.data, kind="qwen"):
        raise BundleError("requirements_invalid", "Qwen requirements and lock dependency sets differ")
    runtime_names = frozenset(
        entry.name
        for entry in entries
        if entry.name.endswith(".py") and not entry.name.startswith("tests/")
    )
    _audit_imports(entries, requirements, "qwen", runtime_names=runtime_names)
    return entries


def _module_name(path: str) -> tuple[str, bool]:
    module = path[:-3].replace("/", ".")
    is_package = module.endswith(".__init__")
    return (module[: -len(".__init__")] if is_package else module, is_package)


def _local_module(module: str) -> bool:
    return module in {"research", "tests", "gdn3"} or any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in _LOCAL_MODULE_PREFIXES
    )


def _module_path(root: Path, module: str) -> tuple[Path, bool] | None:
    if not _local_module(module):
        return None
    relative = PurePosixPath(*module.split("."))
    source = root / Path(str(relative) + ".py")
    package = root / relative / "__init__.py"
    matches = tuple(path for path in (source, package) if path.is_file() or path.is_symlink())
    if len(matches) > 1:
        raise BundleError("local_import_ambiguous", f"ambiguous local module {module}")
    if not matches:
        return None
    return matches[0], matches[0] == package


def _resolve_import_from(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if not node.level:
        assert node.module is not None
        return node.module
    package = module if is_package else module.rpartition(".")[0]
    relative = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError) as error:
        raise BundleError("local_import_invalid", f"invalid relative import in {module}") from error


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(item) for item in target.elts))
    return set()


def _exported_symbols(tree: ast.Module) -> frozenset[str]:
    exports: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                exports.update(_assigned_names(target))
        elif isinstance(node, ast.AnnAssign):
            exports.update(_assigned_names(node.target))
        elif isinstance(node, ast.Import):
            exports.update(alias.asname or alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            exports.update(
                alias.asname or alias.name for alias in node.names if alias.name != "*"
            )
    return frozenset(exports)


def _declared_star_exports(tree: ast.Module, entry_name: str) -> tuple[str, ...] | None:
    declarations: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            declarations.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and node.value is not None
        ):
            declarations.append(node.value)
    if not declarations:
        return None
    if len(declarations) != 1:
        raise BundleError(
            "local_import_invalid", f"multiple __all__ declarations in {entry_name}"
        )
    try:
        value = ast.literal_eval(declarations[0])
    except (TypeError, ValueError) as error:
        raise BundleError(
            "local_import_invalid", f"dynamic __all__ is unsupported in {entry_name}"
        ) from error
    if (
        not isinstance(value, (list, tuple))
        or any(type(name) is not str or not name for name in value)
        or len(value) != len(set(value))
    ):
        raise BundleError(
            "local_import_invalid", f"invalid __all__ declaration in {entry_name}"
        )
    return tuple(value)


def _read_python_entry(
    root: Path,
    path: Path,
    git_modes: Mapping[str, int],
) -> tuple[BundleEntry, ast.Module, str, bool]:
    entry = _entry_from_repo(root, path, git_modes)
    if not entry.name.endswith(".py"):
        raise BundleError("python_source_invalid", f"local module is not Python: {entry.name}")
    try:
        tree = ast.parse(entry.data.decode("utf-8"), filename=entry.name)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise BundleError("python_source_invalid", f"cannot parse {entry.name}") from error
    module, is_package = _module_name(entry.name)
    return entry, tree, module, is_package


def _collect_local_closure(
    root: Path,
    seed_names: Iterable[str],
    *,
    kind: str,
    git_modes: Mapping[str, int],
) -> tuple[BundleEntry, ...]:
    pending = [root / PurePosixPath(name) for name in seed_names]
    collected: dict[str, BundleEntry] = {}
    parsed: dict[str, tuple[ast.Module, str, bool]] = {}
    while pending:
        path = pending.pop()
        entry, tree, module, is_package = _read_python_entry(root, path, git_modes)
        if entry.name in collected:
            continue
        collected[entry.name] = entry
        parsed[entry.name] = (tree, module, is_package)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = tuple(alias.name for alias in node.names)
                aliases: tuple[ast.alias, ...] = ()
            elif isinstance(node, ast.ImportFrom):
                targets = (_resolve_import_from(node, module, is_package),)
                aliases = tuple(node.names)
            else:
                continue
            for target in targets:
                if kind == "tiny" and (module, target) in _TINY_OPTIONAL_LOCAL_IMPORTS:
                    continue
                resolved = _module_path(root, target)
                if resolved is None:
                    if _local_module(target):
                        raise BundleError(
                            "local_import_missing",
                            f"included source omits local import {target} from {entry.name}",
                        )
                    continue
                target_path, _ = resolved
                pending.append(target_path)
                if not aliases:
                    continue
                target_entry, target_tree, _, _ = _read_python_entry(
                    root, target_path, git_modes
                )
                exports = _exported_symbols(target_tree)
                for alias in aliases:
                    imported_names = (
                        _declared_star_exports(target_tree, target_entry.name) or ()
                        if alias.name == "*"
                        else (alias.name,)
                    )
                    for imported_name in imported_names:
                        submodule = f"{target}.{imported_name}"
                        if kind == "tiny" and (
                            module,
                            submodule,
                        ) in _TINY_OPTIONAL_LOCAL_IMPORTS:
                            continue
                        submodule_path = _module_path(root, submodule)
                        if submodule_path is not None:
                            pending.append(submodule_path[0])
                        elif imported_name not in exports:
                            raise BundleError(
                                "local_import_missing",
                                f"{submodule} is neither a local module nor an exported symbol in {target_entry.name}",
                            )
    return _validated_entries(collected.values())


def _audit_imports(
    entries: tuple[BundleEntry, ...],
    requirements: BundleEntry,
    kind: str,
    *,
    runtime_names: frozenset[str] | None = None,
) -> None:
    declared = _requirement_roots(requirements.data, kind=kind)
    python_entries = tuple(
        entry
        for entry in entries
        if entry.name.endswith(".py")
        and (runtime_names is None or entry.name in runtime_names)
    )
    stdlib = frozenset(sys.stdlib_module_names) | {"__future__"}
    for entry in python_entries:
        try:
            tree = ast.parse(entry.data.decode("utf-8"), filename=entry.name)
        except (UnicodeDecodeError, SyntaxError) as error:
            raise BundleError("python_source_invalid", f"cannot parse {entry.name}") from error
        targets: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                targets.append(node.module)
        for target in targets:
            root = target.partition(".")[0]
            canonical_root = re.sub(r"[-_.]+", "_", root).casefold()
            if root in stdlib:
                continue
            if _local_module(target):
                continue
            if kind == "tiny" and canonical_root not in _TINY_ALLOWED_DEPENDENCIES:
                raise BundleError("forbidden_dependency", f"tiny source imports {root}")
            if canonical_root not in declared:
                raise BundleError("undeclared_dependency", f"undeclared import {root} in {entry.name}")


def collect_bundle_files(
    *,
    kind: str,
    repo_root: Path | str,
    config_path: Path | str | None = None,
    assets_manifest: Path | str | None = None,
) -> tuple[BundleEntry, ...]:
    """Collect safe repository payload files and audit their Python imports."""

    if kind not in {"tiny", "qwen", *_MAXIMUM_KINDS}:
        raise BundleError("bundle_kind_invalid", "unsupported bundle kind")
    root = Path(repo_root).resolve(strict=True)
    if kind in _MAXIMUM_KINDS:
        return _collect_maximum_bundle_files(root, kind)
    if config_path is None:
        raise BundleError("config_required", "tiny and Qwen bundles require a config")
    config, config_name = _repo_relative(root, Path(config_path))
    configs = _workspace_kind_configs(root, kind)
    config_names = {_repo_relative(root, path)[1] for path in configs}
    if config_name not in config_names:
        raise BundleError(
            "config_not_committed",
            f"config is not a validated {kind} config in the canonical workspace root",
        )
    git_modes = _git_modes(root)
    source_seed_names = (
        *_COMMON_SOURCE_SEEDS,
        *_TASK_SOURCE_SEEDS,
        *_KIND_SOURCE_SEEDS[kind],
    )
    source_entries = _collect_local_closure(
        root,
        source_seed_names,
        kind=kind,
        git_modes=git_modes,
    )
    test_entries = _collect_local_closure(
        root,
        _KIND_TEST_SEEDS[kind],
        kind=kind,
        git_modes=git_modes,
    )
    if kind == "tiny":
        qwen_runtime = tuple(
            entry.name for entry in source_entries if "qwen" in entry.name.casefold()
        )
        if qwen_runtime:
            raise BundleError(
                "forbidden_dependency",
                f"tiny source includes Qwen runtime module {qwen_runtime[0]}",
            )
        source_entries = tuple(
            entry
            for entry in source_entries
            if "qwen" not in entry.name.casefold()
            and (
                not entry.name.startswith("gdn3/")
                or entry.name in _TINY_EVIDENCE_FILES
            )
        )
        test_entries = tuple(
            entry
            for entry in test_entries
            if entry.name != "gdn3/__init__.py"
            and "qwen" not in entry.name.casefold()
        )
    candidates: set[Path] = set(_required_paths(root, kind, config))
    candidates.update(configs)
    static_entries = tuple(
        _entry_from_repo(root, path, git_modes) for path in candidates
    )
    by_name = {
        entry.name: entry
        for entry in (*static_entries, *source_entries, *test_entries)
    }
    entries = _validated_entries(by_name.values())
    requirements_name = f"research/kmd2_ablation/requirements-{kind}.txt"
    requirements = next(entry for entry in entries if entry.name == requirements_name)
    runtime_names = frozenset(entry.name for entry in source_entries)
    if kind == "tiny":
        runtime_names -= frozenset(_TINY_EVIDENCE_FILES)
    _audit_imports(
        entries,
        requirements,
        kind,
        runtime_names=runtime_names,
    )
    if kind == "qwen" and assets_manifest is None:
        raise BundleError("assets_manifest_required", "Qwen bundle requires assets_manifest")
    return entries


_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "api_token",
        "auth_token",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_key",
        "token",
    }
)


def _reject_secret_material(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
            normalized = re.sub(
                r"[^a-z0-9]+", "_", separated.casefold()
            ).strip("_")
            components = frozenset(normalized.split("_"))
            if (
                components
                & {"credential", "credentials", "passwd", "password", "secret", "token"}
                or normalized in _SECRET_FIELD_NAMES
                or any(
                normalized.endswith(marker) for marker in _SECRET_FIELD_NAMES
                )
            ):
                raise BundleError(
                    "assets_manifest_invalid",
                    f"secret-bearing asset field is forbidden: {key}",
                )
            _reject_secret_material(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_material(item)
    elif isinstance(value, str):
        folded = value.strip().casefold()
        if (
            folded.startswith(("sk-", "ghp_", "github_pat_"))
            or "-----begin private key-----" in folded
            or "-----begin rsa private key-----" in folded
            or "-----begin openssh private key-----" in folded
        ):
            raise BundleError(
                "assets_manifest_invalid",
                "secret-bearing asset value is forbidden",
            )


def _validated_logical_identity(value: str, name: str) -> str:
    stripped = value.strip()
    parts = tuple(part for part in re.split(r"[\\/]", stripped) if part)
    embedded_absolute = re.search(
        r"(?:^|[^A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)",
        stripped,
    ) or re.search(r"(?:^|[\s=:'\"(\[])/(?!/)", stripped)
    if (
        not stripped
        or _DRIVE_PREFIX.match(stripped)
        or stripped.startswith(("/", "\\"))
        or embedded_absolute is not None
        or ".." in parts
        or any(ord(character) < 32 for character in stripped)
    ):
        raise BundleError(
            "assets_manifest_invalid",
            f"asset {name} identity contains a filesystem path",
        )
    return stripped


def _sanitized_external_assets(path: Path | str) -> bytes:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError("assets_manifest_invalid", "cannot read assets manifest") from error
    _reject_secret_material(raw)
    if isinstance(raw, Mapping) and isinstance(raw.get("assets"), Mapping):
        raw = raw["assets"]
    if not isinstance(raw, Mapping) or set(raw) != set(_ASSET_NAMES):
        raise BundleError("assets_manifest_invalid", "assets manifest must declare all logical assets")
    sanitized: dict[str, dict[str, Any]] = {}
    for name in _ASSET_NAMES:
        value = raw[name]
        if not isinstance(value, Mapping):
            raise BundleError("assets_manifest_invalid", f"asset {name} must be a mapping")
        expected_argument = "--" + name.replace("_", "-")
        if value.get("argument") != expected_argument:
            raise BundleError("assets_manifest_invalid", f"asset {name} argument mismatch")
        kind = value.get("kind")
        identity = value.get("identity")
        size = value.get("size_bytes")
        if kind not in {"file", "directory"}:
            raise BundleError("assets_manifest_invalid", f"asset {name} kind is invalid")
        if type(identity) is not str or not identity:
            raise BundleError("assets_manifest_invalid", f"asset {name} identity is invalid")
        identity = _validated_logical_identity(identity, name)
        if type(size) is not int or size < 0:
            raise BundleError("assets_manifest_invalid", f"asset {name} size is invalid")
        output = {
            "argument": expected_argument,
            "expected_identity": identity,
            "expected_kind": kind,
            "expected_size_bytes": size,
        }
        for digest_name in ("sha256", "tree_sha256"):
            digest = value.get(digest_name)
            if digest is not None:
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise BundleError("assets_manifest_invalid", f"asset {name} digest is invalid")
                output[digest_name] = digest
        sanitized[name] = output
    return _canonical_json_bytes({"schema_version": BUNDLE_SCHEMA_VERSION, "assets": sanitized})


def _git_provenance(root: Path) -> dict[str, Any]:
    revision = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _run_git(root, "diff", "--no-ext-diff", "--binary", "HEAD", "--", ".")
    assert isinstance(revision, str) and isinstance(status, str) and isinstance(diff, str)
    normalized = {
        "diff": diff.replace("\r\n", "\n").replace("\r", "\n"),
        "status": sorted(status.replace("\r\n", "\n").splitlines()),
    }
    dirty = bool(normalized["status"])
    return {
        "revision": revision.strip(),
        "dirty": dirty,
        "diff_sha256": (
            hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
            if dirty
            else "0" * 64
        ),
    }


def _entry_metadata(entry: BundleEntry) -> dict[str, Any]:
    return {
        "mode": entry.mode,
        "sha256": hashlib.sha256(entry.data).hexdigest(),
        "size": len(entry.data),
    }


def plan_bundle(
    *,
    kind: str,
    repo_root: Path | str,
    config_path: Path | str | None = None,
    assets_manifest: Path | str | None = None,
) -> BundlePlan:
    """Create a canonical complete archive plan without writing output."""

    root = Path(repo_root).resolve(strict=True)
    selected_config = (
        root / PurePosixPath(_MAXIMUM_FLAGSHIPS[kind])
        if kind in _MAXIMUM_KINDS
        else Path(config_path) if config_path is not None else None
    )
    if selected_config is None:
        raise BundleError("config_required", "bundle config is required")
    config, config_name = _repo_relative(root, selected_config)
    payload = list(
        collect_bundle_files(
            kind=kind,
            repo_root=root,
            config_path=config,
            assets_manifest=assets_manifest,
        )
    )
    if kind in _MAXIMUM_KINDS:
        from .qwen_hybrid_math import DEFERRED_FUSION_WARNING, REFERENCE_IMPLEMENTATION
        package = kind[-1].upper()
        flagship = PurePosixPath(_MAXIMUM_FLAGSHIPS[kind]).stem
        flagship = re.sub(r"^[0-9]+-", "", flagship)
        payload.append(
            BundleEntry(
                "PACKAGE.json",
                _canonical_json_bytes(
                    {
                        "schema_version": BUNDLE_SCHEMA_VERSION,
                        "kind": kind,
                        "package": package,
                        "flagship": flagship,
                        "default_controls": list(_MAXIMUM_DEFAULT_CONTROLS[kind]),
                        "implementation": REFERENCE_IMPLEMENTATION,
                        "performance_warning": DEFERRED_FUSION_WARNING,
                    }
                ),
            )
        )
    if kind == "qwen":
        assert assets_manifest is not None
        payload.append(BundleEntry("external-assets.json", _sanitized_external_assets(assets_manifest)))
    payload.append(BundleEntry("verify_bundle.py", VERIFY_BUNDLE_SOURCE, 0o755))
    payload_entries = _validated_entries(payload)
    try:
        config_document = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError("config_invalid", "bundle config is not valid UTF-8 JSON") from error
    if kind not in _MAXIMUM_KINDS and (
        not isinstance(config_document, Mapping) or config_document.get("backend") != kind
    ):
        raise BundleError("config_invalid", "bundle config backend does not match kind")
    if kind in _MAXIMUM_KINDS:
        campaign_document = json.loads(
            (root / "research/kmd2_ablation/campaigns/qwen08b-maximum-hybrid-v1.json").read_text(encoding="utf-8")
        )
        suite_version = campaign_document.get("schema_version")
    else:
        suite_version = config_document.get("suite_version")
    if type(suite_version) is not str or not suite_version:
        raise BundleError("config_invalid", "bundle config suite_version is missing")
    source_hashes = {
        entry.name: hashlib.sha256(entry.data).hexdigest()
        for entry in payload_entries
        if entry.name.endswith(".py")
        and not entry.name.startswith("tests/")
        and entry.name != "verify_bundle.py"
    }
    production_source_sha256 = hashlib.sha256(
        _canonical_json_bytes(source_hashes)
    ).hexdigest()
    config_argument = config_name
    base_command = [
        "python",
        "-m",
        "research.kmd2_ablation.run_ablation",
    ]
    if kind == "tiny":
        smoke = base_command + [
            "run",
            "--backend",
            "tiny",
            "--config",
            config_argument,
            "--out",
            "results",
            "--job-index",
            "0",
            "--num-jobs",
            "1",
        ]
    else:
        smoke = base_command + [
            "preflight",
            "--backend",
            "qwen",
            "--config",
            config_argument,
            "--out",
            "results",
            "--dry-run",
        ]
    entries_metadata = {
        entry.name: _entry_metadata(entry) for entry in payload_entries
    }
    expected_members = sorted([*entries_metadata, "MANIFEST.json"])
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "suite_version": suite_version,
        "kind": kind,
        "git": _git_provenance(root),
        "config": config_name,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "production_source_sha256": production_source_sha256,
        "entries": entries_metadata,
        "expected_members": expected_members,
        "provenance": {
            "build_command": "python -m research.kmd2_ablation.run_ablation bundle",
            "smoke_command": smoke,
        },
        "smoke": {"command": smoke},
        "manifest_convention": _MANIFEST_CONVENTION,
    }
    entries = _validated_entries(
        (*payload_entries, BundleEntry("MANIFEST.json", _canonical_json_bytes(manifest)))
    )
    return BundlePlan(kind=kind, entries=entries, manifest=manifest)


def verify_bundle(
    archive: Path | str,
    *,
    extract_to: Path | str | None = None,
) -> BundleVerificationResult:
    """Reopen, fully verify, and optionally atomically extract one bundle."""

    archive_path = Path(archive)
    extraction_path = None if extract_to is None else Path(extract_to)
    raw = _VERIFY_ARCHIVE(
        archive_path,
        extraction_path,
    )
    if not isinstance(raw, Mapping):
        raise BundleError("verification_internal_error", "verification returned no report")
    ok = raw.get("ok")
    codes = raw.get("codes")
    sha256 = raw.get("sha256")
    member_count = raw.get("member_count")
    extracted = raw.get("extracted_to")
    if (
        type(ok) is not bool
        or type(codes) is not list
        or any(type(code) is not str for code in codes)
        or type(sha256) is not str
        or type(member_count) is not int
        or (extracted is not None and type(extracted) is not str)
    ):
        raise BundleError(
            "verification_internal_error",
            "verification returned an invalid report",
        )
    return BundleVerificationResult(
        ok=ok,
        codes=tuple(codes),
        sha256=sha256,
        member_count=member_count,
        extracted_to=None if extracted is None else Path(extracted),
    )


def _publish_verifier_sidecar(destination: Path) -> str:
    descriptor: int | None = None
    candidate: Path | None = None
    try:
        descriptor, candidate_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        candidate = Path(candidate_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(VERIFY_BUNDLE_SOURCE)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(candidate, 0o755)
        if candidate.read_bytes() != VERIFY_BUNDLE_SOURCE:
            raise BundleError(
                "verifier_publish_failed",
                "standalone verifier candidate failed byte verification",
            )
        digest = _sha256_file(candidate)
        os.replace(candidate, destination)
        candidate = None
        return digest
    except BundleError:
        raise
    except OSError as error:
        raise BundleError(
            "verifier_publish_failed",
            "standalone verifier could not be published atomically",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if candidate is not None:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def build_bundle(plan: BundlePlan, destination: Path | str) -> BundleResult:
    """Write, verify, and atomically publish one planned archive."""

    if not isinstance(plan, BundlePlan):
        raise TypeError("plan must be a BundlePlan")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    verifier_path = output.with_name(VERIFIER_SIDECAR_NAME)
    if os.path.normcase(os.path.abspath(output)) == os.path.normcase(
        os.path.abspath(verifier_path)
    ):
        raise BundleError(
            "verifier_sidecar_collision",
            "archive destination collides with the standalone verifier sidecar",
        )
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    candidate = Path(candidate_name)
    try:
        written_digest = write_deterministic_zip(plan.entries, candidate)
        verification = verify_bundle(candidate)
        if (
            not verification.ok
            or verification.sha256 != written_digest
            or verification.member_count != len(plan.entries)
        ):
            detail = ",".join(verification.codes) or "identity_mismatch"
            raise BundleError(
                "bundle_verification_failed",
                f"verification failed before publication: {detail}",
            )
        verifier_sha256 = _publish_verifier_sidecar(verifier_path)
        try:
            os.replace(candidate, output)
        except OSError as error:
            raise BundleError(
                "bundle_publish_failed",
                "verified bundle could not be published atomically",
            ) from error
        return BundleResult(
            kind=plan.kind,
            path=output,
            sha256=verification.sha256,
            member_count=verification.member_count,
            verifier_path=verifier_path,
            verifier_sha256=verifier_sha256,
        )
    finally:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def cli_handler(arguments: Any) -> dict[str, Any]:
    """Build one verified bundle and return a path-sanitized JSON report."""

    kind = getattr(arguments, "backend", None)
    maximum_package = getattr(arguments, "maximum_package", None)
    if maximum_package in {"A", "B"}:
        if kind != "qwen":
            raise BundleError("bundle_kind_invalid", "maximum packages require --backend qwen")
        kind = f"maximum-hybrid-{maximum_package.casefold()}"
    config = getattr(arguments, "config", None)
    destination = Path(getattr(arguments, "out", ""))
    assets_manifest = getattr(arguments, "assets_manifest", None)
    repo_root = getattr(arguments, "repo_root", None)
    root = Path.cwd() if repo_root is None else Path(repo_root)
    try:
        plan = plan_bundle(
            kind=kind,
            repo_root=root,
            config_path=config,
            assets_manifest=assets_manifest,
        )
        result = build_bundle(plan, destination)
    except BundleError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise BundleError(
            "bundle_io_error",
            "bundle operation failed without publishing output",
        ) from error
    return {
        "ok": True,
        "codes": [],
        "warnings": [],
        "backend": result.kind,
        "archive": result.path.name,
        "sha256": result.sha256,
        "member_count": result.member_count,
        "verifier": result.verifier_path.name,
        "verifier_sha256": result.verifier_sha256,
    }


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "VERIFIER_SIDECAR_NAME",
    "VERIFY_BUNDLE_SOURCE",
    "ZIP_EPOCH",
    "BundleEntry",
    "BundleError",
    "BundlePlan",
    "BundleResult",
    "BundleVerificationResult",
    "build_bundle",
    "cli_handler",
    "collect_bundle_files",
    "plan_bundle",
    "verify_bundle",
    "write_deterministic_zip",
    "validate_qwen_lock",
]
