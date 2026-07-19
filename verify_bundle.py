#!/usr/bin/env python3
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
