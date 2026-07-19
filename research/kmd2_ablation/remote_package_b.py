"""Offline, fail-closed launcher for the maximum Package B campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import posixpath
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib
from typing import Any, Callable


_ASSETS = ("model", "tokenizer", "native_checkpoint", "data", "teacher_model")
_DEPENDENCIES = ("torch", "transformers", "safetensors", "triton", "fla")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _asset_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"asset may not be a symlink: {path}")
    if path.is_file():
        digest, size = _sha256_file(path)
        return {"kind": "file", "path": str(path.resolve()), "sha256": digest, "size_bytes": size}
    if not path.is_dir():
        raise ValueError(f"asset is not a regular file or directory: {path}")
    digest = hashlib.sha256()
    size = 0
    for member in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if member.is_symlink():
            raise ValueError(f"asset tree may not contain symlinks: {member}")
        if not member.is_file():
            continue
        relative = member.relative_to(path).as_posix()
        file_digest, file_size = _sha256_file(member)
        digest.update(relative.encode("utf-8") + b"\0" + file_digest.encode("ascii") + b"\0")
        size += file_size
    return {"kind": "directory", "path": str(path.resolve()), "tree_sha256": digest.hexdigest(), "size_bytes": size}


def _load(path: Path) -> dict[str, Any]:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"paths", "run"} or not isinstance(document["paths"], dict) or not isinstance(document["run"], dict):
        raise ValueError("config must contain [paths] and [run]")
    required_paths = {"package_root", "cache", "hf_cache", "hf_hub", "output", *_ASSETS}
    required_run = {
        "package", "campaign", "dtype", "student_device", "teacher_device",
        "checkpoint_every",
    }
    if set(document["paths"]) != required_paths or set(document["run"]) != required_run:
        raise ValueError("config has missing or unexpected fields")
    if any(not isinstance(value, str) or not value for value in document["paths"].values()):
        raise ValueError("all path config values must be non-empty strings")
    string_run_fields = required_run - {"checkpoint_every"}
    if any(
        not isinstance(document["run"][field], str) or not document["run"][field]
        for field in string_run_fields
    ):
        raise ValueError("all string run config values must be non-empty")
    if (
        type(document["run"]["checkpoint_every"]) is not int
        or document["run"]["checkpoint_every"] < 1
    ):
        raise ValueError("checkpoint_every must be a positive integer")
    if document["run"]["package"] != "B" or document["run"]["campaign"] != "full":
        raise ValueError("only the Package B full campaign is supported")
    if document["run"]["dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("dtype must be bfloat16 or float32")
    return document


def _command(config: dict[str, Any], manifest: str) -> list[str]:
    paths, run = config["paths"], config["run"]
    launcher = posixpath.join(paths["package_root"], "research/kmd2_ablation/scripts/run_remote_qwen_maximum_hybrids.sh")
    return [
        "bash", launcher, "--assets-manifest", manifest, "--out", paths["output"],
        "--package", "B", "--student-device", run["student_device"],
        "--teacher-device", run["teacher_device"], "--dtype", run["dtype"],
        "--checkpoint-every", str(run["checkpoint_every"]),
    ]


def _validate_runtime(config: dict[str, Any]) -> None:
    missing = []
    for module in _DEPENDENCIES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError("missing Python dependencies: " + ", ".join(missing))
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    for field in ("student_device", "teacher_device"):
        device = config["run"][field]
        if not device.startswith("cuda:") or not device[5:].isdigit() or int(device[5:]) >= torch.cuda.device_count():
            raise RuntimeError(f"invalid CUDA device for {field}: {device}")


def _apply_environment(config: dict[str, Any]) -> None:
    os.environ.update({"XDG_CACHE_HOME": config["paths"]["cache"], "HF_HOME": config["paths"]["hf_cache"], "HUGGINGFACE_HUB_CACHE": config["paths"]["hf_hub"], "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
    runtime_validator: Callable[[dict[str, Any]], None] = _validate_runtime,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = _load(args.config)
    _apply_environment(config)
    manifest = posixpath.join(config["paths"]["output"], ".generated/remote-package-b-assets.json")
    command = _command(config, manifest)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    runtime_validator(config)
    package_root = Path(config["paths"]["package_root"])
    if not package_root.is_dir() or not Path(command[1]).is_file():
        raise RuntimeError(f"invalid package root: {package_root}")
    assets = {}
    arguments = {"native_checkpoint": "--native-checkpoint", "teacher_model": "--teacher-model"}
    for name in _ASSETS:
        identity = _asset_identity(Path(config["paths"][name]))
        identity["argument"] = arguments.get(name, "--" + name.replace("_", "-"))
        assets["checkpoint" if name == "native_checkpoint" else name] = identity
    manifest_path = Path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"schema_version": "1.0.0", "assets": assets}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return runner(command, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"remote Package B preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2)
