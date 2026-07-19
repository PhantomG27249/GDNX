"""Portable JSON command line entry point for the KMD-2 ablation suite.

The parser is intentionally standard-library only.  Scientific backends and
optional dependencies are imported only after a command has been selected.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


Handler = Callable[[argparse.Namespace], Mapping[str, Any]]

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PREFLIGHT = 3
EXIT_EXECUTION = 4
EXIT_SUMMARY = 5
EXIT_BUNDLE = 6

_FAILURE_EXIT = {
    "preflight": EXIT_PREFLIGHT,
    "run": EXIT_EXECUTION,
    "summarize": EXIT_SUMMARY,
    "bundle": EXIT_BUNDLE,
}
_PRODUCTION_HANDLERS = {
    "preflight": ("research.kmd2_ablation.runner", "preflight_command"),
    "run": ("research.kmd2_ablation.runner", "run_command"),
    "summarize": ("research.kmd2_ablation.summarize", "cli_handler"),
    "bundle": ("research.kmd2_ablation.bundle", "cli_handler"),
}
MAXIMUM_HYBRID_CONTROL_IDS = (
    "gdn2-r1", "gdn2-mimo-r2", "gdn2-mimo-r4", "package-a-native-decay",
    "package-a-braid-no-cache", "package-a-recency-w64", "package-a-hola-w64",
    "package-b-recency-w64", "package-b-hola-w64", "shared-query-widening", "stock-qwen",
)


def _path(value: str) -> Path:
    return Path(value)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", required=True, choices=("tiny", "qwen"))
    parser.add_argument("--config", required=True, type=_path, metavar="PATH")
    parser.add_argument("--out", required=True, type=_path, metavar="PATH")
    parser.add_argument("--job-index", type=int, default=0, metavar="N")
    parser.add_argument("--num-jobs", type=int, default=1, metavar="N")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume valid completed jobs (default: enabled)",
    )


def _add_qwen_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--maximum-control",
        choices=MAXIMUM_HYBRID_CONTROL_IDS,
        help="frozen eleven-control campaign identity (the config remains authoritative)",
    )
    parser.add_argument(
        "--mode",
        choices=("reliance", "heal", "initial_exact_cache"),
    )
    parser.add_argument("--model", type=_path, metavar="PATH")
    parser.add_argument("--tokenizer", type=_path, metavar="PATH")
    parser.add_argument(
        "--checkpoint",
        "--native-checkpoint",
        dest="checkpoint",
        type=_path,
        metavar="PATH",
    )
    parser.add_argument("--data", type=_path, metavar="PATH")
    parser.add_argument("--teacher-model", type=_path, metavar="PATH")
    parser.add_argument("--student-device", "--device", dest="student_device")
    parser.add_argument("--teacher-device")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--checkpoint-every", type=int, metavar="N")
    parser.add_argument("--model-sha256")
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--data-sha256")
    parser.add_argument("--teacher-model-sha256")
    parser.add_argument(
        "--assets-manifest",
        "--asset-manifest",
        dest="assets_manifest",
        type=_path,
        metavar="PATH",
    )
    parser.add_argument("--repo-root", type=_path, metavar="PATH")


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser without importing any execution modules."""

    parser = argparse.ArgumentParser(prog="python -m research.kmd2_ablation.run_ablation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "summarize", "bundle"):
        subparser = subparsers.add_parser(command)
        _add_common_arguments(subparser)
        _add_qwen_arguments(subparser)
        if command == "bundle":
            subparser.add_argument(
                "--maximum-package",
                choices=("A", "B"),
                help="build the complete maximum-hybrid Package A or B archive",
            )
        if command == "preflight":
            subparser.add_argument("--dry-run", action="store_true")
    return parser


def _load_production_handler(command: str) -> Handler:
    module_name, attribute = _PRODUCTION_HANDLERS[command]
    module = importlib.import_module(module_name)
    handler = getattr(module, attribute, None)
    if not callable(handler):
        raise RuntimeError(
            f"production handler {module_name}.{attribute} is unavailable"
        )
    return handler


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _error_payload(command: str, error: BaseException) -> dict[str, Any]:
    code = getattr(error, "code", None)
    if type(code) is not str or not code:
        code = f"{command}_handler_error"
    return {
        "ok": False,
        "codes": [code],
        "warnings": [],
        "error": str(error) or type(error).__name__,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Handler] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Parse, dispatch, emit one canonical JSON document, and return an exit code."""

    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            options = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as exit_signal:
        return int(exit_signal.code)

    command = options.command
    try:
        # Optional ML libraries occasionally emit diagnostics to stdout during
        # import. Keep stdout machine-readable by routing all incidental handler
        # output to stderr; this CLI emits the sole JSON document below.
        with contextlib.redirect_stdout(errors), contextlib.redirect_stderr(errors):
            handler = (
                handlers.get(command)
                if handlers is not None
                else _load_production_handler(command)
            )
            if not callable(handler):
                raise RuntimeError(f"no handler registered for {command}")
            raw_report = handler(options)
        if not isinstance(raw_report, Mapping):
            raise TypeError("command handler must return a mapping")
        report = {
            key: value
            for key, value in raw_report.items()
            if type(key) is str and not key.startswith("_")
        }
        if type(report.get("ok")) is not bool:
            raise TypeError("command report must contain a bool ok field")
        requested_exit = raw_report.get("_exit_code")
        if type(requested_exit) is int and requested_exit in {
            EXIT_USAGE,
            EXIT_PREFLIGHT,
            EXIT_EXECUTION,
            EXIT_SUMMARY,
            EXIT_BUNDLE,
        }:
            exit_code = requested_exit
        else:
            exit_code = EXIT_OK if report["ok"] else _FAILURE_EXIT[command]
    except Exception as error:
        report = _error_payload(command, error)
        requested_exit = getattr(error, "exit_code", None)
        exit_code = (
            requested_exit
            if type(requested_exit) is int
            else _FAILURE_EXIT[command]
        )

    try:
        output.write(_json_line(report))
        output.flush()
    except (TypeError, ValueError) as error:
        fallback = _error_payload(command, error)
        output.write(_json_line(fallback))
        output.flush()
        return _FAILURE_EXIT[command]
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_BUNDLE",
    "EXIT_EXECUTION",
    "EXIT_OK",
    "EXIT_PREFLIGHT",
    "EXIT_SUMMARY",
    "EXIT_USAGE",
    "build_parser",
    "main",
]
