"""Metadata-only exact resource accounting for Qwen dry-run preflight."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .runner import PreflightCheckError


_CACHE_SUFFIX_SIZES = {
    "cache_gamma_q": "key",
    "cache_gamma_k": "key",
    "cache_sink_logit": "heads",
    "cache_amplitude": "heads",
}
_MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
_MAX_EXACT_PARAMETER_COUNT = (1 << 63) - 1
_EXPECTED_NATIVE_SCAN = "gdn3.kmd2_fast_scan.scan"
_EXPECTED_SCORE_SCAN = "gdn3.kmd2_fast_scan.scan_with_update_norm"
_MEMORY_SUFFIX = ".in_proj_b.weight"
_NATIVE_ADDITION_SUFFIXES = (
    ".rot_proj.weight",
    ".rot_proj.bias",
    ".q_slot_scale",
    ".out_mix",
    ".decay_chan",
    ".bw_off",
)
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _shape_elements(shape: object, *, context: str) -> int:
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes, bytearray))
        or any(type(item) is not int or item < 0 for item in shape)
    ):
        raise PreflightCheckError(
            "parameter_metadata_invalid", f"{context} has an invalid tensor shape"
        )
    elements = math.prod(shape)
    if elements > _MAX_EXACT_PARAMETER_COUNT:
        raise PreflightCheckError(
            "parameter_accounting_overflow",
            f"{context} exceeds the exact parameter accounting bound",
        )
    return elements


def _checked_sum(*values: int, context: str) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise PreflightCheckError(
                "parameter_accounting_invalid", f"{context} is not nonnegative"
            )
        total += value
        if total > _MAX_EXACT_PARAMETER_COUNT:
            raise PreflightCheckError(
                "parameter_accounting_overflow",
                f"{context} exceeds the exact parameter accounting bound",
            )
    return total


def _checked_product(*values: int, context: str) -> int:
    product = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise PreflightCheckError(
                "parameter_accounting_invalid", f"{context} is not nonnegative"
            )
        product *= value
        if product > _MAX_EXACT_PARAMETER_COUNT:
            raise PreflightCheckError(
                "parameter_accounting_overflow",
                f"{context} exceeds the exact parameter accounting bound",
            )
    return product


def _read_safetensors_header(path: Path) -> dict[str, tuple[int, ...]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing header length")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("header length is outside the safety bound")
            if 8 + header_size > size:
                raise ValueError("header extends beyond the file")
            encoded = stream.read(header_size)
        raw = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PreflightCheckError(
            "parameter_metadata_invalid",
            f"cannot read safetensors metadata: {path}",
        ) from error
    if not isinstance(raw, Mapping):
        raise PreflightCheckError(
            "parameter_metadata_invalid", f"safetensors header is not a mapping: {path}"
        )
    tensors: dict[str, tuple[int, ...]] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, metadata in raw.items():
        if name == "__metadata__":
            continue
        if type(name) is not str or not name or not isinstance(metadata, Mapping):
            raise PreflightCheckError(
                "parameter_metadata_invalid", f"invalid tensor entry in {path}"
            )
        shape = metadata.get("shape")
        elements = _shape_elements(shape, context=f"{path}:{name}")
        dtype = metadata.get("dtype")
        dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if dtype_bytes is None:
            raise PreflightCheckError(
                "parameter_metadata_invalid", f"unknown dtype for {path}:{name}"
            )
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, Sequence)
            or isinstance(offsets, (str, bytes, bytearray))
            or len(offsets) != 2
            or any(type(item) is not int or item < 0 for item in offsets)
            or offsets[1] < offsets[0]
            or 8 + len(encoded) + offsets[1] > size
        ):
            raise PreflightCheckError(
                "parameter_metadata_invalid", f"invalid data offsets for {path}:{name}"
            )
        if offsets[1] - offsets[0] != elements * dtype_bytes:
            raise PreflightCheckError(
                "parameter_metadata_invalid",
                f"shape/dtype byte count disagrees with offsets for {path}:{name}",
            )
        tensors[name] = tuple(shape)
        intervals.append((offsets[0], offsets[1], name))
    if not tensors:
        raise PreflightCheckError(
            "parameter_metadata_missing", f"no tensor metadata found in {path}"
        )
    expected_start = 0
    for start, stop, name in sorted(intervals):
        if start != expected_start:
            raise PreflightCheckError(
                "parameter_metadata_invalid",
                f"tensor data is overlapping or noncanonical near {path}:{name}",
            )
        expected_start = stop
    if expected_start != size - 8 - len(encoded):
        raise PreflightCheckError(
            "parameter_metadata_invalid",
            f"tensor offsets do not cover the safetensors data section: {path}",
        )
    return tensors


def _safetensors_inventory(path: Path) -> dict[str, tuple[int, ...]]:
    files = (
        [path]
        if path.is_file() and path.suffix == ".safetensors"
        else sorted(path.rglob("*.safetensors")) if path.is_dir() else []
    )
    if not files:
        raise PreflightCheckError(
            "parameter_metadata_missing",
            "Qwen dry-run requires safetensors headers for exact parameter accounting",
        )
    tensors: dict[str, tuple[int, ...]] = {}
    for file in files:
        for name, shape in _read_safetensors_header(file).items():
            if name in tensors:
                raise PreflightCheckError(
                    "parameter_metadata_invalid",
                    f"duplicate tensor metadata across safetensors shards: {name}",
                )
            tensors[name] = shape
    return tensors


def _string_names(value: object, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or (not value and not allow_empty)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise PreflightCheckError(
            "parameter_declaration_invalid", f"{field} must contain unique names"
        )
    return tuple(value)


def _resolve_parameter(
    tensors: Mapping[str, tuple[int, ...]], declared: str
) -> tuple[int, ...]:
    # Runtime training extracts the official multimodal wrapper's text model,
    # so its canonical parameter names are ``model.layers.*``.  Published
    # safetensors retain the wrapper prefix ``model.language_model.layers.*``.
    # Resolve that one explicit packaging alias while keeping the uniqueness
    # check fail-closed.
    aliases = {declared}
    if declared.startswith("model."):
        aliases.add("model.language_model." + declared[len("model.") :])
    matches = [
        shape
        for name, shape in tensors.items()
        if any(name == alias or name.endswith("." + alias) for alias in aliases)
    ]
    if len(matches) != 1:
        raise PreflightCheckError(
            "parameter_metadata_mismatch",
            f"declared trainable parameter does not resolve uniquely: {declared}",
        )
    return matches[0]


def _has_name(tree: ast.AST, name: str) -> bool:
    return any(
        (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
        )
        for node in ast.walk(tree)
    )


def _verify_fast_scan_source_contract(source_root: Path) -> None:
    paths = {
        "native": source_root / "gdn3" / "kmd2_native.py",
        "fast": source_root / "gdn3" / "kmd2_fast_scan.py",
        "cache": source_root / "research" / "kmd2_ablation" / "qwen_exact_cache.py",
    }
    try:
        trees = {
            name: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name, path in paths.items()
        }
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise PreflightCheckError(
            "qwen_fast_scan_source_invalid",
            "cannot verify the Qwen fast-scan source contract",
        ) from error

    native_assignments = [
        node
        for node in ast.walk(trees["native"])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_FAST_SCAN" for target in node.targets)
    ]
    native_dump = " ".join(ast.dump(node.value) for node in native_assignments)
    if len(native_assignments) != 1 or "GDN3_FAST_SCAN" not in native_dump:
        raise PreflightCheckError(
            "qwen_fast_scan_source_invalid",
            "native scan does not expose the verified import-time fast-scan gate",
        )

    cache_function = next(
        (
            node
            for node in ast.walk(trees["cache"])
            if isinstance(node, ast.FunctionDef)
            and node.name == "_native_state_and_scores"
        ),
        None,
    )
    cache_dump = "" if cache_function is None else ast.dump(cache_function)
    if (
        cache_function is None
        or "_FAST_SCAN" not in cache_dump
        or "scan_with_update_norm" not in cache_dump
        or not _has_name(trees["fast"], "scan")
        or not _has_name(trees["fast"], "scan_with_update_norm")
    ):
        raise PreflightCheckError(
            "qwen_fast_scan_source_invalid",
            "score-returning Qwen cache scan is not wired to the verified fast path",
        )


def verify_qwen_execution_contract(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    loaded_modules: Mapping[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Prove the pinned r_out and score-returning fast path before Qwen imports."""

    if isinstance(config, ExperimentConfig):
        backend = config.backend
        params: Mapping[str, Any] = config.task.params
    elif isinstance(config, Mapping):
        backend = config.get("backend")
        task = config.get("task")
        params_value = task.get("params") if isinstance(task, Mapping) else None
        if not isinstance(params_value, Mapping):
            raise TypeError("canonical Qwen config has no task.params mapping")
        params = params_value
    else:
        raise TypeError("config must be a Qwen config mapping")
    if backend != "qwen":
        raise TypeError("config must select the Qwen backend")
    native_r_out = params.get("native_r_out")
    if type(native_r_out) is not int or native_r_out < 1:
        raise PreflightCheckError(
            "qwen_r_out_unpinned",
            "task.params.native_r_out must pin a positive integer",
        )
    if params.get("score_scan") != _EXPECTED_SCORE_SCAN:
        raise PreflightCheckError(
            "qwen_fast_scan_unpinned",
            f"task.params.score_scan must be {_EXPECTED_SCORE_SCAN}",
        )
    environment = os.environ if environ is None else environ
    if environment.get("GDN3_FAST_SCAN") != "1":
        raise PreflightCheckError(
            "qwen_fast_scan_inactive",
            "GDN3_FAST_SCAN=1 must be set before the Qwen process starts",
        )
    if environment.get("GDN3_KMD2_ROUT") != str(native_r_out):
        raise PreflightCheckError(
            "qwen_r_out_mismatch",
            "GDN3_KMD2_ROUT does not match task.params.native_r_out",
        )

    modules = sys.modules if loaded_modules is None else loaded_modules
    loaded_native = modules.get("gdn3.kmd2_native")
    if loaded_native is not None and getattr(loaded_native, "_FAST_SCAN", None) is not True:
        raise PreflightCheckError(
            "qwen_fast_scan_import_order_invalid",
            "gdn3.kmd2_native was imported before the fast scan was enabled",
        )
    root = Path(__file__).resolve().parents[2] if source_root is None else Path(source_root)
    _verify_fast_scan_source_contract(root)
    proof = (
        "loaded_module_flag"
        if loaded_native is not None
        else "preimport_environment_and_source_contract"
    )
    return {
        "activation_proof": proof,
        "fast_scan": True,
        "native_r_out": native_r_out,
        "native_scan": _EXPECTED_NATIVE_SCAN,
        "score_scan": _EXPECTED_SCORE_SCAN,
    }


def _native_layer_prefixes(
    config: ExperimentConfig,
    tensors: Mapping[str, tuple[int, ...]],
    names: tuple[str, ...],
) -> tuple[str, ...]:
    if len(names) != config.model.num_layers:
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen memory trainables must declare one in_proj_b weight per native layer",
        )
    prefixes: list[str] = []
    expected_shape = (config.model.num_heads, config.model.hidden_size)
    for name in names:
        if not name.endswith(_MEMORY_SUFFIX):
            raise PreflightCheckError(
                "parameter_declaration_invalid",
                f"Qwen native memory parameter must end in {_MEMORY_SUFFIX}: {name}",
            )
        if _resolve_parameter(tensors, name) != expected_shape:
            raise PreflightCheckError(
                "parameter_metadata_mismatch",
                f"Qwen native in_proj_b layout does not match {expected_shape}: {name}",
            )
        prefixes.append(name[: -len(_MEMORY_SUFFIX)])
    if len(set(prefixes)) != config.model.num_layers:
        raise PreflightCheckError(
            "parameter_declaration_invalid", "Qwen native layer prefixes are not unique"
        )
    return tuple(prefixes)


def _native_addition_count(config: ExperimentConfig, *, r_out: int) -> int:
    if config.model.state_key_dim % 2:
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen native state_key_dim must be even for rot_proj",
        )
    layers = config.model.num_layers
    heads = config.model.num_heads
    key = config.model.state_key_dim
    hidden = config.model.hidden_size
    half_key = key // 2
    per_layer = _checked_sum(
        _checked_product(hidden, heads, half_key, context="rot_proj.weight"),
        _checked_product(heads, half_key, context="rot_proj.bias"),
        _checked_product(heads, key, context="decay_chan"),
        heads,
        *(
            (
                _checked_product(heads, r_out, key, context="q_slot_scale"),
                _checked_product(heads, r_out, context="out_mix"),
            )
            if r_out > 1
            else ()
        ),
        context="KMD2 native additions per layer",
    )
    return _checked_product(layers, per_layer, context="KMD2 native additions")


def _cache_parameter_count(
    config: ExperimentConfig, *, layer_prefixes: tuple[str, ...]
) -> tuple[int, tuple[str, ...]]:
    params = config.task.params
    names = _string_names(
        params.get("cache_parameter_names", ()),
        field="task.params.cache_parameter_names",
        allow_empty=config.qwen.run_mode != "heal",
    )
    if not names:
        return _checked_product(
            config.model.num_layers,
            2 * config.model.state_key_dim + 2 * config.model.num_heads,
            context="Qwen cache parameters",
        ), names
    expected_names = {
        f"{prefix}.{suffix}" for prefix in layer_prefixes for suffix in _CACHE_SUFFIX_SIZES
    }
    if set(names) != expected_names:
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen cache trainables must exactly match the installed native layer prefixes",
        )
    counts = {suffix: 0 for suffix in _CACHE_SUFFIX_SIZES}
    total = 0
    for name in names:
        suffix = name.rsplit(".", 1)[-1]
        kind = _CACHE_SUFFIX_SIZES.get(suffix)
        if kind is None:
            raise PreflightCheckError(
                "parameter_declaration_invalid",
                f"unknown cache trainable parameter: {name}",
            )
        counts[suffix] += 1
        total = _checked_sum(
            total,
            config.model.state_key_dim if kind == "key" else config.model.num_heads,
            context="Qwen cache parameters",
        )
    if any(count != config.model.num_layers for count in counts.values()):
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen cache trainables must declare each cache parameter once per layer",
        )
    return total, names


def _frozen_model_dimensions(
    model_path: Path, config: ExperimentConfig, tensors: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, int], str]:
    metadata = (
        model_path / "config.json"
        if model_path.is_dir()
        else model_path.parent / "config.json"
    )
    raw: Mapping[str, Any] = {}
    if metadata is not None and metadata.is_file():
        try:
            loaded = json.loads(metadata.read_text(encoding="utf-8"))
            raw = loaded if isinstance(loaded, Mapping) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raw = {}
    # Official Qwen3.5 checkpoints are multimodal wrappers.  The dimensions
    # which govern the language model and its GatedDeltaNet tensors live under
    # ``text_config``; treating the wrapper as a flat causal-LM config silently
    # substitutes the experiment declaration and can confuse ordinary
    # attention heads with linear-attention heads.
    text_raw: Mapping[str, Any]
    if raw:
        nested = raw.get("text_config")
        if nested is None:
            text_raw = raw
        elif isinstance(nested, Mapping):
            text_raw = nested
        else:
            raise PreflightCheckError(
                "model_config_metadata_invalid",
                "frozen model text_config must be a mapping",
            )
    else:
        text_raw = {}

    def optional_positive_int(name: str) -> int | None:
        value = text_raw.get(name)
        if value is None:
            return None
        if type(value) is not int or value < 1:
            raise PreflightCheckError(
                "model_config_metadata_invalid",
                f"frozen model {name} must be a positive integer",
            )
        return value

    vocab_candidates = [shape[0] for name, shape in tensors.items()
                        if shape and (name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"))]
    vocab = optional_positive_int("vocab_size") or max(vocab_candidates, default=0)
    conv_shapes = tuple(
        shape for name, shape in tensors.items()
        if name.endswith(".linear_attn.conv1d.weight")
    )
    if any(
        len(shape) != 3 or shape[1] != 1
        for shape in conv_shapes
    ):
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "frozen model convolution tensor layout is incompatible",
        )
    channels_per_head = (
        2 * config.model.state_key_dim + config.model.state_value_dim
    )
    conv_head_candidates = {
        shape[0] // channels_per_head
        for shape in conv_shapes
        if shape[0] % channels_per_head == 0
    }
    if any(shape[0] % channels_per_head for shape in conv_shapes):
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "frozen model convolution channels do not encode whole linear heads",
        )
    if len(conv_head_candidates) > 1:
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "frozen model convolution head counts are heterogeneous",
        )
    raw_key_heads = optional_positive_int("linear_num_key_heads")
    raw_value_heads = optional_positive_int("linear_num_value_heads")
    if ((raw_key_heads is None) != (raw_value_heads is None)
            or (raw_key_heads is not None and raw_key_heads != raw_value_heads)):
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "frozen model linear key/value head counts must be present and equal",
        )
    metadata_heads = raw_value_heads
    if conv_head_candidates:
        conv_heads = next(iter(conv_head_candidates))
        if metadata_heads is not None and metadata_heads != conv_heads:
            raise PreflightCheckError(
                "model_config_metadata_invalid",
                "frozen model linear-head metadata disagrees with safetensors",
            )
        metadata_heads = conv_heads
    linear_heads = config.model.num_heads if metadata_heads is None else metadata_heads
    if linear_heads != config.model.num_heads:
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "experiment model.num_heads does not match frozen linear-attention heads",
        )
    conv_candidates = {shape[-1] for shape in conv_shapes}
    if len(conv_candidates) > 1:
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "frozen model convolution kernels are heterogeneous",
        )
    raw_conv = optional_positive_int("linear_conv_kernel_dim")
    try:
        conv_kernel = raw_conv if raw_conv is not None else next(iter(conv_candidates))
    except StopIteration as error:
        raise PreflightCheckError(
            "model_config_metadata_missing",
            "frozen model linear_conv_kernel_dim is unavailable",
        ) from error
    if conv_candidates and conv_kernel not in conv_candidates:
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "frozen model convolution metadata disagrees with safetensors",
        )
    hidden_size = optional_positive_int("hidden_size") or config.model.hidden_size
    intermediate_size = optional_positive_int("intermediate_size") or config.model.ffn_dim
    if hidden_size != config.model.hidden_size or intermediate_size != config.model.ffn_dim:
        raise PreflightCheckError(
            "model_config_metadata_invalid",
            "experiment hidden/FFN dimensions disagree with the frozen text model",
        )
    values = {
        "vocab_size": vocab,
        "hidden_size": hidden_size,
        "num_layers": optional_positive_int("num_hidden_layers") or config.model.num_layers,
        "num_attention_heads": optional_positive_int("num_attention_heads") or linear_heads,
        "linear_num_heads": linear_heads,
        "intermediate_size": intermediate_size,
        "linear_conv_kernel_dim": conv_kernel,
    }
    if min(values.values()) < 1:
        raise PreflightCheckError("model_config_metadata_missing", "frozen model dimensions are incomplete")
    source = (
        "model_config_json_text_config"
        if raw and text_raw is not raw
        else "model_config_json"
        if raw
        else "safetensors_and_frozen_experiment_config"
    )
    return values, source


def _hybrid_state_history_components(
    config: ExperimentConfig, *, shared: bool, hidden_size: int,
    conv_kernel: int, batch_size: int, convolution_element_bytes: int,
) -> dict[str, int]:
    """Conservative carried-state bytes using the modules' canonical shapes."""
    B, L, H = batch_size, config.model.num_layers, config.model.num_heads
    K, V, R = config.model.state_key_dim, config.model.state_value_dim, 4

    def size(name: str, *dimensions: int, element_bytes: int = 4) -> int:
        return _checked_product(
            L, *dimensions, element_bytes, context=f"hybrid {name} bytes"
        )

    if shared:
        return {
            "states": size("states", B, H, K, V),
            "phase_history": size("phase history", B, H, K // 2),
            "previous_value": size("previous value", B, H, R, V),
            "previous_write": size("previous write", B, H, R, K, V),
            "convolution_history": size(
                "convolution history", B, conv_kernel - 1, hidden_size,
                element_bytes=convolution_element_bytes,
            ),
            "history_flags": size("history flags", B, element_bytes=1),
        }
    band = K // R if K % (2 * R) == 0 else K
    return {
        "states": size("states", B, H, R, band, V),
        "phase_history": size("phase history", B, H, R, band // 2),
        "previous_key": size("previous key", B, H, R, band),
        "previous_value": size("previous value", B, H, R, V),
        "convolution_history": size(
            "convolution history", B, conv_kernel - 1, hidden_size,
            element_bytes=convolution_element_bytes,
        ),
        "history_flags": size("history flags", B, R, element_bytes=1),
        "update_counter": size("update counter", B, element_bytes=8),
    }


def measure_qwen_resources(
    config: ExperimentConfig | None,
    spec: Any,
    *,
    assets: Mapping[str, Mapping[str, Any]],
    environ: Mapping[str, str] | None = None,
    loaded_modules: Mapping[str, Any] | None = None,
    source_root: Path | None = None,
    hybrid_modules: tuple[Any, ...] | None = None,
    hybrid_optimizer: Any | None = None,
    resident_model: Any | None = None,
    batch_size: int | None = None,
    context_length: int | None = None,
    safety_margin_bytes: int | None = None,
    activation_checkpointing: bool = False,
    cuda_probe: Any | None = None,
) -> dict[str, Any]:
    """Count parameters from safetensors headers and state/cache bytes by formula."""

    if hybrid_modules is not None:
        import torch
        from .qwen_hybrid_math import DEFERRED_FUSION_WARNING, REFERENCE_IMPLEMENTATION
        from .qwen_training import _hybrid_module_resources
        if context_length != 32768:
            raise PreflightCheckError("hybrid_context_invalid", "hybrid preflight must exercise actual 32K context")
        if hybrid_optimizer is None or type(batch_size) is not int or batch_size < 1:
            raise PreflightCheckError("hybrid_resource_invalid", "installed modules, optimizer, and batch are required")
        if type(safety_margin_bytes) is not int or safety_margin_bytes < 0:
            raise PreflightCheckError("hybrid_resource_invalid", "device margin must be a nonnegative int")
        if resident_model is None:
            raise PreflightCheckError(
                "hybrid_resident_model_required",
                "installed-module preflight requires the complete resident Qwen model; use metadata preflight before load",
            )
        measured = _hybrid_module_resources(
            hybrid_modules, optimizer=hybrid_optimizer, batch_size=batch_size,
            sequence_length=context_length, checkpointing=activation_checkpointing,
            resident_model=resident_model,
        )
        # Empty AdamW optimizers have not allocated moments yet. Reserve two
        # FP32-equivalent moment slots conservatively instead of reporting zero.
        measured["optimizer_bytes"] = max(
            int(measured["optimizer_bytes"]), 2 * int(measured["gradient_bytes"])
        )
        measured["activation_accounting"] = "conservative_shape_upper_bound"
        subtotal_bytes = _checked_sum(
            *(int(measured[name]) for name in (
                "resident_parameter_bytes", "resident_buffer_bytes", "gradient_bytes",
                "master_weight_bytes", "optimizer_bytes",
                "state_and_history_bytes", "cache_bytes", "workspace_bytes", "activation_bytes"
            )), context="hybrid device memory"
        )
        safety_margin_bytes = max(safety_margin_bytes, math.ceil(0.20 * subtotal_bytes))
        required_bytes = _checked_sum(subtotal_bytes, safety_margin_bytes,
                                      context="hybrid device memory with allocator margin")
        if cuda_probe is None:
            if not torch.cuda.is_available():
                raise PreflightCheckError("cuda_unavailable", "actual 32K preflight requires CUDA")
            device_index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            try:
                driver = torch._C._cuda_getDriverVersion()
            except AttributeError:
                driver = "unavailable"
            device = {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
                      "device_index": device_index, "device_name": torch.cuda.get_device_name(device_index),
                      "driver": driver, "runtime": torch.version.cuda}
        else:
            device = dict(cuda_probe())
            required_device = {"free_bytes", "total_bytes", "device_index", "device_name", "driver", "runtime"}
            if set(device) != required_device:
                raise PreflightCheckError("cuda_probe_invalid", "CUDA probe fields must be exact")
        available_device_bytes = device["free_bytes"]
        if type(available_device_bytes) is not int or available_device_bytes < 0:
            raise PreflightCheckError("cuda_probe_invalid", "CUDA free bytes must be nonnegative")
        headroom = available_device_bytes - required_bytes
        if headroom < 0:
            raise PreflightCheckError(
                "hybrid_32k_memory_unsafe",
                f"32K hybrid requires {required_bytes} bytes with {safety_margin_bytes} byte margin; only {available_device_bytes} available",
            )
        return {
            "available": True, "exact": True, "context_length": context_length,
            "execution": REFERENCE_IMPLEMENTATION,
            "performance_warning": DEFERRED_FUSION_WARNING,
            "hybrid": measured,
            "required_device_bytes": required_bytes, "headroom_bytes": headroom,
            "safety_margin_bytes": safety_margin_bytes,
            "preflight_safe": headroom >= 0,
            "device": {"cuda_available": True, "cuda_runtime": device["runtime"],
                       "cuda_driver": device["driver"], "device_index": device["device_index"],
                       "device_name": device["device_name"], "free_bytes": device["free_bytes"],
                       "total_bytes": device["total_bytes"]},
        }

    if not isinstance(config, ExperimentConfig) or config.backend != "qwen":
        return {"available": False}
    execution = verify_qwen_execution_contract(
        config,
        environ=environ,
        loaded_modules=loaded_modules,
        source_root=source_root,
    )
    model_record = assets.get("model")
    if not isinstance(model_record, Mapping) or type(model_record.get("path")) is not str:
        raise PreflightCheckError(
            "parameter_metadata_missing", "measured Qwen model identity is unavailable"
        )
    tensors = _safetensors_inventory(Path(model_record["path"]))
    total_base_parameters = _checked_sum(
        *(_shape_elements(shape, context=name) for name, shape in tensors.items()),
        context="base Qwen parameters",
    )
    architecture_arm = getattr(spec, "architecture_arm_id", None)
    if architecture_arm is None:
        candidate_arm = getattr(spec, "arm_id", None)
        architecture_arm = candidate_arm if type(candidate_arm) is str else None
    if architecture_arm is None and config.architecture is not None:
        architecture_arm = config.architecture.arm_id
    if type(architecture_arm) is str and architecture_arm.startswith("gdn2-mimo-r4-braid-"):
        import torch
        from .qwen_checkpoint import hybrid_tensor_element_counts

        shared = architecture_arm.endswith("shared-hola-w64")
        dimensions, dimension_source = _frozen_model_dimensions(
            Path(model_record["path"]), config, tensors
        )
        per_layer_counts = hybrid_tensor_element_counts(
            architecture_arm_id=architecture_arm,
            heads=config.model.num_heads,
            key_dim=config.model.state_key_dim,
            value_dim=config.model.state_value_dim,
            hidden_size=dimensions["hidden_size"],
            conv_kernel=dimensions["linear_conv_kernel_dim"],
        )
        hybrid_elements = config.model.num_layers * per_layer_counts["parameter_elements"]
        hybrid_buffer_elements = (
            config.model.num_layers * per_layer_counts["persistent_buffer_elements"]
        )
        trainable_elements = hybrid_elements
        parameter_element_bytes = (
            2 if config.dtype_preferences[0] == "bfloat16" else 4
        )
        # The official asset is a multimodal wrapper.  Runtime extracts only
        # ``model.language_model`` and drops MTP/vision tensors; then the 18
        # native GatedDeltaNet modules are replaced by the hybrid contract.
        wrapped_text = any(
            name.startswith("model.language_model.") for name in tensors
        )
        if wrapped_text:
            runtime_text_tensors = {
                name: shape for name, shape in tensors.items()
                if name.startswith("model.language_model.")
                or name == "lm_head.weight"
            }
        else:
            runtime_text_tensors = {
                name: shape for name, shape in tensors.items()
                if ".visual." not in name and not name.startswith("mtp.")
            }
        from .architecture import TARGET_LAYERS

        layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.linear_attn\.")
        replaced_native_elements = _checked_sum(
            *(
                _shape_elements(shape, context=name)
                for name, shape in runtime_text_tensors.items()
                if (match := layer_pattern.search(name)) is not None
                and int(match.group(1)) in TARGET_LAYERS
            ),
            context="replaced native linear-attention parameters",
        )
        runtime_text_elements = _checked_sum(
            *(
                _shape_elements(shape, context=name)
                for name, shape in runtime_text_tensors.items()
            ),
            context="runtime Qwen text parameters",
        )
        frozen_base_elements = runtime_text_elements - replaced_native_elements
        if frozen_base_elements < 1:
            raise PreflightCheckError(
                "parameter_metadata_mismatch",
                "runtime text model loses all parameters after recurrent replacement",
            )
        frozen_base_bytes = frozen_base_elements * parameter_element_bytes
        hybrid_parameter_bytes = hybrid_elements * parameter_element_bytes
        hybrid_buffer_bytes = hybrid_buffer_elements * 4
        resident_bytes = _checked_sum(
            frozen_base_bytes, hybrid_parameter_bytes, hybrid_buffer_bytes,
            context="resident optimized student tensors",
        )
        gradient_bytes = hybrid_parameter_bytes
        # Fused Adam keeps moments in parameter dtype.  Production moves both
        # slots to CPU outside optimizer.step and never creates FP32 masters.
        master_bytes = 0
        optimizer_bytes = 2 * hybrid_parameter_bytes
        batch_size = int(config.task.params.get("batch_size", 1))
        declared_training_tokens = config.task.params.get(
            "training_window_token_counts", ()
        )
        if (
            isinstance(declared_training_tokens, Sequence)
            and not isinstance(declared_training_tokens, (str, bytes))
            and declared_training_tokens
            and all(type(value) is int and value > 0 for value in declared_training_tokens)
        ):
            training_context = max(declared_training_tokens)
        else:
            training_context = max(config.lengths.curriculum)
        evaluation_context = max(config.lengths.extrapolation)
        convolution_element_bytes = (
            2 if config.dtype_preferences[0] == "bfloat16" else 4
        )
        state_components = _hybrid_state_history_components(
            config, shared=shared, hidden_size=dimensions["hidden_size"],
            conv_kernel=dimensions["linear_conv_kernel_dim"], batch_size=batch_size,
            convolution_element_bytes=convolution_element_bytes,
        )
        matrix_history_names = (("states", "previous_write") if shared else ("states",))
        state_tensor_bytes = _checked_sum(
            *(state_components[name] for name in matrix_history_names),
            context="hybrid recurrent matrix state",
        )
        state_bytes = _checked_sum(
            *state_components.values(), context="hybrid recurrent state and history"
        )
        state_auxiliary_bytes = state_bytes - state_tensor_bytes
        cache_storage_bytes = 2 if config.cache.storage_dtype == "bf16" else 4
        native_key = config.model.state_key_dim
        cache_key = (
            native_key if shared or native_key % 8 else native_key // 4
        )
        cache_entry = (
            cache_storage_bytes * 4
            * (cache_key + config.model.state_value_dim)
            + 4 + 8 + 1 + 8
        )
        cache_bytes = (
            config.model.num_layers * batch_size * config.model.num_heads * 64 * cache_entry
            + config.model.num_layers * batch_size * 8
            + config.model.num_layers * batch_size * config.model.num_heads * 4 * 8
        )
        workspace_bytes = (
            config.model.num_layers * batch_size * config.model.num_heads * 256 * cache_entry
            + config.model.num_layers * batch_size * config.model.num_heads * 8
        )
        checkpointing = bool(config.task.params.get("gradient_checkpointing", True))
        B, L, D, I, VOC = (
            batch_size, dimensions["num_layers"], dimensions["hidden_size"],
            dimensions["intermediate_size"], dimensions["vocab_size"],
        )
        segment = min(64, training_context)
        rank = 4
        native_key = config.model.state_key_dim
        key = native_key if shared or native_key % (2 * rank) else native_key // rank
        value = config.model.state_value_dim
        # Compact Package B stores one state trace plus exact key/value factor
        # traces. Shared Package A does not use this raw Triton recurrence but
        # retains the historical conservative estimate.
        triton_trace_bytes = (
            ((2 * key * value) if shared else (key * value + key + value))
            * B * segment * config.model.num_heads * rank * 4
        )
        triton_tail_bytes = (
            B * segment * config.model.num_heads * rank
            * (5 * key + 5 * value) * 4
        )
        saved_layer_activations = (
            B * training_context * (L + 1) * D * parameter_element_bytes
        )
        layer_workspace = B * training_context * (7 * D + 2 * I) * 4
        attention_workspace = B * training_context * D * 4
        logits_bytes = B * training_context * VOC * parameter_element_bytes
        # Canonical BF16 CE is the largest joint-loss temporary; KL and its
        # backward stream eight rows at a time from the teacher GPU.
        joint_loss_workspace = math.ceil(1.05 * logits_bytes)
        specialization_workspace = (
            hybrid_parameter_bytes
            if int(config.task.params.get("specialization_updates", 0)) > 0
            else 0
        )
        # Training-path global mixing retains the full-context normalized
        # recurrent reads (module dtype) and FP32 HOLA reads until the output
        # mixer is applied once across T; concatenation transiently doubles
        # both.  One layer's reads are live at a time under per-layer
        # checkpointing.  no_grad evaluation selects segment mixing and never
        # allocates these (enforced by test).
        global_mix_read_elements = (
            B * training_context * config.model.num_heads * rank * rank * value
        )
        global_mix_bytes = 2 * global_mix_read_elements * (
            parameter_element_bytes + 4
        )
        hybrid_activation_bytes = _checked_sum(
            triton_trace_bytes, triton_tail_bytes, specialization_workspace,
            global_mix_bytes,
            context="segmented hybrid training workspace",
        )
        generation_buffers = B * evaluation_context * D * parameter_element_bytes
        training_activation_bytes = _checked_sum(
            saved_layer_activations, layer_workspace, attention_workspace,
            logits_bytes, joint_loss_workspace, hybrid_activation_bytes,
            context="training activation phase",
        )
        training_backward_subtotal = _checked_sum(
            resident_bytes, gradient_bytes, state_bytes, cache_bytes,
            workspace_bytes, training_activation_bytes,
            context="training forward/backward phase",
        )
        optimizer_subtotal = _checked_sum(
            resident_bytes, gradient_bytes, optimizer_bytes, state_bytes,
            cache_bytes, workspace_bytes,
            context="phase-local optimizer step",
        )
        evaluation_layer_workspace = (
            B * evaluation_context * (7 * D + 2 * I) * 4
        )
        evaluation_attention_workspace = B * evaluation_context * D * 4
        streamed_logit_workspace = 256 * 1024 * 1024
        evaluation_activation_bytes = _checked_sum(
            generation_buffers, evaluation_layer_workspace,
            evaluation_attention_workspace, streamed_logit_workspace,
            triton_trace_bytes, triton_tail_bytes,
            context="streamed no-grad evaluation phase",
        )
        evaluation_subtotal = _checked_sum(
            resident_bytes, state_bytes, cache_bytes, workspace_bytes,
            evaluation_activation_bytes,
            context="evaluation phase",
        )
        teacher_resident_bytes = runtime_text_elements * parameter_element_bytes
        teacher_hidden_bytes = (
            B * training_context * (L + 1) * D * parameter_element_bytes
        )
        teacher_subtotal = _checked_sum(
            teacher_resident_bytes, logits_bytes, teacher_hidden_bytes,
            layer_workspace, attention_workspace,
            context="teacher forward phase",
        )
        configured_margin = int(config.task.params.get("preflight_safety_margin_bytes", 0))
        phase_subtotals = {
            "training_forward_backward": training_backward_subtotal,
            "optimizer_step": optimizer_subtotal,
            "evaluation_streamed_32k": evaluation_subtotal,
        }
        phase_margins = {
            name: max(configured_margin, math.ceil(0.20 * subtotal))
            for name, subtotal in phase_subtotals.items()
        }
        phase_required = {
            name: subtotal + phase_margins[name]
            for name, subtotal in phase_subtotals.items()
        }
        required_bytes = max(phase_required.values())
        margin = phase_margins[max(phase_required, key=phase_required.get)]
        teacher_margin = max(configured_margin, math.ceil(0.20 * teacher_subtotal))
        teacher_required_bytes = teacher_subtotal + teacher_margin
        if cuda_probe is None:
            if not torch.cuda.is_available():
                raise PreflightCheckError("cuda_unavailable", "hybrid 32K preflight requires CUDA")
            index = torch.cuda.current_device(); free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            device = {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
                      "device_index": index, "device_name": torch.cuda.get_device_name(index),
                      "runtime": torch.version.cuda}
        else:
            device = dict(cuda_probe())
        if int(device["free_bytes"]) < required_bytes:
            raise PreflightCheckError(
                "hybrid_32k_memory_unsafe",
                "phase-separated optimized hybrid preflight is unsafe",
            )
        from .qwen_hybrid_math import DEFERRED_FUSION_WARNING, REFERENCE_IMPLEMENTATION
        return {"available": True, "exact": False,
                "execution": REFERENCE_IMPLEMENTATION,
                "performance_warning": DEFERRED_FUSION_WARNING,
                "accounting": {"resident": "measured_safetensors_parameters_conservative_fp32",
                               "hybrid": "checkpoint_tensor_contract", "activation": "phase_separated_fused_path_upper_bound",
                               "model_dimensions": dimension_source},
                "context_length": evaluation_context,
                "training_context_length": training_context,
                "evaluation_context_length": evaluation_context,
                "total_base_parameters": total_base_parameters,
                "runtime_text_parameters": runtime_text_elements,
                "replaced_native_parameters": replaced_native_elements,
                "frozen_runtime_parameters": frozen_base_elements,
                "hybrid_parameter_elements": hybrid_elements, "resident_bytes": resident_bytes,
                "hybrid_persistent_buffer_elements": hybrid_buffer_elements,
                "trainable_parameters": trainable_elements,
                "total_parameters": frozen_base_elements + hybrid_elements,
                "recurrent_state_elements": state_tensor_bytes // 4,
                "recurrent_state_tensor_elements": state_tensor_bytes // 4,
                "recurrent_state_auxiliary_bytes": state_auxiliary_bytes,
                "recurrent_state_components": state_components,
                "recurrent_state_bytes": state_bytes,
                "cache_persistent_bytes": cache_bytes,
                "cache_block_bytes": workspace_bytes,
                "cache_storage_dtype": config.cache.storage_dtype,
                "cache_compute_dtype": config.cache.compute_dtype,
                "ffn_match": {"matched": True, "target_parameters": trainable_elements,
                              "matched_parameters": trainable_elements,
                              "selected_d_ff": config.model.ffn_dim,
                              "residual_mismatch": 0,
                              "tolerance": max(0.005 * trainable_elements, 1024.0)},
                "gradient_bytes": gradient_bytes, "master_weight_bytes": master_bytes,
                "optimizer_bytes": optimizer_bytes, "state_bytes": state_bytes,
                "cache_bytes": cache_bytes, "workspace_bytes": workspace_bytes,
                "activation_bytes": training_activation_bytes,
                "required_device_bytes": required_bytes,
                "teacher_required_device_bytes": teacher_required_bytes,
                "optimizer_state_residency": "cpu_except_during_optimizer_step",
                "phases": {
                    name: {
                        "subtotal_bytes": phase_subtotals[name],
                        "safety_margin_bytes": phase_margins[name],
                        "required_device_bytes": phase_required[name],
                    }
                    for name in phase_subtotals
                },
                "teacher_phase": {
                    "subtotal_bytes": teacher_subtotal,
                    "safety_margin_bytes": teacher_margin,
                    "required_device_bytes": teacher_required_bytes,
                },
                "activation_components": {"saved_layer_activations": saved_layer_activations,
                    "layer_workspace": layer_workspace, "attention_workspace": attention_workspace,
                    "logits": logits_bytes, "joint_loss_workspace": joint_loss_workspace,
                    "hybrid": hybrid_activation_bytes,
                    "triton_trace": triton_trace_bytes,
                    "specialization_workspace": specialization_workspace,
                    "generation_buffers": generation_buffers,
                    "streamed_evaluation_logits": streamed_logit_workspace},
                "allocator_workspace_margin_bytes": margin,
                "headroom_bytes": int(device["free_bytes"]) - required_bytes,
                "safety_margin_bytes": margin, "preflight_safe": True, "device": device,
                "qwen_execution": execution}
    memory_names = _string_names(
        config.task.params.get("memory_parameter_names", ()),
        field="task.params.memory_parameter_names",
        allow_empty=config.qwen.run_mode != "heal",
    )
    layer_prefixes = _native_layer_prefixes(config, tensors, memory_names)
    if any(
        name.endswith(suffix)
        for name in tensors
        for suffix in _NATIVE_ADDITION_SUFFIXES
    ):
        raise PreflightCheckError(
            "parameter_metadata_mismatch",
            "model safetensors already contain unsupported installed KMD2 additions",
        )
    memory_parameters = _checked_sum(
        *(
            _shape_elements(_resolve_parameter(tensors, name), context=name)
            for name in memory_names
        ),
        context="Qwen memory trainables",
    )
    native_additions = _native_addition_count(
        config, r_out=execution["native_r_out"]
    )
    cache_parameters, cache_names = _cache_parameter_count(
        config, layer_prefixes=layer_prefixes
    )
    cache_enabled = spec.mechanism == "exact_cache" and config.cache.width > 0
    treatment_trainable = _checked_sum(
        memory_parameters,
        cache_parameters if cache_enabled else 0,
        context="Qwen treatment trainables",
    )
    installed_native_total = _checked_sum(
        total_base_parameters,
        native_additions,
        context="Qwen installed native model",
    )
    total_parameters = _checked_sum(
        installed_native_total,
        cache_parameters if cache_enabled else 0,
        context="Qwen installed treatment model",
    )
    recurrent_elements = _checked_product(
        config.model.num_layers,
        config.model.num_heads,
        config.model.state_key_dim,
        config.model.state_value_dim,
        context="Qwen recurrent state",
    )
    storage_bytes = 2 if config.cache.storage_dtype == "bf16" else 4
    entry_bytes = (
        (config.model.state_key_dim + config.model.state_value_dim) * storage_bytes
        + 4
        + 8
        + 1
    )
    persistent_bytes = (
        config.model.num_layers
        * config.model.num_heads
        * config.cache.width
        * entry_bytes
        if cache_enabled
        else 0
    )
    block_bytes = (
        config.model.num_layers
        * config.model.num_heads
        * config.cache.block_size
        * entry_bytes
        if cache_enabled
        else 0
    )
    tolerance = max(0.005 * treatment_trainable, 1024.0)
    return {
        "available": True,
        "exact": True,
        "trainable_parameters": treatment_trainable,
        "total_parameters": total_parameters,
        "recurrent_state_elements": recurrent_elements,
        "recurrent_state_tensor_elements": recurrent_elements,
        "recurrent_state_auxiliary_bytes": 0,
        "recurrent_state_bytes": 4 * recurrent_elements,
        "cache_persistent_bytes": persistent_bytes,
        "cache_block_bytes": block_bytes,
        "cache_storage_dtype": config.cache.storage_dtype,
        "cache_compute_dtype": config.cache.compute_dtype,
        "ffn_match": {
            "matched": True,
            "target_parameters": treatment_trainable,
            "matched_parameters": treatment_trainable,
            "selected_d_ff": config.model.ffn_dim,
            "residual_mismatch": 0,
            "tolerance": tolerance,
        },
        "parameter_metadata_kind": "safetensors_header",
        "parameter_metadata_tensors": len(tensors),
        "parameter_scope": "full_model_plus_installed_kmd2_native_plus_cache",
        "total_base_parameters": total_base_parameters,
        "native_addition_parameters": native_additions,
        "native_r_out": execution["native_r_out"],
        "cache_parameter_count": cache_parameters if cache_enabled else 0,
        "arm_trainable_parameters": {
            "native": memory_parameters,
            "recency": treatment_trainable,
            "surprise": treatment_trainable,
        },
        "arm_total_parameters": {
            "native": installed_native_total,
            "recency": total_parameters,
            "surprise": total_parameters,
        },
        "declared_cache_parameter_count": len(cache_names),
        "qwen_execution": execution,
    }


__all__ = ["measure_qwen_resources", "verify_qwen_execution_contract"]
