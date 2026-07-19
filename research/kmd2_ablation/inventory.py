"""Source-grounded capability inventory for the KMD-2 ablation suite.

The production modules listed here may import optional GPU/model dependencies.
Inventory construction therefore treats them strictly as source artifacts: it
hashes their raw bytes and parses their UTF-8 text without importing them.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


# 2026-07-14: fast scan gated to CUDA inputs (CPU preflight probes take the
# reference loop); see bible.txt footnote.
KMD2_NATIVE_SHA256 = (
    "9d17d8f872a57a54e5a990170903e20865a2ebd44ece5432a9d9a7bf5cc38779"
)
KMD2_FAST_SCAN_SHA256 = (
    "5f6c83fb5afb869765d5599f2a726e5cfd93d939cd6e6fa4863b0fee2267a0b4"
)
GDN3_UPGRADE_SHA256 = (
    "952a617546b1673fc481601fea6ac10ce95dd9c014db4cae35eb39d93bd2d675"
)
REFERENCE_RECURRENCE_SHA256 = (
    "e700dd0ee9b4315dcadda93d7467d558370a21f29aa7c9b9411da0c82a69900a"
)

PINNED_SOURCE_SHA256 = {
    "gdn3/_reference_recurrence.py": REFERENCE_RECURRENCE_SHA256,
    "gdn3/gdn3_upgrade.py": GDN3_UPGRADE_SHA256,
    "gdn3/kmd2_fast_scan.py": KMD2_FAST_SCAN_SHA256,
    "gdn3/kmd2_native.py": KMD2_NATIVE_SHA256,
}

REQUIRED_STRUCTURAL_FINDINGS = {
    "current_convolution": {
        "grouped_conv1d": True,
        "silu_applied_to_conv1d": True,
    },
    "cumulative_data_dependent_rotation": {
        "rot_proj_defined": True,
        "cumsum_dim": 1,
        "rope_targets": ["k", "qs"],
    },
    "shared_query_r_out": {
        "default_r_out": 4,
        "query_unsqueeze_dim": 3,
        "shared_query": True,
        "single_k": True,
        "single_v": True,
        "single_state": True,
        "true_mimo": False,
    },
    "per_channel_decay": {
        "decay_chan_used_in_g": True,
    },
    "decoupled_write": {
        "bw_off_used_in_beta_w": True,
        "separate_beta_e_beta_w": True,
        "erase_uses_beta_e": True,
        "write_uses_beta_w": True,
    },
    "native_exact_cache": {
        "topk_parameter": False,
        "cache_parameter": False,
        "cross_call_cache_return": False,
        "scan_returns_output_only": True,
    },
    "legacy_uvb_overlap": {
        "buffers": ["U", "Vb"],
        "reference": {
            "allocation": True,
            "read": True,
            "update": True,
            "compaction": True,
        },
        "upgrade": {
            "allocation": True,
            "read": True,
            "update": True,
            "compaction": True,
            "native_branch": "KMD2NativeAttn",
        },
    },
    "separate_fast_score": {
        "scan_core": True,
        "scan_impl": True,
        "scan_with_update_norm_impl": True,
        "compiled_scan_assignment": True,
        "compiled_score_assignment": True,
    },
}


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect_pinned_source(
    repo_root: Path, relative_path: str, expected: str
) -> tuple[str, ast.Module]:
    source_path = repo_root / relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Inventory source missing: {relative_path}")

    raw = source_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{relative_path}: SHA-256 drift (expected {expected}, got {actual})"
        )

    try:
        source_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{relative_path}: source is not valid UTF-8") from exc
    try:
        tree = ast.parse(source_text, filename=relative_path)
    except SyntaxError as exc:
        raise ValueError(f"{relative_path}: source is not valid Python") from exc
    return actual, tree


def _dotted_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _top_level_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    if not isinstance(tree, ast.Module):
        return None
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        ),
        None,
    )


def _top_level_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    if not isinstance(tree, ast.Module):
        return None
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def _class_method(
    class_node: ast.ClassDef | None, name: str
) -> ast.FunctionDef | None:
    if class_node is None:
        return None
    return next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def _assignment_values(
    scope: ast.AST | None, target_name: str
) -> list[ast.expr]:
    if scope is None:
        return []
    values: list[ast.expr] = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and any(
            _dotted_name(target) == target_name for target in node.targets
        ):
            values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and _dotted_name(node.target) == target_name
            and node.value is not None
        ):
            values.append(node.value)
    return values


def _top_level_assignment_value(
    tree: ast.AST, target_name: str
) -> ast.expr | None:
    if not isinstance(tree, ast.Module):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(_dotted_name(target) == target_name for target in node.targets):
            return node.value
    return None


def _is_call(node: ast.AST | None, function_name: str) -> bool:
    return isinstance(node, ast.Call) and _dotted_name(node.func) == function_name


def _contains_call(scope: ast.AST | None, function_name: str) -> bool:
    return scope is not None and any(
        _is_call(node, function_name) for node in ast.walk(scope)
    )


def _contains_call_with_arguments(
    scope: ast.AST | None,
    function_name: str,
    argument_names: tuple[str, ...],
) -> bool:
    if scope is None:
        return False
    for node in ast.walk(scope):
        if not _is_call(node, function_name):
            continue
        names = tuple(_dotted_name(argument) for argument in node.args)
        width = len(argument_names)
        if any(
            names[index : index + width] == argument_names
            for index in range(len(names) - width + 1)
        ):
            return True
    return False


def _expression_matches(node: ast.AST, expression: str) -> bool:
    expected = ast.parse(expression, mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def _contains_expression(scope: ast.AST | None, expression: str) -> bool:
    return scope is not None and any(
        _expression_matches(node, expression) for node in ast.walk(scope)
    )


def _integer_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if not isinstance(node.value, bool):
            return node.value
    return None


def _call_keyword_int(call: ast.Call, keyword_name: str) -> int | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return _integer_literal(keyword.value)
    return None


def _parameter_names(function: ast.FunctionDef | None) -> set[str]:
    if function is None:
        return set()
    arguments = function.args
    names = {
        arg.arg
        for arg in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _function_returns(function: ast.FunctionDef | None) -> list[ast.Return]:
    if function is None:
        return []
    returns: list[ast.Return] = []
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            returns.append(node)
            continue
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return returns


def _returns_cache_or_tuple(function: ast.FunctionDef | None) -> bool:
    for return_node in _function_returns(function):
        if return_node.value is None:
            continue
        if isinstance(return_node.value, ast.Tuple):
            return True
        for child in ast.walk(return_node.value):
            if isinstance(child, ast.Name) and child.id.startswith("cache"):
                return True
            if isinstance(child, ast.Attribute) and child.attr.startswith("cache"):
                return True
    return False


def _return_contains_call(
    function: ast.FunctionDef | None,
    function_name: str,
    argument_names: tuple[str, ...],
) -> bool:
    return any(
        return_node.value is not None
        and _contains_call_with_arguments(
            return_node.value, function_name, argument_names
        )
        for return_node in _function_returns(function)
    )


def _scope_has_names(scope: ast.AST | None, required_names: set[str]) -> bool:
    if scope is None:
        return False
    present = {
        node.id for node in ast.walk(scope) if isinstance(node, ast.Name)
    }
    return required_names <= present


def _has_parallel_name_assignment(
    scope: ast.AST | None,
    target_names: tuple[str, ...],
    value_names: tuple[str, ...],
) -> bool:
    if scope is None:
        return False
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Tuple) or not isinstance(node.value, ast.Tuple):
            continue
        targets = tuple(_dotted_name(element) for element in target.elts)
        values = tuple(_dotted_name(element) for element in node.value.elts)
        if targets == target_names and values == value_names:
            return True
    return False


def _compute_structural_findings(
    trees: Mapping[str, ast.AST],
) -> dict[str, Any]:
    native_tree = trees["gdn3/kmd2_native.py"]
    fast_tree = trees["gdn3/kmd2_fast_scan.py"]
    reference_tree = trees["gdn3/_reference_recurrence.py"]
    upgrade_tree = trees["gdn3/gdn3_upgrade.py"]

    native_class = _top_level_class(native_tree, "KMD2NativeAttn")
    native_init = _class_method(native_class, "__init__")
    native_scan = _class_method(native_class, "_scan")
    native_forward = _class_method(native_class, "forward")

    conv_definitions = _assignment_values(native_init, "self.conv1d")
    grouped_conv1d = any(
        isinstance(value, ast.Call)
        and _dotted_name(value.func) == "nn.Conv1d"
        and any(
            keyword.arg == "groups"
            and _dotted_name(keyword.value) == "conv_dim"
            for keyword in value.keywords
        )
        for value in conv_definitions
    )
    silu_applied_to_conv1d = any(
        isinstance(node, ast.Call)
        and _dotted_name(node.func) == "F.silu"
        and any(
            _is_call(child, "self.conv1d")
            for argument in node.args
            for child in ast.walk(argument)
        )
        for node in ast.walk(native_forward)
    ) if native_forward is not None else False

    rot_proj_defined = any(
        _is_call(value, "nn.Linear")
        for value in _assignment_values(native_init, "self.rot_proj")
    )
    theta_cumsum_dim = None
    for value in _assignment_values(native_forward, "Theta"):
        if (
            isinstance(value, ast.Call)
            and _dotted_name(value.func) == "theta.cumsum"
        ):
            theta_cumsum_dim = _call_keyword_int(value, "dim")
            break
    rope_targets = [
        target
        for target in ("k", "qs")
        if any(
            isinstance(value, ast.Call)
            and _dotted_name(value.func) == "rope"
            and bool(value.args)
            and _dotted_name(value.args[0]) == target
            for value in _assignment_values(native_forward, target)
        )
    ]

    r_out_default = None
    for value in _assignment_values(native_init, "self.r_out"):
        if (
            isinstance(value, ast.Call)
            and _dotted_name(value.func) == "_env_int"
            and len(value.args) >= 2
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == "GDN3_KMD2_ROUT"
        ):
            r_out_default = _integer_literal(value.args[1])
            break

    query_unsqueeze_dims = {
        dimension
        for value in _assignment_values(native_forward, "qs")
        for node in ast.walk(value)
        if isinstance(node, ast.Call)
        and _dotted_name(node.func) == "q.unsqueeze"
        and node.args
        if (dimension := _integer_literal(node.args[0])) is not None
    }
    query_unsqueeze_dim = (
        next(iter(query_unsqueeze_dims))
        if len(query_unsqueeze_dims) == 1
        else None
    )
    shared_query = bool(query_unsqueeze_dims)
    single_k = sum(
        _is_call(value, "F.normalize")
        for value in _assignment_values(native_forward, "k")
    ) == 1
    single_v = sum(
        _contains_call(value, "value.reshape")
        for value in _assignment_values(native_forward, "v")
    ) == 1
    single_state = sum(
        _is_call(value, "torch.zeros")
        for value in _assignment_values(native_scan, "S")
    ) == 1

    native_cache_scopes = (native_init, native_scan, native_forward)
    cache_parameter = any(
        name.startswith("cache_")
        for scope in native_cache_scopes
        for name in _parameter_names(scope)
    )
    cross_call_cache_return = any(
        _returns_cache_or_tuple(scope) for scope in native_cache_scopes
    )
    topk_parameter = any(
        scope is not None
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "topk"
            for node in ast.walk(scope)
        )
        for scope in native_cache_scopes
    )

    reference_state = _top_level_function(
        reference_tree, "reference_recurrent_state"
    )
    if not _scope_has_names(reference_state, {"U", "Vb"}):
        reference_state = None

    upgrade_class = _top_level_class(upgrade_tree, "GDN3LinearAttn")
    upgrade_state = _class_method(upgrade_class, "_gdn3_recurrent_state")
    if not _scope_has_names(upgrade_state, {"U", "Vb"}):
        upgrade_state = None
    upgrade_manager = _top_level_class(upgrade_tree, "GDN3UpgradeManager")
    apply_upgrade = _class_method(upgrade_manager, "apply_upgrade")
    native_imported = apply_upgrade is not None and any(
        isinstance(node, ast.ImportFrom)
        and any(alias.name == "KMD2NativeAttn" for alias in node.names)
        for node in ast.walk(apply_upgrade)
    )
    native_constructed = _contains_call(apply_upgrade, "KMD2NativeAttn")

    scan_core = _top_level_function(fast_tree, "_scan_core")
    scan_impl = _top_level_function(fast_tree, "_scan_impl")
    scan_with_update_norm_impl = _top_level_function(
        fast_tree, "_scan_with_update_norm_impl"
    )
    compiled_scan = _top_level_assignment_value(fast_tree, "scan")
    compiled_score_scan = _top_level_assignment_value(
        fast_tree, "scan_with_update_norm"
    )

    findings = {
        "current_convolution": {
            "grouped_conv1d": grouped_conv1d,
            "silu_applied_to_conv1d": silu_applied_to_conv1d,
        },
        "cumulative_data_dependent_rotation": {
            "rot_proj_defined": rot_proj_defined,
            "cumsum_dim": theta_cumsum_dim,
            "rope_targets": rope_targets,
        },
        "shared_query_r_out": {
            "default_r_out": r_out_default,
            "query_unsqueeze_dim": query_unsqueeze_dim,
            "shared_query": shared_query,
            "single_k": single_k,
            "single_v": single_v,
            "single_state": single_state,
            "true_mimo": not (
                shared_query and single_k and single_v and single_state
            ),
        },
        "per_channel_decay": {
            "decay_chan_used_in_g": any(
                _expression_matches(
                    value,
                    "(g_head.unsqueeze(-1) + self.decay_chan).exp()",
                )
                for value in _assignment_values(native_forward, "g")
            ),
        },
        "decoupled_write": {
            "bw_off_used_in_beta_w": any(
                _expression_matches(
                    value, "torch.sigmoid(b + self.bw_off)"
                )
                for value in _assignment_values(native_forward, "beta_w")
            ),
            "separate_beta_e_beta_w": (
                any(
                    _expression_matches(value, "torch.sigmoid(b)")
                    for value in _assignment_values(native_forward, "beta_e")
                )
                and any(
                    _expression_matches(
                        value, "torch.sigmoid(b + self.bw_off)"
                    )
                    for value in _assignment_values(native_forward, "beta_w")
                )
            ),
            "erase_uses_beta_e": _contains_expression(
                native_scan, "be_[t].unsqueeze(-1) * kv_mem"
            ),
            "write_uses_beta_w": _contains_expression(
                native_scan, "bw_[t].unsqueeze(-1) * v_[t]"
            ),
        },
        "native_exact_cache": {
            "topk_parameter": topk_parameter,
            "cache_parameter": cache_parameter,
            "cross_call_cache_return": cross_call_cache_return,
            "scan_returns_output_only": _return_contains_call(
                native_scan, "torch.stack", ("outs",)
            ),
        },
        "legacy_uvb_overlap": {
            "buffers": ["U", "Vb"],
            "reference": {
                "allocation": (
                    any(
                        _is_call(value, "torch.zeros")
                        for value in _assignment_values(reference_state, "U")
                    )
                    and any(
                        _is_call(value, "torch.zeros")
                        for value in _assignment_values(reference_state, "Vb")
                    )
                ),
                "read": _contains_call_with_arguments(
                    reference_state,
                    "layer._kron_read_vec",
                    ("A", "Bk", "U", "Vb"),
                ),
                "update": (
                    any(
                        _is_call(value, "torch.cat")
                        for value in _assignment_values(reference_state, "U")
                    )
                    and any(
                        _is_call(value, "torch.cat")
                        for value in _assignment_values(reference_state, "Vb")
                    )
                ),
                "compaction": _contains_call_with_arguments(
                    reference_state,
                    "layer._compact_vec",
                    ("A", "Bk", "U", "Vb"),
                ),
            },
            "upgrade": {
                "allocation": (
                    any(
                        _is_call(value, "torch.zeros")
                        for value in _assignment_values(upgrade_state, "U")
                    )
                    and any(
                        _is_call(value, "torch.zeros")
                        for value in _assignment_values(upgrade_state, "Vb")
                    )
                ),
                "read": (
                    _contains_call_with_arguments(
                        upgrade_state,
                        "torch.einsum",
                        ("Vb", "x_chunk"),
                    )
                    and _contains_call_with_arguments(
                        upgrade_state,
                        "torch.einsum",
                        ("U", "coeff"),
                    )
                ),
                "update": (
                    bool(_assignment_values(upgrade_state, "U_new"))
                    and bool(_assignment_values(upgrade_state, "Vb_new"))
                    and _has_parallel_name_assignment(
                        upgrade_state,
                        ("U", "Vb"),
                        ("U_new", "Vb_new"),
                    )
                ),
                "compaction": _contains_call_with_arguments(
                    upgrade_state,
                    "self._compact_fast",
                    ("A", "Bk", "U", "Vb"),
                ),
                "native_branch": (
                    "KMD2NativeAttn"
                    if native_imported and native_constructed
                    else None
                ),
            },
        },
        "separate_fast_score": {
            "scan_core": scan_core is not None,
            "scan_impl": scan_impl is not None,
            "scan_with_update_norm_impl": scan_with_update_norm_impl is not None,
            "compiled_scan_assignment": (
                isinstance(compiled_scan, ast.Call)
                and _dotted_name(compiled_scan.func) == "torch.compile"
                and len(compiled_scan.args) == 1
                and _dotted_name(compiled_scan.args[0]) == "_scan_impl"
            ),
            "compiled_score_assignment": (
                isinstance(compiled_score_scan, ast.Call)
                and _dotted_name(compiled_score_scan.func) == "torch.compile"
                and len(compiled_score_scan.args) == 1
                and _dotted_name(compiled_score_scan.args[0])
                == "_scan_with_update_norm_impl"
            ),
        },
    }

    for capability, required in REQUIRED_STRUCTURAL_FINDINGS.items():
        if findings.get(capability) != required:
            raise ValueError(
                f"structural {capability.replace('_', ' ')} mismatch"
            )
    return findings


def build_inventory(repo_root: str | Path) -> dict[str, Any]:
    """Build the deterministic inventory after validating pinned source bytes."""

    root = Path(repo_root)
    inspected_sources = {
        relative_path: _inspect_pinned_source(root, relative_path, expected)
        for relative_path, expected in PINNED_SOURCE_SHA256.items()
    }
    source_files = {
        relative_path: inspected[0]
        for relative_path, inspected in inspected_sources.items()
    }
    source_trees = {
        relative_path: inspected[1]
        for relative_path, inspected in inspected_sources.items()
    }
    structural_findings = _compute_structural_findings(source_trees)

    return {
        "inventory_version": "1.0.0",
        "source_files": source_files,
        "structural_findings": structural_findings,
        "capabilities": {
            "current_convolution": {
                "status": "positive",
                "evidence": ["gdn3/kmd2_native.py"],
                "details": structural_findings["current_convolution"],
            },
            "cumulative_data_dependent_rotation": {
                "status": "positive",
                "evidence": ["gdn3/kmd2_native.py"],
                "details": structural_findings[
                    "cumulative_data_dependent_rotation"
                ],
            },
            "shared_query_r_out": {
                "status": "positive",
                "evidence": ["gdn3/kmd2_native.py"],
                "details": structural_findings["shared_query_r_out"],
            },
            "per_channel_decay": {
                "status": "positive",
                "evidence": [
                    "gdn3/kmd2_native.py",
                    "gdn3/kmd2_fast_scan.py",
                ],
                "details": structural_findings["per_channel_decay"],
            },
            "decoupled_write": {
                "status": "positive",
                "evidence": [
                    "gdn3/kmd2_native.py",
                    "gdn3/kmd2_fast_scan.py",
                ],
                "details": structural_findings["decoupled_write"],
            },
            "native_exact_cache": {
                "status": "negative",
                "evidence": ["gdn3/kmd2_native.py"],
                "details": structural_findings["native_exact_cache"],
            },
            "legacy_uvb_overlap": {
                "status": "legacy_inactive",
                "evidence": [
                    "gdn3/_reference_recurrence.py",
                    "gdn3/gdn3_upgrade.py",
                ],
                "details": structural_findings["legacy_uvb_overlap"],
            },
            "separate_fast_score": {
                "status": "positive",
                "evidence": ["gdn3/kmd2_fast_scan.py"],
                "details": structural_findings["separate_fast_score"],
            },
        },
        "compatibility": {
            "tiny": {
                "tasks": [
                    "affine_associative_regression",
                    "drift_reversal",
                    "far_surprise",
                    "freshness",
                    "irregular_integration",
                    "local_binding",
                    "mqar",
                    "state_tracking",
                    "structured_exceptions",
                    "trajectory",
                ],
                "run_modes": ["promotion", "screen", "smoke"],
            },
            "qwen": {
                "tasks": [
                    "far_surprise",
                    "freshness",
                    "mqar",
                    "ruler",
                    "structured_exceptions",
                ],
                "run_modes": ["heal", "initial_exact_cache", "reliance"],
            },
        },
        "compatibility_metadata": {
            "source": "suite_design",
            "production_derived": False,
        },
        "external_assets": {
            "qwen_model": {
                "kind": "huggingface_model",
                "argument": "--model",
                "required_by": ["qwen"],
                "bundled": False,
            },
            "qwen_tokenizer": {
                "kind": "huggingface_tokenizer",
                "argument": "--tokenizer",
                "required_by": ["qwen"],
                "bundled": False,
            },
            "native_checkpoint": {
                "kind": "torch_checkpoint",
                "argument": "--native-checkpoint",
                "required_by": ["qwen:reliance"],
                "conditional": "optional_for_declared_native_start_heal",
                "bundled": False,
            },
            "dataset": {
                "kind": "dataset",
                "argument": "--data",
                "required_by": ["qwen:heal", "qwen:evaluation"],
                "conditional": "optional_for_synthetic_only",
                "bundled": False,
            },
            "teacher_model": {
                "kind": "huggingface_model",
                "argument": "--teacher-model",
                "required_by": ["qwen:heal"],
                "conditional": "required_unless_synthetic_only",
                "bundled": False,
            },
        },
    }


def verify_inventory_sources(
    inventory: Mapping[str, Any], repo_root: str | Path
) -> None:
    """Verify that an inventory declares exactly the pinned, untampered sources."""

    source_files = inventory.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("Inventory source_files must be a mapping")

    expected_paths = set(PINNED_SOURCE_SHA256)
    declared_paths = set(source_files)
    missing = sorted(expected_paths - declared_paths)
    unexpected = sorted(declared_paths - expected_paths)
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if unexpected:
            problems.append(f"unexpected {unexpected}")
        raise ValueError("Inventory source declarations: " + "; ".join(problems))

    root = Path(repo_root)
    for relative_path, pinned_digest in PINNED_SOURCE_SHA256.items():
        declared_digest = source_files[relative_path]
        if declared_digest != pinned_digest:
            raise ValueError(
                f"{relative_path}: declared SHA-256 {declared_digest!r} does not "
                f"match pinned SHA-256 {pinned_digest}"
            )

        source_path = root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Inventory source missing: {relative_path}")
        actual_digest = _raw_sha256(source_path)
        if actual_digest != declared_digest:
            raise ValueError(
                f"{relative_path}: SHA-256 mismatch "
                f"(declared {declared_digest}, got {actual_digest})"
            )
