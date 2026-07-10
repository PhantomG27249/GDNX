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


KMD2_NATIVE_SHA256 = (
    "326b84cd8114b189496a385d084664d89ac73b3d98b1c720ce71d80af2069b67"
)
KMD2_FAST_SCAN_SHA256 = (
    "d4efb6ce70fbbe69613b7bba7bf7825ddbf1c13f867ee7a67a4a2d1f81bec6c1"
)
GDN3_UPGRADE_SHA256 = (
    "427ba5c5e03e48d76945ba465c53c6b7751443cec4187be88cb4acec8cb20666"
)
REFERENCE_RECURRENCE_SHA256 = (
    "8e64611571904fb5e90ea7641e117f747c1089cee6231f401b571bd5a4b0888a"
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
        "scan_impl": True,
        "compiled_scan_assignment": True,
        "scan_with_update_norm": False,
    },
}


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect_pinned_source(
    repo_root: Path, relative_path: str, expected: str
) -> tuple[str, str, ast.AST]:
    source_path = repo_root / relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Inventory source missing: {relative_path}")

    actual = _raw_sha256(source_path)
    if actual != expected:
        raise ValueError(
            f"{relative_path}: SHA-256 drift (expected {expected}, got {actual})"
        )

    try:
        source_text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{relative_path}: source is not valid UTF-8") from exc
    try:
        tree = ast.parse(source_text, filename=relative_path)
    except SyntaxError as exc:
        raise ValueError(f"{relative_path}: source is not valid Python") from exc
    return actual, source_text, tree


def _function_parameter_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        arguments = node.args
        names.update(arg.arg for arg in arguments.posonlyargs)
        names.update(arg.arg for arg in arguments.args)
        names.update(arg.arg for arg in arguments.kwonlyargs)
        if arguments.vararg is not None:
            names.add(arguments.vararg.arg)
        if arguments.kwarg is not None:
            names.add(arguments.kwarg.arg)
    return names


def _return_mentions_cache_or_tuple(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if isinstance(node.value, ast.Tuple):
            return True
        for child in ast.walk(node.value):
            if isinstance(child, ast.Name) and child.id.startswith("cache"):
                return True
            if isinstance(child, ast.Attribute) and child.attr.startswith("cache"):
                return True
    return False


def _compute_structural_findings(
    texts: Mapping[str, str], trees: Mapping[str, ast.AST]
) -> dict[str, Any]:
    native = texts["gdn3/kmd2_native.py"]
    fast = texts["gdn3/kmd2_fast_scan.py"]
    reference = texts["gdn3/_reference_recurrence.py"]
    upgrade = texts["gdn3/gdn3_upgrade.py"]
    native_tree = trees["gdn3/kmd2_native.py"]
    fast_tree = trees["gdn3/kmd2_fast_scan.py"]

    shared_query = "q.unsqueeze(3)" in native
    single_k = native.count("k = F.normalize(") == 1
    single_v = native.count("v = value.reshape(") == 1
    single_state = native.count("S = torch.zeros") == 1
    fast_function_names = {
        node.name
        for node in ast.walk(fast_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    findings = {
        "current_convolution": {
            "grouped_conv1d": (
                "self.conv1d = nn.Conv1d" in native
                and "groups=conv_dim" in native
            ),
            "silu_applied_to_conv1d": "F.silu(self.conv1d" in native,
        },
        "cumulative_data_dependent_rotation": {
            "rot_proj_defined": "self.rot_proj" in native,
            "cumsum_dim": 1 if "theta.cumsum(dim=1)" in native else None,
            "rope_targets": [
                target
                for target, marker in (
                    ("k", "k = rope(k"),
                    ("qs", "qs = rope(qs"),
                )
                if marker in native
            ],
        },
        "shared_query_r_out": {
            "default_r_out": (
                4 if '_env_int("GDN3_KMD2_ROUT", 4)' in native else None
            ),
            "query_unsqueeze_dim": 3 if shared_query else None,
            "shared_query": shared_query,
            "single_k": single_k,
            "single_v": single_v,
            "single_state": single_state,
            "true_mimo": not (
                shared_query and single_k and single_v and single_state
            ),
        },
        "per_channel_decay": {
            "decay_chan_used_in_g": (
                "g = (g_head.unsqueeze(-1) + self.decay_chan).exp()" in native
            ),
        },
        "decoupled_write": {
            "bw_off_used_in_beta_w": (
                "beta_w = torch.sigmoid(b + self.bw_off)" in native
            ),
            "separate_beta_e_beta_w": (
                "beta_e = torch.sigmoid(b)" in native
                and "beta_w = torch.sigmoid(b + self.bw_off)" in native
            ),
            "erase_uses_beta_e": "be_[t].unsqueeze(-1) * kv_mem" in native,
            "write_uses_beta_w": "bw_[t].unsqueeze(-1) * v_[t]" in native,
        },
        "native_exact_cache": {
            "topk_parameter": ".topk(" in native,
            "cache_parameter": any(
                name.startswith("cache_")
                for name in _function_parameter_names(native_tree)
            ),
            "cross_call_cache_return": _return_mentions_cache_or_tuple(native_tree),
            "scan_returns_output_only": "return torch.stack(outs, 0)" in native,
        },
        "legacy_uvb_overlap": {
            "buffers": ["U", "Vb"],
            "reference": {
                "allocation": (
                    "U = torch.zeros" in reference
                    and "Vb = torch.zeros" in reference
                ),
                "read": "_kron_read_vec(A, Bk, U, Vb" in reference,
                "update": (
                    "U = torch.cat" in reference
                    and "Vb = torch.cat" in reference
                ),
                "compaction": "_compact_vec(A, Bk, U, Vb" in reference,
            },
            "upgrade": {
                "allocation": (
                    "U = torch.zeros" in upgrade
                    and "Vb = torch.zeros" in upgrade
                ),
                "read": "Vb, x_chunk" in upgrade and "U, coeff" in upgrade,
                "update": (
                    "U_new =" in upgrade
                    and "Vb_new =" in upgrade
                    and "U, Vb = U_new, Vb_new" in upgrade
                ),
                "compaction": "_compact_fast(A, Bk, U, Vb)" in upgrade,
                "native_branch": (
                    "KMD2NativeAttn" if "KMD2NativeAttn" in upgrade else None
                ),
            },
        },
        "separate_fast_score": {
            "scan_impl": "_scan_impl" in fast_function_names,
            "compiled_scan_assignment": "scan = torch.compile(_scan_impl)" in fast,
            "scan_with_update_norm": any(
                "scan_with_update_norm" in name for name in fast_function_names
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
    source_texts = {
        relative_path: inspected[1]
        for relative_path, inspected in inspected_sources.items()
    }
    source_trees = {
        relative_path: inspected[2]
        for relative_path, inspected in inspected_sources.items()
    }
    structural_findings = _compute_structural_findings(source_texts, source_trees)

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
                "status": "negative",
                "evidence": ["gdn3/kmd2_fast_scan.py"],
                "details": structural_findings["separate_fast_score"],
            },
        },
        "compatibility": {
            "tiny": {
                "tasks": [
                    "affine",
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
