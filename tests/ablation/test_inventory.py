from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SOURCE_SHA256 = {
    "gdn3/_reference_recurrence.py": (
        "8e64611571904fb5e90ea7641e117f747c1089cee6231f401b571bd5a4b0888a"
    ),
    "gdn3/gdn3_upgrade.py": (
        "427ba5c5e03e48d76945ba465c53c6b7751443cec4187be88cb4acec8cb20666"
    ),
    "gdn3/kmd2_fast_scan.py": (
        "d4efb6ce70fbbe69613b7bba7bf7825ddbf1c13f867ee7a67a4a2d1f81bec6c1"
    ),
    "gdn3/kmd2_native.py": (
        "326b84cd8114b189496a385d084664d89ac73b3d98b1c720ce71d80af2069b67"
    ),
}

EXPECTED_STRUCTURAL_FINDINGS = {
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


def _inventory_module():
    try:
        module = importlib.import_module("research.kmd2_ablation.inventory")
    except ModuleNotFoundError:
        pytest.fail("build_inventory is missing")
    if not hasattr(module, "build_inventory"):
        pytest.fail("build_inventory is missing")
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_inventory_sources(destination_root: Path) -> None:
    for relative_path in EXPECTED_SOURCE_SHA256:
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative_path, destination)


def test_inventory_module_is_standard_library_only_and_never_imports_sources():
    module = _inventory_module()
    module_path = Path(module.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots == {
        "__future__",
        "ast",
        "collections",
        "hashlib",
        "pathlib",
        "typing",
    }
    assert "gdn3" not in imported_roots


def test_inventory_records_current_positive_negative_and_legacy_capabilities(
    tmp_path, monkeypatch
):
    module = _inventory_module()
    inventory = module.build_inventory(REPO_ROOT)
    expected_statuses = {
        "current_convolution": "positive",
        "cumulative_data_dependent_rotation": "positive",
        "shared_query_r_out": "positive",
        "per_channel_decay": "positive",
        "decoupled_write": "positive",
        "native_exact_cache": "negative",
        "legacy_uvb_overlap": "legacy_inactive",
        "separate_fast_score": "negative",
    }

    assert inventory["inventory_version"] == "1.0.0"
    assert inventory["structural_findings"] == EXPECTED_STRUCTURAL_FINDINGS
    assert set(inventory["capabilities"]) == set(expected_statuses)
    for capability, expected_status in expected_statuses.items():
        record = inventory["capabilities"][capability]
        assert record["status"] == expected_status
        assert record["evidence"]
        assert record["details"] == EXPECTED_STRUCTURAL_FINDINGS[capability]
        for source_path in record["evidence"]:
            assert source_path in inventory["source_files"]

    shared_query = inventory["capabilities"]["shared_query_r_out"]["details"]
    assert shared_query["shared_query"] is True
    assert shared_query["true_mimo"] is False

    # Hash drift is the first guard in production. Re-pin a deliberately changed
    # fixture to prove that the semantic predicates independently fail closed.
    _copy_inventory_sources(tmp_path)
    native_path = tmp_path / "gdn3" / "kmd2_native.py"
    source = native_path.read_text(encoding="utf-8")
    source = source.replace("theta.cumsum(dim=1)", "theta.cumsum(dim=0)", 1)
    native_path.write_text(source, encoding="utf-8")
    monkeypatch.setitem(
        module.PINNED_SOURCE_SHA256,
        "gdn3/kmd2_native.py",
        _sha256(native_path),
    )
    with pytest.raises(ValueError, match="structural.*cumulative.*rotation"):
        module.build_inventory(tmp_path)


def test_inventory_hashes_source_bytes_with_deterministic_sha256():
    module = _inventory_module()
    first = module.build_inventory(repo_root=REPO_ROOT)
    second = module.build_inventory(repo_root=REPO_ROOT)

    assert module.KMD2_NATIVE_SHA256 == EXPECTED_SOURCE_SHA256["gdn3/kmd2_native.py"]
    assert module.KMD2_FAST_SCAN_SHA256 == EXPECTED_SOURCE_SHA256["gdn3/kmd2_fast_scan.py"]
    assert module.GDN3_UPGRADE_SHA256 == EXPECTED_SOURCE_SHA256["gdn3/gdn3_upgrade.py"]
    assert (
        module.REFERENCE_RECURRENCE_SHA256
        == EXPECTED_SOURCE_SHA256["gdn3/_reference_recurrence.py"]
    )
    assert module.PINNED_SOURCE_SHA256 == EXPECTED_SOURCE_SHA256
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert first["source_files"] == EXPECTED_SOURCE_SHA256
    for relative_path, digest in first["source_files"].items():
        assert digest == _sha256(REPO_ROOT / relative_path)
        assert len(digest) == 64
        int(digest, 16)


def test_inventory_verification_rejects_stale_or_tampered_source_hash(tmp_path):
    module = _inventory_module()
    inventory = module.build_inventory(REPO_ROOT)
    _copy_inventory_sources(tmp_path)

    module.verify_inventory_sources(inventory, repo_root=tmp_path)

    missing = json.loads(json.dumps(inventory))
    missing["source_files"].pop("gdn3/kmd2_native.py")
    with pytest.raises(ValueError, match="missing.*gdn3/kmd2_native.py"):
        module.verify_inventory_sources(missing, repo_root=tmp_path)

    unexpected = json.loads(json.dumps(inventory))
    unexpected["source_files"]["gdn3/unapproved.py"] = "0" * 64
    with pytest.raises(ValueError, match="unexpected.*gdn3/unapproved.py"):
        module.verify_inventory_sources(unexpected, repo_root=tmp_path)

    declared_stale = json.loads(json.dumps(inventory))
    declared_stale["source_files"]["gdn3/kmd2_native.py"] = "0" * 64
    with pytest.raises(ValueError, match="gdn3/kmd2_native.py.*declared SHA-256"):
        module.verify_inventory_sources(declared_stale, repo_root=tmp_path)

    tampered = tmp_path / "gdn3" / "kmd2_native.py"
    tampered.write_bytes(tampered.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ValueError, match="gdn3/kmd2_native.py.*SHA-256"):
        module.verify_inventory_sources(inventory, repo_root=tmp_path)


def test_inventory_declares_backend_task_compatibility():
    inventory = _inventory_module().build_inventory(REPO_ROOT)
    compatibility = inventory["compatibility"]

    assert compatibility == {
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
    }
    assert inventory["compatibility_metadata"] == {
        "source": "suite_design",
        "production_derived": False,
    }
    for record in compatibility.values():
        assert record["tasks"] == sorted(record["tasks"])
        assert record["run_modes"] == sorted(record["run_modes"])


def test_inventory_uses_logical_external_qwen_assets_without_checkout_paths():
    inventory = _inventory_module().build_inventory(REPO_ROOT)

    assert inventory["external_assets"] == {
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
    }
    serialized = json.dumps(inventory, sort_keys=True)
    assert str(REPO_ROOT) not in serialized
    assert "C:\\Users\\" not in serialized
    assert re.search(r"(?i)[a-z]:\\\\", serialized) is None
    assert re.search(r'"/(?:home|Users|mnt|tmp)/', serialized) is None
