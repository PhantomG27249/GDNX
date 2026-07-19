from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.kmd2_ablation.results import (
    RESULT_SCHEMA_VERSION,
    ResultStore,
    RunRecordError,
    assign_shard,
    atomic_write_json,
    build_job,
    build_jobs_document,
    build_manifest,
    canonical_json_bytes,
    expanded_jobs_digest,
    quarantine_file,
    select_shard,
    semantic_job_id,
    validate_completed_run,
    validate_failed_run,
    write_immutable_json,
)
from research.kmd2_ablation.runner import (
    ForcedOOM,
    MalformedInput,
    NonFiniteGradient,
    NonFiniteLoss,
    build_completed_record,
    build_failed_record,
    execute_jobs,
    load_backend_dispatcher,
)


def _config(*, variant: str = "native") -> dict[str, object]:
    return {
        "backend": "tiny",
        "variant": variant,
        "task": {"name": "parity", "params": {"length": 16, "hold": True}},
        "budget": {"updates": 3, "tokens": 96},
    }


def _jobs() -> list[dict[str, object]]:
    return [
        build_job(
            _config(variant=variant),
            seed=seed,
            stage="mechanism_screen",
            backend="tiny",
            arm_id=variant,
        )
        for variant in ("native", "trapezoid")
        for seed in (11, 19, 23)
    ]


def _provenance() -> dict[str, object]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": "1.0.0",
        "source_hashes": {
            "research/kmd2_ablation/runner.py": "a" * 64,
            "gdn3/kmd2_native.py": "b" * 64,
        },
        "config_hash": hashlib.sha256(canonical_json_bytes(_config())).hexdigest(),
        "asset_hashes": {"task_fixture": "d" * 64},
        "git": {
            "revision": "0123456789abcdef",
            "diff_hash": "e" * 64,
            "dirty": True,
        },
        "environment": {
            "python": "3.12.4",
            "pytorch": "2.7.1",
            "cuda": None,
            "gpu": None,
            "dependencies": {"numpy": "2.3.0"},
        },
    }


@pytest.mark.parametrize(
    "arm_id",
    [
        "mimo-r2", "mimo-r4", "rout-4", "rot-off", "rot-constant",
        "rot-noncumulative", "rot-fixed-rope", "rot-moving-frame-oracle",
    ],
)
def test_canonical_architecture_job_emits_explicit_registry_identity(arm_id):
    from research.kmd2_ablation.architecture import registry_sha256

    config = {
        **_config(),
        "backend": "qwen",
        "architecture": {"arm_id": arm_id, "registry_sha256": registry_sha256()},
    }
    job = build_job(config, seed=11, stage="qwen_heal", backend="qwen", arm_id=arm_id)

    assert job["architecture_registry_sha256"] == registry_sha256()


@pytest.mark.parametrize(
    ("arm_id", "diagnostic"),
    [("rot-off", False), ("rot-moving-frame-oracle", True)],
)
def test_canonical_rotation_job_emits_explicit_diagnostic_mode(arm_id, diagnostic):
    from research.kmd2_ablation.architecture import registry_sha256

    config = {
        **_config(),
        "backend": "qwen",
        "architecture": {
            "arm_id": arm_id,
            "registry_sha256": registry_sha256(),
            "diagnostic_training": diagnostic,
        },
    }
    job = build_job(config, seed=11, stage="qwen_heal", backend="qwen", arm_id=arm_id)

    assert job["architecture_diagnostic_training"] is diagnostic


def _completed_record(
    job: dict[str, object],
    provenance: dict[str, object],
    *,
    num_jobs: int = 2,
) -> dict[str, object]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": "1.0.0",
        "status": "completed",
        "job_id": job["job_id"],
        "experiment_id": job["experiment_id"],
        "seed": job["seed"],
        "stage": job["stage"],
        "backend": job["backend"],
        "arm_id": job["arm_id"],
        "shard": {
            "index": assign_shard(str(job["job_id"]), num_jobs),
            "count": num_jobs,
        },
        "provenance": provenance,
        "canonical_config": job["canonical_config"],
        "metrics": {"accuracy": 0.75},
        "loss_curves": {"train": [1.0, 0.75]},
        "counts": {
            "nonfinite_loss": 0,
            "nonfinite_gradient": 0,
            "skipped_steps": 0,
        },
        "parameters": {"trainable": 123, "total": 456},
        "recurrent_state": {"elements": 64, "bytes": 256},
        "performance": {
            "wall_time_seconds": 1.25,
            "examples_per_second": 8.0,
            "tokens_per_second": 128.0,
            "peak_vram_bytes": 0,
        },
        "identities": {
            "checkpoint": {"kind": "none"},
            "data": {"sha256": "f" * 64},
        },
        "command": ["python", "-m", "research.kmd2_ablation.run_ablation", "run"],
    }


def _backend_payload() -> dict[str, object]:
    return {
        "metrics": {"accuracy": 0.75, "episode_exact": 0.5},
        "loss_curves": {"train": [1.0, 0.75], "validation": [0.9, 0.7]},
        "counts": {
            "nonfinite_loss": 0,
            "nonfinite_gradient": 0,
            "skipped_steps": 0,
        },
        "parameters": {"trainable": 123, "total": 456},
        "recurrent_state": {"elements": 64, "bytes": 256},
        "performance": {
            "wall_time_seconds": 1.25,
            "examples_per_second": 8.0,
            "tokens_per_second": 128.0,
            "peak_vram_bytes": 0,
        },
        "identities": {
            "checkpoint": {"kind": "none"},
            "data": {"sha256": "f" * 64},
        },
    }


def _exact_cache_diagnostics() -> dict[str, object]:
    return {
        "width": 64,
        "block_size": 128,
        "score_definition": "exact_outer",
        "compute_dtype": "fp32",
        "storage_dtype": "bf16",
        "coordinate_frame": "rotated_recurrence",
        "inclusive_causality": True,
        "tie_policy": "score_desc_position_desc",
        "amplitude_initial": [0.0, 0.0],
        "amplitude_final": [0.01, 0.03],
        "selected_index_digest": "1" * 64,
        "selected_index_sample": [3, 17, 42],
        "score_digest": "2" * 64,
        "score_statistics": {"count": 48, "min": 0.1, "max": 2.0, "mean": 0.7},
        "retention_count": 96,
        "eviction_count": 32,
        "persistent_hit_rate": 0.4,
        "conditional_read_accuracy": 0.6,
        "sink_mass": 0.1,
        "attention_entropy": 1.5,
        "top1_mass": 0.7,
        "stale_occupancy": 0.05,
        "stale_error": 0.02,
        "cache_output_norm": 3.0,
        "state_output_norm": 4.0,
        "persistent_bytes": 9_000_000,
        "block_bytes": 36_000_000,
        "implementation_paths": {
            "scan": "reference_loop",
            "score": "exact_outer_fp32",
            "selection": "block_topk_reference",
            "read": "cache_rmsnorm_reference",
        },
    }


@pytest.mark.parametrize(
    ("arm_id", "requires_cache"),
    [("native", False), ("recency", True), ("surprise", True)],
)
def test_qwen_heal_aliases_require_diagnostics_by_actual_cache_treatment(
    arm_id, requires_cache
):
    config = _config()
    config["backend"] = "qwen"
    config["mechanism"] = "exact_cache"
    job = build_job(
        config,
        seed=17,
        stage="qwen_heal",
        backend="qwen",
        arm_id=arm_id,
        pairing_id="a" * 64,
    )
    payload = _backend_payload()
    if requires_cache:
        with pytest.raises(RunRecordError) as caught:
            build_completed_record(
                job,
                _provenance(),
                shard_index=0,
                num_jobs=1,
                command=["python", "run"],
                payload=payload,
            )
        assert caught.value.code == "missing_exact_cache_diagnostics"
        payload["exact_cache"] = _exact_cache_diagnostics()

    record = build_completed_record(
        job,
        _provenance(),
        shard_index=0,
        num_jobs=1,
        command=["python", "run"],
        payload=payload,
    )
    assert record["arm_id"] == arm_id
    assert ("exact_cache" in record) is requires_cache


def test_tiny_matched_native_alias_does_not_require_treatment_cache_diagnostics():
    config = _config()
    config["backend"] = "tiny"
    config["mechanism"] = "exact_cache"
    job = build_job(
        config,
        seed=17,
        stage="selector_replay",
        backend="tiny",
        arm_id="native",
        pairing_id="a" * 64,
    )

    record = build_completed_record(
        job,
        _provenance(),
        shard_index=0,
        num_jobs=1,
        command=["python", "run"],
        payload=_backend_payload(),
    )

    assert record["arm_id"] == "native"
    assert "exact_cache" not in record


def test_semantic_job_identity_and_canonical_documents_ignore_mapping_order(tmp_path):
    config = _config()
    reordered = {
        "budget": {"tokens": 96, "updates": 3},
        "task": {"params": {"hold": True, "length": 16}, "name": "parity"},
        "variant": "native",
        "backend": "tiny",
    }
    first_id = semantic_job_id(
        config,
        backend="tiny",
        arm_id="native",
        seed=11,
        stage="mechanism_screen",
    )
    second_id = semantic_job_id(
        reordered,
        backend="tiny",
        arm_id="native",
        seed=11,
        stage="mechanism_screen",
    )
    assert first_id == second_id
    assert len(first_id) == 64

    first = build_job(
        config,
        seed=11,
        stage="mechanism_screen",
        backend="tiny",
        arm_id="native",
    )
    second = build_job(
        reordered,
        seed=11,
        stage="mechanism_screen",
        backend="tiny",
        arm_id="native",
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["job_id"] == first_id
    assert first["experiment_id"] == hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()

    document = build_jobs_document([first])
    path = tmp_path / "jobs.json"
    assert write_immutable_json(path, document) is True
    first_bytes = path.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert first_bytes == canonical_json_bytes(document) + b"\n"
    assert write_immutable_json(path, json.loads(first_bytes)) is False
    assert path.read_bytes() == first_bytes
    with pytest.raises(FileExistsError, match="immutable JSON conflict"):
        write_immutable_json(path, build_jobs_document(_jobs()))
    assert path.read_bytes() == first_bytes


def test_semantic_job_id_covers_every_explicit_semantic_dimension_without_collisions():
    base = {
        "canonical_config": _config(),
        "backend": "tiny",
        "arm_id": "native",
        "seed": 11,
        "stage": "mechanism_screen",
        "pairing_id": "pair-a",
    }
    reference = semantic_job_id(**base)
    assert len(reference) == 64
    mutations = {
        "backend": "qwen",
        "arm_id": "rotation.off",
        "seed": 12,
        "stage": "tiny_promotion",
        "pairing_id": "pair-b",
        "canonical_config": _config(variant="trapezoid"),
    }
    identities = {
        semantic_job_id(**{**base, field: value})
        for field, value in mutations.items()
    }
    assert reference not in identities
    assert len(identities) == len(mutations)
    assert semantic_job_id(**{**base, "pairing_id": None}) != reference


def test_manifest_contains_complete_provenance_and_expanded_job_digest(tmp_path):
    jobs = _jobs()
    provenance = _provenance()
    manifest = build_manifest(
        canonical_config=_config(),
        jobs=jobs,
        provenance=provenance,
        command=["python", "-m", "research.kmd2_ablation.run_ablation", "run"],
    )
    assert manifest == {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": "1.0.0",
        "canonical_config": _config(),
        "source_hashes": provenance["source_hashes"],
        "config_hash": hashlib.sha256(canonical_json_bytes(_config())).hexdigest(),
        "asset_hashes": provenance["asset_hashes"],
        "git": provenance["git"],
        "environment": provenance["environment"],
        "command": [
            "python",
            "-m",
            "research.kmd2_ablation.run_ablation",
            "run",
        ],
        "expanded_jobs_digest": expanded_jobs_digest(jobs),
    }
    manifest_path = tmp_path / "manifest.json"
    jobs_path = tmp_path / "jobs.json"
    write_immutable_json(manifest_path, manifest)
    write_immutable_json(jobs_path, build_jobs_document(jobs))
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"] == sorted(
        jobs, key=lambda job: job["job_id"]
    )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("git", "revision"),
        ("git", "diff_hash"),
        ("git", "dirty"),
        ("environment", "python"),
        ("environment", "pytorch"),
        ("environment", "cuda"),
        ("environment", "gpu"),
        ("environment", "dependencies"),
    ],
)
def test_manifest_rejects_incomplete_git_or_environment_provenance(section, field):
    provenance = deepcopy(_provenance())
    del provenance[section][field]
    with pytest.raises((TypeError, ValueError), match=field):
        build_manifest(
            canonical_config=_config(),
            jobs=_jobs(),
            provenance=provenance,
            command=["python", "run"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "suite",
        "source_provenance",
        "store_extra_provenance",
        "config_hash",
        "canonical_config",
        "command",
        "missing_identity",
        "extra_field",
        "jobs_digest",
    ],
)
def test_initialize_rejects_malformed_or_mismatched_manifest_before_publish(
    tmp_path, mutation
):
    jobs = _jobs()
    provenance = _provenance()
    manifest = build_manifest(
        canonical_config=_config(),
        jobs=jobs,
        provenance=provenance,
        command=["python", "run"],
    )
    invalid = deepcopy(manifest)
    if mutation == "schema":
        invalid["schema_version"] = "stale"
    elif mutation == "suite":
        invalid["suite_version"] = "stale"
    elif mutation == "source_provenance":
        invalid["source_hashes"] = {"runner.py": "0" * 64}
    elif mutation == "store_extra_provenance":
        pass
    elif mutation == "config_hash":
        invalid["config_hash"] = "0" * 64
    elif mutation == "canonical_config":
        invalid["canonical_config"] = {"different": True}
    elif mutation == "command":
        invalid["command"] = "python run"
    elif mutation == "missing_identity":
        del invalid["git"]
    elif mutation == "extra_field":
        invalid["unexpected"] = True
    else:
        invalid["expanded_jobs_digest"] = "0" * 64

    root = tmp_path / "results"
    store_provenance = deepcopy(provenance)
    if mutation == "store_extra_provenance":
        store_provenance["unexpected"] = "not represented by the manifest"
    store = ResultStore(
        root,
        provenance=store_provenance,
        job_index=0,
        num_jobs=1,
    )
    with pytest.raises((TypeError, ValueError)):
        store.initialize(manifest=invalid, jobs=jobs)

    assert not (root / "manifest.json").exists()
    assert not (root / "jobs.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "0" * 64),
        ("experiment_id", "1" * 64),
        ("schema_version", "stale"),
        ("suite_version", "stale"),
    ],
)
def test_jobs_document_recomputes_and_rejects_stale_job_identity(field, value):
    job = deepcopy(_jobs()[0])
    job[field] = value
    with pytest.raises(ValueError, match=field):
        build_jobs_document([job])


def test_staged_jobs_with_same_experiment_and_seed_have_distinct_run_paths(tmp_path):
    first = build_job(
        _config(),
        seed=11,
        stage="mechanism_screen",
        backend="tiny",
        arm_id="native",
    )
    second = build_job(
        _config(),
        seed=11,
        stage="tiny_promotion",
        backend="tiny",
        arm_id="native",
    )
    assert first["job_id"] != second["job_id"]
    assert len(build_jobs_document([first, second])["jobs"]) == 2
    store = ResultStore(
        tmp_path / "results",
        provenance=_provenance(),
        job_index=0,
        num_jobs=1,
    )
    assert store.run_path(first) != store.run_path(second)
    assert first["job_id"] in store.run_path(first).name
    assert second["job_id"] in store.run_path(second).name


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_assign_shard_rejects_invalid_shard_counts(invalid):
    with pytest.raises((TypeError, ValueError)):
        assign_shard("job", invalid)


def test_sha256_shards_are_disjoint_exhaustive_and_language_independent():
    jobs = _jobs()
    shard_count = 5
    expected = {
        job["job_id"]: int.from_bytes(
            hashlib.sha256(str(job["job_id"]).encode("utf-8")).digest()[:8],
            "big",
            signed=False,
        )
        % shard_count
        for job in jobs
    }
    assert {
        job["job_id"]: assign_shard(str(job["job_id"]), shard_count)
        for job in jobs
    } == expected

    shards = [select_shard(jobs, index, shard_count) for index in range(shard_count)]
    flattened = [job["job_id"] for shard in shards for job in shard]
    assert len(flattened) == len(set(flattened)) == len(jobs)
    assert set(flattened) == {job["job_id"] for job in jobs}

    config_json = json.dumps(_config(), sort_keys=False)
    script = (
        "import json; "
        "from research.kmd2_ablation.results import semantic_job_id, assign_shard; "
        f"c=json.loads({config_json!r}); "
        "j=semantic_job_id(c, backend='tiny', arm_id='native', "
        "seed=11, stage='mechanism_screen'); "
        "print(json.dumps([j, assign_shard(j, 5)]))"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=Path(__file__).parents[2],
                env=env,
                text=True,
            ).strip()
        )
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0]) == [
        semantic_job_id(
            _config(),
            backend="tiny",
            arm_id="native",
            seed=11,
            stage="mechanism_screen",
        ),
        assign_shard(
            semantic_job_id(
                _config(),
                backend="tiny",
                arm_id="native",
                seed=11,
                stage="mechanism_screen",
            ),
            5,
        ),
    ]


def test_subprocess_semantic_identity_and_shard_ignore_json_key_order():
    reordered = {
        "budget": {"tokens": 96, "updates": 3},
        "task": {"params": {"hold": True, "length": 16}, "name": "parity"},
        "variant": "native",
        "backend": "tiny",
    }
    encoded_configs = [
        json.dumps(_config(), separators=(",", ":")),
        json.dumps(reordered, separators=(",", ":")),
    ]
    assert encoded_configs[0] != encoded_configs[1]
    script = (
        "import json,sys; "
        "from research.kmd2_ablation.results import "
        "assign_shard,canonical_json_bytes,semantic_job_id; "
        "c=json.loads(sys.argv[1]); "
        "j=semantic_job_id(c,backend='tiny',arm_id='native',seed=11,"
        "stage='mechanism_screen'); "
        "sys.stdout.buffer.write(canonical_json_bytes([j,assign_shard(j,5)]))"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "123"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script, encoded],
            cwd=Path(__file__).parents[2],
            env=env,
        )
        for encoded in encoded_configs
    ]
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0]) == [
        semantic_job_id(
            _config(),
            backend="tiny",
            arm_id="native",
            seed=11,
            stage="mechanism_screen",
        ),
        assign_shard(
            semantic_job_id(
                _config(),
                backend="tiny",
                arm_id="native",
                seed=11,
                stage="mechanism_screen",
            ),
            5,
        ),
    ]


def test_atomic_json_concurrent_writers_never_publish_partial_records(tmp_path):
    path = tmp_path / "runs" / "job.json"
    records = [
        {"writer": index, "payload": "x" * (100 + index), "complete": True}
        for index in range(24)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(pool.map(lambda record: atomic_write_json(path, record), records))
    published = json.loads(path.read_text(encoding="utf-8"))
    assert published in records
    assert all(len(digest) == 64 for digest in digests)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_lock_release_retries_a_transient_windows_sharing_violation(
    tmp_path, monkeypatch
):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    lock = path.with_name(f".{path.name}.lock")
    real_unlink = Path.unlink
    attempts = 0

    def transiently_blocked(candidate, *args, **kwargs):
        nonlocal attempts
        if candidate == lock and attempts == 0:
            attempts += 1
            raise PermissionError(
                13,
                "simulated Windows sharing violation",
                str(candidate),
            )
        return real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", transiently_blocked)
    with results._creation_lock(path, timeout=1.0):
        pass

    assert attempts == 1
    assert not lock.exists()


def test_lock_acquisition_retries_when_a_transient_sharing_violation_outlives_lock(
    tmp_path, monkeypatch
):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    lock = path.with_name(f".{path.name}.lock")
    real_link = results.os.link
    attempts = 0

    def transiently_blocked(source, destination, *args, **kwargs):
        nonlocal attempts
        if Path(destination) == lock and attempts == 0:
            attempts += 1
            raise PermissionError(
                13,
                "simulated delete-pending Windows lock",
                str(destination),
            )
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(results.os, "link", transiently_blocked)
    with results._creation_lock(path, timeout=1.0):
        pass

    assert attempts == 1
    assert not lock.exists()


def test_lock_owner_publication_retries_short_writes_before_claiming(
    tmp_path, monkeypatch
):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    lock = path.with_name(f".{path.name}.lock")
    real_write = results.os.write
    writes = 0

    def short_write(descriptor, payload):
        nonlocal writes
        writes += 1
        chunk = memoryview(payload)[: max(1, len(payload) // 3)]
        return real_write(descriptor, chunk)

    monkeypatch.setattr(results.os, "write", short_write)
    with results._creation_lock(path, timeout=1.0):
        published = json.loads(lock.read_bytes())

    assert writes > 1
    assert published["version"] == 1
    assert published["pid"] == os.getpid()
    assert published["thread_id"] == threading.get_ident()
    assert isinstance(published["token"], str) and published["token"]
    assert not lock.exists()
    assert not list(tmp_path.glob(f"{lock.name}.*.tmp"))


def test_interrupted_lock_owner_publication_never_exposes_a_partial_lock(
    tmp_path, monkeypatch
):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    lock = path.with_name(f".{path.name}.lock")
    real_fsync = results.os.fsync
    interrupted = False

    def interrupt_first_fsync(descriptor):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OSError("simulated owner publication interruption")
        return real_fsync(descriptor)

    monkeypatch.setattr(results.os, "fsync", interrupt_first_fsync)
    with pytest.raises(OSError, match="owner publication interruption"):
        with results._creation_lock(path, timeout=1.0):
            pytest.fail("an interrupted owner record must never acquire the lock")

    assert interrupted
    assert not lock.exists()
    assert not list(tmp_path.glob(f"{lock.name}.*.tmp"))


def test_stale_malformed_lock_is_recovered_without_leaking_claims(tmp_path):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    lock = path.with_name(f".{path.name}.lock")
    lock.write_bytes(b'{"version":1,"partial":')
    os.utime(lock, (0, 0))

    with results._creation_lock(path, timeout=1.0):
        owner = json.loads(lock.read_bytes())
        assert owner["pid"] == os.getpid()

    assert not lock.exists()
    assert not list(tmp_path.glob(f"{lock.name}.claim-*"))
    assert not list(tmp_path.glob(f"{lock.name}.*.tmp"))


def test_stale_malformed_dead_owner_claim_is_recovered(tmp_path):
    path = tmp_path / "run.json"
    lock = path.with_name(f".{path.name}.lock")
    token = "dead-owner"
    lock.write_bytes(
        _lock_payload(pid=2_000_000_000, thread_id=1, token=token)
    )
    claim_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    claim = lock.with_name(f"{lock.name}.claim-{claim_digest}")
    claim.write_bytes(b'{"version":1,"partial":')
    os.utime(claim, (0, 0))
    script = (
        "import sys; "
        "from pathlib import Path; "
        "from research.kmd2_ablation.results import _creation_lock; "
        "p=Path(sys.argv[1]); "
        "cm=_creation_lock(p,timeout=1.0); cm.__enter__(); cm.__exit__(None,None,None)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).parents[2],
        timeout=3,
        check=False,
    )
    assert completed.returncode == 0
    assert not lock.exists()
    assert not claim.exists()
    assert not list(tmp_path.glob(f"{lock.name}.*.tmp"))


def test_atomic_json_interruption_preserves_prior_record_and_cleans_temp(
    tmp_path, monkeypatch
):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    atomic_write_json(path, {"generation": 1})
    before = path.read_bytes()

    def interrupt(_source, _destination):
        raise OSError("simulated interruption before replace")

    monkeypatch.setattr(results.os, "replace", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_json(path, {"generation": 2})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def _lock_payload(*, pid: int, thread_id: int, token: str) -> bytes:
    return canonical_json_bytes(
        {
            "version": 1,
            "host": socket.gethostname(),
            "pid": pid,
            "thread_id": thread_id,
            "token": token,
        }
    ) + b"\n"


def test_portable_lock_recovers_only_a_confirmed_dead_local_owner(tmp_path):
    from research.kmd2_ablation import results

    path = tmp_path / "manifest.json"
    lock = tmp_path / ".manifest.json.lock"
    lock.write_bytes(
        _lock_payload(pid=2_000_000_000, thread_id=1, token="dead-owner")
    )

    assert write_immutable_json(path, {"ok": True}) is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert not lock.exists()


def test_dead_lock_recovery_retries_a_transient_windows_sharing_violation(
    tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"
    lock = tmp_path / ".manifest.json.lock"
    lock.write_bytes(
        _lock_payload(pid=2_000_000_000, thread_id=1, token="dead-owner")
    )
    real_unlink = Path.unlink
    attempts = 0

    def transiently_blocked(candidate, *args, **kwargs):
        nonlocal attempts
        if candidate == lock and attempts == 0:
            attempts += 1
            raise PermissionError(
                13,
                "simulated Windows sharing violation",
                str(candidate),
            )
        return real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", transiently_blocked)
    assert write_immutable_json(path, {"ok": True}) is True

    assert attempts == 1
    assert not lock.exists()


def test_portable_lock_never_steals_a_live_owner_even_with_ancient_mtime(tmp_path):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    entered = threading.Event()
    release = threading.Event()

    def hold_lock():
        with results._creation_lock(path, timeout=1.0):
            entered.set()
            assert release.wait(2.0)

    owner = threading.Thread(target=hold_lock)
    owner.start()
    assert entered.wait(1.0)
    lock = path.with_name(f".{path.name}.lock")
    os.utime(lock, (0, 0))
    try:
        with pytest.raises(TimeoutError):
            with results._creation_lock(path, timeout=0.05):
                pytest.fail("live lock was stolen")
        assert lock.exists()
    finally:
        release.set()
        owner.join(timeout=2.0)
    assert not owner.is_alive()
    assert not lock.exists()


def test_lock_owner_token_prevents_prior_owner_from_unlinking_successor(tmp_path):
    from research.kmd2_ablation import results

    path = tmp_path / "run.json"
    lock = path.with_name(f".{path.name}.lock")
    successor = _lock_payload(
        pid=os.getpid(), thread_id=threading.get_ident(), token="successor-owner"
    )
    with results._creation_lock(path, timeout=1.0):
        lock.write_bytes(successor)
    assert lock.read_bytes() == successor
    lock.unlink()


def test_atomic_lock_is_process_safe_on_windows_and_posix(tmp_path):
    path = tmp_path / "shared.json"
    script = (
        "import sys; "
        "from research.kmd2_ablation.results import atomic_write_json; "
        "p=sys.argv[1]; w=int(sys.argv[2]); "
        "[atomic_write_json(p, {'writer':w,'step':s}) for s in range(8)]"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(path), str(writer)],
            cwd=Path(__file__).parents[2],
        )
        for writer in range(4)
    ]
    assert [process.wait(timeout=20) for process in processes] == [0, 0, 0, 0]
    published = json.loads(path.read_text(encoding="utf-8"))
    assert published["writer"] in range(4)
    assert published["step"] in range(8)
    assert not path.with_name(f".{path.name}.lock").exists()
    assert not path.with_name(f".{path.name}.lock.takeover").exists()


def test_quarantine_names_are_deterministic_and_preserve_truncated_bytes(tmp_path):
    root = tmp_path / "results"
    first = root / "runs" / "first.json"
    second = root / "runs" / "second.json"
    first.parent.mkdir(parents=True)
    raw = b'{"status":"completed","truncated":'
    first.write_bytes(raw)
    second.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    first_target = quarantine_file(
        first, root=root, job_id="job-17", reason="truncated"
    )
    second_target = quarantine_file(
        second, root=root, job_id="job-17", reason="truncated"
    )
    assert first_target == second_target
    assert first_target == (
        root / "quarantine" / "job-17" / f"truncated-{digest}.json"
    )
    assert first_target.read_bytes() == raw
    assert not first.exists() and not second.exists()


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        ("status", "inconclusive", "not_completed"),
        ("job_id", "wrong", "job_identity"),
        ("experiment_id", "wrong", "job_identity"),
        ("seed", 999, "job_identity"),
        ("canonical_config", {"wrong": True}, "config_mismatch"),
        ("provenance", {"wrong": True}, "provenance_mismatch"),
        ("shard", {"index": 99, "count": 2}, "shard_mismatch"),
    ],
)
def test_completed_record_validation_rejects_status_identity_and_provenance(
    field, replacement, code
):
    job = _jobs()[0]
    provenance = _provenance()
    record = _completed_record(job, provenance)
    validate_completed_run(record, job, provenance)
    invalid = deepcopy(record)
    invalid[field] = replacement
    with pytest.raises(RunRecordError) as caught:
        validate_completed_run(invalid, job, provenance)
    assert caught.value.code == code


@pytest.mark.parametrize(
    "field", ["scientific_classification", "scientific_label"]
)
def test_execution_records_reject_summary_only_scientific_fields(field):
    job = _jobs()[0]
    provenance = _provenance()

    completed = _completed_record(job, provenance)
    completed[field] = "inconclusive"
    with pytest.raises(RunRecordError) as caught:
        validate_completed_run(completed, job, provenance)
    assert caught.value.code == "summary_only_field"

    payload = _backend_payload()
    payload[field] = "inconclusive"
    with pytest.raises(RunRecordError):
        build_completed_record(
            job,
            provenance,
            shard_index=assign_shard(str(job["job_id"]), 2),
            num_jobs=2,
            command=["python", "run"],
            payload=payload,
        )

    failed = build_failed_record(
        job,
        provenance,
        shard_index=assign_shard(str(job["job_id"]), 2),
        num_jobs=2,
        command=["python", "run"],
        error=MalformedInput("bad input", phase="input"),
        traceback_text="trace",
    )
    failed[field] = "inconclusive"
    with pytest.raises(RunRecordError) as caught:
        validate_failed_run(failed, job, provenance)
    assert caught.value.code == "summary_only_field"


def _store(tmp_path, job, provenance, *, num_jobs: int = 3) -> ResultStore:
    return ResultStore(
        tmp_path / "results",
        provenance=provenance,
        job_index=assign_shard(str(job["job_id"]), num_jobs),
        num_jobs=num_jobs,
    )


def test_resume_authority_is_only_a_valid_completed_run_and_jobs_stay_immutable(
    tmp_path,
):
    job = _jobs()[0]
    provenance = _provenance()
    store = _store(tmp_path, job, provenance)
    jobs = _jobs()
    manifest = build_manifest(
        canonical_config=_config(),
        jobs=jobs,
        provenance=provenance,
        command=["python", "-m", "research.kmd2_ablation.run_ablation", "run"],
    )
    store.initialize(manifest=manifest, jobs=jobs)
    jobs_before = (store.root / "jobs.json").read_bytes()

    # Absence and event-stream messages never provide completion authority.
    store.append_event({"event": "completed", "job_id": job["job_id"]})
    assert store.should_run(job) is True

    failed = _completed_record(job, provenance, num_jobs=store.num_jobs)
    failed["status"] = "failed"
    failed["error"] = {"code": "execution_error", "message": "retry me"}
    atomic_write_json(store.run_path(job), failed)
    assert store.should_run(job) is True

    # A later summary may classify this run as inconclusive without mutating
    # the authoritative execution record.
    completed = _completed_record(job, provenance, num_jobs=store.num_jobs)
    atomic_write_json(store.run_path(job), completed)
    assert store.should_run(job) is False
    assert (store.root / "jobs.json").read_bytes() == jobs_before


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("stale_source", "stale-provenance"),
        ("stale_config", "stale-config"),
        ("wrong_job", "conflicting-identity"),
        ("wrong_shard", "stale-shard"),
    ],
)
def test_resume_quarantines_stale_or_conflicting_completed_records(
    tmp_path, mutation, reason
):
    job = _jobs()[0]
    provenance = _provenance()
    store = _store(tmp_path, job, provenance)
    record = _completed_record(job, provenance, num_jobs=store.num_jobs)
    if mutation == "stale_source":
        record["provenance"] = deepcopy(provenance)
        record["provenance"]["source_hashes"] = {"runner.py": "0" * 64}
    elif mutation == "stale_config":
        record["canonical_config"] = {"changed": True}
    elif mutation == "wrong_job":
        record["job_id"] = "f" * 64
    else:
        record["shard"] = {"index": (store.job_index + 1) % store.num_jobs, "count": store.num_jobs}
    atomic_write_json(store.run_path(job), record)

    assert store.should_run(job) is True
    assert not store.run_path(job).exists()
    quarantined = list((store.root / "quarantine" / str(job["job_id"])).glob("*.json"))
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith(f"{reason}-")


def test_resume_quarantines_truncated_noncanonical_and_interrupted_files(tmp_path):
    job = _jobs()[0]
    provenance = _provenance()
    store = _store(tmp_path, job, provenance)
    path = store.run_path(job)
    path.parent.mkdir(parents=True)

    path.write_bytes(b'{"status":"completed"')
    assert store.should_run(job) is True
    assert any(
        item.name.startswith("truncated-")
        for item in (store.root / "quarantine" / str(job["job_id"])).glob("*.json")
    )

    record = _completed_record(job, provenance, num_jobs=store.num_jobs)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    assert store.should_run(job) is True
    assert any(
        item.name.startswith("noncanonical-")
        for item in (store.root / "quarantine" / str(job["job_id"])).glob("*.json")
    )

    interrupted = path.with_name(f".{path.name}.dead-worker.tmp")
    interrupted.write_bytes(b'{"partial":')
    assert store.should_run(job) is True
    assert not interrupted.exists()
    assert any(
        item.name.startswith("interrupted-temp-")
        for item in (store.root / "quarantine" / str(job["job_id"])).glob("*.json")
    )


def test_result_store_detects_and_quarantines_conflicting_completed_writers(tmp_path):
    job = _jobs()[0]
    provenance = _provenance()
    store = _store(tmp_path, job, provenance)
    first = _completed_record(job, provenance, num_jobs=store.num_jobs)
    second = deepcopy(first)
    second["metrics"]["accuracy"] = 0.5
    assert store.persist(job, first) is True
    assert store.persist(job, deepcopy(first)) is False
    with pytest.raises(RunRecordError) as caught:
        store.persist(job, second)
    assert caught.value.code == "conflicting_completed"
    assert not store.run_path(job).exists()
    conflicts = list(
        (store.root / "quarantine" / str(job["job_id"])).glob(
            "conflicting-completed-*.json"
        )
    )
    assert len(conflicts) == 2
    assert store.should_run(job) is True


def test_completed_builder_requires_all_typed_diagnostics_and_provenance():
    job = _jobs()[0]
    provenance = _provenance()
    payload = _backend_payload()
    shard_count = 3
    shard_index = assign_shard(str(job["job_id"]), shard_count)
    record = build_completed_record(
        job,
        provenance,
        shard_index=shard_index,
        num_jobs=shard_count,
        command=["python", "-m", "research.kmd2_ablation.run_ablation", "run"],
        payload=payload,
    )
    validate_completed_run(record, job, provenance)
    assert record["metrics"] == payload["metrics"]
    assert record["canonical_config"] == job["canonical_config"]
    assert record["provenance"]["environment"] == provenance["environment"]

    for missing in (
        "metrics",
        "loss_curves",
        "counts",
        "parameters",
        "recurrent_state",
        "performance",
        "identities",
    ):
        invalid = deepcopy(payload)
        del invalid[missing]
        with pytest.raises(RunRecordError) as caught:
            build_completed_record(
                job,
                provenance,
                shard_index=shard_index,
                num_jobs=shard_count,
                command=["python", "run"],
                payload=invalid,
            )
        assert caught.value.code == "missing_diagnostics"

    invalid = deepcopy(payload)
    invalid["loss_curves"]["train"][0] = float("nan")
    with pytest.raises(RunRecordError) as caught:
        build_completed_record(
            job,
            provenance,
            shard_index=shard_index,
            num_jobs=shard_count,
            command=["python", "run"],
            payload=invalid,
        )
    assert caught.value.code == "invalid_diagnostics"


@pytest.mark.parametrize(
    "missing",
    [
        "score_definition",
        "storage_dtype",
        "selected_index_digest",
        "score_statistics",
        "persistent_hit_rate",
        "conditional_read_accuracy",
        "stale_error",
        "persistent_bytes",
        "implementation_paths",
    ],
)
def test_exact_cache_completed_records_require_full_mechanistic_diagnostics(missing):
    config = _config(variant="top_surprise")
    config["mechanism"] = "exact_cache"
    job = build_job(
        config,
        seed=11,
        stage="capacity_screen",
        backend="tiny",
        arm_id="exact_cache.selector.exact_outer",
    )
    payload = _backend_payload()
    payload["exact_cache"] = _exact_cache_diagnostics()
    del payload["exact_cache"][missing]
    with pytest.raises(RunRecordError) as caught:
        build_completed_record(
            job,
            _provenance(),
            shard_index=0,
            num_jobs=1,
            command=["python", "run"],
            payload=payload,
        )
    assert caught.value.code == "missing_exact_cache_diagnostics"


def test_exact_cache_completed_record_pins_precision_causality_and_ties():
    config = _config(variant="top_surprise")
    config["mechanism"] = "exact_cache"
    job = build_job(
        config,
        seed=11,
        stage="capacity_screen",
        backend="tiny",
        arm_id="exact_cache.selector.exact_outer",
    )
    payload = _backend_payload()
    payload["exact_cache"] = _exact_cache_diagnostics()
    record = build_completed_record(
        job,
        _provenance(),
        shard_index=0,
        num_jobs=1,
        command=["python", "run"],
        payload=payload,
    )
    assert record["exact_cache"]["compute_dtype"] == "fp32"
    assert record["exact_cache"]["storage_dtype"] == "bf16"
    assert record["exact_cache"]["inclusive_causality"] is True
    assert record["exact_cache"]["tie_policy"] == "score_desc_position_desc"

    invalid = deepcopy(record)
    invalid["exact_cache"]["compute_dtype"] = "bf16"
    with pytest.raises(RunRecordError) as caught:
        validate_completed_run(invalid, job, _provenance())
    assert caught.value.code == "invalid_exact_cache_diagnostics"


def _oom_context() -> dict[str, object]:
    return {
        "batch_size": 2,
        "sequence_length": 4096,
        "num_heads": 4,
        "state_key_dim": 32,
        "state_value_dim": 64,
        "cache_width": 64,
        "block_size": 128,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "estimated_bytes": 123_456_789,
        "peak_vram_bytes": 22_000_000_000,
    }


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (NonFiniteLoss("loss became NaN", phase="train"), "nonfinite_loss"),
        (
            NonFiniteGradient("gradient became Inf", phase="backward"),
            "nonfinite_gradient",
        ),
        (MalformedInput("padding is unsupported", phase="input"), "malformed_input"),
    ],
)
def test_failed_records_have_distinct_typed_codes(error, code):
    job = _jobs()[0]
    provenance = _provenance()
    record = build_failed_record(
        job,
        provenance,
        shard_index=0,
        num_jobs=1,
        command=["python", "run"],
        error=error,
        traceback_text="trace\n" * 5000,
    )
    validate_failed_run(record, job, provenance)
    assert record["error"]["code"] == code
    assert len(record["error"]["traceback"]) <= 8192
    assert record["canonical_config"] == job["canonical_config"]


def test_forced_oom_record_preserves_full_shape_resource_and_phase_context():
    job = _jobs()[0]
    provenance = _provenance()
    record = build_failed_record(
        job,
        provenance,
        shard_index=0,
        num_jobs=1,
        command=["python", "run"],
        error=ForcedOOM("forced test OOM", phase="read", context=_oom_context()),
        traceback_text="forced traceback",
    )
    validate_failed_run(record, job, provenance)
    assert record["error"] == {
        "code": "oom",
        "message": "forced test OOM",
        "phase": "read",
        "context": _oom_context(),
        "traceback": "forced traceback",
    }

    missing = _oom_context()
    del missing["cache_width"]
    with pytest.raises(RunRecordError) as caught:
        build_failed_record(
            job,
            provenance,
            shard_index=0,
            num_jobs=1,
            command=["python", "run"],
            error=ForcedOOM("bad OOM", phase="read", context=missing),
            traceback_text="trace",
        )
    assert caught.value.code == "invalid_oom_context"


def test_backend_dispatch_is_lazy_and_resolves_only_requested_module(monkeypatch):
    import research.kmd2_ablation.runner as runner

    calls = []
    sentinel = lambda job: job

    def fake_import(name):
        calls.append(name)
        return SimpleNamespace(run_job=sentinel)

    monkeypatch.setattr(runner.importlib, "import_module", fake_import)
    assert load_backend_dispatcher("tiny") is sentinel
    assert calls == ["research.kmd2_ablation.tiny_training"]
    calls.clear()
    assert load_backend_dispatcher("qwen") is sentinel
    assert calls == ["research.kmd2_ablation.qwen_training"]
    with pytest.raises(ValueError, match="unsupported backend"):
        load_backend_dispatcher("unknown")


def test_runner_persists_failure_continues_and_resume_skips_only_completed(tmp_path):
    jobs = [_jobs()[0], _jobs()[3]]
    provenance = _provenance()
    store = ResultStore(
        tmp_path / "results", provenance=provenance, job_index=0, num_jobs=1
    )
    calls = []

    def dispatch(job):
        calls.append(job["job_id"])
        if job["arm_id"] == "native":
            raise ForcedOOM("forced OOM", phase="backward", context=_oom_context())
        return _backend_payload()

    outcomes = execute_jobs(
        list(reversed(jobs)),
        store=store,
        dispatchers={"tiny": dispatch},
        command=["python", "run"],
        resume=True,
    )
    ordered = sorted(jobs, key=lambda item: item["job_id"])
    assert [outcome["status"] for outcome in outcomes] == [
        "failed" if job["arm_id"] == "native" else "completed" for job in ordered
    ]
    assert calls == [job["job_id"] for job in ordered]
    records = {
        job["arm_id"]: json.loads(store.run_path(job).read_text(encoding="utf-8"))
        for job in jobs
    }
    assert records["native"]["error"]["code"] == "oom"
    assert records["trapezoid"]["status"] == "completed"

    calls.clear()
    resumed = execute_jobs(
        jobs,
        store=store,
        dispatchers={"tiny": dispatch},
        command=["python", "run"],
        resume=True,
    )
    assert [outcome["status"] for outcome in resumed] == [
        "failed" if job["arm_id"] == "native" else "skipped" for job in ordered
    ]
    assert calls == [next(job["job_id"] for job in jobs if job["arm_id"] == "native")]
    assert all(json.loads(path.read_text(encoding="utf-8"))["status"] in {"failed", "completed"} for path in (store.run_path(job) for job in jobs))


def test_runner_propagates_persistence_errors_without_writing_a_failed_record(
    tmp_path, monkeypatch
):
    job = _jobs()[0]
    provenance = _provenance()
    store = _store(tmp_path, job, provenance)
    persisted_statuses = []

    def fail_persist(_job, record):
        persisted_statuses.append(record["status"])
        raise OSError("simulated result filesystem failure")

    monkeypatch.setattr(store, "persist", fail_persist)
    with pytest.raises(OSError, match="result filesystem failure"):
        execute_jobs(
            [job],
            store=store,
            dispatchers={"tiny": lambda _job: _backend_payload()},
            command=["python", "run"],
            resume=False,
        )

    assert persisted_statuses == ["completed"]
    assert not store.run_path(job).exists()


def test_runner_preserves_generator_exit_without_persisting_a_record(tmp_path):
    job = _jobs()[0]
    provenance = _provenance()
    store = _store(tmp_path, job, provenance)

    def stop_control_flow(_job):
        raise GeneratorExit("stop worker")

    with pytest.raises(GeneratorExit, match="stop worker"):
        execute_jobs(
            [job],
            store=store,
            dispatchers={"tiny": stop_control_flow},
            command=["python", "run"],
            resume=False,
        )

    assert not store.run_path(job).exists()


def test_two_concurrent_shards_have_disjoint_complete_union_and_shared_immutable_jobs(
    tmp_path,
):
    jobs = _jobs()
    provenance = _provenance()
    root = tmp_path / "results"
    stores = [
        ResultStore(root, provenance=provenance, job_index=index, num_jobs=2)
        for index in range(2)
    ]
    manifest = build_manifest(
        canonical_config=_config(),
        jobs=jobs,
        provenance=provenance,
        command=["python", "run"],
    )

    def run_shard(store):
        store.initialize(manifest=manifest, jobs=jobs)
        return execute_jobs(
            jobs,
            store=store,
            dispatchers={"tiny": lambda _job: _backend_payload()},
            command=["python", "run"],
            resume=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        shard_results = list(pool.map(run_shard, stores))
    identifiers = [
        outcome["job_id"] for results in shard_results for outcome in results
    ]
    assert len(identifiers) == len(set(identifiers)) == len(jobs)
    assert set(identifiers) == {job["job_id"] for job in jobs}
    assert all(
        outcome["status"] == "completed"
        for results in shard_results
        for outcome in results
    )
    jobs_bytes = (root / "jobs.json").read_bytes()
    assert jobs_bytes == canonical_json_bytes(build_jobs_document(jobs)) + b"\n"

    resumed = [run_shard(store) for store in stores]
    assert all(
        outcome["status"] == "skipped"
        for results in resumed
        for outcome in results
    )
    assert (root / "jobs.json").read_bytes() == jobs_bytes
def test_hybrid_diagnostic_schema_is_exact_finite_and_serializable() -> None:
    from research.kmd2_ablation.results import canonical_json_bytes, validate_hybrid_diagnostics

    record = {
        "rank_update_norms": [1.0, 2.0, 3.0, 4.0], "rank_similarity": 0.2,
        "effective_rank": 3.0, "phase_magnitude": 0.1, "phase_frequency": 0.2,
        "effective_decay_horizons": [1.0, 16.0, 64.0, 256.0],
        "decay_horizon_ratios": [1.0, 16.0, 64.0, 256.0],
        "all_lanes_update_each_token": False, "state_router_active": False,
        "cache_admission_rate": 0.5,
        "cache_occupancy": 0.25,
        "cache_mean_age": 6.0, "cache_read_entropy": 0.7, "cache_gate_mean": 0.8,
        "state_norm": 1.0, "gate_mean": 0.4, "nonfinite_count": 0,
        "flops_per_token": 100.0, "tokens_per_second": 20.0,
        "peak_memory_bytes": 1024, "capacity_confounded": False,
        "cache_admissions": 8,
        "cache_opportunities": 8, "cache_schema": "states", "layer_count": 2,
    }
    assert validate_hybrid_diagnostics(record) == record
    assert b'"capacity_confounded":false' in canonical_json_bytes(record)
    bad = dict(record, rank_update_norms=[1.0, float("nan"), 3.0, 4.0])
    with pytest.raises(Exception, match="rank_update_norms"):
        validate_hybrid_diagnostics(bad)
