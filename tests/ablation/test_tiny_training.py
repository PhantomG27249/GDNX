from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.kmd2_ablation.config import CacheConfig, ExperimentConfig
from research.kmd2_ablation.results import (
    ResultStore,
    _EXACT_CACHE_FIELDS,
    assign_shard,
    build_job,
    canonical_json_bytes,
)
from research.kmd2_ablation.runner import (
    ForcedOOM,
    MalformedInput,
    NonFiniteGradient,
    build_completed_record,
    execute_jobs,
)
from research.kmd2_ablation.tasks import EpisodeBatch, generate_task
from research.kmd2_ablation.tiny_backend import TinyKMD2Config, TinyKMD2Model
from research.kmd2_ablation.tiny_training import (
    TINY_CHECKPOINT_SCHEMA_VERSION,
    TinyExecutionDependencies,
    TinyRuntimeConfigurationError,
    TinyTrainer,
    TinyTrainingConfig,
    _exact_cache_payload,
    build_job_dispatcher,
    run_job,
)
from research.kmd2_ablation.variants import all_variants, get_variant
from tests.ablation.test_config import minimal_config_dict


def _model_config(*, cache: bool = False, modality: str = "token") -> TinyKMD2Config:
    if modality == "affine":
        dk, dv, continuous_dim, output_dim, vocab_size = 3, 2, None, 2, 32
        rotation_mode = "none"
    elif modality == "continuous":
        dk, dv, continuous_dim, output_dim, vocab_size = 2, 2, 3, 1, 32
        rotation_mode = "none"
    else:
        dk, dv, continuous_dim, output_dim, vocab_size = 2, 2, None, None, 8
        rotation_mode = "none"
    cache_config = (
        CacheConfig(
            width=2,
            block_size=2,
            read="rmsnorm",
            storage_dtype="fp32",
            lr_cache=0.02,
        )
        if cache
        else None
    )
    return TinyKMD2Config(
        d_model=8,
        heads=1,
        dk=dk,
        dv=dv,
        layers=1,
        vocab_size=vocab_size,
        d_ff=16,
        r_out=1,
        mimo_rank=1,
        continuous_input_dim=continuous_dim,
        output_dim=output_dim,
        conv_kernel=3,
        dtype=torch.float32,
        eps=1.0e-6,
        rotation_mode=rotation_mode,
        convolution_gate_init=0.0,
        rotation_gate_init=0.0,
        channel_decay_gate_init=0.0,
        write_offset_gate_init=0.0,
        cache=cache_config,
    )


def _training_config(job_id: str = "tiny-job") -> TinyTrainingConfig:
    return TinyTrainingConfig(
        job_id=job_id,
        seed=211,
        updates=10,
        max_tokens=100_000,
        learning_rate=0.01,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.01,
        warmup_updates=2,
        max_grad_norm=1.0,
    )


def test_tiny_run_job_requires_explicit_runtime_binding() -> None:
    with pytest.raises(
        TinyRuntimeConfigurationError, match="runtime_required"
    ) as caught:
        run_job({})
    assert caught.value.code == "runtime_required"


def _dispatcher_job(
    arm_id: str = "native",
    *,
    task: str = "parity",
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    spec = get_variant(arm_id)
    raw = minimal_config_dict()
    raw["backend"] = "tiny"
    raw["mechanism"] = spec.mechanism
    raw["variant"] = spec.variant
    raw["task"] = {"name": task, "params": {} if params is None else params}
    raw["seeds"] = [211]
    raw["budget"] = {
        "tokens": 28 if task == "mqar" else 12,
        "updates": 2,
    }
    raw["schedule"]["warmup_updates"] = 1
    raw["model"] = {
        "hidden_size": 8,
        "num_layers": 1,
        "num_heads": 1,
        "state_key_dim": 2,
        "state_value_dim": 2,
        "ffn_dim": 16,
        "ffn_match_lower": 8,
        "ffn_match_upper": 24,
    }
    raw["lengths"] = {
        "curriculum": [4 if task == "mqar" else 2],
        "extrapolation": [2 if task == "irregular_integration" else 1],
    }
    raw["evaluation"] = {"primary_metric": "exact_match", "direction": "maximize"}
    raw["device_preferences"] = ["cpu"]
    raw["dtype_preferences"] = ["float32"]
    raw["required_stage"] = spec.required_stage
    raw["cache"].update(
        {
            "width": 0 if arm_id == "exact_cache.current_block_only" else 2,
            "block_size": 2,
            "storage_dtype": "fp32",
        }
    )
    selector_scores = {
        "exact_cache.selector.exact_outer": "exact_outer",
        "exact_cache.selector.coupled_paper": "coupled_paper",
        "exact_cache.selector.residual_only": "residual_only",
        "exact_cache.selector.write_value": "write_value",
        "exact_cache.selector.recency": "recency",
        "exact_cache.selector.reservoir": "reservoir",
        "exact_cache.selector.future_query_oracle": "future_query_oracle",
    }
    if arm_id in selector_scores:
        raw["cache"]["score"] = selector_scores[arm_id]
    config = ExperimentConfig.from_dict(raw)
    return build_job(
        config,
        seed=211,
        stage=spec.required_stage,
        backend="tiny",
        arm_id=arm_id,
    )


def _dispatcher_runtime(tmp_path: Path, *, resume: bool = True) -> dict[str, object]:
    return {
        "output": tmp_path,
        "dtype": "float32",
        "asset_hashes": {},
        "resume": resume,
    }


def _provenance_for(job: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "suite_version": "1.0.0",
        "source_hashes": {"tiny_training.py": "a" * 64},
        "config_hash": hashlib.sha256(
            canonical_json_bytes(job["canonical_config"])
        ).hexdigest(),
        "asset_hashes": {},
        "git": {"revision": "abc", "diff_hash": "b" * 64, "dirty": True},
        "environment": {
            "python": "test",
            "pytorch": str(torch.__version__),
            "cuda": None,
            "gpu": None,
            "dependencies": {},
        },
    }


def test_tiny_bound_dispatcher_runs_exact_budget_and_builds_valid_record(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job()
    ticks = iter((10.0, 12.0))
    dispatcher = build_job_dispatcher(
        _dispatcher_runtime(tmp_path),
        dependencies={"monotonic": lambda: next(ticks)},
    )
    payload = dispatcher(job)
    assert payload["loss_curves"]["train"] and len(
        payload["loss_curves"]["train"]
    ) == 2
    assert payload["training"] == {
        "updates_completed": 2,
        "tokens_seen": 12,
        "examples_seen": 2,
    }
    assert payload["parameters"]["trainable"] == payload["parameters"]["total"]
    assert payload["performance"]["wall_time_seconds"] == 2.0
    assert payload["performance"]["peak_vram_bytes"] == 0
    provenance = _provenance_for(job)
    shard_index = assign_shard(str(job["job_id"]), 1)
    record = build_completed_record(
        job,
        provenance,
        shard_index=shard_index,
        num_jobs=1,
        command=["python", "-m", "research.kmd2_ablation.run_ablation", "run"],
        payload=payload,
    )
    assert record["status"] == "completed"


def test_tiny_cache_dispatcher_reports_measured_exact_cache_fields(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job(
        "exact_cache.selector.exact_outer", task="mqar", params={"width": 4}
    )
    ticks = iter((20.0, 21.0))
    payload = build_job_dispatcher(
        _dispatcher_runtime(tmp_path),
        dependencies={"monotonic": lambda: next(ticks)},
    )(job)
    diagnostics = payload["exact_cache"]
    assert diagnostics["width"] == 2
    assert diagnostics["block_size"] == 2
    assert diagnostics["retention_count"] > 0
    assert diagnostics["eviction_count"] > 0
    assert diagnostics["score_statistics"]["count"] > 0
    assert len(diagnostics["selected_index_digest"]) == 64
    assert len(diagnostics["score_digest"]) == 64
    assert diagnostics["implementation_paths"] == {
        "scan": "tiny_backend.TinyKMD2Cell._forward_fp32",
        "score": "exact_cache.admission_scores.exact_outer",
        "selection": "exact_cache.merge_persistent_cache.deterministic_topw",
        "read": "exact_cache.cache_read_blocks.unit_l2",
    }
    build_completed_record(
        job,
        _provenance_for(job),
        shard_index=0,
        num_jobs=1,
        command=["python", "run"],
        payload=payload,
    )


@pytest.mark.parametrize(
    "arm_id",
    [
        "exact_cache.selector.coupled_paper",
        "exact_cache.selector.residual_only",
        "exact_cache.selector.write_value",
        "exact_cache.selector.recency",
        "exact_cache.selector.reservoir",
        "exact_cache.selector.future_query_oracle",
    ],
)
def test_tiny_selector_policy_arms_execute_real_jobs(
    arm_id: str, tmp_path: Path
) -> None:
    job = _dispatcher_job(arm_id, task="mqar", params={"width": 4})
    payload = build_job_dispatcher(_dispatcher_runtime(tmp_path))(job)
    expected = {
        "exact_cache.selector.coupled_paper": "coupled_paper",
        "exact_cache.selector.residual_only": "residual_only",
        "exact_cache.selector.write_value": "write_value",
        "exact_cache.selector.recency": "recency",
        "exact_cache.selector.reservoir": "reservoir",
        "exact_cache.selector.future_query_oracle": "future_query_oracle",
    }[arm_id]
    assert payload["exact_cache"]["score_definition"] == expected
    assert payload["exact_cache"]["selector_seed"] == job["seed"]
    assert payload["exact_cache"]["selector_policy"] == expected
    assert payload["training"]["updates_completed"] == 2


def _cache_alias_job(
    arm_id: str,
    *,
    cache_updates: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    base = _dispatcher_job(
        "exact_cache.selector.exact_outer",
        task="mqar",
        params={"width": 4, **({} if params is None else params)},
    )
    semantic = copy.deepcopy(base["canonical_config"])
    spec = get_variant(arm_id)
    semantic["required_stage"] = spec.required_stage
    if cache_updates:
        semantic["cache"].update(cache_updates)
    return build_job(
        semantic,
        seed=base["seed"],
        stage=spec.required_stage,
        backend="tiny",
        arm_id=arm_id,
    )


@pytest.mark.parametrize(
    ("arm_id", "cache_updates"),
    [
        ("exact_cache.read.unit_l2", {"read": "unit_l2"}),
        ("exact_cache.read.fixed_temperature", {"read": "fixed_temperature"}),
        ("exact_cache.read.rmsnorm", {"read": "rmsnorm"}),
        ("exact_cache.storage.bf16", {"storage_dtype": "bf16"}),
        ("exact_cache.storage.fp32", {"storage_dtype": "fp32"}),
        (
            "exact_cache.pre_rotation_diagnostic",
            {"coordinate_frame": "pre_rotation", "pre_rotation_diagnostic": True},
        ),
    ],
)
def test_tiny_cache_alias_arms_validate_declared_control_and_execute(
    arm_id: str, cache_updates: dict[str, object], tmp_path: Path
) -> None:
    job = _cache_alias_job(arm_id, cache_updates=cache_updates)
    payload = build_job_dispatcher(_dispatcher_runtime(tmp_path))(job)
    diagnostics = payload["exact_cache"]
    if "read" in cache_updates:
        assert diagnostics["implementation_paths"]["read"].endswith(
            str(cache_updates["read"])
        )
    if "storage_dtype" in cache_updates:
        assert diagnostics["storage_dtype"] == cache_updates["storage_dtype"]
    if "coordinate_frame" in cache_updates:
        assert diagnostics["coordinate_frame"] == "pre_rotation"


@pytest.mark.parametrize(
    ("arm_id", "params"),
    [
        ("exact_cache.per_slot_read", {"width": 4, "r_out": 4}),
        ("exact_cache.unbounded_oracle", {"width": 4}),
    ],
)
def test_tiny_cache_diagnostic_arms_execute_their_real_backend_paths(
    arm_id: str, params: dict[str, object], tmp_path: Path
) -> None:
    job = _dispatcher_job(arm_id, task="mqar", params=params)
    diagnostics = build_job_dispatcher(_dispatcher_runtime(tmp_path))(job)[
        "exact_cache"
    ]
    if arm_id.endswith("per_slot_read"):
        assert diagnostics["per_slot_read"] is True
        assert diagnostics["slot_count"] == 4
        assert diagnostics["slot_top1_position_digest"] != hashlib.sha256(
            b"[]"
        ).hexdigest()
    else:
        assert diagnostics["unbounded_cache"] is True
        assert diagnostics["effective_width"] > diagnostics["declared_width"]
        assert diagnostics["eviction_count"] == 0


def test_tiny_per_slot_diagnostics_ignore_invalid_padding() -> None:
    cache = CacheConfig(width=1, block_size=2, storage_dtype="fp32")
    model_config = TinyKMD2Config(
        d_model=8,
        heads=1,
        dk=2,
        dv=2,
        layers=1,
        vocab_size=8,
        d_ff=16,
        r_out=2,
        rotation_mode="none",
        cache=cache,
        per_slot_cache_read=True,
    )
    episode = EpisodeBatch(
        task="mqar",
        split="id",
        seed=431,
        example_ids=("padded",),
        input_ids=torch.tensor([[1, 0, 2]]),
        continuous_inputs=None,
        direct_factors=None,
        targets=torch.tensor([[-100, -100, 3]]),
        valid=torch.tensor([[True, False, True]]),
        positions=torch.tensor([[0, -1, 1]]),
        loss_mask=torch.tensor([[False, False, True]]),
        query_mask=torch.tensor([[False, False, True]]),
        boundaries=torch.tensor([[True, False, False]]),
        source_spans=torch.tensor([[[-1, -1], [-1, -1], [0, 1]]]),
        strata={},
        metadata=({},),
    )
    with torch.no_grad():
        output = TinyKMD2Model(model_config, init_seed=431).forward_episode(episode)
    cell = output.cell_outputs[0]

    def poison_invalid(tensor: torch.Tensor, value: float | int) -> torch.Tensor:
        poisoned = tensor.clone()
        poisoned[~episode.valid] = value
        return poisoned

    poisoned_cell = replace(
        cell,
        slot_cache_read=poison_invalid(cell.slot_cache_read, 1000.0),
        slot_sink_mass=poison_invalid(cell.slot_sink_mass, 1000.0),
        slot_attention_entropy=poison_invalid(
            cell.slot_attention_entropy, 1000.0
        ),
        slot_top1_mass=poison_invalid(cell.slot_top1_mass, 1000.0),
        slot_top1_positions=poison_invalid(cell.slot_top1_positions, 999),
    )
    poisoned_output = replace(output, cell_outputs=(poisoned_cell,))
    config = SimpleNamespace(cache=cache)

    def diagnostics(candidate) -> dict[str, object]:
        return _exact_cache_payload(
            config,
            model_config,
            ((episode, candidate),),
            amplitude_initial=(0.0,),
            amplitude_final=(0.0,),
            cache_active=True,
        )

    baseline = diagnostics(output)
    poisoned = diagnostics(poisoned_output)
    for field in (
        "slot_sink_mass",
        "slot_attention_entropy",
        "slot_top1_mass",
        "slot_cache_output_norm",
        "slot_top1_position_digest",
    ):
        assert poisoned[field] == baseline[field]


def _cache_geometry_job(arm_id: str) -> dict[str, object]:
    spec = get_variant(arm_id)
    declared_width = (
        int(arm_id.rsplit(".", 1)[1])
        if arm_id.startswith("exact_cache.width.")
        else 2
    )
    declared_block = (
        int(arm_id.rsplit(".", 1)[1])
        if arm_id.startswith("exact_cache.block.")
        else 2
    )
    base = _cache_alias_job(arm_id)
    semantic = copy.deepcopy(base["canonical_config"])
    semantic["cache"]["width"] = declared_width
    semantic["cache"]["block_size"] = declared_block
    if declared_width == 0:
        semantic["mechanism"] = "current_block_only"
        semantic["variant"] = "chunk_only"
    screen_length = max(4, 2 * declared_block, declared_width + 1)
    semantic["lengths"]["curriculum"] = [screen_length]
    example = generate_task(
        "mqar", 1, screen_length, 211, "train", {"width": 4}
    )
    semantic["budget"]["tokens"] = 2 * int(example.valid.sum().item())
    return build_job(
        semantic,
        seed=211,
        stage=spec.required_stage,
        backend="tiny",
        arm_id=arm_id,
    )


@pytest.mark.parametrize(
    "arm_id",
    [
        "exact_cache.width.0",
        "exact_cache.width.8",
        "exact_cache.width.16",
        "exact_cache.width.32",
        "exact_cache.width.64",
        "exact_cache.width.128",
        "exact_cache.block.64",
        "exact_cache.block.128",
        "exact_cache.block.256",
    ],
)
def test_tiny_cache_geometry_registry_arms_execute_declared_geometry(
    arm_id: str, tmp_path: Path
) -> None:
    job = _cache_geometry_job(arm_id)
    diagnostics = build_job_dispatcher(_dispatcher_runtime(tmp_path))(job)[
        "exact_cache"
    ]
    if arm_id.startswith("exact_cache.width."):
        assert diagnostics["declared_width"] == int(arm_id.rsplit(".", 1)[1])
    else:
        assert diagnostics["block_size"] == int(arm_id.rsplit(".", 1)[1])


def _factorial_cell_job(arm_id: str) -> dict[str, object]:
    family = arm_id.rsplit(".", 1)[0]
    base = _dispatcher_job(
        family,
        task="far_surprise",
        params={"four_cells": ["M00", "M10", "M01", "M11"]},
    )
    semantic = copy.deepcopy(base["canonical_config"])
    example = generate_task("far_surprise", 1, 2, 211, "train", {})
    semantic["budget"]["tokens"] = 2 * int(example.valid.sum().item())
    return build_job(
        semantic,
        seed=211,
        stage=get_variant(arm_id).required_stage,
        backend="tiny",
        arm_id=arm_id,
    )


@pytest.mark.parametrize(
    "arm_id",
    [
        f"exact_cache.{family}_factorial.{cell}"
        for family in ("rotation", "r_out")
        for cell in ("M00", "M10", "M01", "M11")
    ],
)
def test_tiny_factorial_cells_execute_exact_cache_and_feature_bits(
    arm_id: str, tmp_path: Path
) -> None:
    cell = arm_id.rsplit(".", 1)[1]
    payload = build_job_dispatcher(_dispatcher_runtime(tmp_path))(
        _factorial_cell_job(arm_id)
    )
    diagnostics = payload["exact_cache"]
    assert diagnostics["cache_active"] is (cell[1] == "1")
    if cell in {"M00", "M01"}:
        assert diagnostics["implementation_paths"] == {
            "scan": "tiny_backend.TinyKMD2Cell._forward_fp32",
            "score": "tiny_backend.TinyKMD2Cell._forward_fp32.native_score",
            "selection": "disabled_no_cache",
            "read": "disabled_no_cache",
        }
    if ".r_out_factorial." in arm_id:
        assert diagnostics["model_r_out"] == (4 if cell[2] == "1" else 1)
    else:
        assert diagnostics["rotation_mode"] == (
            "current" if cell[2] == "1" else "none"
        )


def test_tiny_resume_keeps_completed_checkpoint_bytes_and_curve_length(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job()
    first_ticks = iter((1.0, 2.0))
    first = build_job_dispatcher(
        _dispatcher_runtime(tmp_path),
        dependencies={"monotonic": lambda: next(first_ticks)},
    )(job)
    checkpoint = tmp_path / "checkpoints" / str(job["job_id"]) / "latest.pt"
    before = checkpoint.read_bytes()
    second_ticks = iter((3.0, 4.0))
    resumed = build_job_dispatcher(
        _dispatcher_runtime(tmp_path),
        dependencies={"monotonic": lambda: next(second_ticks)},
    )(job)
    assert checkpoint.read_bytes() == before
    assert len(first["loss_curves"]["train"]) == 2
    assert resumed["loss_curves"]["train"] == first["loss_curves"]["train"]
    assert resumed["training"] == first["training"]


def test_tiny_dispatcher_translates_task_generation_failure_to_malformed_input(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job()

    def malformed(*_args, **_kwargs):
        raise ValueError("injected malformed episode")

    dependencies = TinyExecutionDependencies(
        generate_task=malformed,
        build_model=TinyKMD2Model,
        build_trainer=TinyTrainer,
        monotonic=lambda: 0.0,
        peak_vram_bytes=lambda: 0,
    )
    with pytest.raises(MalformedInput, match="injected malformed episode"):
        build_job_dispatcher(
            _dispatcher_runtime(tmp_path), dependencies=dependencies
        )(job)


def test_tiny_dispatcher_arm_table_exhaustively_classifies_registry_support() -> None:
    from research.kmd2_ablation import tiny_training

    tiny_arms = {
        spec.arm_id for spec in all_variants() if "tiny" in spec.compatible_backends
    }
    assert set(tiny_training._TINY_ARM_STATUS) == tiny_arms
    assert all(tiny_training._TINY_ARM_STATUS.values())
    supported = {
        arm_id
        for arm_id, status in tiny_training._TINY_ARM_STATUS.items()
        if status == "supported"
    }
    assert supported == tiny_arms - {
        "exact_cache.rotation_factorial",
        "exact_cache.r_out_factorial",
    }


def test_tiny_dispatcher_rejects_unknown_runtime_and_arm_config_mismatch(
    tmp_path: Path,
) -> None:
    runtime = _dispatcher_runtime(tmp_path)
    runtime["device"] = "cpu"
    with pytest.raises(TinyRuntimeConfigurationError) as runtime_error:
        build_job_dispatcher(runtime)
    assert runtime_error.value.code == "runtime_configuration_invalid"

    native = _dispatcher_job()
    mismatched = build_job(
        native["canonical_config"],
        seed=native["seed"],
        stage=native["stage"],
        backend="tiny",
        arm_id="trapezoid",
    )
    with pytest.raises(TinyRuntimeConfigurationError) as arm_error:
        build_job_dispatcher(_dispatcher_runtime(tmp_path))(mismatched)
    assert arm_error.value.code == "arm_configuration_mismatch"


def test_tiny_cache_payload_has_exactly_every_required_measured_field(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job(
        "exact_cache.selector.exact_outer", task="mqar", params={"width": 4}
    )
    payload = build_job_dispatcher(_dispatcher_runtime(tmp_path))(job)
    assert _EXACT_CACHE_FIELDS <= set(payload["exact_cache"])


def test_tiny_dispatcher_translates_forced_oom_and_nonfinite_gradient(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job()

    def out_of_memory(*_args, **_kwargs):
        raise torch.OutOfMemoryError("injected tiny OOM")

    with pytest.raises(ForcedOOM, match="injected tiny OOM") as oom:
        build_job_dispatcher(
            _dispatcher_runtime(tmp_path), dependencies={"build_model": out_of_memory}
        )(job)
    assert oom.value.phase == "model_initialization"
    assert oom.value.context["device"] == "cpu"

    def nonfinite_trainer(model, config):
        trainer = TinyTrainer(model, config)

        def fail(_episode):
            raise FloatingPointError("training gradients are not finite")

        trainer.train_step = fail  # type: ignore[method-assign]
        return trainer

    with pytest.raises(NonFiniteGradient, match="gradients"):
        build_job_dispatcher(
            _dispatcher_runtime(tmp_path),
            dependencies={"build_trainer": nonfinite_trainer},
        )(job)


def test_tiny_dispatcher_integrates_with_execute_jobs_and_resume_skip(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job()
    provenance = _provenance_for(job)
    store = ResultStore(
        tmp_path / "results", provenance=provenance, job_index=0, num_jobs=1
    )
    dispatcher = build_job_dispatcher(_dispatcher_runtime(tmp_path / "runtime"))
    completed = execute_jobs(
        [job],
        store=store,
        command=["python", "run"],
        dispatchers={"tiny": dispatcher},
        resume=True,
    )
    assert completed == [{"job_id": job["job_id"], "status": "completed"}]
    skipped = execute_jobs(
        [job],
        store=store,
        command=["python", "run"],
        dispatchers={"tiny": dispatcher},
        resume=True,
    )
    assert skipped == [{"job_id": job["job_id"], "status": "skipped"}]


def test_tiny_dispatcher_translates_complete_current_native_baseline(
    tmp_path: Path,
) -> None:
    captured: list[TinyKMD2Config] = []

    def build_model(config: TinyKMD2Config, *, init_seed: int):
        captured.append(config)
        return TinyKMD2Model(config, init_seed=init_seed)

    build_job_dispatcher(
        _dispatcher_runtime(tmp_path), dependencies={"build_model": build_model}
    )(_dispatcher_job())
    assert len(captured) == 1
    translated = captured[0]
    assert translated.rotation_mode == "current"
    assert translated.rotation_gate_init == 1.0
    assert translated.convolution_gate_init == 1.0
    assert translated.channel_decay_gate_init == 1.0
    assert translated.write_offset_gate_init == 1.0


def test_tiny_dispatcher_accepts_matched_native_job_for_treatment_config(
    tmp_path: Path,
) -> None:
    treatment = _dispatcher_job(
        "exact_cache.selector.exact_outer",
        task="mqar",
        params={"width": 4},
    )
    raw = copy.deepcopy(treatment["canonical_config"])
    raw["runtime"] = {"output_path": "ignored", "device_ordinal": 0}
    config = ExperimentConfig.from_dict(raw)
    native = build_job(
        config,
        seed=treatment["seed"],
        stage=treatment["stage"],
        backend="tiny",
        arm_id="native",
        pairing_id="matched-native-pair",
    )

    payload = build_job_dispatcher(_dispatcher_runtime(tmp_path))(native)

    assert "exact_cache" not in payload
    assert payload["metrics"]
    assert payload["recurrent_state"]["bytes"] > 0


@pytest.mark.parametrize(
    ("configured_arm", "paired_arm", "task", "params"),
    [
        ("rotation.current", "rotation.off", "parity", {}),
        ("rotation.off", "rotation.current", "parity", {}),
        ("convolution.on", "convolution.off", "mqar", {"width": 4}),
        ("convolution.off", "convolution.on", "mqar", {"width": 4}),
    ],
)
def test_tiny_dispatcher_accepts_matched_reliance_counterpart_job(
    configured_arm: str,
    paired_arm: str,
    task: str,
    params: dict[str, object],
    tmp_path: Path,
) -> None:
    configured = _dispatcher_job(configured_arm, task=task, params=params)
    raw = copy.deepcopy(configured["canonical_config"])
    raw["runtime"] = {"output_path": "ignored", "device_ordinal": 0}
    config = ExperimentConfig.from_dict(raw)
    counterpart = build_job(
        config,
        seed=configured["seed"],
        stage=configured["stage"],
        backend="tiny",
        arm_id=paired_arm,
        pairing_id="matched-reliance-pair",
    )

    payload = build_job_dispatcher(_dispatcher_runtime(tmp_path))(counterpart)

    assert payload["metrics"]
    assert payload["recurrent_state"]["bytes"] > 0


def test_tiny_dispatcher_uses_explicit_true_mimo_rank_and_parameter_match(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job(
        "true_mimo.sweep",
        task="mqar",
        params={
            "width": 4,
            "mimo_rank": 2,
            "parameter_match_target": {
                "state_key_dim": 2,
                "state_value_dim": 2,
                "mimo_rank": 1,
            },
        },
    )
    captured: list[TinyKMD2Config] = []

    def build_model(config: TinyKMD2Config, *, init_seed: int):
        captured.append(config)
        return TinyKMD2Model(config, init_seed=init_seed)

    payload = build_job_dispatcher(
        _dispatcher_runtime(tmp_path), dependencies={"build_model": build_model}
    )(job)
    assert captured[0].mimo_rank == 2
    assert captured[0].r_out == 1
    assert captured[0].cache is None
    assert payload["parameters"]["trainable"] == sum(
        parameter.numel()
        for parameter in TinyKMD2Model(captured[0], init_seed=211).parameters()
        if parameter.requires_grad
    )


def test_tiny_dispatcher_activates_exact_gdn2_channelwise_gate_path(
    tmp_path: Path,
) -> None:
    job = _dispatcher_job(
        "gdn2_decoupled.channelwise", task="mqar", params={"width": 4}
    )
    captured: list[TinyKMD2Config] = []

    def build_model(config: TinyKMD2Config, *, init_seed: int):
        captured.append(config)
        return TinyKMD2Model(config, init_seed=init_seed)

    payload = build_job_dispatcher(
        _dispatcher_runtime(tmp_path), dependencies={"build_model": build_model}
    )(job)
    assert captured[0].gdn2_decoupled is True
    assert captured[0].mimo_rank == 1
    model = TinyKMD2Model(captured[0], init_seed=211)
    names = set(dict(model.named_parameters()))
    assert any(name.endswith("erase_proj.weight") for name in names)
    assert any(name.endswith("write_proj.weight") for name in names)
    assert not any(name.endswith("b_proj.weight") for name in names)
    assert not any(name.endswith("bw_off") for name in names)
    assert payload["metrics"]


def test_matched_native_state_size_comparator_uses_declared_target_dimensions(
    tmp_path: Path,
) -> None:
    raw = minimal_config_dict()
    raw.update(
        backend="tiny",
        mechanism="state_size",
        variant="state_size_sweep",
        seeds=[211],
        required_stage="mechanism_screen",
    )
    raw["task"] = {
        "name": "mqar",
        "params": {
            "width": 4,
            "parameter_match_target": {
                "state_key_dim": 2,
                "state_value_dim": 2,
                "mimo_rank": 1,
            },
        },
    }
    raw["budget"] = {"tokens": 28, "updates": 2}
    raw["schedule"]["warmup_updates"] = 1
    raw["model"] = {
        "hidden_size": 8,
        "num_layers": 1,
        "num_heads": 1,
        "state_key_dim": 4,
        "state_value_dim": 2,
        "ffn_dim": 16,
        "ffn_match_lower": 8,
        "ffn_match_upper": 24,
    }
    raw["lengths"] = {"curriculum": [4], "extrapolation": [8, 16]}
    raw["device_preferences"] = ["cpu"]
    raw["dtype_preferences"] = ["float32"]
    config = ExperimentConfig.from_dict(raw)
    native = build_job(
        config,
        seed=211,
        stage="mechanism_screen",
        backend="tiny",
        arm_id="native",
        pairing_id="state-size-pair",
    )
    captured: list[TinyKMD2Config] = []

    def build_model(model_config: TinyKMD2Config, *, init_seed: int):
        captured.append(model_config)
        return TinyKMD2Model(model_config, init_seed=init_seed)

    build_job_dispatcher(
        _dispatcher_runtime(tmp_path), dependencies={"build_model": build_model}
    )(native)

    assert len(captured) == 1
    assert (captured[0].dk, captured[0].dv, captured[0].mimo_rank) == (2, 2, 1)


def test_tiny_cache_off_and_current_block_controls_emit_honest_diagnostics(
    tmp_path: Path,
) -> None:
    off_job = _dispatcher_job("exact_cache.off", task="mqar", params={"width": 4})
    off = build_job_dispatcher(_dispatcher_runtime(tmp_path / "off"))(off_job)
    assert off["exact_cache"]["amplitude_initial"] == [0.0]
    assert off["exact_cache"]["amplitude_final"] == [0.0]
    assert off["exact_cache"]["retention_count"] == 0
    assert off["exact_cache"]["eviction_count"] == 0
    assert off["exact_cache"]["persistent_bytes"] == 0
    assert off["exact_cache"]["block_bytes"] == 0
    assert off["exact_cache"]["implementation_paths"] == {
        "scan": "tiny_backend.TinyKMD2Cell._forward_fp32",
        "score": "tiny_backend.TinyKMD2Cell._forward_fp32.native_score",
        "selection": "disabled_no_cache",
        "read": "disabled_no_cache",
    }
    build_completed_record(
        off_job,
        _provenance_for(off_job),
        shard_index=0,
        num_jobs=1,
        command=["python", "run"],
        payload=off,
    )

    block_job = _dispatcher_job(
        "exact_cache.current_block_only", task="mqar", params={"width": 4}
    )
    block = build_job_dispatcher(_dispatcher_runtime(tmp_path / "block"))(block_job)
    assert block["exact_cache"]["width"] == 0
    assert block["exact_cache"]["retention_count"] == 0
    assert block["exact_cache"]["eviction_count"] > 0
    assert block["exact_cache"]["persistent_bytes"] == 0


def test_tiny_warm_start_arm_copies_every_shape_compatible_native_tensor(
    tmp_path: Path,
) -> None:
    snapshots: list[tuple[TinyKMD2Config, dict[str, torch.Tensor]]] = []

    def build_trainer(model: TinyKMD2Model, config: TinyTrainingConfig):
        snapshots.append(
            (
                model.config,
                {name: value.detach().clone() for name, value in model.state_dict().items()},
            )
        )
        return TinyTrainer(model, config)

    job = _dispatcher_job("trapezoid", task="irregular_integration", params={})
    build_job_dispatcher(
        _dispatcher_runtime(tmp_path), dependencies={"build_trainer": build_trainer}
    )(job)
    arm_config, arm_state = snapshots[0]
    native_config = replace(arm_config, trapezoid=False)
    native_state = TinyKMD2Model(native_config, init_seed=211).state_dict()
    common = set(arm_state) & set(native_state)
    assert common
    assert all(
        arm_state[name].shape != native_state[name].shape
        or torch.equal(arm_state[name], native_state[name])
        for name in common
    )


def test_tiny_optimizer_groups_cache_projection_and_opening_gradient() -> None:
    model = TinyKMD2Model(_model_config(cache=True, modality="affine"), init_seed=5)
    trainer = TinyTrainer(model, _training_config())
    assert [group["name"] for group in trainer.optimizer.param_groups] == [
        "memory",
        "cache",
    ]
    memory, cache = trainer.optimizer.param_groups
    assert memory["lr"] == pytest.approx(0.005)
    assert cache["lr"] == pytest.approx(0.01)
    assert memory["betas"] == cache["betas"] == (0.9, 0.95)
    assert memory["eps"] == cache["eps"] == 1.0e-8
    assert memory["weight_decay"] == 0.01
    assert cache["weight_decay"] == 0.0
    assert trainer.optimizer_parameter_names[1] == (
        "blocks.0.cell.cache_gamma_q",
        "blocks.0.cell.cache_gamma_k",
        "blocks.0.cell.cache_sink_logit",
        "blocks.0.cell.cache_amplitude",
    )

    amplitude = model.blocks[0].cell.cache_amplitude
    assert amplitude.dtype == torch.float32
    assert torch.count_nonzero(amplitude) == 0
    batch = generate_task(
        "affine_associative_regression",
        2,
        3,
        223,
        "train",
        {"input_dim": 3, "output_dim": 2},
    )
    output = model.forward_episode(batch)
    assert output.loss is not None
    output.loss.backward()
    assert amplitude.grad is not None and torch.isfinite(amplitude.grad).all()
    assert torch.count_nonzero(amplitude.grad) > 0
    trainer.optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        amplitude.fill_(1.5)
    trainer.optimizer.step()
    assert torch.equal(amplitude, torch.ones_like(amplitude))
    with torch.no_grad():
        amplitude.fill_(-0.5)
    trainer.optimizer.step()
    assert torch.equal(amplitude, torch.zeros_like(amplitude))


def test_tiny_optimizer_without_cache_has_one_stable_memory_group() -> None:
    model = TinyKMD2Model(_model_config(cache=False), init_seed=7)
    trainer = TinyTrainer(model, _training_config("native-job"))
    assert len(trainer.optimizer.param_groups) == 1
    assert trainer.optimizer.param_groups[0]["name"] == "memory"
    expected = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert trainer.optimizer_parameter_names == (expected,)


def _advance_for_checkpoint(trainer: TinyTrainer, batch: EpisodeBatch) -> None:
    trainer.optimizer.zero_grad(set_to_none=True)
    output = trainer.model.forward_episode(batch)
    assert output.loss is not None
    output.loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.step = 1
    trainer.tokens_seen = int(batch.valid.sum())
    trainer.metric_history.append(
        {
            "step": 1,
            "tokens_seen": trainer.tokens_seen,
            "loss": float(output.loss.detach()),
        }
    )
    torch.rand(7, generator=trainer.rng)


def _checkpoint_fixture(tmp_path: Path) -> tuple[TinyTrainer, Path, EpisodeBatch]:
    trainer = TinyTrainer(
        TinyKMD2Model(_model_config(cache=True, modality="affine"), init_seed=13),
        _training_config("checkpoint-job"),
    )
    batch = generate_task(
        "affine_associative_regression",
        2,
        3,
        229,
        "train",
        {"input_dim": 3, "output_dim": 2},
    )
    _advance_for_checkpoint(trainer, batch)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    return trainer, path, batch


def _assert_nested_exact(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert list(left) == list(right)
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_exact(left_item, right_item)
    else:
        assert left == right


def test_tiny_checkpoint_schema_is_complete_atomic_and_cpu_portable(
    tmp_path: Path,
) -> None:
    trainer, path, _ = _checkpoint_fixture(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert tuple(payload) == (
        "schema_version",
        "job_id",
        "model_config_signature",
        "training_config_signature",
        "step",
        "tokens_seen",
        "model_state_names",
        "model_state",
        "optimizer_parameter_names",
        "optimizer_active_parameter_names",
        "optimizer_active_parameter_steps",
        "optimizer_state",
        "scheduler_spec",
        "scheduler_state",
        "rng_state",
        "metric_state",
    )
    assert payload["schema_version"] == TINY_CHECKPOINT_SCHEMA_VERSION
    assert payload["schema_version"] == "1.4.0"
    assert payload["job_id"] == "checkpoint-job"
    assert len(payload["model_config_signature"]) == 64
    assert len(payload["training_config_signature"]) == 64
    assert payload["model_state_names"] == tuple(trainer.model.state_dict())
    assert payload["optimizer_parameter_names"] == trainer.optimizer_parameter_names
    active_ids = set(payload["optimizer_state"]["state"])
    expected_active = tuple(
        name
        for names, group in zip(
            trainer.optimizer_parameter_names,
            payload["optimizer_state"]["param_groups"],
            strict=True,
        )
        for name, parameter_id in zip(names, group["params"], strict=True)
        if parameter_id in active_ids
    )
    assert payload["optimizer_active_parameter_names"] == expected_active
    expected_active_steps = tuple(
        int(float(payload["optimizer_state"]["state"][parameter_id]["step"]))
        for group in payload["optimizer_state"]["param_groups"]
        for parameter_id in group["params"]
        if parameter_id in active_ids
    )
    assert payload["optimizer_active_parameter_steps"] == expected_active_steps
    assert expected_active
    assert payload["scheduler_spec"] == {
        "name": "warmup_cosine",
        "warmup_updates": 2,
        "total_updates": 10,
    }
    for tensor in payload["model_state"].values():
        assert tensor.device.type == "cpu"
    assert payload["rng_state"].device.type == "cpu"
    assert payload["rng_state"].dtype == torch.uint8
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))

    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"old")
    trainer.save_checkpoint(replacement)
    assert replacement.stat().st_size > 3
    assert not list(tmp_path.glob(f".{replacement.name}.*.tmp"))


def test_tiny_checkpoint_resume_restores_every_state_exactly(tmp_path: Path) -> None:
    source, path, _ = _checkpoint_fixture(tmp_path)
    resumed = TinyTrainer(
        TinyKMD2Model(_model_config(cache=True, modality="affine"), init_seed=99),
        _training_config("checkpoint-job"),
    )
    resumed.load_checkpoint(path)
    assert resumed.step == source.step
    assert resumed.tokens_seen == source.tokens_seen
    assert resumed.metric_history == source.metric_history
    _assert_nested_exact(source.model.state_dict(), resumed.model.state_dict())
    _assert_nested_exact(source.optimizer.state_dict(), resumed.optimizer.state_dict())
    _assert_nested_exact(source.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert torch.equal(source.rng.get_state(), resumed.rng.get_state())


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda p: p.__setitem__("extra", 1), id="unknown-field"),
        pytest.param(
            lambda p: p.__setitem__("schema_version", "999"), id="schema"
        ),
        pytest.param(lambda p: p.__setitem__("job_id", "other"), id="job"),
        pytest.param(
            lambda p: p.__setitem__("model_config_signature", "0" * 64),
            id="model-config",
        ),
        pytest.param(
            lambda p: p.__setitem__("training_config_signature", "0" * 64),
            id="training-config",
        ),
        pytest.param(
            lambda p: p.__setitem__(
                "model_state_names", p["model_state_names"][:-1]
            ),
            id="model-names",
        ),
        pytest.param(
            lambda p: p["model_state"].__setitem__(
                p["model_state_names"][0],
                p["model_state"][p["model_state_names"][0]][:-1],
            ),
            id="model-shape",
        ),
        pytest.param(
            lambda p: p["model_state"].__setitem__(
                p["model_state_names"][0],
                p["model_state"][p["model_state_names"][0]].double(),
            ),
            id="model-dtype",
        ),
        pytest.param(
            lambda p: p["model_state"][p["model_state_names"][0]].fill_(float("nan")),
            id="model-nonfinite",
        ),
        pytest.param(
            lambda p: p["model_state"][
                "blocks.0.cell.cache_amplitude"
            ].fill_(1.01),
            id="amplitude-range",
        ),
        pytest.param(
            lambda p: p.__setitem__(
                "optimizer_parameter_names", p["optimizer_parameter_names"][:-1]
            ),
            id="optimizer-names",
        ),
        pytest.param(
            lambda p: p["optimizer_state"]["param_groups"][0].__setitem__("lr", 9.0),
            id="optimizer-hyperparameters",
        ),
        pytest.param(
            lambda p: p["scheduler_spec"].__setitem__("name", "linear"),
            id="scheduler-spec",
        ),
        pytest.param(
            lambda p: p["scheduler_state"].__setitem__("last_epoch", 9),
            id="scheduler-state",
        ),
        pytest.param(
            lambda p: p.__setitem__("rng_state", p["rng_state"].float()),
            id="rng",
        ),
    ],
)
def test_tiny_checkpoint_rejects_corruption_without_mutation(
    tmp_path: Path, corrupt
) -> None:
    _, path, _ = _checkpoint_fixture(tmp_path)
    target = TinyTrainer(
        TinyKMD2Model(_model_config(cache=True, modality="affine"), init_seed=17),
        _training_config("checkpoint-job"),
    )
    before = {
        "model": copy.deepcopy(target.model.state_dict()),
        "optimizer": copy.deepcopy(target.optimizer.state_dict()),
        "scheduler": copy.deepcopy(target.scheduler.state_dict()),
        "rng": target.rng.get_state().clone(),
    }
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrupt(payload)
    corrupt_path = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt_path)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        target.load_checkpoint(corrupt_path)
    _assert_nested_exact(before["model"], target.model.state_dict())
    _assert_nested_exact(before["optimizer"], target.optimizer.state_dict())
    _assert_nested_exact(before["scheduler"], target.scheduler.state_dict())
    assert torch.equal(before["rng"], target.rng.get_state())
    assert target.step == 0 and target.tokens_seen == 0 and not target.metric_history


def test_tiny_checkpoint_apply_failure_rolls_back_all_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path, _ = _checkpoint_fixture(tmp_path)
    target = TinyTrainer(
        TinyKMD2Model(_model_config(cache=True, modality="affine"), init_seed=19),
        _training_config("checkpoint-job"),
    )
    before_model = copy.deepcopy(target.model.state_dict())
    before_optimizer = copy.deepcopy(target.optimizer.state_dict())
    before_rng = target.rng.get_state().clone()
    real_load = target.scheduler.load_state_dict
    calls = 0

    def fail_once(state: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected scheduler failure")
        real_load(state)

    monkeypatch.setattr(target.scheduler, "load_state_dict", fail_once)
    with pytest.raises(RuntimeError, match="injected scheduler failure"):
        target.load_checkpoint(path)
    _assert_nested_exact(before_model, target.model.state_dict())
    _assert_nested_exact(before_optimizer, target.optimizer.state_dict())
    assert torch.equal(before_rng, target.rng.get_state())
    assert target.step == 0 and target.tokens_seen == 0 and not target.metric_history


def test_tiny_checkpoint_step_zero_requires_zero_tokens(tmp_path: Path) -> None:
    config = _training_config("zero-step-job")
    source = TinyTrainer(TinyKMD2Model(_model_config(), init_seed=20), config)
    path = tmp_path / "zero.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["optimizer_active_parameter_names"] == ()
    assert payload["optimizer_active_parameter_steps"] == ()
    assert payload["optimizer_state"]["state"] == {}
    payload["tokens_seen"] = 1
    torch.save(payload, path)
    target = TinyTrainer(TinyKMD2Model(_model_config(), init_seed=21), config)
    with pytest.raises(ValueError, match="step zero.*zero tokens"):
        target.load_checkpoint(path)


@pytest.mark.parametrize("corruption", ["zero-first", "duplicate-final"])
def test_tiny_checkpoint_metric_tokens_are_positive_and_strictly_increasing(
    tmp_path: Path, corruption: str
) -> None:
    model_config, batch = _learning_case("token")
    config = _training_config("metric-token-job")
    source = TinyTrainer(TinyKMD2Model(model_config, init_seed=22), config)
    source.train_step(batch)
    source.train_step(batch)
    path = tmp_path / f"{corruption}.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if corruption == "zero-first":
        payload["metric_state"][0]["tokens_seen"] = 0
    else:
        duplicate = payload["metric_state"][0]["tokens_seen"]
        payload["metric_state"][1]["tokens_seen"] = duplicate
        payload["tokens_seen"] = duplicate
    torch.save(payload, path)
    target = TinyTrainer(TinyKMD2Model(model_config, init_seed=24), config)
    with pytest.raises(ValueError, match="strictly increase"):
        target.load_checkpoint(path)


@pytest.mark.parametrize("corruption", ["missing-slot", "stale-slot-step"])
def test_tiny_checkpoint_rejects_incomplete_or_stale_active_adam_state(
    tmp_path: Path, corruption: str
) -> None:
    model_config, batch = _learning_case("token")
    config = _training_config("active-adam-job")
    source = TinyTrainer(TinyKMD2Model(model_config, init_seed=28), config)
    source.train_step(batch)
    source.train_step(batch)
    path = tmp_path / f"{corruption}.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    active_ids = tuple(payload["optimizer_state"]["state"])
    assert len(active_ids) > 1
    if corruption == "missing-slot":
        del payload["optimizer_state"]["state"][active_ids[0]]
    else:
        payload["optimizer_state"]["state"][active_ids[0]]["step"].fill_(1.0)
    torch.save(payload, path)
    target = TinyTrainer(TinyKMD2Model(model_config, init_seed=30), config)
    with pytest.raises(ValueError, match="active Adam"):
        target.load_checkpoint(path)


def test_tiny_checkpoint_mixed_token_direct_optimizer_steps_resume_exactly(
    tmp_path: Path,
) -> None:
    model_config = _model_config(modality="affine")
    config = _training_config("mixed-modality-job")
    token_batch = generate_task("parity", 4, 3, 317, "train", {})
    direct_batch = generate_task(
        "affine_associative_regression",
        4,
        3,
        319,
        "train",
        {"input_dim": 3, "output_dim": 2},
    )
    source = TinyTrainer(TinyKMD2Model(model_config, init_seed=32), config)
    source.train_step(token_batch)
    source.train_step(direct_batch)
    path = tmp_path / "mixed.pt"
    source.save_checkpoint(path)

    resumed = TinyTrainer(TinyKMD2Model(model_config, init_seed=33), config)
    resumed.load_checkpoint(path)
    _assert_nested_exact(source.model.state_dict(), resumed.model.state_dict())
    _assert_nested_exact(source.optimizer.state_dict(), resumed.optimizer.state_dict())
    _assert_nested_exact(source.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert source.metric_history == resumed.metric_history


def test_tiny_checkpoint_accepts_default_float64_adam_step_portably(
    tmp_path: Path,
) -> None:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model_config, batch = _learning_case("token")
        config = _training_config("float64-adam-step-job")
        source = TinyTrainer(TinyKMD2Model(model_config, init_seed=34), config)
        source.train_step(batch)
        optimizer_state = source.optimizer.state_dict()["state"]
        assert optimizer_state
        assert {
            slot["step"].dtype for slot in optimizer_state.values()
        } == {torch.float64}
        path = tmp_path / "float64-step.pt"
        source.save_checkpoint(path)
        resumed = TinyTrainer(TinyKMD2Model(model_config, init_seed=35), config)
        resumed.load_checkpoint(path)
        _assert_nested_exact(
            source.optimizer.state_dict(), resumed.optimizer.state_dict()
        )
    finally:
        torch.set_default_dtype(previous_dtype)


def _learning_case(modality: str) -> tuple[TinyKMD2Config, EpisodeBatch]:
    if modality == "token":
        return _model_config(modality="token"), generate_task(
            "parity", 8, 4, 307, "train", {}
        )
    if modality == "continuous":
        return _model_config(modality="continuous"), generate_task(
            "irregular_integration", 8, 4, 311, "train", {"components": 1}
        )
    if modality == "affine":
        return _model_config(modality="affine"), generate_task(
            "affine_associative_regression",
            8,
            3,
            313,
            "train",
            {"input_dim": 3, "output_dim": 2},
        )
    raise AssertionError(modality)


def test_tiny_training_step_updates_metrics_schedule_and_enforces_budgets() -> None:
    model_config, batch = _learning_case("token")
    trainer = TinyTrainer(TinyKMD2Model(model_config, init_seed=23), _training_config())
    before = copy.deepcopy(trainer.model.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    result = trainer.train_step(batch)
    assert result.keys() == {"step", "tokens_seen", "loss", "grad_norm"}
    assert result["step"] == trainer.step == 1
    assert result["tokens_seen"] == trainer.tokens_seen == int(batch.valid.sum())
    assert result["loss"] == trainer.metric_history[0]["loss"]
    assert result["grad_norm"] >= 0 and torch.isfinite(torch.tensor(result["grad_norm"]))
    assert trainer.scheduler.last_epoch == 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.model.state_dict().items()
    )
    assert torch.equal(global_rng, torch.random.get_rng_state())

    evaluated = trainer.evaluate(batch)
    assert evaluated.keys() == {"loss", "tokens"}
    assert evaluated["tokens"] == int(batch.valid.sum())
    assert torch.isfinite(torch.tensor(evaluated["loss"]))
    assert trainer.step == 1 and len(trainer.metric_history) == 1

    tiny_budget = replace(
        _training_config("token-budget"), max_tokens=int(batch.valid.sum()) - 1
    )
    blocked = TinyTrainer(TinyKMD2Model(model_config, init_seed=23), tiny_budget)
    blocked_before = copy.deepcopy(blocked.model.state_dict())
    with pytest.raises(RuntimeError, match="token budget"):
        blocked.train_step(batch)
    _assert_nested_exact(blocked_before, blocked.model.state_dict())
    assert blocked.step == 0 and blocked.tokens_seen == 0

    one_update = replace(_training_config("update-budget"), updates=1, warmup_updates=1)
    exhausted = TinyTrainer(TinyKMD2Model(model_config, init_seed=23), one_update)
    exhausted.train_step(batch)
    with pytest.raises(RuntimeError, match="update budget"):
        exhausted.train_step(batch)
    assert exhausted.step == 1 and len(exhausted.metric_history) == 1


def _trainer_state_snapshot(trainer: TinyTrainer) -> dict[str, object]:
    return {
        "model": copy.deepcopy(trainer.model.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer._scheduler_state()),
        "rng": trainer.rng.get_state().clone(),
        "step": trainer.step,
        "tokens_seen": trainer.tokens_seen,
        "metrics": copy.deepcopy(trainer.metric_history),
        "training": trainer.model.training,
    }


def _assert_trainer_snapshot(trainer: TinyTrainer, snapshot: dict[str, object]) -> None:
    _assert_nested_exact(snapshot["model"], trainer.model.state_dict())
    _assert_nested_exact(snapshot["optimizer"], trainer.optimizer.state_dict())
    _assert_nested_exact(snapshot["scheduler"], trainer._scheduler_state())
    assert torch.equal(snapshot["rng"], trainer.rng.get_state())
    assert trainer.step == snapshot["step"]
    assert trainer.tokens_seen == snapshot["tokens_seen"]
    assert trainer.metric_history == snapshot["metrics"]
    assert trainer.model.training is snapshot["training"]


def test_tiny_training_step_rolls_back_injected_scheduler_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config, batch = _learning_case("token")
    trainer = TinyTrainer(
        TinyKMD2Model(model_config, init_seed=25),
        _training_config("scheduler-rollback"),
    )
    trainer.model.eval()
    before = _trainer_state_snapshot(trainer)

    def fail_scheduler() -> None:
        torch.rand(7, generator=trainer.rng)
        raise RuntimeError("injected train scheduler failure")

    monkeypatch.setattr(trainer.scheduler, "step", fail_scheduler)
    with pytest.raises(RuntimeError, match="injected train scheduler failure"):
        trainer.train_step(batch)
    _assert_trainer_snapshot(trainer, before)


@pytest.mark.parametrize(
    "corruption", ["parameter", "optimizer-state", "learning-rate", "amplitude"]
)
def test_tiny_training_step_rejects_post_step_corruption_and_rolls_back(
    corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_config, batch = _learning_case("affine")
    model_config = replace(
        model_config,
        cache=CacheConfig(
            width=2,
            block_size=2,
            read="rmsnorm",
            storage_dtype="fp32",
            lr_cache=0.02,
        ),
    )
    trainer = TinyTrainer(
        TinyKMD2Model(model_config, init_seed=26),
        _training_config(f"post-step-{corruption}"),
    )
    before = _trainer_state_snapshot(trainer)
    if corruption == "learning-rate":
        real_scheduler_step = trainer.scheduler.step

        def corrupt_scheduler() -> None:
            real_scheduler_step()
            trainer.optimizer.param_groups[0]["lr"] = float("nan")

        monkeypatch.setattr(trainer.scheduler, "step", corrupt_scheduler)
    else:
        real_optimizer_step = trainer.optimizer.step

        def corrupt_optimizer(*args, **kwargs):
            result = real_optimizer_step(*args, **kwargs)
            with torch.no_grad():
                if corruption == "parameter":
                    next(trainer.model.parameters()).fill_(float("inf"))
                elif corruption == "optimizer-state":
                    first_slot = next(iter(trainer.optimizer.state.values()))
                    first_slot["exp_avg"].fill_(float("inf"))
                else:
                    trainer.model.blocks[0].cell.cache_amplitude.fill_(1.01)
            return result

        corrupt_optimizer._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
        monkeypatch.setattr(trainer.optimizer, "step", corrupt_optimizer)
    with pytest.raises(FloatingPointError):
        trainer.train_step(batch)
    _assert_trainer_snapshot(trainer, before)


def test_tiny_training_step_rolls_back_natural_extreme_finite_lr_failure() -> None:
    model_config, batch = _learning_case("token")
    config = replace(
        _training_config("extreme-finite-lr"),
        learning_rate=1.0e308,
        weight_decay=0.0,
        warmup_updates=0,
    )
    trainer = TinyTrainer(TinyKMD2Model(model_config, init_seed=27), config)
    before = _trainer_state_snapshot(trainer)
    with pytest.raises((OverflowError, RuntimeError, FloatingPointError)):
        trainer.train_step(batch)
    _assert_trainer_snapshot(trainer, before)


@pytest.mark.parametrize("modality", ["token", "continuous", "affine"])
def test_tiny_training_ten_steps_learns_deterministically_and_resumes_exactly(
    modality: str, tmp_path: Path
) -> None:
    model_config, batch = _learning_case(modality)
    config = _training_config(f"learning-{modality}")

    uninterrupted = TinyTrainer(
        TinyKMD2Model(model_config, init_seed=29), config
    )
    initial_loss = uninterrupted.evaluate(batch)["loss"]
    for _ in range(10):
        uninterrupted.train_step(batch)
    final_loss = uninterrupted.evaluate(batch)["loss"]
    assert final_loss < initial_loss
    assert uninterrupted.step == 10
    assert uninterrupted.tokens_seen == 10 * int(batch.valid.sum())

    source = TinyTrainer(TinyKMD2Model(model_config, init_seed=29), config)
    for _ in range(5):
        source.train_step(batch)
    checkpoint = tmp_path / f"{modality}.pt"
    source.save_checkpoint(checkpoint)

    resumed = TinyTrainer(TinyKMD2Model(model_config, init_seed=777), config)
    resumed.load_checkpoint(checkpoint)
    for _ in range(5):
        resumed.train_step(batch)

    _assert_nested_exact(uninterrupted.model.state_dict(), resumed.model.state_dict())
    _assert_nested_exact(
        uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    _assert_nested_exact(
        uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict()
    )
    assert uninterrupted.metric_history == resumed.metric_history
    assert torch.equal(uninterrupted.rng.get_state(), resumed.rng.get_state())
    assert resumed.evaluate(batch)["loss"] == final_loss
