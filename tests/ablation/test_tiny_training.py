from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from research.kmd2_ablation.config import CacheConfig
from research.kmd2_ablation.tasks import EpisodeBatch, generate_task
from research.kmd2_ablation.tiny_backend import TinyKMD2Config, TinyKMD2Model
from research.kmd2_ablation.tiny_training import (
    TINY_CHECKPOINT_SCHEMA_VERSION,
    TinyTrainer,
    TinyTrainingConfig,
)


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
    assert payload["schema_version"] == "1.2.0"
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
