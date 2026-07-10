"""Deterministic CPU-friendly training support for the tiny KMD-2 backend."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .tasks import EpisodeBatch
from .tiny_backend import (
    TinyFactors,
    TinyKMD2Model,
    TinyModelOutput,
    tiny_factors_from_episode,
)


TINY_CHECKPOINT_SCHEMA_VERSION = "1.2.0"
_CACHE_PARAMETER_NAMES = frozenset(
    {
        "cache_gamma_q",
        "cache_gamma_k",
        "cache_sink_logit",
        "cache_amplitude",
    }
)
_CHECKPOINT_FIELDS = (
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
_SCHEDULER_STATE_FIELDS = (
    "base_lrs",
    "last_epoch",
    "_step_count",
    "_is_initial",
    "_get_lr_called_within_step",
    "_last_lr",
    "lr_lambdas",
)


def _finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def _config_signature(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _finite_tensor(name: str, tensor: Tensor) -> None:
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class TinyTrainingConfig:
    """The complete optimization budget and hyperparameters for one tiny run."""

    job_id: str
    seed: int
    updates: int
    max_tokens: int
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    warmup_updates: int
    max_grad_norm: float

    def __post_init__(self) -> None:
        if type(self.job_id) is not str or not self.job_id:
            raise TypeError("job_id must be a non-empty string")
        if type(self.seed) is not int:
            raise TypeError("seed must be an int")
        for name in ("updates", "max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int")
        if type(self.warmup_updates) is not int or self.warmup_updates < 0:
            raise ValueError("warmup_updates must be a nonnegative int")
        if self.warmup_updates > self.updates:
            raise ValueError("warmup_updates cannot exceed updates")
        if type(self.betas) is not tuple or len(self.betas) != 2:
            raise TypeError("betas must be a tuple of two finite numbers")
        betas = tuple(
            _finite_real(f"betas[{index}]", beta, minimum=0.0)
            for index, beta in enumerate(self.betas)
        )
        if any(beta >= 1.0 for beta in betas):
            raise ValueError("betas must be less than one")
        object.__setattr__(self, "betas", betas)
        for name, minimum, strict in (
            ("learning_rate", 0.0, True),
            ("eps", 0.0, True),
            ("weight_decay", 0.0, False),
            ("max_grad_norm", 0.0, True),
        ):
            object.__setattr__(
                self,
                name,
                _finite_real(name, getattr(self, name), minimum=minimum, strict_minimum=strict),
            )


class TinyTrainer:
    """AdamW trainer with stable parameter groups and deterministic local RNG."""

    def __init__(self, model: TinyKMD2Model, config: TinyTrainingConfig):
        if not isinstance(model, TinyKMD2Model):
            raise TypeError("model must be TinyKMD2Model")
        if not isinstance(config, TinyTrainingConfig):
            raise TypeError("config must be TinyTrainingConfig")
        self.model = model
        self.config = config
        self.step = 0
        self.tokens_seen = 0
        self.metric_history: list[dict[str, float | int]] = []
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(config.seed)

        memory: list[Tensor] = []
        cache: list[Tensor] = []
        memory_names: list[str] = []
        cache_names: list[str] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.rsplit(".", 1)[-1] in _CACHE_PARAMETER_NAMES:
                cache.append(parameter)
                cache_names.append(name)
            else:
                memory.append(parameter)
                memory_names.append(name)
        if not memory:
            raise ValueError("model must expose at least one trainable memory parameter")

        groups: list[dict[str, Any]] = [
            {
                "params": memory,
                "name": "memory",
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        ]
        names: list[tuple[str, ...]] = [tuple(memory_names)]
        if cache:
            cache_config = model.config.cache
            if cache_config is None:
                raise RuntimeError("cache parameters exist without a cache configuration")
            groups.append(
                {
                    "params": cache,
                    "name": "cache",
                    "lr": cache_config.lr_cache,
                    "weight_decay": 0.0,
                }
            )
            names.append(tuple(cache_names))
        self.optimizer_parameter_names = tuple(names)
        self.optimizer = torch.optim.AdamW(
            groups,
            betas=config.betas,
            eps=config.eps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._schedule_multiplier,
        )
        self._optimizer_post_hook = self.optimizer.register_step_post_hook(
            self._project_cache_amplitudes
        )

    def _schedule_multiplier(self, scheduler_step: int) -> float:
        warmup = self.config.warmup_updates
        if warmup and scheduler_step < warmup:
            return float(scheduler_step + 1) / float(warmup)
        decay_steps = max(1, self.config.updates - warmup)
        progress = min(1.0, max(0.0, (scheduler_step - warmup) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _project_cache_amplitudes(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del optimizer, args, kwargs
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name.rsplit(".", 1)[-1] == "cache_amplitude":
                    parameter.clamp_(0.0, 1.0)

    def _forward_episode(self, episode: EpisodeBatch) -> TinyModelOutput:
        if not isinstance(episode, EpisodeBatch):
            raise TypeError("episode must be an EpisodeBatch")
        device = next(self.model.parameters()).device
        factors = None
        if episode.direct_factors is not None:
            source = tiny_factors_from_episode(episode)
            factors = TinyFactors(
                q=source.q.to(device),
                k=source.k.to(device),
                v=source.v.to(device),
                decay=source.decay.to(device),
                beta_e=source.beta_e.to(device),
                beta_w=source.beta_w.to(device),
                out_mix=source.out_mix.to(device),
                valid=source.valid.to(device),
                positions=source.positions.to(device),
            )
        return self.model(
            input_ids=(
                None if episode.input_ids is None else episode.input_ids.to(device)
            ),
            continuous_inputs=(
                None
                if episode.continuous_inputs is None
                else episode.continuous_inputs.to(device)
            ),
            factors=factors,
            targets=episode.targets.to(device),
            loss_mask=episode.loss_mask.to(device),
            boundaries=episode.boundaries.to(device),
            valid=None if factors is not None else episode.valid.to(device),
            positions=None if factors is not None else episode.positions.to(device),
        )

    def train_step(self, episode: EpisodeBatch) -> dict[str, float | int]:
        """Run one finite, clipped optimization update over an episode batch."""
        if not isinstance(episode, EpisodeBatch):
            raise TypeError("episode must be an EpisodeBatch")
        if self.step >= self.config.updates:
            raise RuntimeError("update budget is exhausted")
        batch_tokens = int(episode.valid.sum().item())
        if self.tokens_seen + batch_tokens > self.config.max_tokens:
            raise RuntimeError("token budget would be exceeded")

        previous_training_mode = self.model.training
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            output = self._forward_episode(episode)
            if output.loss is None:
                raise RuntimeError("training episode did not produce a loss")
            if not bool(torch.isfinite(output.loss.detach()).all()):
                raise FloatingPointError("training loss is not finite")
            output.loss.backward()
            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not parameters:
                raise RuntimeError("training loss produced no parameter gradients")
            for parameter in parameters:
                assert parameter.grad is not None
                if not bool(torch.isfinite(parameter.grad.detach()).all()):
                    raise FloatingPointError("training gradients are not finite")
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
            loss = float(output.loss.detach().cpu())
            previous_model = copy.deepcopy(self.model.state_dict())
            previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
            previous_scheduler = copy.deepcopy(self._scheduler_state())
            previous_rng = self.rng.get_state().clone()
            try:
                self.optimizer.step()
                self.scheduler.step()
                self._validate_post_step_state()
            except BaseException:
                self.model.load_state_dict(previous_model, strict=True)
                self.optimizer.load_state_dict(previous_optimizer)
                self.scheduler.load_state_dict(previous_scheduler)
                self.rng.set_state(previous_rng)
                raise
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train(previous_training_mode)
            raise

        self.step += 1
        self.tokens_seen += batch_tokens
        record: dict[str, float | int] = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "loss": loss,
        }
        self.metric_history.append(record)
        return {**record, "grad_norm": grad_norm}

    def _validate_post_step_state(self) -> None:
        for name, tensor in self.model.state_dict().items():
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor.detach()).all()
            ):
                raise FloatingPointError(
                    f"post-step model state {name!r} is not finite"
                )
            if name.rsplit(".", 1)[-1] == "cache_amplitude" and (
                bool((tensor.detach() < 0).any())
                or bool((tensor.detach() > 1).any())
            ):
                raise FloatingPointError(
                    "post-step cache amplitude is outside [0,1]"
                )

        for group_index, group in enumerate(self.optimizer.param_groups):
            self._validate_finite_learning_rate(
                f"optimizer group {group_index} learning rate", group["lr"]
            )
        for slot in self.optimizer.state.values():
            for name, value in slot.items():
                if isinstance(value, Tensor):
                    if (value.is_floating_point() or value.is_complex()) and not bool(
                        torch.isfinite(value.detach()).all()
                    ):
                        raise FloatingPointError(
                            f"post-step optimizer state {name!r} is not finite"
                        )
                elif type(value) in (int, float) and not math.isfinite(float(value)):
                    raise FloatingPointError(
                        f"post-step optimizer state {name!r} is not finite"
                    )

        scheduler_state = self._scheduler_state()
        for field_name in ("base_lrs", "_last_lr"):
            for index, learning_rate in enumerate(scheduler_state[field_name]):
                self._validate_finite_learning_rate(
                    f"scheduler {field_name}[{index}]", learning_rate
                )

    @staticmethod
    def _validate_finite_learning_rate(name: str, value: object) -> None:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise FloatingPointError(f"post-step {name} must be scalar")
            learning_rate = float(value.detach().cpu())
        elif type(value) in (int, float):
            learning_rate = float(value)
        else:
            raise FloatingPointError(f"post-step {name} has an invalid type")
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise FloatingPointError(f"post-step {name} is not finite and nonnegative")

    @torch.no_grad()
    def evaluate(self, episode: EpisodeBatch) -> dict[str, float | int]:
        """Return a finite full-batch loss without changing trainer state."""
        if not isinstance(episode, EpisodeBatch):
            raise TypeError("episode must be an EpisodeBatch")
        was_training = self.model.training
        self.model.eval()
        try:
            output = self._forward_episode(episode)
            if output.loss is None:
                raise RuntimeError("evaluation episode did not produce a loss")
            if not bool(torch.isfinite(output.loss).all()):
                raise FloatingPointError("evaluation loss is not finite")
            loss = float(output.loss.cpu())
        finally:
            self.model.train(was_training)
        return {"loss": loss, "tokens": int(episode.valid.sum().item())}

    @property
    def _scheduler_spec(self) -> dict[str, int | str]:
        return {
            "name": "warmup_cosine",
            "warmup_updates": self.config.warmup_updates,
            "total_updates": self.config.updates,
        }

    def _checkpoint_payload(self) -> dict[str, Any]:
        model_state = self.model.state_dict()
        optimizer_state = _cpu_clone(self.optimizer.state_dict())
        active_entries = tuple(
            (name, parameter_id)
            for names, group in zip(
                self.optimizer_parameter_names,
                optimizer_state["param_groups"],
                strict=True,
            )
            for name, parameter_id in zip(names, group["params"], strict=True)
            if parameter_id in optimizer_state["state"]
        )
        active_names = tuple(name for name, _ in active_entries)
        active_steps = tuple(
            int(float(optimizer_state["state"][parameter_id]["step"]))
            for _, parameter_id in active_entries
        )
        return {
            "schema_version": TINY_CHECKPOINT_SCHEMA_VERSION,
            "job_id": self.config.job_id,
            "model_config_signature": _config_signature(self.model.config),
            "training_config_signature": _config_signature(self.config),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "model_state_names": tuple(model_state),
            "model_state": _cpu_clone(dict(model_state)),
            "optimizer_parameter_names": self.optimizer_parameter_names,
            "optimizer_active_parameter_names": active_names,
            "optimizer_active_parameter_steps": active_steps,
            "optimizer_state": optimizer_state,
            "scheduler_spec": self._scheduler_spec,
            "scheduler_state": _cpu_clone(self._scheduler_state()),
            "rng_state": self.rng.get_state().cpu().clone(),
            "metric_state": copy.deepcopy(self.metric_history),
        }

    def save_checkpoint(self, path: str | os.PathLike[str]) -> Path:
        """Atomically replace ``path`` with a complete, validated CPU checkpoint."""
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("checkpoint path must be a string or path-like object")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(destination)
        payload = self._checkpoint_payload()
        self._validate_checkpoint_payload(payload)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return destination

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Strictly validate and transactionally restore a checkpoint."""
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("checkpoint path must be a string or path-like object")
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            raw = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("checkpoint could not be decoded safely") from error
        payload = self._validate_checkpoint_payload(raw)

        previous_model = copy.deepcopy(self.model.state_dict())
        previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
        previous_scheduler = copy.deepcopy(self._scheduler_state())
        previous_rng = self.rng.get_state().clone()
        previous_step = self.step
        previous_tokens = self.tokens_seen
        previous_metrics = copy.deepcopy(self.metric_history)
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
            self.rng.set_state(payload["rng_state"])
            self.step = payload["step"]
            self.tokens_seen = payload["tokens_seen"]
            self.metric_history = copy.deepcopy(payload["metric_state"])
        except BaseException:
            self.model.load_state_dict(previous_model, strict=True)
            self.optimizer.load_state_dict(previous_optimizer)
            self.scheduler.load_state_dict(previous_scheduler)
            self.rng.set_state(previous_rng)
            self.step = previous_step
            self.tokens_seen = previous_tokens
            self.metric_history = previous_metrics
            raise

    def _scheduler_state(self) -> dict[str, Any]:
        state = self.scheduler.state_dict()
        return {name: state[name] for name in _SCHEDULER_STATE_FIELDS}

    def _validate_checkpoint_payload(self, payload: object) -> dict[str, Any]:
        if type(payload) is not dict:
            raise TypeError("checkpoint payload must be a dict")
        if set(payload) != set(_CHECKPOINT_FIELDS):
            missing = sorted(set(_CHECKPOINT_FIELDS) - set(payload))
            unknown = sorted(set(payload) - set(_CHECKPOINT_FIELDS))
            raise ValueError(
                f"checkpoint fields mismatch; missing={missing}, unknown={unknown}"
            )
        if payload["schema_version"] != TINY_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema_version is incompatible")
        if payload["job_id"] != self.config.job_id:
            raise ValueError("checkpoint job_id does not match this trainer")
        if payload["model_config_signature"] != _config_signature(self.model.config):
            raise ValueError("checkpoint model configuration does not match")
        if payload["training_config_signature"] != _config_signature(self.config):
            raise ValueError("checkpoint training configuration does not match")

        step = payload["step"]
        tokens_seen = payload["tokens_seen"]
        if type(step) is not int or not 0 <= step <= self.config.updates:
            raise ValueError("checkpoint step is outside the configured budget")
        if (
            type(tokens_seen) is not int
            or not 0 <= tokens_seen <= self.config.max_tokens
        ):
            raise ValueError("checkpoint tokens_seen is outside the configured budget")
        if step == 0 and tokens_seen != 0:
            raise ValueError("checkpoint at step zero must have zero tokens")

        expected_model = self.model.state_dict()
        expected_names = tuple(expected_model)
        if payload["model_state_names"] != expected_names:
            raise ValueError("checkpoint model state names/order do not match")
        saved_model = payload["model_state"]
        if type(saved_model) is not dict or tuple(saved_model) != expected_names:
            raise ValueError("checkpoint model_state keys/order do not match")
        for name, expected in expected_model.items():
            saved = saved_model[name]
            if not isinstance(saved, Tensor):
                raise TypeError(f"checkpoint model_state[{name!r}] must be a tensor")
            if saved.shape != expected.shape:
                raise ValueError(f"checkpoint model_state[{name!r}] shape does not match")
            if saved.dtype != expected.dtype:
                raise ValueError(f"checkpoint model_state[{name!r}] dtype does not match")
            _finite_tensor(f"checkpoint model_state[{name!r}]", saved)
            if name.rsplit(".", 1)[-1] == "cache_amplitude" and (
                bool((saved < 0).any()) or bool((saved > 1).any())
            ):
                raise ValueError("checkpoint cache amplitudes must be in [0,1]")

        if payload["optimizer_parameter_names"] != self.optimizer_parameter_names:
            raise ValueError("checkpoint optimizer parameter names/order do not match")
        self._validate_optimizer_state(
            payload["optimizer_state"],
            payload["optimizer_active_parameter_names"],
            payload["optimizer_active_parameter_steps"],
            step,
        )
        if payload["scheduler_spec"] != self._scheduler_spec:
            raise ValueError("checkpoint scheduler specification does not match")
        self._validate_scheduler_state(payload["scheduler_state"], step)
        self._validate_rng_state(payload["rng_state"])
        self._validate_metric_state(payload["metric_state"], step, tokens_seen)
        return payload

    def _validate_optimizer_state(
        self,
        state: object,
        active_names: object,
        active_steps: object,
        step: int,
    ) -> None:
        if type(state) is not dict or set(state) != {"state", "param_groups"}:
            raise ValueError("checkpoint optimizer_state structure does not match AdamW")
        if (
            type(active_names) is not tuple
            or any(type(name) is not str or not name for name in active_names)
            or len(set(active_names)) != len(active_names)
        ):
            raise ValueError(
                "checkpoint active Adam parameter-name manifest is invalid"
            )
        if (
            type(active_steps) is not tuple
            or len(active_steps) != len(active_names)
            or any(type(active_step) is not int for active_step in active_steps)
        ):
            raise ValueError(
                "checkpoint active Adam per-parameter step manifest is invalid"
            )
        saved_groups = state["param_groups"]
        template = self.optimizer.state_dict()
        template_groups = template["param_groups"]
        if type(saved_groups) is not list or len(saved_groups) != len(template_groups):
            raise ValueError("checkpoint optimizer group count does not match")

        parameters_by_id: dict[int, Tensor] = {}
        names_by_id: dict[int, str] = {}
        for group_index, (saved_group, template_group, live_group) in enumerate(
            zip(
                saved_groups,
                template_groups,
                self.optimizer.param_groups,
                strict=True,
            )
        ):
            if type(saved_group) is not dict or set(saved_group) != set(template_group):
                raise ValueError("checkpoint optimizer group fields do not match")
            if saved_group["params"] != template_group["params"]:
                raise ValueError("checkpoint optimizer parameter indices do not match")
            names = self.optimizer_parameter_names[group_index]
            for name, parameter_id, parameter in zip(
                names,
                template_group["params"],
                live_group["params"],
                strict=True,
            ):
                parameters_by_id[parameter_id] = parameter
                names_by_id[parameter_id] = name
            expected_lr = float(template_group["initial_lr"]) * self._schedule_multiplier(step)
            if saved_group["lr"] != expected_lr:
                raise ValueError("checkpoint optimizer learning rate is inconsistent")
            for key in template_group:
                if key in {"params", "lr"}:
                    continue
                if saved_group[key] != template_group[key]:
                    raise ValueError(
                        f"checkpoint optimizer group field {key!r} does not match"
                    )

        saved_slots = state["state"]
        if type(saved_slots) is not dict or not set(saved_slots).issubset(parameters_by_id):
            raise ValueError("checkpoint optimizer state parameter indices do not match")
        expected_active_ids = tuple(
            parameter_id
            for parameter_id in parameters_by_id
            if parameter_id in saved_slots
        )
        expected_active_names = tuple(
            names_by_id[parameter_id] for parameter_id in expected_active_ids
        )
        if active_names != expected_active_names:
            raise ValueError(
                "checkpoint active Adam manifest does not match saved slot IDs"
            )
        if step == 0 and (saved_slots or active_names or active_steps):
            raise ValueError(
                "checkpoint at step zero must have empty active Adam state"
            )
        if step > 0 and not saved_slots:
            raise ValueError("checkpoint active Adam state must not be empty")
        for manifest_index, parameter_id in enumerate(expected_active_ids):
            slot = saved_slots[parameter_id]
            if type(slot) is not dict or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("checkpoint AdamW slot fields do not match")
            parameter = parameters_by_id[parameter_id]
            saved_step = slot["step"]
            if (
                not isinstance(saved_step, Tensor)
                or saved_step.shape != torch.Size([])
                or not saved_step.is_floating_point()
                or saved_step.device.type != "cpu"
            ):
                raise ValueError(
                    "checkpoint AdamW step must be a scalar CPU floating tensor"
                )
            _finite_tensor("checkpoint AdamW step", saved_step)
            step_value = float(saved_step)
            manifest_step = active_steps[manifest_index]
            if not 1 <= manifest_step <= step:
                raise ValueError(
                    "checkpoint active Adam manifest step is outside global progress"
                )
            if step_value != float(manifest_step):
                raise ValueError(
                    "checkpoint active Adam scalar step does not match its manifest"
                )
            for name in ("exp_avg", "exp_avg_sq"):
                moment = slot[name]
                if not isinstance(moment, Tensor):
                    raise TypeError(f"checkpoint AdamW {name} must be a tensor")
                if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                    raise ValueError(f"checkpoint AdamW {name} shape/dtype does not match")
                _finite_tensor(f"checkpoint AdamW {name}", moment)

    def _validate_scheduler_state(self, state: object, step: int) -> None:
        template = self._scheduler_state()
        if type(state) is not dict or tuple(state) != _SCHEDULER_STATE_FIELDS:
            raise ValueError("checkpoint scheduler state fields do not match")
        if state["last_epoch"] != step or state["_step_count"] != step + 1:
            raise ValueError("checkpoint scheduler step is inconsistent")
        if state["base_lrs"] != template["base_lrs"]:
            raise ValueError("checkpoint scheduler base learning rates do not match")
        expected_lrs = [
            base_lr * self._schedule_multiplier(step) for base_lr in template["base_lrs"]
        ]
        if state["_last_lr"] != expected_lrs:
            raise ValueError("checkpoint scheduler learning rates are inconsistent")
        if state["lr_lambdas"] != template["lr_lambdas"]:
            raise ValueError("checkpoint scheduler lambda structure does not match")
        for key in ("_is_initial", "_get_lr_called_within_step"):
            if state[key] != template[key]:
                raise ValueError(f"checkpoint scheduler field {key!r} does not match")

    @staticmethod
    def _validate_rng_state(state: object) -> None:
        if (
            not isinstance(state, Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.device.type != "cpu"
        ):
            raise ValueError("checkpoint RNG state must be a one-dimensional CPU uint8 tensor")
        try:
            probe = torch.Generator(device="cpu")
            probe.set_state(state)
        except RuntimeError as error:
            raise ValueError("checkpoint RNG state is invalid") from error

    @staticmethod
    def _validate_metric_state(state: object, step: int, tokens_seen: int) -> None:
        if type(state) is not list or len(state) != step:
            raise ValueError("checkpoint metric state must contain one record per step")
        prior_tokens = 0
        for index, record in enumerate(state, start=1):
            if type(record) is not dict or set(record) != {"step", "tokens_seen", "loss"}:
                raise ValueError("checkpoint metric record fields do not match")
            if record["step"] != index:
                raise ValueError("checkpoint metric steps must be contiguous")
            record_tokens = record["tokens_seen"]
            if (
                type(record_tokens) is not int
                or not prior_tokens < record_tokens <= tokens_seen
            ):
                raise ValueError(
                    "checkpoint metric token counts must be positive and "
                    "strictly increase"
                )
            prior_tokens = record_tokens
            loss = record["loss"]
            if type(loss) is not float or not math.isfinite(loss):
                raise ValueError("checkpoint metric losses must be finite floats")
        if state and prior_tokens != tokens_seen:
            raise ValueError("checkpoint final metric token count does not match")


__all__ = [
    "TINY_CHECKPOINT_SCHEMA_VERSION",
    "TinyTrainer",
    "TinyTrainingConfig",
]
