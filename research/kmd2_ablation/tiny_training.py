"""Deterministic CPU-friendly training support for the tiny KMD-2 backend."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from .tasks import EpisodeBatch, generate_task
from .tiny_backend import (
    TinyFactors,
    TinyKMD2Config,
    TinyKMD2Model,
    TinyModelOutput,
    future_query_relevance,
    project_lookahead_gates_,
    project_momentum_gates_,
    project_trapezoid_gates_,
    tiny_factors_from_episode,
)


TINY_CHECKPOINT_SCHEMA_VERSION = "1.3.0"
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


class TinyRuntimeConfigurationError(ValueError):
    """A stable, machine-readable Tiny dispatcher configuration failure."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("error code must be a non-empty string")
        if type(message) is not str or not message:
            raise TypeError("error message must be a non-empty string")
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class TinyExecutionDependencies:
    """Injectable heavy boundaries for deterministic dispatcher tests."""

    generate_task: Callable[..., EpisodeBatch]
    build_model: Callable[..., TinyKMD2Model]
    build_trainer: Callable[..., "TinyTrainer"]
    monotonic: Callable[[], float]
    peak_vram_bytes: Callable[[], int]


_TINY_MAX_GRAD_NORM = 1.0
_TINY_ARM_STATUS: Mapping[str, str] = {
    "native": "supported",
    "rotation.current": "supported",
    "rotation.off": "supported",
    "rotation.constant_rate": "supported",
    "rotation.non_cumulative": "supported",
    "rotation.fixed_rope": "supported",
    "rotation.moving_frame_oracle": "supported",
    "convolution.on": "supported",
    "convolution.off": "supported",
    "trapezoid": "supported",
    "bc_bias": "supported",
    "bc_bias.diagonal_rescale": "supported",
    "bc_bias.constant_coordinate_oracle": "supported",
    "corrected_momentum": "supported",
    "causal_lookahead": "supported",
    "state_size.sweep": "supported",
    "true_mimo.sweep": "supported",
    "exact_cache.off": "supported",
    "exact_cache.current_block_only": "supported",
    "exact_cache.selector.exact_outer": "supported",
    "exact_cache.selector.coupled_paper": "supported",
    "exact_cache.selector.residual_only": "supported",
    "exact_cache.selector.write_value": "supported",
    "exact_cache.selector.recency": "supported",
    "exact_cache.selector.reservoir": "supported",
    "exact_cache.selector.future_query_oracle": "supported",
    "exact_cache.read.unit_l2": "supported",
    "exact_cache.read.fixed_temperature": "supported",
    "exact_cache.read.rmsnorm": "supported",
    "exact_cache.storage.bf16": "supported",
    "exact_cache.storage.fp32": "supported",
    "exact_cache.pre_rotation_diagnostic": "supported",
    "exact_cache.per_slot_read": "supported",
    "exact_cache.unbounded_oracle": "supported",
    "exact_cache.width.0": "supported",
    "exact_cache.width.8": "supported",
    "exact_cache.width.16": "supported",
    "exact_cache.width.32": "supported",
    "exact_cache.width.64": "supported",
    "exact_cache.width.128": "supported",
    "exact_cache.block.64": "supported",
    "exact_cache.block.128": "supported",
    "exact_cache.block.256": "supported",
    "exact_cache.rotation_factorial": "schema_missing_factorial_cell",
    "exact_cache.r_out_factorial": "schema_missing_r_out",
    "exact_cache.rotation_factorial.M00": "supported",
    "exact_cache.rotation_factorial.M10": "supported",
    "exact_cache.rotation_factorial.M01": "supported",
    "exact_cache.rotation_factorial.M11": "supported",
    "exact_cache.r_out_factorial.M00": "supported",
    "exact_cache.r_out_factorial.M10": "supported",
    "exact_cache.r_out_factorial.M01": "supported",
    "exact_cache.r_out_factorial.M11": "supported",
}


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


def _compatible_model_config_signatures(value: Any) -> frozenset[str]:
    """Return current and safe pre-variant hashes for disabled defaults."""

    current = _config_signature(value)
    canonical = _canonical_value(value)
    if not isinstance(canonical, dict):
        return frozenset({current})
    historical_defaults = {
        "corrected_momentum": False,
        "momentum_gamma_init": 0.0,
        "causal_lookahead": False,
        "lookahead_rho_init": 0.0,
        "bc_bias_mode": "none",
        "selector_seed": 0,
        "unbounded_cache": False,
        "per_slot_cache_read": False,
    }
    signatures = {current}

    def add_default_omission(names: tuple[str, ...]) -> None:
        if any(
            canonical.get(name) != historical_defaults[name]
            for name in names
        ):
            return
        historical = dict(canonical)
        for name in names:
            historical.pop(name)
        signatures.add(_config_signature(historical))

    pre_variant_fields = (
        "corrected_momentum",
        "momentum_gamma_init",
        "causal_lookahead",
        "lookahead_rho_init",
        "bc_bias_mode",
    )
    later_selector_fields = (
        "selector_seed",
        "unbounded_cache",
        "per_slot_cache_read",
    )
    add_default_omission(("bc_bias_mode",))
    add_default_omission(pre_variant_fields)
    add_default_omission(pre_variant_fields + later_selector_fields)
    return frozenset(signatures)


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
            self._project_constrained_parameters
        )

    def _schedule_multiplier(self, scheduler_step: int) -> float:
        warmup = self.config.warmup_updates
        if warmup and scheduler_step < warmup:
            return float(scheduler_step + 1) / float(warmup)
        decay_steps = max(1, self.config.updates - warmup)
        progress = min(1.0, max(0.0, (scheduler_step - warmup) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _project_constrained_parameters(
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
            project_trapezoid_gates_(self.model)
            project_momentum_gates_(self.model)
            project_lookahead_gates_(self.model)

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
                read_gate=(
                    None
                    if source.read_gate is None
                    else source.read_gate.to(device)
                ),
                trapezoid_rho=(
                    None
                    if source.trapezoid_rho is None
                    else source.trapezoid_rho.to(device)
                ),
                momentum_gamma=(
                    None
                    if source.momentum_gamma is None
                    else source.momentum_gamma.to(device)
                ),
                lookahead_rho=(
                    None
                    if source.lookahead_rho is None
                    else source.lookahead_rho.to(device)
                ),
                moving_frame_phase=(
                    None
                    if source.moving_frame_phase is None
                    else source.moving_frame_phase.to(device)
                ),
                cache_q=(
                    None if source.cache_q is None else source.cache_q.to(device)
                ),
                cache_k=(
                    None if source.cache_k is None else source.cache_k.to(device)
                ),
            )
        relevance = (
            future_query_relevance(episode).to(device)
            if self.model.config.cache is not None
            and self.model.config.cache.score == "future_query_oracle"
            else None
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
            future_relevance=relevance,
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
            if name.rsplit(".", 1)[-1] == "rho_head" and (
                bool((tensor.detach() < 0).any())
                or bool((tensor.detach() > 1).any())
            ):
                raise FloatingPointError(
                    f"post-step rho_head {name!r} is outside [0,1]"
                )
            if name.rsplit(".", 1)[-1] == "momentum_gamma" and (
                bool((tensor.detach() < 0).any())
                or bool((tensor.detach() > 1).any())
            ):
                raise FloatingPointError(
                    f"post-step momentum_gamma {name!r} is outside [0,1]"
                )
            if name.rsplit(".", 1)[-1] == "lookahead_rho" and (
                bool((tensor.detach() < 0).any())
                or bool((tensor.detach() > 1).any())
            ):
                raise FloatingPointError(
                    f"post-step lookahead_rho {name!r} is outside [0,1]"
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
        if payload["model_config_signature"] not in _compatible_model_config_signatures(
            self.model.config
        ):
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
            if name.rsplit(".", 1)[-1] == "rho_head" and (
                bool((saved < 0).any()) or bool((saved > 1).any())
            ):
                raise ValueError(
                    f"checkpoint rho_head {name!r} must be in [0,1]"
                )
            if name.rsplit(".", 1)[-1] == "momentum_gamma" and (
                bool((saved < 0).any()) or bool((saved > 1).any())
            ):
                raise ValueError(
                    f"checkpoint momentum_gamma {name!r} must be in [0,1]"
                )
            if name.rsplit(".", 1)[-1] == "lookahead_rho" and (
                bool((saved < 0).any()) or bool((saved > 1).any())
            ):
                raise ValueError(
                    f"checkpoint lookahead_rho {name!r} must be in [0,1]"
                )

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
def _default_peak_vram_bytes() -> int:
    return 0


def _default_dependencies() -> TinyExecutionDependencies:
    return TinyExecutionDependencies(
        generate_task=generate_task,
        build_model=TinyKMD2Model,
        build_trainer=TinyTrainer,
        monotonic=__import__("time").monotonic,
        peak_vram_bytes=_default_peak_vram_bytes,
    )


def _resolve_dependencies(
    dependencies: TinyExecutionDependencies | Mapping[str, object] | None,
) -> TinyExecutionDependencies:
    defaults = _default_dependencies()
    if dependencies is None:
        return defaults
    if isinstance(dependencies, TinyExecutionDependencies):
        values = dependencies
    elif isinstance(dependencies, Mapping):
        names = tuple(TinyExecutionDependencies.__dataclass_fields__)
        unknown = set(dependencies) - set(names)
        if unknown:
            raise TinyRuntimeConfigurationError(
                "runtime_configuration_invalid",
                "unknown Tiny dependencies: " + ", ".join(sorted(unknown)),
            )
        values = TinyExecutionDependencies(
            **{
                name: dependencies.get(name, getattr(defaults, name))
                for name in names
            }
        )
    else:
        raise TypeError(
            "dependencies must be TinyExecutionDependencies, a mapping, or None"
        )
    if any(
        not callable(getattr(values, name))
        for name in TinyExecutionDependencies.__dataclass_fields__
    ):
        raise TypeError("every Tiny execution dependency must be callable")
    return values


def _runtime_values(runtime: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    allowed = {"output", "dtype", "asset_hashes", "resume"}
    unknown = set(runtime) - allowed
    if unknown:
        raise TinyRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "unknown runtime keys: " + ", ".join(sorted(unknown)),
        )
    missing = allowed - set(runtime)
    if missing:
        raise TinyRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "missing runtime keys: " + ", ".join(sorted(missing)),
        )
    if runtime["dtype"] not in {"float32", "bfloat16"}:
        raise TinyRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "runtime.dtype must be float32 or bfloat16",
        )
    if type(runtime["resume"]) is not bool:
        raise TinyRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime.resume must be boolean"
        )
    if not isinstance(runtime["asset_hashes"], Mapping) or runtime["asset_hashes"]:
        raise TinyRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "Tiny jobs have no external assets; asset_hashes must be empty",
        )
    try:
        output = Path(runtime["output"]).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError) as error:
        raise TinyRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime.output is not writable"
        ) from error
    return {
        "output": output,
        "dtype": runtime["dtype"],
        "asset_hashes": {},
        "resume": runtime["resume"],
    }


_SELECTOR_ARM_SCORES = {
    "exact_cache.selector.exact_outer": "exact_outer",
    "exact_cache.selector.coupled_paper": "coupled_paper",
    "exact_cache.selector.residual_only": "residual_only",
    "exact_cache.selector.write_value": "write_value",
    "exact_cache.selector.recency": "recency",
    "exact_cache.selector.reservoir": "reservoir",
    "exact_cache.selector.future_query_oracle": "future_query_oracle",
}
_CACHE_ALIAS_BASE = ("exact_cache", "top_surprise")
_FACTORIAL_BASE_VARIANTS = {
    "exact_cache.rotation_factorial": "cache_rotation_factorial",
    "exact_cache.r_out_factorial": "cache_r_out_factorial",
}
_RELIANCE_PAIR_IDENTITIES = {
    "rotation": frozenset({"current_rotation", "rotation_off"}),
    "convolution": frozenset({"convolution_on", "convolution_off"}),
}


def _validate_tiny_arm_semantics(config: object, spec: object) -> None:
    """Bind registry-only cache aliases to explicit canonical controls.

    The scientific schema intentionally represents read/storage/geometry screens
    through the complete cache record and represents factorial cells through the
    generic factorial variant plus the job arm.  This validator accepts exactly
    those encodings while retaining the strict mechanism/variant check for every
    ordinary arm.
    """

    identity = (config.mechanism, config.variant)
    if identity == (spec.mechanism, spec.variant):
        expected_score = _SELECTOR_ARM_SCORES.get(spec.arm_id)
        if expected_score is not None and config.cache.score != expected_score:
            raise TinyRuntimeConfigurationError(
                "arm_configuration_mismatch",
                f"{spec.arm_id} requires cache.score={expected_score}",
            )
        return

    arm_id = spec.arm_id
    if arm_id == "native":
        # Preflight emits the complete-current native comparator with the
        # treatment's canonical task/budget configuration so the two jobs are
        # mechanically paired.  The job-level arm is the sole execution
        # override, just like the explicit cache read/storage aliases below.
        return
    reliance_variants = _RELIANCE_PAIR_IDENTITIES.get(spec.mechanism)
    if (
        spec.evidence_kind == "reliance"
        and config.mechanism == spec.mechanism
        and reliance_variants is not None
        and config.variant in reliance_variants
        and spec.variant in reliance_variants
    ):
        # Reliance jobs are an explicit two-arm intervention.  Their canonical
        # task, budget, and seed stay byte-identical while the job arm selects
        # the active or ablated runtime gate.
        return
    if arm_id.startswith("exact_cache.read.") and identity == _CACHE_ALIAS_BASE:
        expected = arm_id.rsplit(".", 1)[1]
        if config.cache.read == expected:
            return
    elif arm_id.startswith("exact_cache.storage.") and identity == _CACHE_ALIAS_BASE:
        expected = arm_id.rsplit(".", 1)[1]
        if config.cache.storage_dtype == expected:
            return
    elif arm_id == "exact_cache.pre_rotation_diagnostic" and identity == _CACHE_ALIAS_BASE:
        if (
            config.cache.coordinate_frame == "pre_rotation"
            and config.cache.pre_rotation_diagnostic
        ):
            return
    elif arm_id.startswith("exact_cache.width."):
        declared = int(arm_id.rsplit(".", 1)[1])
        expected_identity = (
            ("current_block_only", "chunk_only")
            if declared == 0
            else _CACHE_ALIAS_BASE
        )
        if identity == expected_identity and config.cache.width == declared:
            return
    elif arm_id.startswith("exact_cache.block.") and identity == _CACHE_ALIAS_BASE:
        declared = int(arm_id.rsplit(".", 1)[1])
        if config.cache.block_size == declared:
            return
    else:
        for prefix, base_variant in _FACTORIAL_BASE_VARIANTS.items():
            if not arm_id.startswith(prefix + "."):
                continue
            cell = arm_id.rsplit(".", 1)[1]
            cells = config.task.params.get("four_cells")
            if (
                identity == ("exact_cache", base_variant)
                and cell in {"M00", "M10", "M01", "M11"}
                and isinstance(cells, (list, tuple))
                and set(cells) == {"M00", "M10", "M01", "M11"}
                and len(cells) == 4
            ):
                return

    raise TinyRuntimeConfigurationError(
        "arm_configuration_mismatch",
        "job arm does not match canonical mechanism/variant and declared cache controls",
    )


def _validated_job(job: Mapping[str, object], runtime: Mapping[str, object]):
    from .config import ExperimentConfig
    from .results import (
        RESULT_SCHEMA_VERSION,
        SUITE_VERSION,
        canonical_json_bytes,
        semantic_job_id,
    )
    from .variants import (
        VariantCompatibilityError,
        all_variants,
        get_variant,
        validate_variant_compatibility,
    )

    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    allowed = {
        "schema_version",
        "suite_version",
        "job_id",
        "experiment_id",
        "seed",
        "stage",
        "backend",
        "arm_id",
        "canonical_config",
        "pairing_id",
    }
    required = allowed - {"pairing_id"}
    missing = required - set(job)
    unknown = set(job) - allowed
    if missing or unknown:
        raise TinyRuntimeConfigurationError(
            "job_configuration_invalid",
            f"job fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}",
        )
    if job["schema_version"] != RESULT_SCHEMA_VERSION or job["suite_version"] != SUITE_VERSION:
        raise TinyRuntimeConfigurationError(
            "job_configuration_invalid", "job schema or suite version is incompatible"
        )
    if job["backend"] != "tiny":
        raise TinyRuntimeConfigurationError(
            "job_configuration_invalid", "Tiny dispatcher received a non-Tiny job"
        )
    semantic = job["canonical_config"]
    if not isinstance(semantic, Mapping):
        raise TinyRuntimeConfigurationError(
            "job_configuration_invalid", "job.canonical_config must be a mapping"
        )
    raw = copy.deepcopy(dict(semantic))
    raw["runtime"] = {
        "output_path": str(runtime["output"]),
        "device_ordinal": 0,
    }
    try:
        config = ExperimentConfig.from_dict(raw)
    except (TypeError, ValueError) as error:
        raise TinyRuntimeConfigurationError(
            "job_configuration_invalid", f"canonical config is invalid: {error}"
        ) from error
    if canonical_json_bytes(config.semantic_dict()) != canonical_json_bytes(semantic):
        raise TinyRuntimeConfigurationError(
            "job_configuration_invalid", "canonical config changed during validation"
        )
    if config.backend != "tiny" or job["seed"] not in config.seeds:
        raise TinyRuntimeConfigurationError(
            "job_configuration_invalid", "job backend/seed does not match config"
        )
    pairing = job.get("pairing_id")
    expected_job_id = semantic_job_id(
        semantic,
        backend="tiny",
        arm_id=job["arm_id"],
        seed=job["seed"],
        stage=job["stage"],
        pairing_id=pairing,
    )
    expected_experiment_id = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    if job["job_id"] != expected_job_id or job["experiment_id"] != expected_experiment_id:
        raise TinyRuntimeConfigurationError(
            "job_identity_mismatch", "job or experiment identity does not match semantics"
        )
    try:
        spec = get_variant(job["arm_id"])
    except (KeyError, TypeError) as error:
        raise TinyRuntimeConfigurationError(
            "arm_invalid", f"unknown Tiny arm: {job['arm_id']!r}"
        ) from error
    _validate_tiny_arm_semantics(config, spec)
    compatibility_spec = spec
    if spec.arm_id == "native" and (
        config.mechanism,
        config.variant,
    ) != (spec.mechanism, spec.variant):
        matches = tuple(
            candidate
            for candidate in all_variants()
            if (candidate.mechanism, candidate.variant)
            == (config.mechanism, config.variant)
        )
        if len(matches) != 1:
            raise TinyRuntimeConfigurationError(
                "arm_configuration_mismatch",
                "native comparator cannot resolve the configured treatment arm",
            )
        compatibility_spec = matches[0]
    try:
        validate_variant_compatibility(
            compatibility_spec.arm_id,
            backend="tiny",
            task=config.task.name,
            stage=job["stage"],
            experiment_kind=compatibility_spec.experiment_kind,
        )
    except VariantCompatibilityError as error:
        raise TinyRuntimeConfigurationError(
            "arm_incompatible", str(error)
        ) from error
    status = _TINY_ARM_STATUS.get(spec.arm_id)
    if status is None:
        raise TinyRuntimeConfigurationError(
            "arm_coverage_missing", f"Tiny arm coverage missing for {spec.arm_id}"
        )
    if status != "supported":
        code = (
            "arm_schema_unsupported"
            if status.startswith("schema_")
            else "arm_semantics_unsupported"
        )
        raise TinyRuntimeConfigurationError(code, f"{spec.arm_id}: {status}")
    if config.task.name == "ruler":
        raise TinyRuntimeConfigurationError(
            "task_backend_unsupported", "RULER is Qwen-only"
        )
    return config, spec


def _generate_exact_train_episode(
    config: object,
    *,
    seed: int,
    generator: Callable[..., EpisodeBatch],
) -> tuple[EpisodeBatch, int]:
    from .runner import MalformedInput

    per_update, remainder = divmod(config.budget.tokens, config.budget.updates)
    if remainder:
        raise TinyRuntimeConfigurationError(
            "budget_unrepresentable",
            "Tiny token budget must divide exactly across update budget",
        )
    first_generation_error: BaseException | None = None
    for length in config.lengths.curriculum:
        def create(batch_size: int) -> EpisodeBatch:
            return generator(
                config.task.name,
                batch_size,
                length,
                seed,
                "train",
                _task_generator_params(config.task.params),
            )

        try:
            first = create(1)
        except (TypeError, ValueError) as error:
            first_generation_error = first_generation_error or error
            continue
        first_tokens = int(first.valid.sum().item())
        if first_tokens == per_update:
            return first, 1
        if first_tokens > per_update:
            continue
        low, high = 1, max(2, per_update // first_tokens + 1)
        while True:
            candidate = create(high)
            count = int(candidate.valid.sum().item())
            if count >= per_update:
                break
            low, high = high, high * 2
            if high > per_update:
                break
        while low <= high:
            middle = (low + high) // 2
            candidate = create(middle)
            count = int(candidate.valid.sum().item())
            if count == per_update:
                return candidate, middle
            if count < per_update:
                low = middle + 1
            else:
                high = middle - 1
    if first_generation_error is not None:
        raise MalformedInput(
            str(first_generation_error),
            phase="data_generation",
            context={"task": config.task.name, "split": "train"},
        ) from first_generation_error
    raise TinyRuntimeConfigurationError(
        "budget_unrepresentable",
        "no configured curriculum length and integer batch size exactly represents "
        f"{per_update} tokens per update",
    )


def _generate_evaluation_episodes(
    config: object,
    *,
    seed: int,
    batch_size: int,
    generator: Callable[..., EpisodeBatch],
) -> tuple[EpisodeBatch, ...]:
    from .runner import MalformedInput

    requests = (
        ("id", config.lengths.curriculum[-1], seed + 1),
        ("ood_2x", config.lengths.extrapolation[0], seed + 2),
        ("ood_4x", config.lengths.extrapolation[-1], seed + 3),
    )
    episodes: list[EpisodeBatch] = []
    for split, length, evaluation_seed in requests:
        try:
            episode = generator(
                config.task.name,
                batch_size,
                length,
                evaluation_seed,
                split,
                _task_generator_params(config.task.params),
            )
        except (TypeError, ValueError) as error:
            raise MalformedInput(
                str(error),
                phase="data_generation",
                context={"task": config.task.name, "split": split},
            ) from error
        episodes.append(episode)
    return tuple(episodes)


_TINY_EXECUTION_TASK_PARAMS = frozenset(
    {
        "continuous_input_dim",
        "four_cells",
        "mimo_rank",
        "output_dim",
        "parameter_match_target",
        "r_out",
        "vocab_size",
    }
)


def _task_generator_params(params: Mapping[str, object]) -> dict[str, object]:
    """Separate declared Tiny model/control semantics from generator semantics."""

    return {
        name: value
        for name, value in params.items()
        if name not in _TINY_EXECUTION_TASK_PARAMS
    }


def _episode_digest(episodes: Sequence[EpisodeBatch]) -> dict[str, object]:
    digest = hashlib.sha256()
    size_bytes = 0

    def update_bytes(name: str, payload: bytes) -> None:
        nonlocal size_bytes
        header = json.dumps([name, len(payload)], separators=(",", ":")).encode()
        digest.update(header)
        digest.update(payload)
        size_bytes += len(payload)

    for episode_index, episode in enumerate(episodes):
        prefix = f"episode[{episode_index}]"
        metadata = {
            "task": episode.task,
            "split": episode.split,
            "seed": episode.seed,
            "example_ids": episode.example_ids,
            "metadata": episode.metadata,
        }
        update_bytes(
            f"{prefix}.metadata",
            json.dumps(
                _canonical_value(metadata),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode(),
        )
        tensors: dict[str, Tensor] = {
            "targets": episode.targets,
            "valid": episode.valid,
            "positions": episode.positions,
            "loss_mask": episode.loss_mask,
            "query_mask": episode.query_mask,
            "boundaries": episode.boundaries,
            "source_spans": episode.source_spans,
        }
        if episode.input_ids is not None:
            tensors["input_ids"] = episode.input_ids
        if episode.continuous_inputs is not None:
            tensors["continuous_inputs"] = episode.continuous_inputs
        if episode.direct_factors is not None:
            tensors.update(
                {f"direct_factors.{name}": value for name, value in episode.direct_factors.items()}
            )
        tensors.update({f"strata.{name}": value for name, value in episode.strata.items()})
        for name in sorted(tensors):
            tensor = tensors[name].detach().cpu().contiguous()
            update_bytes(
                f"{prefix}.{name}.schema",
                json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode(),
            )
            update_bytes(
                f"{prefix}.{name}.values",
                tensor.view(torch.uint8).numpy().tobytes(),
            )
    return {
        "kind": "deterministic_synthetic_episodes",
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "episode_count": len(episodes),
        "example_count": sum(len(episode.example_ids) for episode in episodes),
    }


def _model_dimensions(episodes: Sequence[EpisodeBatch]) -> tuple[int, int | None, int | None]:
    vocab_size = 2
    continuous_input_dim: int | None = None
    output_dim: int | None = None
    for episode in episodes:
        if episode.input_ids is not None:
            valid_inputs = episode.input_ids[episode.valid]
            valid_targets = episode.targets[episode.loss_mask]
            for values in (valid_inputs, valid_targets):
                if values.numel():
                    vocab_size = max(vocab_size, int(values.max().item()) + 1)
        elif episode.continuous_inputs is not None:
            dimension = episode.continuous_inputs.shape[-1]
            if continuous_input_dim not in {None, dimension}:
                raise TinyRuntimeConfigurationError(
                    "task_shape_mismatch", "continuous input dimensions disagree"
                )
            continuous_input_dim = dimension
            output_dim = episode.targets.shape[-1]
        else:
            output_dim = episode.targets.shape[-1]
    return vocab_size, continuous_input_dim, output_dim


def _tiny_model_config(
    config: object,
    spec: object,
    episodes: Sequence[EpisodeBatch],
    *,
    dtype: str,
    selector_seed: int,
) -> TinyKMD2Config:
    from .config import CacheConfig

    arm_id = spec.arm_id
    params = config.task.params
    rotation_modes = {
        "rotation.current": "current",
        "rotation.off": "none",
        "rotation.constant_rate": "constant_rate",
        "rotation.non_cumulative": "non_cumulative",
        "rotation.fixed_rope": "fixed_rope",
        "rotation.moving_frame_oracle": "moving_frame",
    }
    rotation_mode = rotation_modes.get(arm_id, "current")
    if arm_id == "bc_bias.constant_coordinate_oracle":
        rotation_mode = "none"
    factorial_cell = (
        arm_id.rsplit(".", 1)[1]
        if arm_id.startswith("exact_cache.rotation_factorial.")
        or arm_id.startswith("exact_cache.r_out_factorial.")
        else None
    )
    cache_factor_enabled = factorial_cell is not None and factorial_cell[1] == "1"
    feature_factor_enabled = factorial_cell is not None and factorial_cell[2] == "1"
    if arm_id.startswith("exact_cache.rotation_factorial."):
        rotation_mode = "current" if feature_factor_enabled else "none"
    rotation_gate = 0.0 if rotation_mode == "none" else 1.0
    convolution_gate = 0.0 if arm_id == "convolution.off" else 1.0
    cache = None
    cache_enabled = (
        arm_id.startswith("exact_cache.")
        and arm_id != "exact_cache.off"
        and not arm_id in _FACTORIAL_BASE_VARIANTS
        and (factorial_cell is None or cache_factor_enabled)
    )
    if cache_enabled:
        cache = CacheConfig(**_canonical_value(config.cache))
        expected_score = _SELECTOR_ARM_SCORES.get(arm_id)
        if expected_score is not None and cache.score != expected_score:
            raise TinyRuntimeConfigurationError(
                "arm_configuration_mismatch",
                f"{arm_id} requires cache.score={expected_score}",
            )
    vocab_size, continuous_input_dim, output_dim = _model_dimensions(episodes)
    declared_vocab_size = params.get("vocab_size", vocab_size)
    if type(declared_vocab_size) is not int or declared_vocab_size < vocab_size:
        raise TinyRuntimeConfigurationError(
            "task_shape_mismatch",
            "task.params.vocab_size must cover every generated input and target ID",
        )
    for name, observed in (
        ("continuous_input_dim", continuous_input_dim),
        ("output_dim", output_dim),
    ):
        declared = params.get(name, observed)
        if declared != observed:
            raise TinyRuntimeConfigurationError(
                "task_shape_mismatch",
                f"task.params.{name} does not match generated episode tensors",
            )
    r_out = params.get("r_out", 1)
    if arm_id.startswith("exact_cache.r_out_factorial."):
        r_out = 4 if feature_factor_enabled else 1
    mimo_rank = params.get("mimo_rank", 1)
    for name, value in (("r_out", r_out), ("mimo_rank", mimo_rank)):
        if type(value) is not int or value < 1:
            raise TinyRuntimeConfigurationError(
                "arm_configuration_mismatch",
                f"task.params.{name} must be a positive integer",
            )
    if arm_id == "true_mimo.sweep":
        if mimo_rank <= 1 or r_out != 1:
            raise TinyRuntimeConfigurationError(
                "arm_configuration_mismatch",
                "true_mimo.sweep requires mimo_rank>1 and r_out=1",
            )
    elif mimo_rank != 1:
        raise TinyRuntimeConfigurationError(
            "arm_configuration_mismatch",
            "mimo_rank>1 is only valid for true_mimo.sweep",
        )
    bc_mode = {
        "bc_bias": "additive",
        "bc_bias.diagonal_rescale": "diagonal_rescale",
        "bc_bias.constant_coordinate_oracle": "constant_coordinate_oracle",
    }.get(arm_id, "none")
    try:
        translated = TinyKMD2Config(
            d_model=config.model.hidden_size,
            heads=config.model.num_heads,
            dk=config.model.state_key_dim,
            dv=config.model.state_value_dim,
            layers=config.model.num_layers,
            vocab_size=declared_vocab_size,
            d_ff=config.model.ffn_dim,
            r_out=r_out,
            mimo_rank=mimo_rank,
            continuous_input_dim=continuous_input_dim,
            output_dim=output_dim,
            dtype=torch.float32 if dtype == "float32" else torch.bfloat16,
            rotation_mode=rotation_mode,
            convolution_gate_init=convolution_gate,
            rotation_gate_init=rotation_gate,
            channel_decay_gate_init=1.0,
            write_offset_gate_init=1.0,
            trapezoid=arm_id == "trapezoid",
            corrected_momentum=arm_id == "corrected_momentum",
            causal_lookahead=arm_id == "causal_lookahead",
            bc_bias_mode=bc_mode,
            cache=cache,
            selector_seed=selector_seed,
            unbounded_cache=arm_id == "exact_cache.unbounded_oracle",
            per_slot_cache_read=arm_id == "exact_cache.per_slot_read",
        )
        if arm_id == "native" and config.mechanism in {"state_size", "true_mimo"}:
            target_declaration = params.get("parameter_match_target")
            if not isinstance(target_declaration, Mapping):
                raise TinyRuntimeConfigurationError(
                    "arm_configuration_mismatch",
                    "matched native comparator requires task.params.parameter_match_target",
                )
            allowed_target = {"state_key_dim", "state_value_dim", "mimo_rank"}
            unknown_target = set(target_declaration) - allowed_target
            if unknown_target:
                raise TinyRuntimeConfigurationError(
                    "arm_configuration_mismatch",
                    "unknown parameter_match_target keys: "
                    + ", ".join(sorted(unknown_target)),
                )
            translated = replace(
                translated,
                dk=target_declaration.get("state_key_dim", translated.dk),
                dv=target_declaration.get("state_value_dim", translated.dv),
                mimo_rank=target_declaration.get("mimo_rank", 1),
                r_out=1,
                cache=None,
            )
        if arm_id in {"state_size.sweep", "true_mimo.sweep"}:
            from .variants import match_tiny_parameter_count

            target_declaration = params.get("parameter_match_target")
            if not isinstance(target_declaration, Mapping):
                raise TinyRuntimeConfigurationError(
                    "arm_configuration_mismatch",
                    f"{arm_id} requires task.params.parameter_match_target",
                )
            allowed_target = {"state_key_dim", "state_value_dim", "mimo_rank"}
            unknown_target = set(target_declaration) - allowed_target
            if unknown_target:
                raise TinyRuntimeConfigurationError(
                    "arm_configuration_mismatch",
                    "unknown parameter_match_target keys: "
                    + ", ".join(sorted(unknown_target)),
                )
            target = replace(
                translated,
                dk=target_declaration.get("state_key_dim", translated.dk),
                dv=target_declaration.get("state_value_dim", translated.dv),
                mimo_rank=target_declaration.get("mimo_rank", 1),
                r_out=1,
                cache=None,
            )
            matched = match_tiny_parameter_count(
                target,
                replace(translated, cache=None),
                comparison=(
                    "state_size" if arm_id == "state_size.sweep" else "mimo_rank"
                ),
                d_ff_match_min=config.model.ffn_match_lower,
                d_ff_match_max=config.model.ffn_match_upper,
            )
            translated = matched.matched.config
        return translated
    except TinyRuntimeConfigurationError:
        raise
    except (TypeError, ValueError, NotImplementedError) as error:
        raise TinyRuntimeConfigurationError(
            "arm_translation_invalid", f"Tiny model configuration is invalid: {error}"
        ) from error


def _training_configuration(config: object, job_id: str, seed: int) -> TinyTrainingConfig:
    return TinyTrainingConfig(
        job_id=job_id,
        seed=seed,
        updates=config.budget.updates,
        max_tokens=config.budget.tokens,
        learning_rate=config.optimizer.learning_rate,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        warmup_updates=config.schedule.warmup_updates,
        max_grad_norm=_TINY_MAX_GRAD_NORM,
    )


def _checkpoint_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "kind": "tiny_atomic_checkpoint",
        "schema_version": TINY_CHECKPOINT_SCHEMA_VERSION,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _evaluate_episodes(
    trainer: TinyTrainer, episodes: Sequence[EpisodeBatch]
) -> tuple[dict[str, float], list[float], list[tuple[EpisodeBatch, TinyModelOutput]], int]:
    losses: list[float] = []
    evaluated: list[tuple[EpisodeBatch, TinyModelOutput]] = []
    correct = total = exact_examples = example_count = 0
    squared_error = regression_elements = 0.0
    evaluation_tokens = 0
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            for episode in episodes:
                output = trainer._forward_episode(episode)
                if output.loss is None or not bool(torch.isfinite(output.loss)):
                    raise FloatingPointError("evaluation loss is not finite")
                losses.append(float(output.loss.detach().cpu()))
                evaluated.append((episode, output))
                evaluation_tokens += int(episode.valid.sum().item())
                mask = episode.loss_mask.to(output.logits.device)
                if episode.targets.ndim == 2:
                    predictions = output.logits.argmax(dim=-1).cpu()
                    selected = episode.loss_mask
                    outcomes = predictions[selected] == episode.targets[selected]
                    correct += int(outcomes.sum().item())
                    total += outcomes.numel()
                    for row in range(episode.valid.shape[0]):
                        row_outcomes = predictions[row][episode.loss_mask[row]] == episode.targets[row][episode.loss_mask[row]]
                        exact_examples += int(bool(row_outcomes.all()))
                        example_count += 1
                else:
                    delta = output.logits.cpu()[episode.loss_mask] - episode.targets[episode.loss_mask]
                    squared_error += float(delta.double().square().sum().item())
                    regression_elements += float(delta.numel())
                    example_count += episode.valid.shape[0]
    finally:
        trainer.model.train(was_training)
    metrics: dict[str, float] = {"eval_loss": math.fsum(losses) / len(losses)}
    if total:
        metrics["token_accuracy"] = correct / total
        metrics["episode_exact_match"] = exact_examples / example_count
    if regression_elements:
        metrics["regression_mse"] = squared_error / regression_elements
    return metrics, losses, evaluated, evaluation_tokens


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else math.fsum(values) / len(values)


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _canonical_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _exact_cache_payload(
    config: object,
    model_config: TinyKMD2Config,
    evaluated: Sequence[tuple[EpisodeBatch, TinyModelOutput]],
    *,
    amplitude_initial: Sequence[float],
    amplitude_final: Sequence[float],
    cache_active: bool,
) -> dict[str, object]:
    selected_indices: list[int] = []
    scores: list[float] = []
    persistent_hits: list[bool] = []
    conditional_correct: list[bool] = []
    sinks: list[float] = []
    entropies: list[float] = []
    top1_masses: list[float] = []
    cache_norms: list[float] = []
    state_norms: list[float] = []
    stale_flags: list[bool] = []
    stale_errors: list[bool] = []
    slot_sinks: list[float] = []
    slot_entropies: list[float] = []
    slot_top1_masses: list[float] = []
    slot_cache_norms: list[float] = []
    slot_top1_positions: list[int] = []
    retention_count = eviction_count = persistent_bytes = block_bytes = 0
    effective_width = 0
    slot_count = 0
    for episode, output in evaluated:
        predictions = (
            output.logits.argmax(dim=-1).cpu()
            if episode.targets.ndim == 2
            else None
        )
        for cell in output.cell_outputs:
            effective_width = max(effective_width, cell.selected_positions.shape[-1])
            slot_count = max(slot_count, cell.slot_cache_read.shape[-2])
            selected_indices.extend(
                int(value)
                for value in cell.selected_positions.detach().cpu().flatten().tolist()
                if value >= 0
            )
            expanded_valid = episode.valid.unsqueeze(-1).expand_as(cell.scores.cpu())
            scores.extend(float(value) for value in cell.scores.cpu()[expanded_valid].tolist())
            retention_count += cell.retention_count
            eviction_count += cell.eviction_count
            persistent_bytes = max(persistent_bytes, cell.cache_persistent_bytes)
            block_bytes = max(block_bytes, cell.cache_block_bytes)
            if cell.slot_cache_read.numel():
                slot_cache_read = cell.slot_cache_read.detach().cpu()
                slot_sink_mass = cell.slot_sink_mass.detach().cpu()
                slot_attention_entropy = cell.slot_attention_entropy.detach().cpu()
                slot_top1_mass = cell.slot_top1_mass.detach().cpu()
                slot_top1_position = cell.slot_top1_positions.detach().cpu()
                slot_valid = (
                    episode.valid.detach()
                    .cpu()
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand_as(slot_sink_mass)
                )
                slot_cache_norms.extend(
                    float(value)
                    for value in torch.linalg.vector_norm(
                        slot_cache_read, dim=-1
                    )[slot_valid].tolist()
                )
                slot_sinks.extend(
                    float(value)
                    for value in slot_sink_mass[slot_valid].tolist()
                )
                slot_entropies.extend(
                    float(value)
                    for value in slot_attention_entropy[slot_valid].tolist()
                )
                slot_top1_masses.extend(
                    float(value)
                    for value in slot_top1_mass[slot_valid].tolist()
                )
                slot_top1_positions.extend(
                    int(value)
                    for value in slot_top1_position[slot_valid].tolist()
                )
            for batch_index in range(episode.valid.shape[0]):
                segment_start = 0
                for token_index in range(episode.valid.shape[1]):
                    if not bool(episode.valid[batch_index, token_index]):
                        continue
                    if bool(episode.boundaries[batch_index, token_index]):
                        segment_start = token_index
                    for head in range(cell.sink_mass.shape[-1]):
                        sinks.append(float(cell.sink_mass[batch_index, token_index, head]))
                        entropies.append(float(cell.attention_entropy[batch_index, token_index, head]))
                        top1_masses.append(float(cell.top1_mass[batch_index, token_index, head]))
                        cache_norms.append(float(torch.linalg.vector_norm(cell.cache_read[batch_index, token_index, head]).cpu()))
                        state_norms.append(float(torch.linalg.vector_norm(cell.state_read[batch_index, token_index, head]).cpu()))
                    if not bool(episode.query_mask[batch_index, token_index]):
                        continue
                    start, stop = episode.source_spans[batch_index, token_index].tolist()
                    gold_positions = set(
                        int(value)
                        for value in episode.positions[batch_index, start:stop].tolist()
                    )
                    query_is_latest = bool(
                        episode.strata.get("query_type", torch.full_like(episode.positions, -1))[batch_index, token_index] == 0
                    )
                    stale_positions: set[int] = set()
                    admission = episode.strata.get("admission_score")
                    if query_is_latest and admission is not None:
                        for raw_index in range(segment_start, token_index):
                            position = int(episode.positions[batch_index, raw_index])
                            if float(admission[batch_index, raw_index]) > 0 and position not in gold_positions:
                                stale_positions.add(position)
                    for head in range(cell.sink_mass.shape[-1]):
                        persistent = {
                            int(value)
                            for value in cell.persistent_selected_positions[
                                batch_index, token_index, head
                            ].detach().cpu().tolist()
                            if value >= 0
                        }
                        hit = bool(persistent & gold_positions)
                        persistent_hits.append(hit)
                        top1 = int(cell.top1_positions[batch_index, token_index, head])
                        if hit:
                            conditional_correct.append(top1 in gold_positions)
                        candidates = [
                            int(value)
                            for value in cell.hit_ready_positions[
                                batch_index, token_index, head
                            ].detach().cpu().tolist()
                            if value >= 0
                        ]
                        stale_flags.extend(value in stale_positions for value in candidates)
                        if query_is_latest:
                            wrong = bool(
                                predictions is not None
                                and predictions[batch_index, token_index]
                                != episode.targets[batch_index, token_index]
                            )
                            stale_errors.append(wrong and top1 in stale_positions)
    if not scores:
        raise TinyRuntimeConfigurationError(
            "cache_diagnostics_invalid", "cache evaluation produced no scores"
        )
    score_min, score_max = min(scores), max(scores)
    return {
        "width": effective_width,
        "block_size": config.cache.block_size,
        "score_definition": config.cache.score,
        "compute_dtype": config.cache.compute_dtype,
        "storage_dtype": config.cache.storage_dtype,
        "coordinate_frame": config.cache.coordinate_frame,
        "inclusive_causality": config.cache.inclusive,
        "tie_policy": config.cache.tie_policy,
        "amplitude_initial": list(amplitude_initial),
        "amplitude_final": list(amplitude_final),
        "selected_index_digest": _digest_json(selected_indices),
        "selected_index_sample": selected_indices[:32],
        "score_digest": _digest_json(scores),
        "score_statistics": {
            "count": len(scores),
            "min": score_min,
            "max": score_max,
            "mean": _mean(scores),
        },
        "retention_count": retention_count,
        "eviction_count": eviction_count,
        "persistent_hit_rate": _mean([float(value) for value in persistent_hits]),
        "conditional_read_accuracy": _mean([float(value) for value in conditional_correct]),
        "sink_mass": _mean(sinks),
        "attention_entropy": _mean(entropies),
        "top1_mass": _mean(top1_masses),
        "stale_occupancy": _mean([float(value) for value in stale_flags]),
        "stale_error": _mean([float(value) for value in stale_errors]),
        "cache_output_norm": _mean(cache_norms),
        "state_output_norm": _mean(state_norms),
        "persistent_bytes": persistent_bytes,
        "block_bytes": block_bytes,
        "cache_active": cache_active,
        "declared_width": config.cache.width,
        "effective_width": effective_width,
        "selector_seed": model_config.selector_seed,
        "selector_policy": config.cache.score,
        "model_r_out": model_config.r_out,
        "rotation_mode": model_config.rotation_mode,
        "unbounded_cache": model_config.unbounded_cache,
        "per_slot_read": model_config.per_slot_cache_read,
        "slot_count": slot_count,
        "slot_sink_mass": _mean(slot_sinks),
        "slot_attention_entropy": _mean(slot_entropies),
        "slot_top1_mass": _mean(slot_top1_masses),
        "slot_cache_output_norm": _mean(slot_cache_norms),
        "slot_top1_position_digest": _digest_json(slot_top1_positions),
        "implementation_paths": {
            "scan": "tiny_backend.TinyKMD2Cell._forward_fp32",
            "score": (
                f"exact_cache.admission_scores.{config.cache.score}"
                if cache_active
                else "tiny_backend.TinyKMD2Cell._forward_fp32.native_score"
            ),
            "selection": (
                "exact_cache.merge_persistent_cache.deterministic_topw"
                if cache_active
                else "disabled_no_cache"
            ),
            "read": (
                f"exact_cache.cache_read_blocks.{config.cache.read}"
                if cache_active
                else "disabled_no_cache"
            ),
        },
    }


def _amplitudes(model: TinyKMD2Model) -> list[float]:
    return [
        float(value)
        for name, parameter in model.named_parameters()
        if name.rsplit(".", 1)[-1] == "cache_amplitude"
        for value in parameter.detach().cpu().flatten().tolist()
    ]


def _warm_start_from_native_(
    model: TinyKMD2Model, spec: object, *, seed: int
) -> None:
    """Copy every compatible unchanged tensor from the complete native baseline."""

    if spec.experiment_kind == "cold_redesign" or spec.arm_id == "native":
        return
    try:
        native_config = replace(
            model.config,
            rotation_mode="current",
            rotation_gate_init=1.0,
            convolution_gate_init=1.0,
            channel_decay_gate_init=1.0,
            write_offset_gate_init=1.0,
            trapezoid=False,
            corrected_momentum=False,
            causal_lookahead=False,
            bc_bias_mode="none",
            cache=None,
            unbounded_cache=False,
            per_slot_cache_read=False,
        )
        native_state = TinyKMD2Model(native_config, init_seed=seed).state_dict()
    except (TypeError, ValueError) as error:
        raise TinyRuntimeConfigurationError(
            "native_warm_start_invalid",
            f"cannot construct the complete native initialization: {error}",
        ) from error
    arm_state = model.state_dict()

    def changed(name: str) -> bool:
        return any(
            name == declared or name.endswith(f".{declared}")
            for declared in spec.changed_parameters
        )

    transferred = {
        name: native_state[name]
        for name, value in arm_state.items()
        if name in native_state
        and native_state[name].shape == value.shape
        and native_state[name].dtype == value.dtype
        and not changed(name)
    }
    merged = dict(arm_state)
    merged.update(transferred)
    model.load_state_dict(merged, strict=True)


def execute_job(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object],
    dependencies: TinyExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute one fully bound Tiny job with real deterministic episodes."""
    from .runner import ForcedOOM, NonFiniteGradient, NonFiniteLoss

    runtime_value = _runtime_values(runtime)
    config, spec = _validated_job(job, runtime_value)
    if runtime_value["dtype"] not in config.dtype_preferences:
        raise TinyRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "runtime.dtype is not declared in config.dtype_preferences",
        )
    dependencies_value = _resolve_dependencies(dependencies)
    started = dependencies_value.monotonic()
    train_episode, batch_size = _generate_exact_train_episode(
        config,
        seed=job["seed"],
        generator=dependencies_value.generate_task,
    )
    evaluation_episodes = _generate_evaluation_episodes(
        config,
        seed=job["seed"],
        batch_size=batch_size,
        generator=dependencies_value.generate_task,
    )
    all_episodes = (train_episode, *evaluation_episodes)
    data_identity = _episode_digest(all_episodes)
    model_config = _tiny_model_config(
        config,
        spec,
        all_episodes,
        dtype=runtime_value["dtype"],
        selector_seed=job["seed"],
    )
    try:
        model = dependencies_value.build_model(model_config, init_seed=job["seed"])
        if not isinstance(model, TinyKMD2Model):
            raise TypeError("Tiny model dependency returned an incompatible object")
        _warm_start_from_native_(model, spec, seed=job["seed"])
        trainer = dependencies_value.build_trainer(
            model,
            _training_configuration(config, job["job_id"], job["seed"]),
        )
    except (MemoryError, torch.OutOfMemoryError) as error:
        raise ForcedOOM(
            str(error) or "Tiny model allocation failed",
            phase="model_initialization",
            context={
                "batch_size": batch_size,
                "sequence_length": train_episode.valid.shape[1],
                "num_heads": config.model.num_heads,
                "state_key_dim": config.model.state_key_dim,
                "state_value_dim": config.model.state_value_dim,
                "cache_width": config.cache.width if model_config.cache else 0,
                "block_size": config.cache.block_size,
                "dtype": runtime_value["dtype"],
                "device": "cpu",
                "estimated_bytes": 0,
                "peak_vram_bytes": 0,
            },
        ) from error
    if not isinstance(trainer, TinyTrainer):
        raise TypeError("Tiny trainer dependency returned an incompatible object")
    amplitude_initial = _amplitudes(model)
    checkpoint = (
        runtime_value["output"] / "checkpoints" / job["job_id"] / "latest.pt"
    )
    if runtime_value["resume"] and checkpoint.is_file():
        try:
            trainer.load_checkpoint(checkpoint)
        except (TypeError, ValueError, OSError) as error:
            raise TinyRuntimeConfigurationError(
                "resume_checkpoint_invalid", str(error)
            ) from error
    try:
        while trainer.step < config.budget.updates:
            trainer.train_step(train_episode)
            trainer.save_checkpoint(checkpoint)
    except FloatingPointError as error:
        failure = NonFiniteGradient if "gradient" in str(error) else NonFiniteLoss
        raise failure(str(error), phase="training") from error
    except (MemoryError, torch.OutOfMemoryError) as error:
        raise ForcedOOM(
            str(error) or "Tiny training allocation failed",
            phase="training",
            context={
                "batch_size": batch_size,
                "sequence_length": train_episode.valid.shape[1],
                "num_heads": config.model.num_heads,
                "state_key_dim": config.model.state_key_dim,
                "state_value_dim": config.model.state_value_dim,
                "cache_width": config.cache.width if model_config.cache else 0,
                "block_size": config.cache.block_size,
                "dtype": runtime_value["dtype"],
                "device": "cpu",
                "estimated_bytes": 0,
                "peak_vram_bytes": 0,
            },
        ) from error
    if trainer.step != config.budget.updates or trainer.tokens_seen != config.budget.tokens:
        raise TinyRuntimeConfigurationError(
            "budget_mismatch", "Tiny trainer did not consume the exact configured budgets"
        )
    try:
        metrics, evaluation_losses, evaluated, evaluation_tokens = _evaluate_episodes(
            trainer, evaluation_episodes
        )
    except FloatingPointError as error:
        raise NonFiniteLoss(str(error), phase="evaluation") from error
    finished = dependencies_value.monotonic()
    wall_time = float(finished) - float(started)
    if not math.isfinite(wall_time) or wall_time < 0:
        raise TinyRuntimeConfigurationError(
            "clock_invalid", "monotonic execution duration is invalid"
        )
    peak_vram = dependencies_value.peak_vram_bytes()
    if type(peak_vram) is not int or peak_vram < 0:
        raise TinyRuntimeConfigurationError(
            "runtime_measurement_invalid", "peak_vram_bytes must be nonnegative int"
        )
    duration = max(wall_time, 1.0e-12)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    state_bytes = max(
        cell.state_bytes
        for _, output in evaluated
        for cell in output.cell_outputs
    )
    payload: dict[str, object] = {
        "metrics": metrics,
        "loss_curves": {
            "train": [float(record["loss"]) for record in trainer.metric_history],
            "evaluation": evaluation_losses,
        },
        "counts": {
            "nonfinite_loss": 0,
            "nonfinite_gradient": 0,
            "skipped_steps": 0,
        },
        "parameters": {"trainable": trainable, "total": total},
        "recurrent_state": {"elements": state_bytes // 4, "bytes": state_bytes},
        "performance": {
            "wall_time_seconds": wall_time,
            "examples_per_second": (
                config.budget.updates * batch_size
                + sum(len(episode.example_ids) for episode in evaluation_episodes)
            ) / duration,
            "tokens_per_second": (trainer.tokens_seen + evaluation_tokens) / duration,
            "peak_vram_bytes": peak_vram,
        },
        "identities": {
            "checkpoint": _checkpoint_identity(checkpoint),
            "data": data_identity,
        },
        "training": {
            "updates_completed": trainer.step,
            "tokens_seen": trainer.tokens_seen,
            "examples_seen": trainer.step * batch_size,
        },
        "evaluations": [
            {
                "split": episode.split,
                "seed": episode.seed,
                "examples": len(episode.example_ids),
                "tokens": int(episode.valid.sum().item()),
                "loss": loss,
            }
            for episode, loss in zip(evaluation_episodes, evaluation_losses, strict=True)
        ],
    }
    if spec.arm_id.startswith("exact_cache."):
        initial = amplitude_initial if amplitude_initial else [0.0]
        final = _amplitudes(model) if model_config.cache is not None else [0.0]
        payload["exact_cache"] = _exact_cache_payload(
            config,
            model_config,
            evaluated,
            amplitude_initial=initial,
            amplitude_final=final,
            cache_active=model_config.cache is not None,
        )
    return payload


def run_job(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object] | None = None,
    dependencies: TinyExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Runner-discoverable entry point; runtime state must be explicitly bound."""

    if runtime is None:
        raise TinyRuntimeConfigurationError(
            "runtime_required",
            "Tiny run_job requires build_job_dispatcher(runtime, dependencies)",
        )
    return execute_job(job, runtime=runtime, dependencies=dependencies)


def build_job_dispatcher(
    runtime: Mapping[str, object],
    dependencies: TinyExecutionDependencies | Mapping[str, object] | None = None,
) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    """Bind non-semantic runtime state into the runner one-argument protocol."""

    frozen_runtime = _runtime_values(runtime)
    resolved_dependencies = _resolve_dependencies(dependencies)

    def dispatch(job: Mapping[str, object]) -> Mapping[str, object]:
        return run_job(
            job,
            runtime=frozen_runtime,
            dependencies=resolved_dependencies,
        )

    dispatch.__name__ = "run_bound_tiny_job"
    return dispatch


__all__ = [
    "TINY_CHECKPOINT_SCHEMA_VERSION",
    "TinyExecutionDependencies",
    "TinyRuntimeConfigurationError",
    "TinyTrainer",
    "TinyTrainingConfig",
    "build_job_dispatcher",
    "execute_job",
    "run_job",
]
