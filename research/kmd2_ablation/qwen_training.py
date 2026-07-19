"""Small, deterministic Qwen heal losses and one-update training adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .qwen_fused_loss import fused_heal_ce_kl


_OPTIMIZER_STATE_OFFLOAD_THRESHOLD_BYTES = 1 << 30


class QwenTrainingError(RuntimeError):
    """Typed failure that must invalidate, rather than alter, a paired run."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class TeacherRequiredError(QwenTrainingError):
    """Ordinary Qwen heal was requested without a frozen teacher."""


class QwenRuntimeConfigurationError(QwenTrainingError):
    """Runtime-only execution bindings are absent, stale, or incompatible."""


def _finite_real(name: str, value: object, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _tensors_are_finite(values: Sequence[torch.Tensor]) -> bool:
    """Check many tensors with one host synchronization per device."""
    grouped: dict[torch.device, list[torch.Tensor]] = {}
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError("finite checks require tensors")
        grouped.setdefault(value.device, []).append(value.detach())
    return all(bool(torch.stack([
        torch.isfinite(value).all() for value in tensors
    ]).all()) for tensors in grouped.values())


@dataclass(frozen=True)
class QwenHealTrainingConfig:
    """Fixed objective and stopping contract shared by all paired arms."""

    objective: str
    ce_weight: float
    kl_weight: float
    layerwise_weight: float
    temperature: float
    accumulation_steps: int
    max_updates: int
    max_tokens: int
    gradient_checkpointing: bool
    lambda_spec: float = 0.0
    lambda_gate: float = 0.0
    specialization_updates: int = 0

    def __post_init__(self) -> None:
        if type(self.objective) is not str or not self.objective:
            raise ValueError("objective must be a nonempty string")
        for name in ("ce_weight", "kl_weight", "layerwise_weight"):
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))
        if self.ce_weight + self.kl_weight + self.layerwise_weight <= 0.0:
            raise ValueError("at least one heal loss weight must be positive")
        temperature = _finite_real("temperature", self.temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        object.__setattr__(self, "temperature", temperature)
        for name in ("accumulation_steps", "max_updates", "max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be boolean")
        for name in ("lambda_spec", "lambda_gate"):
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))
        if type(self.specialization_updates) is not int or self.specialization_updates < 0:
            raise ValueError("specialization_updates must be a nonnegative integer")
        if self.objective == "synthetic_only" and (
            self.kl_weight != 0.0 or self.layerwise_weight != 0.0
        ):
            raise ValueError(
                "synthetic_only without a teacher requires zero KL and layerwise weights"
            )


def validate_teacher_requirement(
    config: QwenHealTrainingConfig,
    *,
    teacher_present: bool,
    phase: str,
) -> None:
    """Apply the same teacher guard in preflight and at runtime."""
    if not isinstance(config, QwenHealTrainingConfig):
        raise TypeError("config must be a QwenHealTrainingConfig")
    if type(teacher_present) is not bool:
        raise TypeError("teacher_present must be boolean")
    if type(phase) is not str or not phase:
        raise ValueError("phase must be a nonempty string")
    if not teacher_present and config.objective != "synthetic_only":
        raise TeacherRequiredError(
            "teacher_required",
            f"{phase}: objective {config.objective!r} requires a teacher model",
        )


def _validate_logits(name: str, logits: object) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[2] < 2:
        raise ValueError(f"{name} must have shape [batch, time>=2, vocab>=2]")
    return logits


def _validate_labels(labels: object, logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
        raise TypeError("labels must be a torch.long tensor")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits batch/time dimensions")
    valid = labels[:, 1:] != -100
    if not bool(valid.any()):
        raise ValueError("labels must contain at least one valid causal target")
    return labels


def causal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Teacher-forced next-token CE using the standard one-token shift."""
    logits = _validate_logits("logits", logits)
    labels = _validate_labels(labels, logits)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Canonical full-logit ``KL(teacher || student)`` distillation loss."""
    student = _validate_logits("student_logits", student_logits)
    teacher = _validate_logits("teacher_logits", teacher_logits)
    if teacher.shape != student.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    scale = _finite_real("temperature", temperature)
    if scale <= 0.0:
        raise ValueError("temperature must be positive")
    student_log = F.log_softmax(student.float() / scale, dim=-1)
    teacher_log = F.log_softmax(
        teacher.detach().to(device=student.device, dtype=torch.float32) / scale,
        dim=-1,
    )
    return F.kl_div(
        student_log,
        teacher_log,
        reduction="batchmean",
        log_target=True,
    ) * (scale * scale) / student.shape[1]


_LAYERWISE_CHUNK_ELEMENTS = 1 << 20


def _chunked_layerwise_statistics(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 residual/teacher means with bounded temporary storage.

    This helper is only dispatched for contiguous CUDA BF16 tensors.  The
    teacher is copied between GPUs in BF16 before conversion, rather than
    materializing a full cross-device FP32 copy.  FP32 reductions are combined
    through FP64 *scalar* accumulation so chunking does not accumulate a
    sequence-length-dependent rounding error.
    """
    student_flat = student.reshape(-1)
    teacher_flat = teacher.detach().reshape(-1)
    residual_sums: list[torch.Tensor] = []
    teacher_sums: list[torch.Tensor] = []
    for start in range(0, student_flat.numel(), _LAYERWISE_CHUNK_ELEMENTS):
        stop = min(start + _LAYERWISE_CHUNK_ELEMENTS, student_flat.numel())
        student_float = student_flat[start:stop].float()
        teacher_local = teacher_flat[start:stop].to(
            device=student.device,
            dtype=torch.bfloat16,
            non_blocking=True,
        )
        teacher_float = teacher_local.float()
        residual_sums.append((student_float - teacher_float).square().sum())
        teacher_sums.append(teacher_float.square().sum())
    residual_mean = (
        torch.stack(residual_sums).to(torch.float64).sum()
        / student_flat.numel()
    ).float()
    teacher_mean = (
        torch.stack(teacher_sums).to(torch.float64).sum()
        / student_flat.numel()
    ).float()
    return residual_mean, teacher_mean


class _MemoryBoundedLayerwiseAlignment(torch.autograd.Function):
    """Rematerialize BF16 residuals instead of saving full FP32 differences."""

    @staticmethod
    def forward(
        ctx: object,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        residual_mean, teacher_mean = _chunked_layerwise_statistics(
            student, teacher
        )
        denominator = teacher_mean.clamp_min(1.0e-8)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(student, teacher, denominator)
        return residual_mean / denominator

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None]:
        if grad_output is None:
            return None, None
        student, teacher, denominator = ctx.saved_tensors
        student_flat = student.reshape(-1)
        teacher_flat = teacher.detach().reshape(-1)
        gradient = torch.empty_like(student_flat)
        scale = grad_output.float().div(denominator).div(student_flat.numel())
        for start in range(0, student_flat.numel(), _LAYERWISE_CHUNK_ELEMENTS):
            stop = min(start + _LAYERWISE_CHUNK_ELEMENTS, student_flat.numel())
            teacher_local = teacher_flat[start:stop].to(
                device=student.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            residual = student_flat[start:stop].float() - teacher_local.float()
            gradient[start:stop] = residual.mul(2.0).mul(scale)
        return gradient.reshape_as(student), None


def _can_use_memory_bounded_layerwise(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> bool:
    return (
        torch.is_grad_enabled()
        and student.requires_grad
        and student.is_cuda
        and teacher.is_cuda
        and student.dtype == torch.bfloat16
        and teacher.dtype == torch.bfloat16
        and student.is_contiguous()
        and teacher.is_contiguous()
    )


def layerwise_alignment_loss(
    student_hidden: Sequence[torch.Tensor],
    teacher_hidden: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Canonical normalized residual MSE over ``hidden_states[1:]``."""
    if isinstance(student_hidden, torch.Tensor) or not isinstance(
        student_hidden, Sequence
    ):
        raise TypeError("student_hidden must be a sequence of tensors")
    if isinstance(teacher_hidden, torch.Tensor) or not isinstance(
        teacher_hidden, Sequence
    ):
        raise TypeError("teacher_hidden must be a sequence of tensors")
    if len(student_hidden) < 2 or len(student_hidden) != len(teacher_hidden):
        raise ValueError("student and teacher hidden-state sequences must align")
    losses: list[torch.Tensor] = []
    for index, (student, teacher) in enumerate(
        zip(student_hidden[1:], teacher_hidden[1:]), start=1
    ):
        if (
            not isinstance(student, torch.Tensor)
            or not isinstance(teacher, torch.Tensor)
            or not student.is_floating_point()
            or not teacher.is_floating_point()
        ):
            raise TypeError(f"hidden layer {index} must contain floating tensors")
        if student.shape != teacher.shape or student.ndim < 2:
            raise ValueError(f"hidden layer {index} shape does not align")
        if _can_use_memory_bounded_layerwise(student, teacher):
            losses.append(_MemoryBoundedLayerwiseAlignment.apply(student, teacher))
            continue
        teacher_float = teacher.detach().to(
            device=student.device, dtype=torch.float32
        )
        difference = student.float() - teacher_float
        losses.append(
            difference.square().mean()
            / teacher_float.square().mean().clamp_min(1.0e-8)
        )
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class HealLossBreakdown:
    total: torch.Tensor
    ce: torch.Tensor
    kl: torch.Tensor
    layerwise: torch.Tensor


def _output_field(output: object, name: str) -> object:
    if isinstance(output, Mapping):
        if name not in output:
            raise ValueError(f"model output is missing {name}")
        return output[name]
    if not hasattr(output, name):
        raise ValueError(f"model output is missing {name}")
    return getattr(output, name)


def compute_heal_loss(
    student_output: object,
    teacher_output: object | None,
    labels: torch.Tensor,
    config: QwenHealTrainingConfig,
) -> HealLossBreakdown:
    """Compose the three preregistered losses without importing a trainer."""
    if not isinstance(config, QwenHealTrainingConfig):
        raise TypeError("config must be a QwenHealTrainingConfig")
    student_logits = _validate_logits(
        "student_logits", _output_field(student_output, "logits")
    )
    zero = student_logits.sum() * 0.0
    if config.kl_weight > 0.0 or config.layerwise_weight > 0.0:
        if teacher_output is None:
            raise TeacherRequiredError(
                "teacher_required", "KL/layerwise Qwen heal losses require a teacher"
            )

    if config.ce_weight > 0.0 and config.kl_weight > 0.0:
        assert teacher_output is not None
        labels = _validate_labels(labels, student_logits)
        teacher_logits = _validate_logits(
            "teacher_logits", _output_field(teacher_output, "logits")
        )
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("teacher and student logits must have identical shapes")
        teacher_logits = teacher_logits.detach()
        ce, kl = fused_heal_ce_kl(
            student_logits,
            teacher_logits,
            labels,
            temperature=config.temperature,
        )
    else:
        ce = (
            causal_cross_entropy(student_logits, labels)
            if config.ce_weight > 0.0
            else zero
        )
        if config.kl_weight > 0.0:
            assert teacher_output is not None
            kl = distillation_kl(
                student_logits,
                _output_field(teacher_output, "logits"),
                temperature=config.temperature,
            )
        else:
            kl = zero
    if config.layerwise_weight > 0.0:
        assert teacher_output is not None
        layerwise = layerwise_alignment_loss(
            _output_field(student_output, "hidden_states"),
            _output_field(teacher_output, "hidden_states"),
        )
    else:
        layerwise = zero
    total = (
        config.ce_weight * ce
        + config.kl_weight * kl
        + config.layerwise_weight * layerwise
    )
    return HealLossBreakdown(total=total, ce=ce, kl=kl, layerwise=layerwise)


def _validate_parameter_names(
    model: torch.nn.Module,
    memory_parameter_names: tuple[str, ...],
    cache_parameter_names: tuple[str, ...],
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], tuple[tuple[str, torch.nn.Parameter], ...]]:
    if type(memory_parameter_names) is not tuple or not memory_parameter_names:
        raise ValueError("memory_parameter_names must be a nonempty tuple")
    if type(cache_parameter_names) is not tuple:
        raise TypeError("cache_parameter_names must be a tuple")
    combined = memory_parameter_names + cache_parameter_names
    if any(type(name) is not str or not name for name in combined):
        raise ValueError("optimizer parameter names must be nonempty strings")
    if len(set(combined)) != len(combined):
        raise ValueError("optimizer parameter groups overlap or contain duplicates")
    named = dict(model.named_parameters())
    missing = sorted(set(combined) - set(named))
    if missing:
        raise KeyError("optimizer parameter names are missing: " + ", ".join(missing))
    actual_trainable = {name for name, parameter in named.items() if parameter.requires_grad}
    if actual_trainable != set(combined):
        raise ValueError(
            "optimizer groups must cover exactly the declared trainable parameters"
        )
    memory = tuple((name, named[name]) for name in memory_parameter_names)
    cache = tuple((name, named[name]) for name in cache_parameter_names)
    return memory, cache


def build_qwen_heal_optimizer(
    model: torch.nn.Module,
    *,
    memory_parameter_names: tuple[str, ...],
    cache_parameter_names: tuple[str, ...],
    learning_rate: float,
    lr_cache: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build stable memory/cache AdamW groups with no cache weight decay."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    memory, cache = _validate_parameter_names(
        model, memory_parameter_names, cache_parameter_names
    )
    memory_lr = _finite_real("learning_rate", learning_rate)
    cache_lr = _finite_real("lr_cache", lr_cache)
    epsilon = _finite_real("eps", eps)
    decay = _finite_real("weight_decay", weight_decay)
    if memory_lr <= 0.0 or cache_lr <= 0.0 or epsilon <= 0.0:
        raise ValueError("optimizer learning rates and eps must be positive")
    if (
        type(betas) is not tuple
        or len(betas) != 2
        or any(type(beta) not in (int, float) for beta in betas)
        or any(not math.isfinite(float(beta)) or not 0.0 <= float(beta) < 1.0 for beta in betas)
    ):
        raise ValueError("betas must be two finite values in [0,1)")
    groups: list[dict[str, Any]] = [
        {
            "name": "memory",
            "parameter_names": tuple(name for name, _ in memory),
            "params": [parameter for _, parameter in memory],
            "lr": memory_lr,
            "weight_decay": decay,
        }
    ]
    if cache:
        groups.append(
            {
                "name": "cache",
                "parameter_names": tuple(name for name, _ in cache),
                "params": [parameter for _, parameter in cache],
                "lr": cache_lr,
                "weight_decay": 0.0,
            }
        )
    optimizer_options: dict[str, object] = {}
    bound_parameters = tuple(parameter for _name, parameter in (*memory, *cache))
    if bound_parameters and all(parameter.device.type == "cuda" for parameter in bound_parameters):
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        groups,
        betas=(float(betas[0]), float(betas[1])),
        eps=epsilon,
        **optimizer_options,
    )
    # Package B has roughly 4.2 GiB of BF16 Adam moments.  They are not used
    # during forward/backward, where keeping them resident would overlap with
    # recurrent scratch and the full-vocabulary loss.  Mark only genuinely
    # large CUDA optimizers for phase-local CPU storage; small/test jobs keep
    # ordinary AdamW behavior.
    estimated_moment_bytes = 2 * sum(
        parameter.numel() * parameter.element_size()
        for parameter in bound_parameters
    )
    optimizer._gdnx_cpu_state_offload = bool(  # type: ignore[attr-defined]
        optimizer_options.get("fused") is True
        and estimated_moment_bytes >= _OPTIMIZER_STATE_OFFLOAD_THRESHOLD_BYTES
    )
    optimizer._gdnx_estimated_moment_bytes = estimated_moment_bytes  # type: ignore[attr-defined]
    return optimizer


def _move_optimizer_state_(
    optimizer: torch.optim.Optimizer, *, to_parameter_devices: bool
) -> None:
    """Move Adam state between phase-local GPU use and CPU residency."""
    if not bool(getattr(optimizer, "_gdnx_cpu_state_offload", False)):
        return
    for parameter, state in optimizer.state.items():
        destination = parameter.device if to_parameter_devices else torch.device("cpu")
        for name, value in tuple(state.items()):
            if not isinstance(value, torch.Tensor) or value.device == destination:
                continue
            state[name] = value.detach().to(
                device=destination,
                non_blocking=(to_parameter_devices and value.device.type == "cpu"),
            )


def _optimizer_state_is_offloaded(optimizer: torch.optim.Optimizer) -> bool:
    """Return whether every materialized tensor slot is currently on CPU."""
    tensors = tuple(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return bool(getattr(optimizer, "_gdnx_cpu_state_offload", False)) and all(
        tensor.device.type == "cpu" for tensor in tensors
    )


def package_b_auxiliary_loss(
    model: torch.nn.Module, *, lambda_spec: float, lambda_gate: float,
    successful_updates: int = 0, specialization_updates: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Centered rank specialization plus Package-B trapezoid-gate warmup."""
    spec_scale = _finite_real("lambda_spec", lambda_spec)
    gate_scale = _finite_real("lambda_gate", lambda_gate)
    if type(successful_updates) is not int or successful_updates < 0:
        raise ValueError("successful_updates must be a nonnegative integer")
    if specialization_updates is not None and (
        type(specialization_updates) is not int or specialization_updates < 0
    ):
        raise ValueError("specialization_updates must be a nonnegative integer or None")
    components = tuple(
        module for module in model.modules()
        if getattr(module, "package", None) == "four_state"
        and hasattr(module, "specialization_probe")
    )
    if not components:
        raise ValueError("Package-B auxiliary loss requires four-state HybridComponents")
    identities = tuple({
        "probe_sha256": module.specialization_probe_sha256,
        "probe_hashes": {
            "key": module.specialization_probe_sha256,
            "value": module.specialization_value_probe_sha256,
        },
        "coefficients": [value / (20.0 ** .5) for value in (-3.0, -1.0, 1.0, 3.0)],
    } for module in components)
    identity: dict[str, object] = dict(identities[0])
    identity.update({"lambda_spec": spec_scale, "lambda_gate": gate_scale,
                     "specialization_updates": specialization_updates})
    if any(item != identities[0] for item in identities[1:]):
        raise ValueError("Package-B specialization identity differs across layers")
    active = specialization_updates is None or successful_updates < specialization_updates
    if not active:
        reference = components[0].q_weight
        return torch.zeros((), device=reference.device, dtype=reference.dtype), identity
    module_losses = []
    gates = []
    for module in components:
        q = module.q_weight
        coefficients = module.specialization_coefficients.to(device=q.device, dtype=q.dtype)
        projection_losses = []
        for name in ("q_weight", "k_weight", "v_weight", "erase_weight", "write_weight", "z_weight"):
            weight = getattr(module, name)
            centered = weight - weight.mean(dim=0, keepdim=True)
            base_probe = (module.specialization_probe if weight.shape[1] == q.shape[1]
                          else module.specialization_value_probe)
            probe = base_probe.to(device=weight.device, dtype=weight.dtype)
            projection_losses.append((coefficients[:, None, None] * centered * probe[None]).mean())
        module_losses.append(torch.stack(projection_losses).sum())
        # The removed state-router gate is not part of decay-only braiding.
        # lambda is the CURRENT-endpoint coefficient, so the actual trapezoid
        # strength is (1-lambda).  Reward that previous-endpoint contribution.
        # A zero logit means lambda=.5 (equal endpoints), not Euler identity.
        gates.append((1.0 - module.trapezoid_proj.bias.sigmoid()).mean())
    total = spec_scale * torch.stack(module_losses).mean() - gate_scale * torch.stack(gates).mean()
    return total, identity


def project_cache_amplitudes_(model: torch.nn.Module) -> tuple[str, ...]:
    """Project every declared cache amplitude to the closed identity-gate range."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    amplitudes = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name == "cache_amplitude" or name.endswith(".cache_amplitude")
    )
    with torch.no_grad():
        for name, parameter in amplitudes:
            if not bool(torch.isfinite(parameter).all()):
                raise QwenTrainingError(
                    "nonfinite_parameter", f"cache amplitude {name} is nonfinite"
                )
            parameter.clamp_(0.0, 1.0)
    return tuple(name for name, _ in amplitudes)


_HYBRID_OPTIMIZER_PATHS = frozenset(
    {"ordinary", "amp", "skipped", "resumed", "sharded"}
)


def project_hybrid_constraints_(model: torch.nn.Module) -> tuple[str, ...]:
    """Project all bounded Package-A/B coefficients exactly once per module."""
    projected: list[str] = []
    with torch.no_grad():
        for prefix, module in model.named_modules():
            projector = getattr(module, "project_coefficients_", None)
            if callable(projector):
                projector()
                projected.append(prefix or "<root>")
        for name, parameter in model.named_parameters():
            suffix = name.rsplit(".", 1)[-1]
            if suffix in {"trapezoid_gate", "lookahead_gate", "d_raw"}:
                if not bool(torch.isfinite(parameter).all()):
                    raise QwenTrainingError("nonfinite_parameter", f"hybrid gate {name} is nonfinite")
                parameter.clamp_(0.0, 1.0)
                projected.append(name)
    return tuple(dict.fromkeys(projected))


def run_qwen_arm(
    *, model: torch.nn.Module, optimizer_path: str,
    update: Callable[[], object], execution: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run the common projected optimizer boundary for every hybrid path."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if optimizer_path not in _HYBRID_OPTIMIZER_PATHS or not callable(update):
        raise QwenTrainingError("unsupported_execution", "unsupported_execution optimizer path")
    settings = dict(execution or {})
    tp = settings.get("tensor_parallel", 1)
    pp = settings.get("pipeline_parallel", 1)
    dp = settings.get("data_parallel", 1)
    checkpointing = settings.get("activation_checkpointing", False)
    if any(type(value) is not int or value < 1 for value in (tp, pp, dp)):
        raise QwenTrainingError("unsupported_execution", "unsupported_execution parallel setting")
    if tp != 1 or pp != 1:
        raise QwenTrainingError("unsupported_execution", "unsupported_execution TP/PP")
    if dp != 1:
        raise QwenTrainingError(
            "unsupported_execution", "unsupported_execution data parallel is not implemented"
        )
    if type(checkpointing) is not bool:
        raise QwenTrainingError("unsupported_execution", "unsupported_execution checkpointing")
    if settings.get("packed") is True and not settings.get("document_boundaries"):
        raise QwenTrainingError("unsupported_execution", "unsupported_execution boundaryless packing")
    if checkpointing:
        checkpoint_hook = settings.get("activation_checkpointing_hook")
        if not callable(checkpoint_hook):
            raise QwenTrainingError("unsupported_execution", "unsupported_execution checkpointing hook missing")
        checkpoint_hook(model)
    completed = bool(update())
    projected = project_hybrid_constraints_(model)
    return {"optimizer_path": optimizer_path, "completed": completed,
            "projected": projected, "execution": settings}


@dataclass(frozen=True)
class HealStepLog:
    job_id: str
    pairing_id: str
    arm: str
    update: int
    tokens_seen: int
    example_ids: tuple[str, ...]
    microbatches: int
    losses: Mapping[str, float]
    learning_rates: Mapping[str, float]
    skipped_steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "pairing_id": self.pairing_id,
            "arm": self.arm,
            "update": self.update,
            "tokens_seen": self.tokens_seen,
            "example_ids": list(self.example_ids),
            "microbatches": self.microbatches,
            "losses": dict(sorted(self.losses.items())),
            "learning_rates": dict(sorted(self.learning_rates.items())),
            "skipped_steps": self.skipped_steps,
        }


def _emit_live_metrics(path: Path, log: HealStepLog, max_updates: int) -> None:
    """Best-effort JSONL append for the live-metrics watcher.

    Opt-in via GDNX_LIVE_METRICS=1; every failure is swallowed so the emitter
    can never alter or abort a training run.
    """
    try:
        record = log.as_dict()
        record["max_updates"] = max_updates
        record["wall_time"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:
        pass


class QwenHealTrainer:
    """One deterministic paired-heal update path, independent of the main trainer."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        teacher: torch.nn.Module | None,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        config: QwenHealTrainingConfig,
        job_id: str,
        pairing_id: str,
        arm: str,
        expected_example_windows: tuple[tuple[str, ...], ...],
        teacher_device: str | torch.device | None = None,
        distributed: bool = False,
        grad_scaler: object | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if teacher is not None and not isinstance(teacher, torch.nn.Module):
            raise TypeError("teacher must be a torch.nn.Module or None")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if getattr(scheduler, "optimizer", None) is not optimizer:
            raise ValueError("scheduler must be bound to the supplied optimizer")
        if not isinstance(config, QwenHealTrainingConfig):
            raise TypeError("config must be a QwenHealTrainingConfig")
        validate_teacher_requirement(
            config, teacher_present=teacher is not None, phase="runtime"
        )
        if type(job_id) is not str or not job_id:
            raise ValueError("job_id must be a nonempty string")
        if (
            type(pairing_id) is not str
            or len(pairing_id) != 64
            or any(character not in "0123456789abcdef" for character in pairing_id)
        ):
            raise ValueError("pairing_id must be lowercase SHA-256")
        if arm not in {"native", "recency", "surprise"}:
            raise ValueError("arm must be native, recency, or surprise")
        expected_count = config.max_updates * config.accumulation_steps
        if (
            type(expected_example_windows) is not tuple
            or len(expected_example_windows) != expected_count
        ):
            raise ValueError(
                "expected_example_windows must cover every configured microbatch"
            )
        for window in expected_example_windows:
            if (
                type(window) is not tuple
                or not window
                or any(type(item) is not str or not item for item in window)
            ):
                raise ValueError("each expected example window must be nonempty IDs")
        if config.gradient_checkpointing:
            underlying = getattr(model, "module", model)
            enable = getattr(underlying, "gradient_checkpointing_enable", None)
            if not callable(enable):
                raise TypeError(
                    "gradient checkpointing was requested but the model cannot enable it"
                )
            if not getattr(underlying, "_kmd2_checkpointing_enabled", False):
                enable()
                underlying._kmd2_checkpointing_enabled = True
        if teacher is not None:
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        try:
            resolved_teacher_device = (
                None if teacher_device is None else torch.device(teacher_device)
            )
        except (TypeError, RuntimeError) as error:
            raise ValueError("teacher_device must name a valid torch device") from error
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        _move_optimizer_state_(self.optimizer, to_parameter_devices=False)
        self.scheduler = scheduler
        self.config = config
        self.job_id = job_id
        self.pairing_id = pairing_id
        self.arm = arm
        self.expected_example_windows = expected_example_windows
        self.teacher_device = resolved_teacher_device
        self.step = 0
        self.tokens_seen = 0
        self.example_cursor = 0
        self.skipped_steps = 0
        self.distributed = distributed
        if grad_scaler is not None and any(
            not callable(getattr(grad_scaler, name, None))
            for name in ("scale", "unscale_", "step", "update", "get_scale", "state_dict", "load_state_dict")
        ):
            raise TypeError("grad_scaler must implement the GradScaler state/step interface")
        self.grad_scaler = grad_scaler
        self.last_rank_update_norms: tuple[float, ...] = ()

    def _prevalidate_batches(
        self, microbatches: Sequence[Mapping[str, object]]
    ) -> tuple[int, tuple[str, ...]]:
        if isinstance(microbatches, (str, bytes)) or not isinstance(
            microbatches, Sequence
        ):
            raise TypeError("microbatches must be a sequence of mappings")
        if len(microbatches) != self.config.accumulation_steps:
            raise QwenTrainingError(
                "accumulation_mismatch",
                f"expected {self.config.accumulation_steps} microbatches",
            )
        total_tokens = 0
        flattened_ids: list[str] = []
        for offset, batch in enumerate(microbatches):
            if not isinstance(batch, Mapping):
                raise TypeError("each microbatch must be a mapping")
            missing = {"input_ids", "labels", "example_ids"} - set(batch)
            if missing:
                raise ValueError("microbatch is missing: " + ", ".join(sorted(missing)))
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            ids = batch["example_ids"]
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dtype != torch.long
                or input_ids.ndim != 2
            ):
                raise TypeError("input_ids must be a rank-2 torch.long tensor")
            if (
                not isinstance(labels, torch.Tensor)
                or labels.dtype != torch.long
                or labels.shape != input_ids.shape
            ):
                raise TypeError("labels must be torch.long and match input_ids")
            if type(ids) is not tuple or len(ids) != input_ids.shape[0]:
                raise ValueError("example_ids must match the microbatch size")
            expected = self.expected_example_windows[self.example_cursor + offset]
            if ids != expected:
                raise QwenTrainingError(
                    "example_window_mismatch",
                    f"expected {expected!r}, received {ids!r}",
                )
            total_tokens += input_ids.numel()
            flattened_ids.extend(ids)
        return total_tokens, tuple(flattened_ids)

    @staticmethod
    def _model_inputs(batch: Mapping[str, object]) -> dict[str, object]:
        inputs = {
            name: value
            for name, value in batch.items()
            if name not in {"labels", "example_ids"}
        }
        for reserved in ("output_hidden_states", "use_cache"):
            if reserved in inputs:
                raise ValueError(f"microbatch cannot override {reserved}")
        inputs["output_hidden_states"] = True
        inputs["use_cache"] = False
        return inputs

    def _fail_nonfinite(self, code: str, message: str) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.skipped_steps += 1
        raise QwenTrainingError(code, message)

    def _distributed_any(self, value: bool) -> bool:
        if not self.distributed:
            return value
        flag = torch.tensor(int(value), device=next(self.model.parameters()).device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    def _distributed_sum_int(self, value: int) -> int:
        if not self.distributed:
            return value
        total = torch.tensor(value, dtype=torch.int64, device=next(self.model.parameters()).device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return int(total.item())

    def _synchronize_scaler_state(self) -> None:
        if not self.distributed or self.grad_scaler is None:
            return
        objects = [copy.deepcopy(self.grad_scaler.state_dict()) if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(objects, src=0)
        self.grad_scaler.load_state_dict(objects[0])

    def train_update(
        self, microbatches: Sequence[Mapping[str, object]]
    ) -> HealStepLog:
        if self.step >= self.config.max_updates:
            raise QwenTrainingError(
                "update_budget_exhausted", "configured update budget is exhausted"
            )
        token_count, example_ids = self._prevalidate_batches(microbatches)
        token_count = self._distributed_sum_int(token_count)
        if self.tokens_seen + token_count > self.config.max_tokens:
            raise QwenTrainingError(
                "token_budget_exhausted", "update would exceed the fixed token budget"
            )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {"total": 0.0, "ce": 0.0, "kl": 0.0, "layerwise": 0.0}
        if self.config.specialization_updates > 0:
            totals["specialization"] = 0.0
        loss_names = tuple(totals)
        logged_loss_vectors: list[torch.Tensor] = []
        divisor = float(self.config.accumulation_steps)
        for batch in microbatches:
            labels = batch["labels"]
            assert isinstance(labels, torch.Tensor)
            model_inputs = self._model_inputs(batch)
            from .qwen_exact_cache import guarded_model_forward

            if self.teacher is None:
                student_output = guarded_model_forward(self.model, **model_inputs)
                teacher_output = None
            else:
                teacher_inputs = {
                    name: (
                        value.to(self.teacher_device)
                        if self.teacher_device is not None
                        and isinstance(value, torch.Tensor)
                        else value
                    )
                    for name, value in model_inputs.items()
                }
                student_device = next(self.model.parameters()).device
                overlap_devices = (
                    self.teacher_device is not None
                    and student_device.type == "cuda"
                    and self.teacher_device.type == "cuda"
                    and student_device != self.teacher_device
                )
                if overlap_devices:
                    # CUDA calls are asynchronous across devices.  Queue the
                    # frozen teacher first so it executes on the second GPU
                    # while the Python-heavy recurrent student forward is
                    # being dispatched on the first.
                    with torch.no_grad():
                        teacher_output = self.teacher(**teacher_inputs)
                    student_output = guarded_model_forward(self.model, **model_inputs)
                else:
                    student_output = guarded_model_forward(self.model, **model_inputs)
                    with torch.no_grad():
                        teacher_output = self.teacher(**teacher_inputs)
            breakdown = compute_heal_loss(
                student_output, teacher_output, labels, self.config
            )
            auxiliary = breakdown.total.new_zeros(())
            if self.config.specialization_updates > 0:
                try:
                    auxiliary, _ = package_b_auxiliary_loss(
                        self.model, lambda_spec=self.config.lambda_spec,
                        lambda_gate=self.config.lambda_gate, successful_updates=self.step,
                        specialization_updates=self.config.specialization_updates,
                    )
                except ValueError as error:
                    if "requires four-state" not in str(error):
                        raise
            values = {
                "total": breakdown.total + auxiliary,
                "ce": breakdown.ce,
                "kl": breakdown.kl,
                "layerwise": breakdown.layerwise,
            }
            if self.config.specialization_updates > 0:
                values["specialization"] = auxiliary
            loss_vector = torch.stack([values[name].detach() for name in loss_names])
            local_nonfinite = not _tensors_are_finite((loss_vector,))
            if self._distributed_any(local_nonfinite):
                self._fail_nonfinite(
                    "nonfinite_loss", "Qwen heal loss contains a nonfinite value"
                )
            scaled_loss = values["total"] / divisor
            if self.grad_scaler is not None:
                scaled_loss = self.grad_scaler.scale(scaled_loss)
            scaled_loss.backward()
            logged_loss_vectors.append(loss_vector)
            # Backward has consumed the output graphs.  Drop the multi-GiB
            # vocabulary tensors before Adam moments are brought onto the GPU
            # for their short optimizer phase.
            del scaled_loss, values, breakdown, student_output, teacher_output

        # Preserve the prior Python-double accumulation order while collapsing
        # one device-to-host transfer per logged loss and microbatch into one
        # compact transfer per update.
        logged_losses = torch.stack(logged_loss_vectors).cpu().tolist()
        for row in logged_losses:
            for name, value in zip(loss_names, row, strict=True):
                totals[name] += float(value) / divisor

        optimizer_parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
        if self._distributed_any(any(parameter.grad is None for parameter in optimizer_parameters)):
            self._fail_nonfinite(
                "missing_gradient", "a declared trainable parameter has no gradient"
            )
        gradients = tuple(parameter.grad for parameter in optimizer_parameters
                          if parameter.grad is not None)
        if self._distributed_any(not _tensors_are_finite(gradients)):
            self._fail_nonfinite(
                "nonfinite_gradient", "Qwen heal gradients contain a nonfinite value"
            )

        # Keep transactional rollback off-device: duplicating all Package-B
        # parameters and AdamW slots in VRAM makes the real 18-layer update
        # needlessly unrunnable.  CPU snapshots retain exact rollback behavior.
        parameter_snapshot = [
            parameter.detach().to(device="cpu", copy=True).contiguous()
            for parameter in optimizer_parameters
        ]
        parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        rank_snapshot_indices = [
            index for index, parameter in enumerate(optimizer_parameters)
            if ".components." in f".{parameter_names.get(id(parameter), '')}"
            and parameter.ndim > 0 and parameter.shape[0] == 4
        ]
        from .qwen_checkpoint import _portable_cpu_copy

        optimizer_snapshot = _portable_cpu_copy(self.optimizer.state_dict())
        scheduler_snapshot = copy.deepcopy(self.scheduler.state_dict())
        scaler_snapshot = None if self.grad_scaler is None else copy.deepcopy(self.grad_scaler.state_dict())
        try:
            skipped_by_scaler = False
            _move_optimizer_state_(self.optimizer, to_parameter_devices=True)
            try:
                if self.grad_scaler is None:
                    self.optimizer.step()
                else:
                    scale_before = float(self.grad_scaler.get_scale())
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    local_scaler_skip = float(self.grad_scaler.get_scale()) < scale_before
                    skipped_by_scaler = self._distributed_any(local_scaler_skip)
                    if self.distributed and skipped_by_scaler:
                        self.grad_scaler.load_state_dict(scaler_snapshot)
                    self._synchronize_scaler_state()
            finally:
                _move_optimizer_state_(self.optimizer, to_parameter_devices=False)
            if skipped_by_scaler:
                with torch.no_grad():
                    for parameter, snapshot in zip(optimizer_parameters, parameter_snapshot):
                        parameter.copy_(snapshot.to(device=parameter.device))
                self.optimizer.load_state_dict(optimizer_snapshot)
                _move_optimizer_state_(self.optimizer, to_parameter_devices=False)
                self.scheduler.load_state_dict(scheduler_snapshot)
                self.optimizer.zero_grad(set_to_none=True)
                self.skipped_steps += 1
                rates = {
                    str(group.get("name", f"group_{index}")): float(group["lr"])
                    for index, group in enumerate(self.optimizer.param_groups)
                }
                return HealStepLog(
                    job_id=self.job_id, pairing_id=self.pairing_id, arm=self.arm,
                    update=self.step, tokens_seen=self.tokens_seen, example_ids=example_ids,
                    microbatches=self.config.accumulation_steps, losses=totals,
                    learning_rates=rates, skipped_steps=self.skipped_steps,
                )
            project_cache_amplitudes_(self.model)
            from .qwen_variants import project_variant_gates_

            project_variant_gates_(self.model)
            project_hybrid_constraints_(self.model)
            if self.distributed:
                for parameter in optimizer_parameters:
                    lower = parameter.detach().clone()
                    upper = parameter.detach().clone()
                    torch.distributed.all_reduce(lower, op=torch.distributed.ReduceOp.MIN)
                    torch.distributed.all_reduce(upper, op=torch.distributed.ReduceOp.MAX)
                    if not torch.equal(lower, upper):
                        raise QwenTrainingError(
                            "distributed_projection_mismatch",
                            "post-projection parameters differ across data-parallel ranks",
                        )
            if rank_snapshot_indices:
                rank_squares = torch.zeros(4, dtype=torch.float64)
                for index in rank_snapshot_indices:
                    after = optimizer_parameters[index].detach().to(device="cpu")
                    before = parameter_snapshot[index]
                    rank_squares += (after - before).double().reshape(4, -1).square().sum(1)
                self.last_rank_update_norms = tuple(float(value) for value in rank_squares.sqrt())
            if not _tensors_are_finite(tuple(optimizer_parameters)):
                raise QwenTrainingError(
                    "nonfinite_parameter", "optimizer produced a nonfinite parameter"
                )
            self.scheduler.step()
        except BaseException:
            with torch.no_grad():
                for parameter, snapshot in zip(optimizer_parameters, parameter_snapshot):
                    parameter.copy_(snapshot.to(device=parameter.device))
            self.optimizer.load_state_dict(optimizer_snapshot)
            _move_optimizer_state_(self.optimizer, to_parameter_devices=False)
            self.scheduler.load_state_dict(scheduler_snapshot)
            if self.grad_scaler is not None:
                self.grad_scaler.load_state_dict(scaler_snapshot)
                self._synchronize_scaler_state()
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            raise
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens_seen += token_count
        self.example_cursor += self.config.accumulation_steps
        rates: dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{index}")
            if type(name) is not str or name in rates:
                raise RuntimeError("optimizer group names must be unique strings")
            rates[name] = float(group["lr"])
        return HealStepLog(
            job_id=self.job_id,
            pairing_id=self.pairing_id,
            arm=self.arm,
            update=self.step,
            tokens_seen=self.tokens_seen,
            example_ids=example_ids,
            microbatches=self.config.accumulation_steps,
            losses=totals,
            learning_rates=rates,
            skipped_steps=self.skipped_steps,
        )


@dataclass(frozen=True)
class QwenJobData:
    """Materialized, identity-bearing train/evaluation windows for one heal job."""

    train_microbatches: tuple[Mapping[str, object], ...]
    eval_microbatches: tuple[Mapping[str, object], ...]
    data_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        for field_name in ("train_microbatches", "eval_microbatches"):
            value = getattr(self, field_name)
            if type(value) is not tuple or not value:
                raise ValueError(f"{field_name} must be a nonempty tuple")
            if any(not isinstance(batch, Mapping) for batch in value):
                raise TypeError(f"{field_name} must contain mappings")
        if not isinstance(self.data_identity, Mapping) or not self.data_identity:
            raise ValueError("data_identity must be a nonempty mapping")
        try:
            json.dumps(
                self.data_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("data_identity must be finite JSON") from error


_QWEN_ARM_IDS = {
    "native": "native",
    "recency": "recency",
    "exact_cache.selector.recency": "recency",
    "surprise": "surprise",
    "exact_cache.selector.exact_outer": "surprise",
    "gdn2-channel-r1": "native",
    "rout-4": "native",
    "mimo-r2": "native",
    "mimo-r4": "native",
    "rot-off": "native",
    "rot-constant": "native",
    "rot-noncumulative": "native",
    "rot-fixed-rope": "native",
    "rot-moving-frame-oracle": "native",
    "trapezoid": "native",
    "lookahead": "native",
    "qk-bc-additive": "native",
    "qk-diagonal": "native",
    "gdn2-mimo-r4-braid-shared-hola-w64": "native",
    "gdn2-mimo-r4-braid-four-state-hola-w64": "native",
}


@dataclass(frozen=True)
class _ArchitectureDispatchContract:
    arm: str
    architecture_arm_id: str
    registry_sha256: str
    mimo_rank: int
    trainable_names: tuple[str, ...]
    diagnostic_training: bool = False


def _architecture_dispatch_contract(
    job: Mapping[str, object], config: Mapping[str, object]
) -> _ArchitectureDispatchContract | None:
    """Validate architecture identity before any data/model construction."""
    from .architecture import TARGET_LAYERS, architecture_record, registry_sha256
    from .qwen_architecture import QwenArchitectureConfig

    arm_id = job.get("arm_id")
    architecture = config.get("architecture")
    architecture_arm_id = (
        architecture.get("arm_id") if isinstance(architecture, Mapping) else None
    )
    architecture_ids = {
        "rout-4", "mimo-r2", "mimo-r4", "rot-off", "rot-constant", "rot-noncumulative",
        "rot-fixed-rope", "rot-moving-frame-oracle",
        "trapezoid", "lookahead", "qk-bc-additive", "qk-diagonal",
        "gdn2-mimo-r4-braid-shared-hola-w64",
        "gdn2-mimo-r4-braid-four-state-hola-w64",
    }
    if arm_id not in architecture_ids and architecture_arm_id not in architecture_ids:
        return None
    if (
        arm_id not in architecture_ids
        or architecture_arm_id not in architecture_ids
        or architecture_arm_id != arm_id
    ):
        raise QwenRuntimeConfigurationError(
            "architecture_arm_mismatch", "job arm_id does not match canonical architecture arm_id"
        )
    assert isinstance(architecture, Mapping)
    digest = architecture.get("registry_sha256")
    submitted_digest = job.get("architecture_registry_sha256")
    if digest != submitted_digest or digest != registry_sha256():
        raise QwenRuntimeConfigurationError(
            "architecture_registry_hash_mismatch", "architecture registry identity is stale or inconsistent"
        )
    record = architecture_record(arm_id)
    submitted_width = architecture.get("output_width", record.output_width)
    if type(submitted_width) is not int:
        raise QwenRuntimeConfigurationError(
            "architecture_width_invalid", "architecture output_width must be an int"
        )
    if submitted_width != record.output_width:
        raise QwenRuntimeConfigurationError(
            "architecture_width_mismatch", "job width does not match the canonical architecture arm"
        )
    rank = architecture.get("mimo_rank", record.mimo_rank)
    if rank != record.mimo_rank:
        raise QwenRuntimeConfigurationError(
            "architecture_rank_mismatch", "job rank does not match the canonical true MIMO arm"
        )
    rotation_arm = str(arm_id).startswith("rot-")
    hybrid_arm = str(arm_id).startswith("gdn2-mimo-r4-braid-")
    forbidden = (
        submitted_width != (4 if arm_id == "rout-4" or hybrid_arm else 1)
        or architecture.get("gate_mode", record.gate_mode) != ("channelwise" if hybrid_arm else "scalar")
        or bool(architecture.get("cache_enabled", record.cache.enabled)) is not hybrid_arm
        or architecture.get("gdn2_decoupled", False) is not False
        or (rotation_arm and architecture.get("rotation_mode", record.rotation_mode) != record.rotation_mode)
    )
    if forbidden:
        raise QwenRuntimeConfigurationError(
            "architecture_combination_invalid",
            "architecture arm contains a forbidden mechanism combination",
        )
    # Re-run the production record validator while still in the preconstruction phase.
    diagnostic_training = architecture.get("diagnostic_training", False)
    if rotation_arm and job.get("architecture_diagnostic_training") is not diagnostic_training:
        raise QwenRuntimeConfigurationError(
            "architecture_diagnostic_training_mismatch",
            "submitted diagnostic-training identity does not match canonical architecture config",
        )
    QwenArchitectureConfig(arm_id, digest, record, diagnostic_training=diagnostic_training)
    if type(diagnostic_training) is not bool or (diagnostic_training and arm_id != "rot-moving-frame-oracle"):
        raise QwenRuntimeConfigurationError(
            "architecture_combination_invalid", "diagnostic training is valid only for moving-frame oracle"
        )
    incremental_suffixes = {
        "trapezoid": ("rho_head", "rho_proj.weight"),
        "lookahead": ("lookahead_rho", "lookahead_projection.weight"),
        "qk-bc-additive": ("bc_q_amplitude", "bc_k_amplitude", "bc_q_bias", "bc_k_bias"),
        "qk-diagonal": ("bc_q_amplitude", "bc_k_amplitude", "bc_q_scale", "bc_k_scale"),
    }
    shared_hybrid = str(arm_id).endswith("shared-hola-w64")
    hybrid_suffixes = (
        "components.q_weight", "components.k_weight", "components.v_weight",
        "components.erase_weight", "components.write_weight", "components.z_weight",
        "components.write_offset", "components.native_decay_weight", "components.native_A_log",
        "components.native_dt_bias",
        "components.phase_proj.weight",
        "components.phase_proj.bias", "components.output_mixer", "components.d_q",
        "components.d_k", "components.b_q", "components.b_k", "components.alpha_q",
        "components.beta_q", "components.alpha_k", "components.beta_k", "components.cache_gate_logit",
        "components.conv1d.weight", "components.norm.weight", "components.out_proj.weight",
        "rot_proj.weight", "rot_proj.bias", "hola.gamma_q", "hola.gamma_k", "hola.sink_logit",
    ) + ((
        "components.native_decay_chan", "components.braid_residual",
        "components.trapezoid_gate", "components.lookahead_gate",
        "components.c_logits", "components.d_raw", "components.braid_router.weight",
        "components.braid_router.bias", "hola_output_mixer",
    ) if shared_hybrid else (
        "components.native_decay_pair", "components.trapezoid_proj.weight",
        "components.trapezoid_proj.bias",
    ))
    if hybrid_arm:
        from .qwen_variants import validate_maximum_control_config
        maximum = validate_maximum_control_config(dict(config))
        if maximum is not None:
            disabled_groups = (
                (("braid",), not maximum.braid),
                (("trapezoid",), not maximum.trapezoid),
                (("lookahead",), not maximum.lookahead),
                (("d_q", "d_k", "b_q", "b_k", "alpha_q", "alpha_k", "beta_q", "beta_k"), not maximum.affine_qk),
                (("cache_gate_logit", "hola", "hola_output"), maximum.cache_policy == "none"),
            )
            hybrid_suffixes = tuple(
                suffix for suffix in hybrid_suffixes
                if not any(disabled and any(token in suffix for token in tokens)
                           for tokens, disabled in disabled_groups)
            )
    suffixes = (hybrid_suffixes if hybrid_arm else
                incremental_suffixes[arm_id] if arm_id in incremental_suffixes else
                {"rot-constant": ("rotation_rate",),
                 "rot-noncumulative": ("rot_proj.weight", "rot_proj.bias"),
                 "rot-moving-frame-oracle": (("rot_proj.weight", "rot_proj.bias") if diagnostic_training else ())}.get(str(arm_id), ())
                if rotation_arm else
                ("q_slot_scale", "out_mix") if arm_id == "rout-4" else
                ("mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out"))
    names = tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS
        for suffix in suffixes
    ))
    return _ArchitectureDispatchContract("native", arm_id, digest, rank, names, diagnostic_training)


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", f"{name} must be a mapping"
        )
    return value


def _job_config(job: Mapping[str, object]) -> Mapping[str, object]:
    config = job.get("canonical_config")
    if not isinstance(config, Mapping):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.canonical_config must be a mapping"
        )
    if config.get("backend") != "qwen" or job.get("backend") != "qwen":
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "Qwen dispatcher received a non-Qwen job"
        )
    from .qwen_variants import validate_maximum_control_config
    try:
        maximum = validate_maximum_control_config(dict(config))
    except ValueError as error:
        raise QwenRuntimeConfigurationError(
            "maximum_control_invalid", str(error)
        ) from error
    qwen = _required_mapping(config.get("qwen"), "canonical_config.qwen")
    expected_mode = "reliance" if maximum is not None and not maximum.replacement else "heal"
    if qwen.get("run_mode") != expected_mode:
        raise QwenRuntimeConfigurationError(
            "qwen_mode_invalid" if maximum is not None else "qwen_heal_required",
            f"Qwen adapter requires run_mode={expected_mode!r}"
        )
    return config


def _positive_int(mapping: Mapping[str, object], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int or value < 1:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must be a positive integer"
        )
    return value


def _string_tuple(value: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (
        not allow_empty and not value
    ):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must be a string sequence"
        )
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must contain nonempty strings"
        )
    if len(set(result)) != len(result):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must not contain duplicates"
        )
    return result


def _selected_arm(job: Mapping[str, object]) -> str:
    arm_id = job.get("arm_id")
    if arm_id not in _QWEN_ARM_IDS:
        raise QwenRuntimeConfigurationError(
            "qwen_arm_invalid", f"unsupported paired Qwen arm: {arm_id!r}"
        )
    return _QWEN_ARM_IDS[arm_id]


def derive_three_arm_pairing(
    job: Mapping[str, object],
    *,
    example_ids: tuple[str, ...],
    pre_replacement_checkpoint_sha256: str,
    data_sha256: str,
):
    """Derive the native/recency/surprise scientific contract from one job."""
    from .qwen_backend import QwenHealArmContract, validate_three_arm_pairing

    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    config = _job_config(job)
    budget = _required_mapping(config.get("budget"), "canonical_config.budget")
    optimizer = _required_mapping(
        config.get("optimizer"), "canonical_config.optimizer"
    )
    schedule = _required_mapping(config.get("schedule"), "canonical_config.schedule")
    lengths = _required_mapping(config.get("lengths"), "canonical_config.lengths")
    task = _required_mapping(config.get("task"), "canonical_config.task")
    task_params = _required_mapping(task.get("params"), "canonical_config.task.params")
    cache = _required_mapping(config.get("cache"), "canonical_config.cache")
    seed = job.get("seed")
    if type(seed) is not int or seed < 0:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.seed must be a nonnegative integer"
        )
    curriculum_raw = lengths.get("curriculum")
    if not isinstance(curriculum_raw, (list, tuple)):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "lengths.curriculum must be a sequence"
        )
    curriculum = tuple(curriculum_raw)
    extrapolation = lengths.get("extrapolation")
    if not isinstance(extrapolation, (list, tuple)) or not extrapolation:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "lengths.extrapolation must be nonempty"
        )
    stopping = task_params.get(
        "stopping", {"max_nonfinite": 0, "early_stopping": False}
    )
    if not isinstance(stopping, Mapping) or not stopping:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "task.params.stopping must be a mapping"
        )
    cache_match = {
        "width": cache.get("width"),
        "block_size": cache.get("block_size"),
        "read": cache.get("read"),
        "read_init": cache.get("read_init"),
        "storage_dtype": cache.get("storage_dtype"),
        "lr_cache": cache.get("lr_cache"),
    }
    surprise_policy = cache.get("score")
    if type(surprise_policy) is not str or surprise_policy in {"", "recency"}:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "cache.score must name the winning surprise policy"
        )
    job_id = job.get("job_id")
    if type(job_id) is not str or not job_id:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.job_id must be a nonempty string"
        )
    contracts = tuple(
        QwenHealArmContract(
            arm=arm,
            job_id=job_id if arm == _selected_arm(job) else f"{job_id}:{arm}",
            seed=seed,
            pre_replacement_checkpoint_sha256=pre_replacement_checkpoint_sha256,
            data_sha256=data_sha256,
            example_ids=example_ids,
            token_budget=_positive_int(budget, "tokens"),
            update_budget=_positive_int(budget, "updates"),
            curriculum=curriculum,
            optimizer=optimizer,
            schedule=schedule,
            stopping=stopping,
            eval_cells=tuple(str(length) for length in extrapolation),
            cache_match=None if arm == "native" else cache_match,
            selection_policy=(
                None if arm == "native" else "recency" if arm == "recency" else surprise_policy
            ),
        )
        for arm in ("native", "recency", "surprise")
    )
    return validate_three_arm_pairing(contracts)


def _batch_example_ids(batch: Mapping[str, object]) -> tuple[str, ...]:
    identifiers = batch.get("example_ids")
    if type(identifiers) is not tuple or not identifiers or any(
        type(item) is not str or not item for item in identifiers
    ):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "every data window requires tuple example_ids"
        )
    return identifiers


def _batch_token_count(batch: Mapping[str, object]) -> int:
    input_ids = batch.get("input_ids")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.dtype != torch.long
        or input_ids.ndim != 2
    ):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "input_ids must be rank-2 torch.long"
        )
    labels = batch.get("labels")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.long
        or labels.shape != input_ids.shape
    ):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "labels must match input_ids"
        )
    if len(_batch_example_ids(batch)) != input_ids.shape[0]:
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "example_ids must match batch size"
        )
    return input_ids.numel()


def _validate_job_data(
    data: QwenJobData,
    *,
    config: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if not isinstance(data, QwenJobData):
        raise TypeError("data loader must return QwenJobData")
    task = _required_mapping(config.get("task"), "canonical_config.task")
    params = _required_mapping(task.get("params"), "canonical_config.task.params")
    accumulation = params.get("accumulation_steps", 1)
    if type(accumulation) is not int or accumulation < 1:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "accumulation_steps must be positive"
        )
    budget = _required_mapping(config.get("budget"), "canonical_config.budget")
    updates = _positive_int(budget, "updates")
    expected_microbatches = updates * accumulation
    if len(data.train_microbatches) != expected_microbatches:
        raise QwenRuntimeConfigurationError(
            "data_window_invalid",
            "training windows do not exactly cover update x accumulation budget",
        )
    token_count = sum(_batch_token_count(batch) for batch in data.train_microbatches)
    if token_count != _positive_int(budget, "tokens"):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "training windows do not exactly match token budget"
        )
    windows = tuple(_batch_example_ids(batch) for batch in data.train_microbatches)
    flattened = tuple(item for window in windows for item in window)
    if len(flattened) != len(set(flattened)):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "paired Qwen example IDs must be globally unique"
        )
    preregistered = _string_tuple(
        params.get("example_ids"), "task.params.example_ids"
    )
    if flattened != preregistered:
        raise QwenRuntimeConfigurationError(
            "example_window_mismatch",
            "runtime data windows do not match preregistered example_ids order",
        )
    for batch in data.eval_microbatches:
        _batch_token_count(batch)
    return flattened, windows


def _digest_string(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", f"asset_hashes.{name} is not SHA-256"
        )
    return value


def _runtime_assets(
    runtime: Mapping[str, object], *, teacher_required: bool
) -> tuple[dict[str, object], dict[str, object]]:
    from .qwen_backend import ExternalAssetIdentity, validate_external_assets

    allowed = {
        "model",
        "tokenizer",
        "checkpoint",
        "data",
        "teacher_model",
        "output",
        "student_device",
        "teacher_device",
        "dtype",
        "asset_hashes",
        "resume",
        "checkpoint_every",
    }
    unknown = set(runtime) - allowed
    if unknown:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "unknown runtime keys: " + ", ".join(sorted(unknown)),
        )
    for name in ("model", "checkpoint", "data", "output", "student_device", "dtype"):
        if runtime.get(name) is None:
            raise QwenRuntimeConfigurationError(
                "runtime_configuration_invalid", f"runtime.{name} is required"
            )
    if teacher_required:
        for name in ("teacher_model", "teacher_device"):
            if runtime.get(name) is None:
                raise TeacherRequiredError(
                    "teacher_required", f"runtime.{name} is required for Qwen heal"
                )
    if type(runtime.get("student_device")) is not str or (
        runtime.get("teacher_device") is not None
        and type(runtime.get("teacher_device")) is not str
    ):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime devices must be strings"
        )
    if runtime.get("dtype") not in {"float32", "bfloat16"}:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime.dtype must be float32 or bfloat16"
        )
    if type(runtime.get("resume")) is not bool:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime.resume must be boolean"
        )
    checkpoint_every = runtime.get("checkpoint_every", 1)
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "checkpoint_every must be positive"
        )
    hashes = _required_mapping(runtime.get("asset_hashes"), "runtime.asset_hashes")
    asset_paths = {
        name: runtime[name]
        for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model")
        if runtime.get(name) is not None
    }
    if set(hashes) != set(asset_paths):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "asset_hashes must exactly match supplied runtime asset paths",
        )
    specs = []
    for name, raw_path in asset_paths.items():
        try:
            path = Path(raw_path)
        except TypeError as error:
            raise QwenRuntimeConfigurationError(
                "runtime_configuration_invalid", f"runtime.{name} must be path-like"
            ) from error
        kind = "directory" if path.is_dir() else "file"
        specs.append(
            ExternalAssetIdentity(
                name=name,
                path=path,
                kind=kind,
                sha256=_digest_string(name, hashes[name]),
            )
        )
    validated = {asset.name: asset for asset in validate_external_assets(specs)}
    normalized = dict(runtime)
    normalized["output"] = Path(runtime["output"]).expanduser().resolve()
    normalized["checkpoint_every"] = checkpoint_every
    return normalized, validated


def _asset_spec(asset: object):
    from .qwen_backend import ExternalAssetIdentity, ValidatedAssetIdentity

    if not isinstance(asset, ValidatedAssetIdentity):
        raise TypeError("asset must be a ValidatedAssetIdentity")
    return ExternalAssetIdentity(
        name=asset.name,
        path=asset.path,
        kind=asset.kind,
        size_bytes=asset.size_bytes,
        sha256=asset.sha256,
    )


def _training_parameter_names(
    config: Mapping[str, object], arm: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    architecture = config.get("architecture")
    if isinstance(architecture, Mapping) and architecture.get("arm_id") == "gdn2-channel-r1":
        from .architecture import TARGET_LAYERS
        memory = tuple(sorted(
            f"model.layers.{index}.linear_attn.{suffix}"
            for index in TARGET_LAYERS
            for suffix in ("erase_proj.weight", "write_proj.weight", "write_offset")
        ))
        return memory, ()
    task = _required_mapping(config.get("task"), "canonical_config.task")
    params = _required_mapping(task.get("params"), "canonical_config.task.params")
    memory = _string_tuple(
        params.get("memory_parameter_names"), "task.params.memory_parameter_names"
    )
    cache = _string_tuple(
        params.get("cache_parameter_names", ()),
        "task.params.cache_parameter_names",
        allow_empty=True,
    )
    return memory, cache if arm != "native" else ()


def _training_config(config: Mapping[str, object]) -> QwenHealTrainingConfig:
    task = _required_mapping(config.get("task"), "canonical_config.task")
    params = _required_mapping(task.get("params"), "canonical_config.task.params")
    budget = _required_mapping(config.get("budget"), "canonical_config.budget")
    return QwenHealTrainingConfig(
        objective=str(params.get("objective", "language_model_heal")),
        ce_weight=params.get("ce_weight", 0.1),
        kl_weight=params.get("kl_weight", 1.0),
        layerwise_weight=params.get("layerwise_weight", 0.0),
        temperature=params.get("temperature", 2.0),
        accumulation_steps=params.get("accumulation_steps", 1),
        max_updates=_positive_int(budget, "updates"),
        max_tokens=_positive_int(budget, "tokens"),
        gradient_checkpointing=params.get("gradient_checkpointing", True),
        lambda_spec=params.get("lambda_spec", 0.0),
        lambda_gate=params.get("lambda_gate", 0.0),
        specialization_updates=params.get("specialization_updates", 0),
    )


def _validate_parallel_and_packing(config: Mapping[str, object]) -> int:
    task = _required_mapping(config.get("task"), "canonical_config.task")
    params = _required_mapping(task.get("params"), "canonical_config.task.params")
    for field in ("tensor_parallel", "pipeline_parallel", "data_parallel"):
        value = params.get(field, 1)
        if type(value) is not int or value < 1:
            raise QwenRuntimeConfigurationError("unsupported_execution", f"{field} must be positive")
        if field in {"tensor_parallel", "pipeline_parallel"} and value != 1:
            raise QwenRuntimeConfigurationError(
                "unsupported_execution", f"{field}>1 has no operational Qwen dispatcher implementation"
            )
    data_parallel = int(params.get("data_parallel", 1))
    if data_parallel > 1:
        if (not torch.distributed.is_available() or not torch.distributed.is_initialized()
                or torch.distributed.get_world_size() != data_parallel):
            raise QwenRuntimeConfigurationError(
                "unsupported_execution", "data_parallel requires an initialized matching torch.distributed world"
            )
    if params.get("packed", False) is True and params.get("document_boundaries") is not True:
        raise QwenRuntimeConfigurationError(
            "unsupported_execution", "packed Qwen training requires explicit document boundaries"
        )
    return data_parallel


def _default_load_data(
    *,
    asset: object,
    job: Mapping[str, object] | None = None,
    **_kwargs: object,
) -> QwenJobData:
    """Load a small explicit JSON/JSONL/PT window bundle for production smoke runs."""
    path = asset.path
    if path.is_dir():
        candidates = (path / "qwen_windows.pt", path / "qwen_windows.jsonl")
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", f"no supported Qwen window bundle at {path}"
        )
    if path.suffix == ".pt":
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise QwenRuntimeConfigurationError(
                "data_window_invalid", "Qwen .pt data is not a safe tensor bundle"
            ) from error
    elif path.suffix == ".jsonl":
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "Qwen data must be .pt, .json, or .jsonl"
        )
    evaluation_seed: int | None = None
    available_evaluation_seeds: tuple[int, ...] = ()
    if isinstance(raw, Mapping):
        train_raw = raw.get("train")
        eval_by_seed = raw.get("eval_by_seed")
        if eval_by_seed is not None:
            if "eval" in raw:
                raise QwenRuntimeConfigurationError(
                    "data_window_invalid",
                    "Qwen data bundle cannot contain both eval and eval_by_seed",
                )
            if not isinstance(eval_by_seed, Mapping) or not eval_by_seed:
                raise QwenRuntimeConfigurationError(
                    "data_window_invalid",
                    "Qwen data bundle eval_by_seed must be a nonempty mapping",
                )
            seed_keys = tuple(eval_by_seed)
            if any(
                type(seed_key) is not str
                or not seed_key.isdigit()
                or str(int(seed_key)) != seed_key
                for seed_key in seed_keys
            ):
                raise QwenRuntimeConfigurationError(
                    "data_window_invalid",
                    "Qwen data bundle eval_by_seed keys must be canonical nonnegative integers",
                )
            if not isinstance(job, Mapping):
                raise QwenRuntimeConfigurationError(
                    "data_window_invalid",
                    "seed-indexed Qwen evaluation data requires a job mapping",
                )
            evaluation_seed = job.get("seed")
            if type(evaluation_seed) is not int or evaluation_seed < 0:
                raise QwenRuntimeConfigurationError(
                    "data_window_invalid",
                    "seed-indexed Qwen evaluation data requires a nonnegative job seed",
                )
            seed_key = str(evaluation_seed)
            if seed_key not in eval_by_seed:
                raise QwenRuntimeConfigurationError(
                    "data_window_invalid",
                    f"Qwen data bundle has no evaluation partition for seed {evaluation_seed}",
                )
            available_evaluation_seeds = tuple(
                sorted(int(item) for item in seed_keys)
            )
            eval_raw = eval_by_seed[seed_key]
        else:
            eval_raw = raw.get("eval", train_raw)
    else:
        train_raw = eval_raw = raw
    if not isinstance(train_raw, Sequence) or not isinstance(eval_raw, Sequence):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "Qwen data bundle train/eval fields must be sequences"
        )

    def convert(record: object, index: int) -> Mapping[str, object]:
        if not isinstance(record, Mapping):
            raise QwenRuntimeConfigurationError(
                "data_window_invalid", f"data record {index} must be a mapping"
            )
        identifiers = record.get("example_ids")
        if identifiers is None:
            identifier = record.get("example_id", f"window-{index:08d}")
            identifiers = (identifier,)
        identifiers = tuple(identifiers)
        input_ids = torch.as_tensor(record.get("input_ids"), dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        labels_raw = record.get("labels")
        labels = input_ids.clone() if labels_raw is None else torch.as_tensor(
            labels_raw, dtype=torch.long
        )
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        converted: dict[str, object] = {
            "input_ids": input_ids,
            "labels": labels,
            "example_ids": identifiers,
        }
        tensor_annotations = {
            "query_mask": 2,
            "source_spans": 3,
            "stale_mask": 3,
        }
        for name, expected_rank in tensor_annotations.items():
            if name not in record:
                continue
            tensor = torch.as_tensor(record[name])
            if tensor.ndim == expected_rank - 1:
                tensor = tensor.unsqueeze(0)
            converted[name] = tensor
        if "stale_positions" in record:
            stale_positions = torch.as_tensor(
                record["stale_positions"], dtype=torch.int64
            )
            if stale_positions.numel() == 0:
                stale_positions = stale_positions.reshape(0, 3)
            converted["stale_positions"] = stale_positions
        if "ruler_metadata" in record:
            metadata = record["ruler_metadata"]
            if isinstance(metadata, Mapping):
                metadata = (metadata,)
            if (
                isinstance(metadata, (str, bytes, bytearray))
                or not isinstance(metadata, Sequence)
                or any(not isinstance(item, Mapping) for item in metadata)
            ):
                raise QwenRuntimeConfigurationError(
                    "ruler_annotations_invalid",
                    f"data record {index} ruler_metadata must contain mappings",
                )
            converted["ruler_metadata"] = tuple(
                copy.deepcopy(dict(item)) for item in metadata
            )
            # Canonical bundles store only compact token IDs plus the exact
            # answer/source spans.  Materialize the dense per-token view on
            # load so it occupies memory for only the selected seed, not disk
            # for every campaign seed.
            ruler_metadata = converted["ruler_metadata"]
            assert isinstance(ruler_metadata, tuple)
            if len(ruler_metadata) != input_ids.shape[0]:
                raise QwenRuntimeConfigurationError(
                    "ruler_annotations_invalid",
                    f"data record {index} metadata batch size is inconsistent",
                )
            compact_span_flags = tuple(
                "answer_spans" in item or "source_spans" in item
                for item in ruler_metadata
            )
            if any(compact_span_flags) and not all(compact_span_flags):
                raise QwenRuntimeConfigurationError(
                    "ruler_annotations_invalid",
                    f"data record {index} mixes compact and legacy RULER metadata",
                )
            compact_spans = bool(compact_span_flags and all(compact_span_flags))
            if compact_spans and (
                "query_mask" not in converted or "source_spans" not in converted
            ):
                derived_query_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                derived_source_spans = torch.full(
                    (*input_ids.shape, 2), -1, dtype=torch.int64
                )
                for batch_index, item in enumerate(ruler_metadata):
                    answer_spans = item.get("answer_spans")
                    source_span_rows = item.get("source_spans")
                    if (
                        not isinstance(answer_spans, Sequence)
                        or isinstance(answer_spans, (str, bytes, bytearray))
                        or not isinstance(source_span_rows, Sequence)
                        or isinstance(source_span_rows, (str, bytes, bytearray))
                        or len(answer_spans) != len(source_span_rows)
                    ):
                        raise QwenRuntimeConfigurationError(
                            "ruler_annotations_invalid",
                            f"data record {index} has malformed compact spans",
                        )
                    for answer_span, source_span in zip(
                        answer_spans, source_span_rows
                    ):
                        if (
                            not isinstance(answer_span, Sequence)
                            or isinstance(answer_span, (str, bytes, bytearray))
                            or len(answer_span) != 2
                            or not isinstance(source_span, Sequence)
                            or isinstance(source_span, (str, bytes, bytearray))
                            or len(source_span) != 2
                            or any(type(value) is not int for value in (*answer_span, *source_span))
                        ):
                            raise QwenRuntimeConfigurationError(
                                "ruler_annotations_invalid",
                                f"data record {index} has malformed compact span rows",
                            )
                        answer_start, answer_stop = answer_span
                        source_start, source_stop = source_span
                        if not (
                            0 <= answer_start < answer_stop <= input_ids.shape[1]
                            and 0 <= source_start < source_stop <= answer_start
                        ):
                            raise QwenRuntimeConfigurationError(
                                "ruler_annotations_invalid",
                                f"data record {index} has out-of-range compact spans",
                            )
                        derived_query_mask[
                            batch_index, answer_start:answer_stop
                        ] = True
                        derived_source_spans[
                            batch_index, answer_start:answer_stop
                        ] = torch.tensor(
                            (source_start, source_stop), dtype=torch.int64
                        )
                converted.setdefault("query_mask", derived_query_mask)
                converted.setdefault("source_spans", derived_source_spans)
            if compact_spans or {
                "query_mask", "source_spans"
            } <= set(converted):
                converted.setdefault(
                    "stale_positions", torch.zeros(0, 3, dtype=torch.int64)
                )
        if "state_tracking_metadata" in record:
            metadata = record["state_tracking_metadata"]
            if isinstance(metadata, Mapping):
                metadata = (metadata,)
            if (isinstance(metadata, (str, bytes, bytearray))
                    or not isinstance(metadata, Sequence)
                    or any(not isinstance(item, Mapping) for item in metadata)):
                raise QwenRuntimeConfigurationError(
                    "state_tracking_annotations_invalid",
                    f"data record {index} state_tracking_metadata must contain mappings",
                )
            converted["state_tracking_metadata"] = tuple(
                copy.deepcopy(dict(item)) for item in metadata
            )
        return converted

    train = tuple(convert(record, index) for index, record in enumerate(train_raw))
    evaluation = tuple(convert(record, index) for index, record in enumerate(eval_raw))
    data_identity: dict[str, object] = {
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
        "kind": asset.kind,
        "example_count": len(train),
    }
    if evaluation_seed is not None:
        data_identity.update({
            "evaluation_seed": evaluation_seed,
            "available_evaluation_seeds": list(available_evaluation_seeds),
            "evaluation_example_count": len(evaluation),
        })
    return QwenJobData(
        train_microbatches=train,
        eval_microbatches=evaluation,
        data_identity=data_identity,
    )


def _default_load_teacher(*, asset: object, runtime: Mapping[str, object], **_kwargs: object):
    from .qwen_backend import _default_base_model_loader

    dtype = torch.float32 if runtime["dtype"] == "float32" else torch.bfloat16
    teacher = _default_base_model_loader(
        Path(asset.path), torch_dtype=dtype, low_cpu_mem_usage=True
    )
    teacher.to(runtime["teacher_device"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _default_build_scheduler(
    *, optimizer: torch.optim.Optimizer, config: Mapping[str, object], **_kwargs: object
):
    schedule = _required_mapping(config.get("schedule"), "canonical_config.schedule")
    if schedule.get("name") != "cosine":
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "Qwen heal requires cosine schedule"
        )
    warmup = schedule.get("warmup_updates")
    if type(warmup) is not int or warmup < 0:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "warmup_updates must be nonnegative"
        )
    total = _positive_int(
        _required_mapping(config.get("budget"), "canonical_config.budget"), "updates"
    )

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _move_batch(
    batch: Mapping[str, object], device: str
) -> dict[str, object]:
    return {
        name: (
            value.to(device)
            if isinstance(value, torch.Tensor)
            and name not in _EVALUATION_ANNOTATION_FIELDS
            else value
        )
        for name, value in batch.items()
    }


def _shard_training_windows(
    batches: tuple[Mapping[str, object], ...], *, rank: int, world_size: int,
) -> tuple[tuple[Mapping[str, object], ...], tuple[tuple[str, ...], ...]]:
    """Shard paired windows by rows; rank unions reproduce each full window."""
    if not 0 <= rank < world_size or world_size < 1:
        raise QwenRuntimeConfigurationError("data_parallel_shard_invalid", "rank/world size is invalid")
    sharded: list[Mapping[str, object]] = []
    windows: list[tuple[str, ...]] = []
    for batch in batches:
        ids = _batch_example_ids(batch)
        if len(ids) % world_size:
            raise QwenRuntimeConfigurationError(
                "data_parallel_shard_uneven",
                "each global training window must divide evenly across data-parallel ranks",
            )
        indices = tuple(range(rank, len(ids), world_size))
        if not indices:
            raise QwenRuntimeConfigurationError(
                "data_parallel_shard_empty", "each window must contain a row for every rank"
            )
        device = batch["input_ids"].device
        index_tensor = torch.tensor(indices, device=device)
        row: dict[str, object] = {}
        for name, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(ids):
                row[name] = value.index_select(0, index_tensor)
            elif isinstance(value, tuple) and len(value) == len(ids):
                row[name] = tuple(value[index] for index in indices)
            else:
                row[name] = value
        local_ids = tuple(ids[index] for index in indices)
        row["example_ids"] = local_ids
        sharded.append(row); windows.append(local_ids)
    return tuple(sharded), tuple(windows)


_EVALUATION_ANNOTATION_FIELDS = {
    "query_mask",
    "source_spans",
    "stale_mask",
    "stale_positions",
    "ruler_metadata",
    "state_tracking_metadata",
}

# A full Qwen-3.5 vocabulary projection at 32k is about 15 GiB in BF16.  The
# backbone must still see the complete context (the exact-cache contract does
# not permit cross-call state), but its final [B,T,H] hidden tensor is small
# enough to retain while the LM head is evaluated in bounded token chunks.
_EVALUATION_LOGIT_WORKSPACE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class _StreamingCausalScores:
    loss: torch.Tensor
    aligned_predictions: torch.Tensor
    correct: int
    total: int
    chunk_tokens: int
    peak_logit_bytes: int


def _stream_causal_scores(
    model: torch.nn.Module,
    *,
    inputs: Mapping[str, object],
    labels: torch.Tensor,
) -> _StreamingCausalScores | None:
    """Run one exact full-context backbone pass and stream the vocabulary head.

    Hugging Face causal-LM wrappers are a backbone followed by ``lm_head``.
    Calling those same two modules separately is numerically identical to the
    wrapper, while avoiding a resident [B,T,V] tensor.  Unknown/custom model
    wrappers return ``None`` and retain the established dense evaluator path.
    """

    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if (
        not isinstance(backbone, torch.nn.Module)
        or not isinstance(lm_head, torch.nn.Module)
        or not callable(get_output_embeddings)
    ):
        return None
    try:
        if get_output_embeddings() is not lm_head:
            return None
    except (AttributeError, NotImplementedError, TypeError):
        return None
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None

    from .qwen_exact_cache import guarded_model_forward

    backbone_output = guarded_model_forward(backbone, **dict(inputs))
    if isinstance(backbone_output, Mapping):
        hidden = backbone_output.get("last_hidden_state")
    else:
        hidden = getattr(backbone_output, "last_hidden_state", None)
    if hidden is None and isinstance(backbone_output, (tuple, list)) and backbone_output:
        hidden = backbone_output[0]
    if (
        not isinstance(hidden, torch.Tensor)
        or not hidden.is_floating_point()
        or hidden.ndim != 3
        or labels.dtype != torch.long
        or labels.shape != hidden.shape[:2]
        or hidden.shape[1] < 2
    ):
        raise QwenRuntimeConfigurationError(
            "evaluation_output_invalid",
            "streamed Qwen backbone must return floating last_hidden_state [B,T,H]",
        )
    valid = labels[:, 1:] != -100
    if not bool(valid.any()):
        raise QwenRuntimeConfigurationError(
            "evaluation_output_invalid",
            "evaluation labels require at least one valid causal target",
        )

    batch_size, steps, _ = hidden.shape
    vocab_size = int(weight.shape[0])
    if vocab_size < 2:
        raise QwenRuntimeConfigurationError(
            "evaluation_output_invalid", "Qwen LM head vocabulary must be at least two"
        )
    bytes_per_logit = max(hidden.element_size(), weight.element_size())
    chunk_tokens = max(
        1,
        _EVALUATION_LOGIT_WORKSPACE_BYTES
        // max(1, batch_size * vocab_size * bytes_per_logit),
    )
    chunk_tokens = min(chunk_tokens, steps - 1)
    aligned_predictions = torch.zeros_like(labels)
    loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
    correct_count = torch.zeros((), dtype=torch.int64, device=hidden.device)
    total_count = torch.zeros((), dtype=torch.int64, device=hidden.device)
    peak_logit_bytes = 0

    for start in range(0, steps - 1, chunk_tokens):
        stop = min(steps - 1, start + chunk_tokens)
        logits = lm_head(hidden[:, start:stop, :])
        if (
            not isinstance(logits, torch.Tensor)
            or not logits.is_floating_point()
            or logits.shape != (batch_size, stop - start, vocab_size)
        ):
            raise QwenRuntimeConfigurationError(
                "evaluation_output_invalid",
                "Qwen LM head must return floating [B,chunk,V] logits",
            )
        peak_logit_bytes = max(peak_logit_bytes, logits.numel() * logits.element_size())
        targets = labels[:, start + 1 : stop + 1]
        local_valid = targets != -100
        # Per-token CE followed by FP32 accumulation is the same objective as
        # the dense reduction, without retaining either logits or losses.
        local_losses = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(targets)
        loss_sum = loss_sum + local_losses[local_valid].float().sum()
        predictions = logits.argmax(dim=-1)
        aligned_predictions[:, start + 1 : stop + 1] = predictions
        correct_count = correct_count + ((predictions == targets) & local_valid).sum()
        total_count = total_count + local_valid.sum()

    counts = torch.stack((correct_count, total_count)).cpu().tolist()
    total = int(counts[1])
    return _StreamingCausalScores(
        loss=loss_sum / total,
        aligned_predictions=aligned_predictions,
        correct=int(counts[0]),
        total=total,
        chunk_tokens=chunk_tokens,
        peak_logit_bytes=peak_logit_bytes,
    )

_STATE_TRACKING_METADATA_FIELDS = {
    "task", "cell_id", "seed", "example_id", "prompt_end", "targets",
    "modulus", "evidence_scope",
}

_RULER_METADATA_FIELDS = {
    "cell_id",
    "context_length",
    "needles",
    "queries",
    "depth_stratum",
    "example_id",
    "episode_id",
    "evaluation_mode",
    "evidence_scope",
    "seed",
    "example_index",
    "prompt_end",
    "answers",
    "answer_token_ids",
    "answer_spans",
    "source_spans",
    "depth_strata",
    "query_keys",
    "target_digest",
    "paired_interval",
}


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else math.fsum(values) / len(values)


def _checked_byte_sum(name: str, values: Sequence[object]) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                f"cache {name} byte counts must be nonnegative integers",
            )
        total += value
    return total


class _OnlineMean:
    __slots__ = ("total", "count")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float, *, count: int = 1) -> None:
        if not math.isfinite(value) or type(count) is not int or count < 0:
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid", "streamed diagnostic value is invalid"
            )
        self.total = math.fsum((self.total, value))
        self.count += count

    def add_tensor(self, value: torch.Tensor) -> None:
        detached = value.detach()
        if detached.numel() == 0:
            return
        if not bool(torch.isfinite(detached).all()):
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid", "streamed diagnostic tensor is nonfinite"
            )
        self.add(
            float(detached.double().sum().cpu()),
            count=detached.numel(),
        )

    def mean(self) -> float:
        return 0.0 if self.count == 0 else self.total / self.count


class _StreamingTensorSequence:
    __slots__ = (
        "_dtype",
        "_label",
        "_hasher",
        "_sample_size",
        "count",
        "sample",
    )

    def __init__(
        self, *, dtype: torch.dtype, label: str, sample_size: int = 0
    ) -> None:
        self._dtype = dtype
        self._label = label
        self._hasher = hashlib.sha256()
        self.count = 0
        self.sample: list[int] = []
        self._sample_size = sample_size

    def add(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.detach().to(device="cpu", dtype=self._dtype).contiguous().reshape(-1)
        if normalized.numel():
            self._hasher.update(normalized.view(torch.uint8).numpy().tobytes())
            remaining = self._sample_size - len(self.sample)
            if remaining > 0:
                self.sample.extend(
                    int(item) for item in normalized[:remaining].tolist()
                )
            self.count += normalized.numel()
        return normalized

    def digest(self) -> str:
        envelope = hashlib.sha256()
        envelope.update(
            f"tensor-sequence-v1:{self._label}:{self._dtype}:{self.count}:".encode(
                "ascii"
            )
        )
        envelope.update(self._hasher.digest())
        return envelope.hexdigest()


class _StreamingScoreStatistics:
    __slots__ = ("sequence", "total", "minimum", "maximum")

    def __init__(self) -> None:
        self.sequence = _StreamingTensorSequence(
            dtype=torch.float32, label="cache-update-scores"
        )
        self.total = _OnlineMean()
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: torch.Tensor) -> None:
        normalized = self.sequence.add(value)
        if normalized.numel() == 0:
            return
        if not bool(torch.isfinite(normalized).all()) or bool((normalized < 0).any()):
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                "cache scores must be finite and nonnegative",
            )
        self.total.add_tensor(normalized)
        self.minimum = min(self.minimum, float(normalized.min()))
        self.maximum = max(self.maximum, float(normalized.max()))

    def as_dict(self) -> dict[str, object]:
        if self.sequence.count == 0:
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                "cache evaluation produced no measured scores",
            )
        return {
            "count": self.sequence.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total.mean(),
        }


def _cache_amplitudes(model: torch.nn.Module) -> tuple[float, ...]:
    values: list[float] = []
    for name, parameter in sorted(model.named_parameters()):
        if name == "cache_amplitude" or name.endswith(".cache_amplitude"):
            values.extend(float(value) for value in parameter.detach().float().cpu().flatten())
    return tuple(values)


def _validate_evaluation_annotations(
    batch: Mapping[str, object],
    *,
    job: Mapping[str, object],
    require_cache: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Mapping[tuple[int, int], frozenset[int]],
    tuple[tuple[dict[str, object], object], ...],
]:
    config = _job_config(job)
    _validate_parallel_and_packing(config)
    task = _required_mapping(config.get("task"), "canonical_config.task")
    is_ruler = task.get("name") == "ruler"
    required = {"query_mask", "source_spans"}
    missing = sorted(required - set(batch))
    stale_fields = {"stale_mask", "stale_positions"} & set(batch)
    if not stale_fields:
        missing.append("stale_mask or stale_positions")
    if missing:
        code = "cache_annotations_missing" if require_cache else "ruler_annotations_missing"
        raise QwenRuntimeConfigurationError(
            code,
            "annotated evaluation windows require: " + ", ".join(missing),
        )
    if len(stale_fields) != 1:
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid",
            "provide exactly one of stale_mask or stale_positions",
        )
    input_ids = batch.get("input_ids")
    query_mask = batch.get("query_mask")
    source_spans = batch.get("source_spans")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "input_ids must be rank-2 for evaluation"
        )
    batch_size, steps = input_ids.shape
    if (
        not isinstance(query_mask, torch.Tensor)
        or query_mask.dtype != torch.bool
        or query_mask.shape != (batch_size, steps)
    ):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "query_mask must be bool [B,T]"
        )
    if (
        not isinstance(source_spans, torch.Tensor)
        or source_spans.dtype != torch.int64
        or source_spans.shape != (batch_size, steps, 2)
    ):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "source_spans must be int64 [B,T,2]"
        )
    if not bool(query_mask.any()):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "evaluation requires at least one annotated query"
        )
    if bool((source_spans[~query_mask] != -1).any()):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid",
            "non-query positions require [-1,-1] source spans",
        )
    query_coordinates = tuple(
        (int(batch_index), int(token_index))
        for batch_index, token_index in torch.nonzero(
            query_mask, as_tuple=False
        ).detach().cpu().tolist()
    )
    stale_by_query: dict[tuple[int, int], set[int]] = {
        coordinate: set() for coordinate in query_coordinates
    }
    if "stale_positions" in stale_fields:
        stale_positions = batch.get("stale_positions")
        if (
            not isinstance(stale_positions, torch.Tensor)
            or stale_positions.dtype != torch.int64
            or stale_positions.ndim != 2
            or stale_positions.shape[1] != 3
        ):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "stale_positions must be int64 [N,3] rows of batch/query/stale positions",
            )
        seen: set[tuple[int, int, int]] = set()
        for row in stale_positions.detach().cpu().tolist():
            batch_index, token_index, stale_position = (int(value) for value in row)
            triple = (batch_index, token_index, stale_position)
            coordinate = (batch_index, token_index)
            if triple in seen or coordinate not in stale_by_query:
                raise QwenRuntimeConfigurationError(
                    "cache_annotations_invalid",
                    "stale_positions must uniquely label declared query positions",
                )
            seen.add(triple)
            stale_by_query[coordinate].add(stale_position)
    else:
        stale_mask = batch.get("stale_mask")
        if (
            not isinstance(stale_mask, torch.Tensor)
            or stale_mask.dtype != torch.bool
            or stale_mask.shape != (batch_size, steps, steps)
        ):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid", "stale_mask must be bool [B,T,T]"
            )
        if steps > 4096:
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "dense stale_mask is unsupported above 4096 tokens; use stale_positions",
            )
        if bool(stale_mask[~query_mask].any()):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "non-query positions cannot carry stale labels",
            )
        for coordinate in query_coordinates:
            batch_index, token_index = coordinate
            stale_by_query[coordinate].update(
                int(value)
                for value in torch.nonzero(
                    stale_mask[batch_index, token_index], as_tuple=False
                ).flatten().detach().cpu().tolist()
            )
    for batch_index, token_index in query_coordinates:
        start, stop = (
            int(value) for value in source_spans[batch_index, token_index]
        )
        stale = stale_by_query[(batch_index, token_index)]
        if token_index < 1 or not 0 <= start < stop <= token_index:
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "query source spans must be causal and nonempty",
            )
        if any(
            position < 0
            or position >= token_index
            or start <= position < stop
            for position in stale
        ):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "stale labels must be causal and disjoint from the gold source span",
            )
    frozen_stale = {
        coordinate: frozenset(values)
        for coordinate, values in stale_by_query.items()
    }

    if not is_ruler:
        return query_mask, source_spans, frozen_stale, ()
    raw_metadata = batch.get("ruler_metadata")
    if (
        not isinstance(raw_metadata, tuple)
        or len(raw_metadata) != batch_size
        or any(not isinstance(item, Mapping) for item in raw_metadata)
    ):
        raise QwenRuntimeConfigurationError(
            "ruler_annotations_missing",
            "RULER evaluation requires one ruler_metadata mapping per example",
        )
    from .tasks.ruler import RULER_DEPTH_STRATA, RulerCell, RulerEpisode

    example_ids = _batch_example_ids(batch)
    validated: list[tuple[dict[str, object], object]] = []
    for batch_index, raw in enumerate(raw_metadata):
        metadata = copy.deepcopy(dict(raw))
        if set(metadata) != _RULER_METADATA_FIELDS:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid",
                "ruler_metadata fields must exactly match the production schema",
            )
        if metadata["evaluation_mode"] not in {"teacher_forced", "free_generation"}:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid",
                "RULER evaluation_mode must be teacher_forced or free_generation",
            )
        if metadata["evidence_scope"] not in {"feasibility", "promotion"}:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER evidence_scope is invalid"
            )
        if metadata["seed"] != job.get("seed") or metadata["example_id"] != example_ids[batch_index]:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER seed/example identity is mismatched"
            )
        if metadata["depth_stratum"] not in RULER_DEPTH_STRATA:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER depth_stratum is invalid"
            )
        if not isinstance(metadata["paired_interval"], Mapping):
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER paired_interval must be annotated"
            )
        cell = RulerCell(
            metadata["context_length"], metadata["needles"], metadata["queries"]
        )
        if metadata["cell_id"] != cell.cell_id:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER cell identity is inconsistent"
            )
        try:
            episode = RulerEpisode(
                episode_id=metadata["episode_id"],
                seed=metadata["seed"],
                example_index=metadata["example_index"],
                cell=cell,
                input_ids=tuple(int(value) for value in input_ids[batch_index].cpu()),
                prompt_end=metadata["prompt_end"],
                answers=tuple(metadata["answers"]),
                answer_token_ids=tuple(tuple(values) for values in metadata["answer_token_ids"]),
                answer_spans=tuple(tuple(values) for values in metadata["answer_spans"]),
                source_spans=tuple(tuple(values) for values in metadata["source_spans"]),
                depth_strata=tuple(metadata["depth_strata"]),
                query_keys=tuple(metadata["query_keys"]),
            )
        except (TypeError, ValueError) as error:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER episode metadata is malformed"
            ) from error
        expected_target_digest = _json_digest(
            [list(values) for values in episode.answer_token_ids]
        )
        if metadata["target_digest"] != expected_target_digest:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER target_digest is inconsistent"
            )
        expected_queries = torch.zeros(steps, dtype=torch.bool, device=query_mask.device)
        expected_spans = torch.full(
            (steps, 2), -1, dtype=torch.int64, device=source_spans.device
        )
        for answer_span, source_span in zip(episode.answer_spans, episode.source_spans):
            answer_start, answer_stop = answer_span
            expected_queries[answer_start:answer_stop] = True
            expected_spans[answer_start:answer_stop] = torch.tensor(
                source_span, dtype=torch.int64, device=source_spans.device
            )
        if not torch.equal(query_mask[batch_index], expected_queries) or not torch.equal(
            source_spans[batch_index], expected_spans
        ):
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid",
                "RULER query_mask/source_spans do not match answer/source spans",
            )
        validated.append((metadata, episode))
    return query_mask, source_spans, frozen_stale, tuple(validated)


def _validate_state_tracking_annotations(
    batch: Mapping[str, object], *, job: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    raw = batch.get("state_tracking_metadata")
    batch_size = int(batch["input_ids"].shape[0])
    if (not isinstance(raw, tuple) or len(raw) != batch_size
            or any(not isinstance(item, Mapping) for item in raw)):
        raise QwenRuntimeConfigurationError(
            "state_tracking_annotations_missing", "one state-tracking metadata row is required per example"
        )
    example_ids = _batch_example_ids(batch)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        row = copy.deepcopy(dict(item))
        if set(row) != _STATE_TRACKING_METADATA_FIELDS:
            raise QwenRuntimeConfigurationError(
                "state_tracking_annotations_invalid", "state-tracking metadata fields must be exact"
            )
        if row["task"] not in {"parity", "modular"} or row["seed"] != job.get("seed"):
            raise QwenRuntimeConfigurationError("state_tracking_annotations_invalid", "task or seed mismatch")
        if row["example_id"] != example_ids[index] or type(row["cell_id"]) is not str:
            raise QwenRuntimeConfigurationError("state_tracking_annotations_invalid", "example or cell mismatch")
        if type(row["prompt_end"]) is not int or not 0 < row["prompt_end"] <= batch["input_ids"].shape[1]:
            raise QwenRuntimeConfigurationError("state_tracking_annotations_invalid", "prompt_end is invalid")
        targets = row["targets"]
        if not isinstance(targets, (list, tuple)) or not targets or any(type(value) is not int for value in targets):
            raise QwenRuntimeConfigurationError("state_tracking_annotations_invalid", "targets must be nonempty ints")
        if row["task"] == "modular" and (type(row["modulus"]) is not int or row["modulus"] < 2):
            raise QwenRuntimeConfigurationError("state_tracking_annotations_invalid", "modulus is invalid")
        if row["task"] == "parity" and row["modulus"] is not None:
            raise QwenRuntimeConfigurationError("state_tracking_annotations_invalid", "parity modulus must be null")
        if row["evidence_scope"] not in {"feasibility", "promotion"}:
            raise QwenRuntimeConfigurationError("state_tracking_annotations_invalid", "evidence scope is invalid")
        rows.append(row)
    return tuple(rows)


def _default_evaluate(
    *,
    loaded_arm: object,
    data: QwenJobData,
    job: Mapping[str, object],
    runtime: Mapping[str, object],
    amplitude_initial: Sequence[float] = (),
    generate_answers: Callable[..., Sequence[str]] | None = None,
    generate_state_values: Callable[..., Sequence[int]] | None = None,
    tokenizer_asset: object | None = None,
    **_kwargs: object,
) -> dict[str, object]:
    from gdn3.kmd2_native import KMD2NativeAttn
    from .qwen_exact_cache import KMD2ExactCacheAttn, guarded_model_forward
    from .tasks.ruler import score_free_generation, score_state_tracking, score_teacher_forced

    model = loaded_arm.model
    task_name = _required_mapping(
        _job_config(job).get("task"), "canonical_config.task"
    ).get("name")
    cache_layers = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, KMD2ExactCacheAttn)
    )
    require_cache = loaded_arm.arm != "native"
    native_layers = tuple(
        module for module in model.modules() if isinstance(module, KMD2NativeAttn)
    )
    state_elements = sum(module.H * module.dk * module.dv for module in native_layers)
    if state_elements < 1:
        config = _job_config(job)
        model_config = _required_mapping(config.get("model", {}), "canonical_config.model")
        dimensions = ("num_layers", "num_heads", "state_key_dim", "state_value_dim")
        if all(type(model_config.get(name)) is int for name in dimensions):
            state_elements = math.prod(int(model_config[name]) for name in dimensions)

    losses: list[float] = []
    correct = total = 0
    evaluations: list[dict[str, object]] = []
    selected_indices = _StreamingTensorSequence(
        dtype=torch.int64,
        label="cache-selected-indices",
        sample_size=32,
    )
    scores = _StreamingScoreStatistics()
    persistent_hits = _OnlineMean()
    conditional_correct = _OnlineMean()
    sinks = _OnlineMean()
    entropies = _OnlineMean()
    top1_masses = _OnlineMean()
    stale_flags = _OnlineMean()
    stale_errors = _OnlineMean()
    cache_norms = _OnlineMean()
    state_norms = _OnlineMean()
    retention_count = eviction_count = persistent_bytes = block_bytes = 0
    streamed_projection_batches = 0
    dense_projection_batches = 0
    streamed_chunk_tokens = 0
    streamed_peak_logit_bytes = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for raw_batch in data.eval_microbatches:
                state_rows: tuple[dict[str, object], ...] = ()
                if task_name in {"parity", "modular"}:
                    state_rows = _validate_state_tracking_annotations(raw_batch, job=job)
                    input_shape = raw_batch["input_ids"].shape
                    query_mask = torch.zeros(input_shape, dtype=torch.bool)
                    source_spans = torch.full((*input_shape, 2), -1, dtype=torch.int64)
                    stale_by_query = {}
                    ruler_rows = ()
                else:
                    query_mask, source_spans, stale_by_query, ruler_rows = _validate_evaluation_annotations(
                        raw_batch, job=job, require_cache=require_cache,
                    )
                if require_cache and not cache_layers:
                    raise QwenRuntimeConfigurationError(
                        "cache_diagnostics_unavailable",
                        "cache arm contains no installed KMD2 exact-cache layers",
                    )
                metric_names = (
                    "hits",
                    "conditional",
                    "sink",
                    "entropy",
                    "top1_mass",
                    "stale",
                    "stale_error",
                )
                row_accumulators = [
                    {name: _OnlineMean() for name in metric_names}
                    for _ in range(query_mask.shape[0])
                ]
                stale_error_records: list[tuple[int, int, bool]] = []
                layer_streams: dict[str, dict[str, object]] = {}

                def make_observer(layer_name: str):
                    nonlocal retention_count, eviction_count
                    state: dict[str, object] = {
                        "next_start": 0,
                        "blocks": 0,
                        "block_peak": 0,
                        "pending": None,
                    }
                    layer_streams[layer_name] = state

                    def observe(block: object) -> None:
                        nonlocal retention_count, eviction_count
                        block_start = getattr(block, "block_start", None)
                        block_stop = getattr(block, "block_stop", None)
                        if (
                            type(block_start) is not int
                            or type(block_stop) is not int
                            or block_start != state["next_start"]
                            or block_stop <= block_start
                        ):
                            raise QwenRuntimeConfigurationError(
                                "cache_diagnostics_invalid",
                                f"cache layer {layer_name} streamed block identity drifted",
                            )
                        state["next_start"] = block_stop
                        state["blocks"] = int(state["blocks"]) + 1
                        scores.add(block.update_scores)
                        cache_norms.add_tensor(block.cache_output_norm)
                        state_norms.add_tensor(block.state_output_norm)
                        sinks.add_tensor(block.sink_mass)
                        entropies.add_tensor(block.attention_entropy)
                        top1_masses.add_tensor(block.top1_mass)
                        if type(block.block_bytes) is not int or block.block_bytes < 0:
                            raise QwenRuntimeConfigurationError(
                                "cache_diagnostics_invalid",
                                "cache block byte counts must be nonnegative integers",
                            )
                        state["block_peak"] = max(
                            int(state["block_peak"]), block.block_bytes
                        )
                        prior_valid = int(
                            (block.persistent_selected_positions >= 0).sum()
                        )
                        batch_count, _, head_count = block.top1_positions.shape
                        incoming = (
                            batch_count * (block_stop - block_start) * head_count
                        )
                        pending = state["pending"]
                        if pending is not None:
                            previous_prior, previous_incoming = pending
                            retention_count += prior_valid
                            eviction_count += (
                                previous_prior + previous_incoming - prior_valid
                            )
                        state["pending"] = (prior_valid, incoming)

                        for (batch_index, target_position), stale in stale_by_query.items():
                            read_position = target_position - 1
                            if not block_start <= read_position < block_stop:
                                continue
                            local = read_position - block_start
                            source_start, source_stop = (
                                int(value)
                                for value in source_spans[
                                    batch_index, target_position
                                ]
                            )
                            gold = set(range(source_start, source_stop))
                            persistent_rows = block.persistent_selected_positions[
                                batch_index
                            ].detach().cpu()
                            top1_rows = block.top1_positions[
                                batch_index, local
                            ].detach().cpu()
                            candidate_rows = block.candidate_positions[
                                batch_index, local
                            ].detach().cpu()
                            candidate_valid_rows = block.candidate_valid[
                                batch_index, local
                            ].detach().cpu()
                            sink_rows = block.sink_mass[
                                batch_index, local
                            ].detach().float().cpu()
                            entropy_rows = block.attention_entropy[
                                batch_index, local
                            ].detach().float().cpu()
                            top1_mass_rows = block.top1_mass[
                                batch_index, local
                            ].detach().float().cpu()
                            accumulator = row_accumulators[batch_index]
                            for head in range(top1_rows.shape[0]):
                                persistent = {
                                    int(value)
                                    for value in persistent_rows[head].tolist()
                                    if value >= 0
                                }
                                hit = bool(persistent & gold)
                                top1 = int(top1_rows[head])
                                candidates = [
                                    int(value)
                                    for value in candidate_rows[head][
                                        candidate_valid_rows[head]
                                    ].tolist()
                                    if value >= 0
                                ]
                                stale_count = sum(
                                    value in stale for value in candidates
                                )
                                persistent_hits.add(float(hit))
                                accumulator["hits"].add(float(hit))
                                if hit:
                                    conditional = float(top1 in gold)
                                    conditional_correct.add(conditional)
                                    accumulator["conditional"].add(conditional)
                                stale_flags.add(
                                    float(stale_count), count=len(candidates)
                                )
                                accumulator["stale"].add(
                                    float(stale_count), count=len(candidates)
                                )
                                accumulator["sink"].add(float(sink_rows[head]))
                                accumulator["entropy"].add(
                                    float(entropy_rows[head])
                                )
                                accumulator["top1_mass"].add(
                                    float(top1_mass_rows[head])
                                )
                                stale_error_records.append(
                                    (batch_index, target_position, top1 in stale)
                                )

                    return observe

                for name, layer in cache_layers:
                    layer.set_cache_diagnostic_observer(
                        make_observer(name), retain_full=False
                    )
                try:
                    batch = _move_batch(raw_batch, runtime["student_device"])
                    inputs = {
                        name: value
                        for name, value in batch.items()
                        if name not in {"labels", "example_ids"} | _EVALUATION_ANNOTATION_FIELDS
                    }
                    inputs.update({"output_hidden_states": False, "use_cache": False})
                    labels = batch["labels"]
                    assert isinstance(labels, torch.Tensor)
                    streamed_scores = _stream_causal_scores(
                        model, inputs=inputs, labels=labels
                    )
                    if streamed_scores is None:
                        output = guarded_model_forward(model, **inputs)
                    else:
                        output = None
                finally:
                    for _, layer in cache_layers:
                        layer.set_cache_diagnostic_observer(None)
                if streamed_scores is None:
                    logits = _validate_logits(
                        "evaluation logits", _output_field(output, "logits")
                    )
                    loss = causal_cross_entropy(logits, labels)
                    shifted_predictions = logits[:, :-1].argmax(dim=-1)
                    targets = labels[:, 1:]
                    valid = targets != -100
                    batch_correct = int(
                        ((shifted_predictions == targets) & valid).sum().cpu()
                    )
                    batch_total = int(valid.sum().cpu())
                    aligned_predictions = torch.zeros_like(labels)
                    aligned_predictions[:, 1:] = shifted_predictions
                    dense_projection_batches += 1
                else:
                    loss = streamed_scores.loss
                    batch_correct = streamed_scores.correct
                    batch_total = streamed_scores.total
                    aligned_predictions = streamed_scores.aligned_predictions
                    streamed_projection_batches += 1
                    streamed_chunk_tokens = max(
                        streamed_chunk_tokens, streamed_scores.chunk_tokens
                    )
                    streamed_peak_logit_bytes = max(
                        streamed_peak_logit_bytes, streamed_scores.peak_logit_bytes
                    )
                if not bool(torch.isfinite(loss)):
                    raise QwenTrainingError("nonfinite_loss", "evaluation loss is nonfinite")
                losses.append(float(loss.cpu()))
                correct += batch_correct
                total += batch_total
                layer_persistent_bytes: list[object] = []
                layer_block_bytes: list[object] = []
                for layer_name, layer in cache_layers:
                    diagnostics = layer.last_cache_diagnostics
                    stream = layer_streams[layer_name]
                    if (
                        diagnostics is None
                        or hasattr(diagnostics, "blocks")
                        or getattr(diagnostics, "blocks_processed", None)
                        != stream["blocks"]
                    ):
                        raise QwenRuntimeConfigurationError(
                            "cache_diagnostics_invalid",
                            f"cache layer {layer_name} omitted bounded synchronized diagnostics",
                        )
                    pending = stream["pending"]
                    if pending is None:
                        raise QwenRuntimeConfigurationError(
                            "cache_diagnostics_invalid",
                            f"cache layer {layer_name} streamed no blocks",
                        )
                    previous_prior, previous_incoming = pending
                    final_valid = int(diagnostics.final_selected_valid.sum())
                    retention_count += final_valid
                    eviction_count += previous_prior + previous_incoming - final_valid
                    selected_indices.add(
                        diagnostics.final_selected_positions[
                            diagnostics.final_selected_valid
                        ]
                    )
                    layer_persistent_bytes.append(diagnostics.persistent_bytes)
                    layer_block_bytes.append(stream["block_peak"])

                for batch_index, target_position, top1_is_stale in stale_error_records:
                    wrong = bool(
                        aligned_predictions[batch_index, target_position].cpu()
                        != labels[batch_index, target_position].cpu()
                    )
                    stale_error = float(wrong and top1_is_stale)
                    stale_errors.add(stale_error)
                    row_accumulators[batch_index]["stale_error"].add(stale_error)

                persistent_bytes = max(
                    persistent_bytes,
                    _checked_byte_sum("persistent", layer_persistent_bytes),
                )
                block_bytes = max(
                    block_bytes,
                    _checked_byte_sum("block", layer_block_bytes),
                )

                for batch_index, (metadata, episode) in enumerate(ruler_rows):
                    if metadata["evaluation_mode"] == "free_generation":
                        if not callable(generate_answers) or tokenizer_asset is None:
                            raise QwenRuntimeConfigurationError(
                                "free_generation_unavailable",
                                "free-generation RULER requires tokenizer and generation implementation",
                            )
                        generated = generate_answers(
                            model=model, episode=episode, tokenizer_asset=tokenizer_asset,
                            device=runtime["student_device"],
                        )
                        score = score_free_generation(episode, generated)
                    else:
                        score = score_teacher_forced(
                            episode,
                            [int(value) for value in aligned_predictions[batch_index].cpu()],
                        )
                    if require_cache:
                        accumulator = row_accumulators[batch_index]
                        cache_diagnostics: dict[str, object] = {
                            "active": True,
                            "persistent_hit": accumulator["hits"].mean(),
                            "persistent_hit_count": accumulator["hits"].count,
                            "conditional_read": accumulator["conditional"].mean(),
                            "conditional_read_count": accumulator["conditional"].count,
                            "sink_mass": accumulator["sink"].mean(),
                            "attention_entropy": accumulator["entropy"].mean(),
                            "top1_mass": accumulator["top1_mass"].mean(),
                            "stale_occupancy": accumulator["stale"].mean(),
                            "stale_error": accumulator["stale_error"].mean(),
                        }
                    else:
                        cache_diagnostics = {"active": False}
                    evaluations.append(
                        {
                            "task": "ruler",
                            "cell_id": metadata["cell_id"],
                            "context_length": metadata["context_length"],
                            "needles": metadata["needles"],
                            "queries": metadata["queries"],
                            "depth_stratum": metadata["depth_stratum"],
                            "example_id": metadata["example_id"],
                            "episode_id": metadata["episode_id"],
                            "evaluation_mode": score.evaluation_mode,
                            "evidence_scope": metadata["evidence_scope"],
                            "numerator": score.numerator,
                            "denominator": score.denominator,
                            "episode_exact": score.episode_exact,
                            "source_spans": [list(span) for span in episode.source_spans],
                            "target_digest": metadata["target_digest"],
                            "cache_diagnostics": cache_diagnostics,
                            "paired_interval": copy.deepcopy(dict(metadata["paired_interval"])),
                            "seed": job["seed"],
                            "arm_id": job["arm_id"],
                        }
                    )
                for batch_index, metadata in enumerate(state_rows):
                    if not callable(generate_state_values) or tokenizer_asset is None:
                        raise QwenRuntimeConfigurationError(
                            "state_tracking_generation_unavailable",
                            "state tracking requires tokenizer and generation implementation",
                        )
                    targets_row = tuple(int(value) for value in metadata["targets"])
                    predictions = generate_state_values(
                        model=model, input_ids=batch["input_ids"][batch_index],
                        prompt_end=metadata["prompt_end"], count=len(targets_row),
                        tokenizer_asset=tokenizer_asset, device=runtime["student_device"],
                    )
                    score = score_state_tracking(
                        metadata["task"], predictions, targets_row,
                        modulus=metadata["modulus"], lm_loss=float(loss.cpu()),
                    )
                    evaluations.append({
                        "task": "state_tracking", "state_task": score.task,
                        "cell_id": metadata["cell_id"], "example_id": metadata["example_id"],
                        "seed": metadata["seed"], "evaluation_mode": "free_generation",
                        "evidence_scope": metadata["evidence_scope"],
                        "numerator": sum(score.correct), "denominator": len(score.correct),
                        "episode_exact": all(score.correct), "lm_loss": score.lm_loss,
                        "modulus": score.modulus, "predictions": list(predictions),
                        "targets": list(targets_row), "arm_id": job["arm_id"],
                    })
    finally:
        model.train(was_training)

    if not losses:
        raise QwenRuntimeConfigurationError(
            "evaluation_invalid", "Qwen evaluation requires at least one batch"
        )
    result: dict[str, object] = {
        "metrics": {
            "eval_loss": math.fsum(losses) / len(losses),
            "token_accuracy": correct / max(1, total),
        },
        "recurrent_state": {"elements": state_elements, "bytes": 4 * state_elements},
        "evaluation_execution": {
            "full_context_backbone": True,
            "streamed_vocabulary_projection_batches": streamed_projection_batches,
            "dense_vocabulary_projection_batches": dense_projection_batches,
            "streamed_chunk_tokens": streamed_chunk_tokens,
            "peak_streamed_logit_bytes": streamed_peak_logit_bytes,
            "logit_workspace_limit_bytes": _EVALUATION_LOGIT_WORKSPACE_BYTES,
        },
    }
    if evaluations:
        result["evaluations"] = evaluations
    if require_cache:
        initial = tuple(float(value) for value in amplitude_initial)
        final = _cache_amplitudes(model)
        if not initial or len(initial) != len(final):
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                "cache amplitude initial/final measurements are incomplete",
            )
        cache_config = cache_layers[0][1].cache_config
        result["exact_cache"] = {
            "width": cache_config.width,
            "block_size": cache_config.block_size,
            "score_definition": cache_config.score,
            "compute_dtype": cache_config.compute_dtype,
            "storage_dtype": cache_config.storage_dtype,
            "coordinate_frame": cache_config.coordinate_frame,
            "inclusive_causality": cache_config.inclusive,
            "tie_policy": cache_config.tie_policy,
            "amplitude_initial": list(initial),
            "amplitude_final": list(final),
            "selected_index_digest": selected_indices.digest(),
            "selected_index_sample": selected_indices.sample,
            "score_digest": scores.sequence.digest(),
            "score_statistics": scores.as_dict(),
            "retention_count": retention_count,
            "eviction_count": eviction_count,
            "persistent_hit_rate": persistent_hits.mean(),
            "conditional_read_accuracy": conditional_correct.mean(),
            "sink_mass": sinks.mean(),
            "attention_entropy": entropies.mean(),
            "top1_mass": top1_masses.mean(),
            "stale_occupancy": stale_flags.mean(),
            "stale_error": stale_errors.mean(),
            "cache_output_norm": cache_norms.mean(),
            "state_output_norm": state_norms.mean(),
            "persistent_bytes": persistent_bytes,
            "block_bytes": block_bytes,
            "implementation_paths": {
                "scan": "gdn3.kmd2_native.KMD2NativeAttn.forward",
                "score": (
                    "qwen_backend.KMD2RecencyCacheAttn.position"
                    if cache_config.score == "recency"
                    else "qwen_exact_cache.KMD2ExactCacheAttn._native_state_and_scores"
                ),
                "selection": "exact_cache.merge_persistent_cache.deterministic_topw",
                "read": f"exact_cache.cache_read_blocks.{cache_config.read}",
            },
        }
    return result


def _default_peak_vram_bytes(device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(torch.device(device)))
    return 0


def _default_reset_peak_vram(device: str) -> None:
    """Start one job's peak window at its fully loaded resident baseline."""
    if device.startswith("cuda") and torch.cuda.is_available():
        resolved = torch.device(device)
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)


@dataclass(frozen=True)
class QwenExecutionDependencies:
    """Injectable heavy boundaries used by :func:`execute_job`."""

    load_arm: Callable[..., object]
    load_teacher: Callable[..., torch.nn.Module]
    load_data: Callable[..., QwenJobData]
    build_optimizer: Callable[..., torch.optim.Optimizer]
    build_scheduler: Callable[..., object]
    load_checkpoint: Callable[..., object]
    save_checkpoint: Callable[..., Path]
    evaluate: Callable[..., Mapping[str, object]]
    monotonic: Callable[[], float]
    reset_peak_vram: Callable[[str], None]
    peak_vram_bytes: Callable[[str], int]
    generate_answers: Callable[..., Sequence[str]]
    generate_state_values: Callable[..., Sequence[int]]
    build_grad_scaler: Callable[..., object] | None = None


def _default_build_grad_scaler(*, device: str | torch.device, dtype: torch.dtype) -> object:
    resolved = torch.device(device)
    enabled = resolved.type == "cuda" and dtype == torch.float16
    return torch.amp.GradScaler(resolved.type, enabled=enabled)


@lru_cache(maxsize=4)
def _cached_auto_tokenizer(path: str) -> object:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(path)


def _default_generate_answers(*, model: torch.nn.Module, episode: object,
                              tokenizer_asset: object, device: str) -> tuple[str, ...]:
    """Generate and segment the numeric answers used by the pinned RULER format."""
    tokenizer = _cached_auto_tokenizer(str(tokenizer_asset.path))
    prompt = torch.tensor([episode.input_ids[:episode.prompt_end]], dtype=torch.long, device=device)
    max_tokens = sum(len(tokens) for tokens in episode.answer_token_ids) + 4 * episode.cell.queries
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            # The hybrid architecture deliberately has no cross-call state
            # cache.  Qwen's one-token logit projection keeps generation from
            # constructing a [context,vocab] tensor at every decode step.
            use_cache=False,
            logits_to_keep=1,
        )
    text = tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True)
    answers = tuple(re.findall(r"(?<!\d)\d{7}(?!\d)", text))
    if len(answers) < episode.cell.queries:
        answers = answers + tuple("" for _ in range(episode.cell.queries - len(answers)))
    return answers[:episode.cell.queries]


def _default_generate_state_values(*, model: torch.nn.Module, input_ids: torch.Tensor,
                                   prompt_end: int, count: int, tokenizer_asset: object,
                                   device: str) -> tuple[int, ...]:
    tokenizer = _cached_auto_tokenizer(str(tokenizer_asset.path))
    prompt = input_ids[:prompt_end].to(device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=max(8, count * 4),
            do_sample=False,
            use_cache=False,
            logits_to_keep=1,
        )
    text = tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True)
    values = tuple(int(value) for value in re.findall(r"-?\d+", text)[:count])
    if len(values) != count:
        raise QwenRuntimeConfigurationError(
            "state_tracking_generation_invalid", "generated output did not contain every state value"
        )
    return values


def _default_dependencies() -> QwenExecutionDependencies:
    from .qwen_backend import load_qwen_arm
    from .qwen_checkpoint import load_qwen_checkpoint, save_qwen_checkpoint

    return QwenExecutionDependencies(
        load_arm=load_qwen_arm,
        load_teacher=_default_load_teacher,
        load_data=_default_load_data,
        build_optimizer=build_qwen_heal_optimizer,
        build_scheduler=_default_build_scheduler,
        load_checkpoint=load_qwen_checkpoint,
        save_checkpoint=save_qwen_checkpoint,
        evaluate=_default_evaluate,
        monotonic=time.monotonic,
        reset_peak_vram=_default_reset_peak_vram,
        peak_vram_bytes=_default_peak_vram_bytes,
        generate_answers=_default_generate_answers,
        generate_state_values=_default_generate_state_values,
        build_grad_scaler=_default_build_grad_scaler,
    )


def _resolve_dependencies(
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None,
) -> QwenExecutionDependencies:
    defaults = _default_dependencies()
    if dependencies is None:
        return defaults
    if isinstance(dependencies, QwenExecutionDependencies):
        return dependencies
    if not isinstance(dependencies, Mapping):
        raise TypeError("dependencies must be QwenExecutionDependencies, a mapping, or None")
    fields = tuple(QwenExecutionDependencies.__dataclass_fields__)
    unknown = set(dependencies) - set(fields)
    if unknown:
        raise ValueError("unknown Qwen dependencies: " + ", ".join(sorted(unknown)))
    values = {
        name: dependencies.get(name, getattr(defaults, name)) for name in fields
    }
    if any(not callable(value) for value in values.values()):
        raise TypeError("every Qwen execution dependency must be callable")
    return QwenExecutionDependencies(**values)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "research/kmd2_ablation/architecture.py",
        "research/kmd2_ablation/config.py",
        "research/kmd2_ablation/exact_cache.py",
        "research/kmd2_ablation/qwen_backend.py",
        "research/kmd2_ablation/qwen_architecture.py",
        "research/kmd2_ablation/qwen_checkpoint.py",
        "research/kmd2_ablation/qwen_exact_cache.py",
        "research/kmd2_ablation/qwen_fused_loss.py",
        "research/kmd2_ablation/qwen_gdn2_triton.py",
        "research/kmd2_ablation/qwen_hybrid_chunkwise.py",
        "research/kmd2_ablation/qwen_hybrid_components.py",
        "research/kmd2_ablation/qwen_hybrid_four_state.py",
        "research/kmd2_ablation/qwen_hybrid_hola.py",
        "research/kmd2_ablation/qwen_hybrid_liger_chunked.py",
        "research/kmd2_ablation/qwen_hybrid_liger_dplr.py",
        "research/kmd2_ablation/qwen_hybrid_liger_wy.py",
        "research/kmd2_ablation/qwen_hybrid_math.py",
        "research/kmd2_ablation/qwen_hybrid_shared.py",
        "research/kmd2_ablation/qwen_hybrid_triton.py",
        "research/kmd2_ablation/qwen_training.py",
        "research/kmd2_ablation/qwen_variants.py",
        "research/kmd2_ablation/results.py",
        "research/kmd2_ablation/runner.py",
        "research/kmd2_ablation/tasks/ruler.py",
        "research/kmd2_ablation/variants.py",
        "gdn3/_reference_recurrence.py",
        "gdn3/gdn3_upgrade.py",
        "gdn3/kmd2_fast_scan.py",
        "gdn3/kmd2_native.py",
    )
    result: dict[str, str] = {}
    for name in relative_paths:
        path = root / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def _identity_record(asset: object) -> dict[str, object]:
    return {
        "kind": asset.kind,
        "size_bytes": asset.size_bytes,
        "sha256": asset.sha256,
    }


def _relevant_cuda_rng_devices(runtime: Mapping[str, object]) -> tuple[int, ...]:
    if not torch.cuda.is_available():
        return ()
    devices: set[int] = set()
    for name in ("student_device", "teacher_device"):
        raw = runtime.get(name)
        if type(raw) is not str:
            continue
        try:
            device = torch.device(raw)
        except (TypeError, RuntimeError):
            continue
        if device.type != "cuda":
            continue
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise QwenRuntimeConfigurationError(
                "runtime_configuration_invalid",
                f"runtime.{name} names unavailable CUDA device {index}",
            )
        devices.add(index)
    return tuple(sorted(devices))


@contextmanager
def _scoped_paired_rng(seed: int, runtime: Mapping[str, object]):
    """Seed one job transaction and restore caller RNGs on every exit path."""
    python_state = random.getstate()
    cuda_devices = _relevant_cuda_rng_devices(runtime)
    try:
        with torch.random.fork_rng(devices=list(cuda_devices), enabled=True):
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if torch.cuda.is_available():
                for device_index in cuda_devices:
                    torch.cuda.default_generators[device_index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)


def _true_mimo_resources(
    *, rank: int, layers: int, batch_size: int, sequence_length: int,
    heads: int, key_dim: int, value_dim: int, native_conv_parameters: int,
) -> dict[str, int]:
    """Exact independent resource contract for genuine rank-R MIMO."""
    if rank not in (2, 4):
        raise ValueError("true_mimo_resource_rank_invalid")
    values = (layers, batch_size, sequence_length, heads, key_dim, value_dim)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("true_mimo_resource_dimension_invalid")
    if type(native_conv_parameters) is not int or native_conv_parameters < 0:
        raise ValueError("true_mimo_native_conv_parameters_invalid")
    new_per_layer = 2 * heads * rank * key_dim * key_dim + 3 * heads * rank * value_dim
    state_elements = batch_size * heads * key_dim * value_dim
    activation_elements = (
        batch_size * sequence_length * heads * rank * (2 * key_dim + 3 * value_dim)
    )
    return {
        "new_parameters_per_layer": new_per_layer,
        "new_parameters": layers * new_per_layer,
        "recurrent_state_elements": state_elements,
        "recurrent_state_bytes": state_elements * 4,
        "rankwise_live_activation_elements": activation_elements,
        "rankwise_live_activation_bytes": activation_elements * 4,
        "native_conv_parameters": native_conv_parameters,
    }


def _shared_query_widening_resources(
    *, width: int, layers: int, batch_size: int, sequence_length: int,
    heads: int, key_dim: int, value_dim: int, max_tokens: int | None = None,
) -> dict[str, int]:
    """Independent resource contract for shared-query output widening."""
    if type(width) is not int or width != 4:
        raise ValueError("shared_query_widening_resource_width_invalid")
    values = (layers, batch_size, sequence_length, heads, key_dim, value_dim)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("shared_query_widening_resource_dimension_invalid")
    if max_tokens is None:
        max_tokens = batch_size * sequence_length
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("shared_query_widening_resource_dimension_invalid")
    new_per_layer = heads * width * key_dim + heads * width
    state_elements = batch_size * heads * key_dim * value_dim
    rank_reads = max_tokens * heads * width * value_dim
    q_slots = max_tokens * heads * width * key_dim
    extra = max_tokens * heads * (width - 1) * (key_dim + value_dim)
    return {
        "new_parameters_per_layer": new_per_layer,
        "new_parameters": layers * new_per_layer,
        "recurrent_state_elements": state_elements,
        "recurrent_state_bytes": state_elements * 4,
        "total_rank_read_elements": rank_reads,
        "total_q_slot_elements": q_slots,
        "extra_vs_r1_elements": extra,
        "extra_vs_r1_bytes": extra * 4,
    }


def _widening_resource_shape_maxima(
    shapes: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """Return independent maximum batch size and maximum token count."""
    if (not shapes or any(type(shape) is not tuple or len(shape) != 2
                          or any(type(value) is not int or value < 1 for value in shape)
                          for shape in shapes)):
        raise ValueError("shared_query_widening_resource_shape_invalid")
    return max(shape[0] for shape in shapes), max(shape[0] * shape[1] for shape in shapes)


def _architecture_new_state_resources(
    modules: tuple[torch.nn.Module, ...],
) -> dict[str, int]:
    """Count manifest-declared new parameters and persistent buffers exactly."""
    parameter_elements = buffer_elements = buffer_bytes = 0
    for module in modules:
        manifest = module.transformation_manifest()
        # Incremental arms declare "new" inline; the maximum hybrids declare
        # theirs in the separate architecture_tensor_manifest() (their
        # transformation manifest describes the component replication).
        if not isinstance(manifest, Mapping) or "new" not in manifest:
            tensor_manifest = getattr(module, "architecture_tensor_manifest", None)
            manifest = tensor_manifest() if callable(tensor_manifest) else manifest
        if not isinstance(manifest, Mapping) or type(manifest.get("new")) not in (tuple, list):
            raise QwenRuntimeConfigurationError(
                "architecture_tensor_manifest_invalid", "architecture new-state manifest is malformed"
            )
        parameters, buffers = dict(module.named_parameters()), dict(module.named_buffers())
        for name in manifest["new"]:
            in_parameter, in_buffer = name in parameters, name in buffers
            if in_parameter == in_buffer:
                raise QwenRuntimeConfigurationError(
                    "architecture_tensor_manifest_invalid",
                    "manifest new state must identify exactly one parameter or buffer",
                )
            if in_parameter:
                parameter_elements += parameters[name].numel()
            else:
                tensor = buffers[name]
                buffer_elements += tensor.numel()
                buffer_bytes += tensor.numel() * tensor.element_size()
    return {
        "new_parameters": parameter_elements,
        "architecture_new_buffer_elements": buffer_elements,
        "architecture_new_buffer_bytes": buffer_bytes,
    }


def _hybrid_module_resources(
    modules: tuple[torch.nn.Module, ...], *, optimizer: torch.optim.Optimizer,
    batch_size: int, sequence_length: int, checkpointing: bool,
    resident_model: torch.nn.Module | None = None,
    data_parallel: int = 1,
) -> dict[str, object]:
    """Account Package A/B resources from installed modules and live optimizer slots."""
    if not modules or min(batch_size, sequence_length) < 1:
        raise QwenRuntimeConfigurationError("hybrid_resource_invalid", "hybrid modules and dimensions are required")
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for module in modules for parameter in module.parameters()
    )
    gradient_bytes = sum(
        parameter.numel() * parameter.element_size()
        for module in modules for parameter in module.parameters() if parameter.requires_grad
    )
    optimizer_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values() for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    if resident_model is None:
        raise QwenRuntimeConfigurationError(
            "hybrid_resident_model_required",
            "post-load hybrid accounting requires the complete resident Qwen model",
        )
    resident = resident_model
    resident_parameter_bytes = sum(p.numel() * p.element_size() for p in resident.parameters())
    resident_buffer_bytes = sum(b.numel() * b.element_size() for b in resident.buffers())
    trainable_elements = sum(p.numel() for p in resident.parameters() if p.requires_grad)
    optimizer_bytes = max(optimizer_bytes, 8 * trainable_elements)
    master_weight_bytes = 4 * trainable_elements
    persistent_bytes = cache_bytes = workspace_bytes = 0
    layer_reports: list[dict[str, object]] = []
    for module in modules:
        report_fn = getattr(module, "resource_report", None)
        if callable(report_fn):
            report = report_fn(batch_size=batch_size)
        else:
            element = module.components.q_weight.element_size()
            shared_histories = (
                module.recurrent_state_bytes(batch_size=batch_size)
                + batch_size * module.H * (module.dk // 2) * 4
                + batch_size * module.H * 4 * module.dv * 4
                + batch_size * module.H * module.dk * module.dv * 4
                + batch_size * (module.conv_k - 1) * module.components.hidden * element
                + batch_size
            )
            report = {"persistent_bytes": shared_histories}
        hola_report = module.hola.resource_report(batch_size=batch_size)
        layer_persistent = int(report.get("persistent_bytes", 0))
        persistent_bytes += layer_persistent
        cache_bytes += int(hola_report["persistent_bytes"])
        workspace_bytes += int(hola_report["workspace_bytes"])
        layer_reports.append({"state_and_history_bytes": layer_persistent,
                              "cache_bytes": int(hola_report["persistent_bytes"]),
                              "workspace_bytes": int(hola_report["workspace_bytes"])})
    model_config = getattr(resident_model, "config", None)
    hidden = int(getattr(model_config, "hidden_size", modules[0].components.hidden))
    layers = int(getattr(model_config, "num_hidden_layers", len(modules)))
    intermediate = int(getattr(model_config, "intermediate_size", 4 * hidden))
    vocab = int(getattr(model_config, "vocab_size", 0))
    if vocab < 1:
        raise QwenRuntimeConfigurationError(
            "hybrid_model_config_incomplete", "resident model config must declare vocab_size"
        )
    B,T,L,D,I = batch_size,sequence_length,layers,hidden,intermediate
    saved_layer_activations = B*T*L*D*4 * (1 if checkpointing else 8)
    layer_workspace = B*T*(7*D+2*I)*4
    attention_workspace = B*T*L*D*4
    logits_bytes = B*T*vocab*4
    hybrid_activation_bytes = sum(
        B*T*module.H*4*(4*module.dk+4*module.dv)*4 for module in modules
    )
    # Training-path global mixing retains every normalized [B,T,H,4,4,V]
    # recurrent read (module dtype) and FP32 HOLA read until the mixer is
    # applied once across T; the concatenation transiently doubles both.
    # Inference selects segment mixing and never allocates these.
    global_mix_bytes = sum(
        2 * (B * T * module.H * 16 * module.dv)
        * (module.components.q_weight.element_size() + 4)
        for module in modules
    )
    generation_buffers = B*T*D*4+B*vocab*4
    activation_bytes = sum((saved_layer_activations,layer_workspace,attention_workspace,
                            logits_bytes,hybrid_activation_bytes,global_mix_bytes,
                            generation_buffers))
    return {
        "layer_count": len(modules), "batch_size": batch_size,
        "sequence_length": sequence_length, "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes, "optimizer_bytes": optimizer_bytes,
        "resident_parameter_bytes": resident_parameter_bytes,
        "resident_buffer_bytes": resident_buffer_bytes,
        "master_weight_bytes": master_weight_bytes,
        "state_and_history_bytes": persistent_bytes, "cache_bytes": cache_bytes,
        "workspace_bytes": workspace_bytes, "activation_bytes": activation_bytes,
        "activation_checkpointing": checkpointing, "data_parallel": data_parallel,
        "activation_accounting": "conservative_component_upper_bound",
        "activation_components": {"saved_layer_activations": saved_layer_activations,
            "layer_workspace": layer_workspace, "attention_workspace": attention_workspace,
            "logits": logits_bytes, "hybrid": hybrid_activation_bytes,
            "global_mix_reads": global_mix_bytes,
            "generation_buffers": generation_buffers},
        "layers": layer_reports,
    }


def _reduce_transition_router_diagnostics(
    rows: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    """Reduce exact source-router statistics over valid layer/head/destination rows."""
    if not rows:
        raise QwenRuntimeConfigurationError(
            "hybrid_router_diagnostics_unavailable", "missing live transition-router rows"
        )
    reference: tuple[int, int, int] | None = None
    entropy_sum = 0.0
    opportunities = 0
    argmax_counts: torch.Tensor | None = None
    source_mass: torch.Tensor | None = None
    for probabilities, valid in rows:
        if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 5
                or probabilities.shape[-1] != 4 or not isinstance(valid, torch.Tensor)
                or valid.dtype != torch.bool or tuple(valid.shape) != tuple(probabilities.shape[:2])):
            raise QwenRuntimeConfigurationError(
                "hybrid_router_diagnostics_invalid", "heterogeneous transition-router schema"
            )
        schema = (int(probabilities.shape[2]), int(probabilities.shape[3]), int(probabilities.shape[4]))
        if reference is None:
            reference = schema
            argmax_counts = torch.zeros(schema[-1], dtype=torch.float64)
            source_mass = torch.zeros(schema[-1], dtype=torch.float64)
        elif schema != reference:
            raise QwenRuntimeConfigurationError(
                "hybrid_router_diagnostics_invalid", "heterogeneous transition-router schema"
            )
        selected = probabilities.detach().float()[valid]
        if selected.numel() == 0:
            continue
        flat = selected.reshape(-1, selected.shape[-1]).cpu()
        if not bool(torch.isfinite(flat).all()) or bool((flat < 0).any()):
            raise QwenRuntimeConfigurationError(
                "hybrid_router_diagnostics_invalid", "transition-router probabilities are invalid"
            )
        if not torch.allclose(flat.sum(-1), torch.ones(flat.shape[0]), atol=1e-5, rtol=1e-5):
            raise QwenRuntimeConfigurationError(
                "hybrid_router_diagnostics_invalid", "transition-router probabilities are not normalized"
            )
        # torch.argmax returns the first index, giving the specified lowest-source tie break.
        assert argmax_counts is not None and source_mass is not None
        argmax_counts += torch.bincount(flat.argmax(-1), minlength=4).double()
        source_mass += flat.double().sum(0)
        entropy_sum += float((-(flat * flat.clamp_min(1e-12).log()).sum(-1)).sum())
        opportunities += flat.shape[0]
    if opportunities < 1:
        raise QwenRuntimeConfigurationError(
            "hybrid_router_diagnostics_unavailable", "zero transition-router opportunities"
        )
    assert argmax_counts is not None and source_mass is not None
    return {
        "opportunities": opportunities,
        "entropy": entropy_sum / opportunities,
        "argmax_occupancy": (argmax_counts / opportunities).tolist(),
        "source_probability_mass": (source_mass / opportunities).tolist(),
    }


def _measure_live_hybrid_caches(
    modules: Sequence[torch.nn.Module], *, valid_token_count: int,
) -> dict[str, object]:
    """Measure only genuine cache carries produced by the immediately preceding live pass."""
    if not modules:
        raise QwenRuntimeConfigurationError(
            "hybrid_cache_diagnostics_unavailable", "no upgraded layers were supplied"
        )
    if type(valid_token_count) is not int or valid_token_count < 0:
        raise QwenRuntimeConfigurationError(
            "hybrid_cache_diagnostics_invalid", "valid token count must be nonnegative"
        )
    schema: str | None = None
    admissions = opportunities = occupied = capacity = age_count = 0
    age_sum = 0.0
    state_norms: list[torch.Tensor] = []
    state_rms_max: list[torch.Tensor] = []
    state_abs_max: list[torch.Tensor] = []
    for layer_index, module in enumerate(modules):
        cache = getattr(module, "last_recurrent_cache", None)
        if cache is None:
            raise QwenRuntimeConfigurationError(
                "hybrid_cache_diagnostics_unavailable", f"upgraded layer {layer_index} has missing live cache"
            )
        layer_schema = "state" if hasattr(cache, "state") else "states" if hasattr(cache, "states") else None
        if layer_schema is None:
            raise QwenRuntimeConfigurationError(
                "hybrid_cache_diagnostics_invalid", "live cache has neither state nor states"
            )
        if schema is None:
            schema = layer_schema
        elif schema != layer_schema:
            raise QwenRuntimeConfigurationError(
                "hybrid_cache_diagnostics_invalid", "heterogeneous live cache schemas"
            )
        state = getattr(cache, layer_schema)
        if not isinstance(state, torch.Tensor) or not bool(torch.isfinite(state).all()):
            raise QwenRuntimeConfigurationError(
                "hybrid_cache_diagnostics_invalid", "live recurrent state is missing or nonfinite"
            )
        detached = state.detach().float()
        state_norms.append(detached.norm())
        # A cross-layer mean Frobenius norm can hide one exploding head or
        # lane; record the per-layer extremes alongside it.  For the
        # four-state schema [B,H,R,K,V] reduce per (head, lane) matrix.
        matrix_rms = detached.square().mean((-2, -1)).sqrt()
        state_rms_max.append(matrix_rms.max())
        state_abs_max.append(detached.abs().max())
        hola = getattr(cache, "hola_state", None)
        if hola is None:
            raise QwenRuntimeConfigurationError(
                "hybrid_cache_diagnostics_unavailable", f"upgraded layer {layer_index} has missing HOLA state"
            )
        required = ("valid", "epochs", "current_epoch", "block_valid", "block_epochs",
                    "admission_count", "age_sum", "age_count")
        if any(not isinstance(getattr(hola, name, None), torch.Tensor) for name in required):
            raise QwenRuntimeConfigurationError(
                "hybrid_cache_diagnostics_invalid", "heterogeneous HOLA diagnostic schema"
            )
        persistent = hola.valid & (hola.epochs == hola.current_epoch[..., None])
        block = hola.block_valid & (hola.block_epochs == hola.current_epoch[..., None])
        occupied += int(persistent.sum().cpu()) + int(block.sum().cpu())
        capacity += persistent.numel() + block.numel()
        layer_admissions = int(hola.admission_count.sum().cpu())
        admissions += layer_admissions
        opportunities += valid_token_count * int(module.H)
        age_sum += float(hola.age_sum.sum().cpu())
        age_count += int(hola.age_count.sum().cpu())
    if opportunities < 1:
        raise QwenRuntimeConfigurationError(
            "hybrid_cache_diagnostics_unavailable", "zero live cache opportunities"
        )
    return {
        "cache_schema": schema, "layer_count": len(modules),
        "cache_admissions": admissions, "cache_opportunities": opportunities,
        "cache_admission_rate": admissions / opportunities,
        "cache_occupancy": occupied / capacity if capacity else 0.0,
        "cache_mean_age": age_sum / age_count if age_count else 0.0,
        "state_norm": float(torch.stack(state_norms).mean().cpu()),
        "state_matrix_rms_max": float(torch.stack(state_rms_max).max().cpu()),
        "state_abs_max": float(torch.stack(state_abs_max).max().cpu()),
    }


def _optimizer_mutation_fingerprint(
    optimizer: torch.optim.Optimizer,
) -> tuple[object, ...]:
    """Describe optimizer topology and tensor identity/version without copying data."""
    def freeze(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return (
                "tensor", id(value), value.data_ptr(), value._version,
                tuple(value.shape), str(value.dtype), str(value.device),
            )
        if isinstance(value, Mapping):
            return ("mapping", tuple(
                (str(key), freeze(item)) for key, item in value.items()
            ))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(freeze(item) for item in value))
        if value is None or type(value) in (bool, int, float, str):
            return (type(value).__name__, value)
        return ("object", type(value).__module__, type(value).__qualname__, id(value))

    groups = tuple(
        tuple(
            (key, tuple(id(parameter) for parameter in value))
            if key == "params"
            else (key, freeze(value))
            for key, value in group.items()
        )
        for group in optimizer.param_groups
    )
    state = tuple(
        (id(parameter), freeze(values))
        for parameter, values in optimizer.state.items()
    )
    return (groups, state)


def _run_live_hybrid_diagnostic_pass(
    *, model: torch.nn.Module, modules: Sequence[torch.nn.Module], batch: Mapping[str, object],
    optimizer: torch.optim.Optimizer | None = None, scheduler: object | None = None,
    scaler: object | None = None, sampler: object | None = None, trainer: object | None = None,
    metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute a use-cache eval pass transactionally and return exact live diagnostics."""
    if not isinstance(batch, Mapping) or not isinstance(batch.get("input_ids"), torch.Tensor):
        raise QwenRuntimeConfigurationError(
            "hybrid_diagnostic_batch_invalid", "diagnostic batch requires input_ids"
        )
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    valid = (attention_mask.bool() if isinstance(attention_mask, torch.Tensor)
             else torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device))
    if valid.ndim != 2 or tuple(valid.shape) != tuple(input_ids.shape):
        raise QwenRuntimeConfigurationError(
            "hybrid_diagnostic_batch_invalid", "diagnostic attention_mask must be [B,T]"
        )
    mode_snapshot = {module: module.training for module in model.modules()}
    cache_snapshot = tuple(
        (module, hasattr(module, "last_recurrent_cache"),
         getattr(module, "last_recurrent_cache", None))
        for module in modules
    )
    braid_snapshot = tuple(
        (module.components,
         module.components._braid_entropy_sum.detach().clone(),
         module.components._braid_occupancy_sum.detach().clone(),
         module.components._braid_sample_count.detach().clone())
        for module in modules
    )
    optimizer_fingerprint = (
        _optimizer_mutation_fingerprint(optimizer) if optimizer is not None else None
    )
    optimizer_mutated = False
    scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
    sampler_state = copy.deepcopy(sampler.state_dict()) if sampler is not None else None
    trainer_fields = tuple(name for name in (
        "successful_updates", "successful_update_count", "step", "tokens_seen",
        "example_cursor", "skipped_steps", "last_rank_update_norms",
    ) if trainer is not None and hasattr(trainer, name))
    trainer_state = {name: copy.deepcopy(getattr(trainer, name)) for name in trainer_fields}
    metrics_mutable = metrics is None or isinstance(metrics, dict)
    metrics_state = copy.deepcopy(dict(metrics)) if metrics is not None else None
    metrics_restore_error: Exception | None = None
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    router_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    decay_rows: list[torch.Tensor] = []
    handles = []

    def capture(module: torch.nn.Module, args: tuple[object, ...],
                kwargs: Mapping[str, object] | None = None) -> None:
        hidden = args[0] if args else (kwargs or {}).get("hidden_states")
        if not isinstance(hidden, torch.Tensor):
            raise QwenRuntimeConfigurationError(
                "hybrid_router_diagnostics_unavailable", "hybrid layer omitted hidden-state input"
            )
        components = getattr(module, "components", None)
        if getattr(components, "package", None) == "four_state":
            gamma = components.decay_gamma(hidden).detach().float()
            decay_rows.append(gamma[valid.detach()])
        else:
            probabilities = components.braid_probabilities(hidden)
            router_rows.append((probabilities.detach(), valid.detach()))

    try:
        for module in modules:
            # with_kwargs: decoder layers may pass hidden_states by keyword.
            handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        model.eval()
        with torch.no_grad():
            kwargs: dict[str, object] = {"input_ids": input_ids, "use_cache": True}
            if isinstance(attention_mask, torch.Tensor):
                kwargs["attention_mask"] = attention_mask
            model(**kwargs)
        cache_metrics = _measure_live_hybrid_caches(
            modules, valid_token_count=int(valid.sum().item())
        )
        result = dict(cache_metrics)
        if decay_rows:
            rates = torch.cat(decay_rows).clamp_min(2.0 ** -24).log().neg().mean((0, 1, 3))
            horizons = rates.clamp_min(1e-12).reciprocal()
            result["time_braid"] = {
                "effective_horizons": horizons.cpu().tolist(),
                "horizon_ratios": (horizons / horizons[0]).cpu().tolist(),
                "all_lanes_update_each_token": False,
                "state_router_active": False,
            }
        else:
            result["router"] = _reduce_transition_router_diagnostics(router_rows)
    finally:
        for handle in handles:
            handle.remove()
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        if scaler is not None:
            scaler.load_state_dict(scaler_state)
        if sampler is not None:
            sampler.load_state_dict(sampler_state)
        for name, value in trainer_state.items():
            setattr(trainer, name, value)
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(list(cuda_rng))
        for module, training in mode_snapshot.items():
            module.train(training)
        for module, present, value in cache_snapshot:
            if present:
                module.last_recurrent_cache = value
            elif hasattr(module, "last_recurrent_cache"):
                delattr(module, "last_recurrent_cache")
        with torch.no_grad():
            for components, entropy, occupancy, count in braid_snapshot:
                components._braid_entropy_sum.copy_(entropy)
                components._braid_occupancy_sum.copy_(occupancy)
                components._braid_sample_count.copy_(count)
        if metrics is not None and isinstance(metrics, dict):
            try:
                metrics.clear(); metrics.update(metrics_state)
            except Exception as error:
                metrics_restore_error = error
        if optimizer is not None:
            optimizer_mutated = (
                _optimizer_mutation_fingerprint(optimizer) != optimizer_fingerprint
            )
    if not metrics_mutable:
        raise QwenRuntimeConfigurationError(
            "hybrid_diagnostics_invalid", "reported metrics must be a mutable mapping"
        )
    if metrics_restore_error is not None:
        raise QwenRuntimeConfigurationError(
            "hybrid_diagnostics_invalid", "reported metrics could not be restored"
        ) from metrics_restore_error
    if optimizer_mutated:
        raise QwenRuntimeConfigurationError(
            "hybrid_diagnostics_invalid", "optimizer state mutated during live diagnostics"
        )
    return result


def _collect_hybrid_diagnostics(
    modules: tuple[torch.nn.Module, ...], *, trainer: QwenHealTrainer,
    tokens_per_second: float, peak_memory_bytes: int, flops_per_token: float,
    capacity_confounded: bool, live: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Measure hybrid diagnostics from live parameters, gradients, and retained caches."""
    if len(trainer.last_rank_update_norms) != 4:
        raise QwenRuntimeConfigurationError(
            "hybrid_rank_diagnostics_unavailable", "all four ranks must receive measured gradients"
        )
    q_rows = torch.cat(
        [module.components.q_weight.detach().float().reshape(4, -1) for module in modules], dim=1
    )
    normalized = torch.nn.functional.normalize(q_rows, dim=1)
    pairwise = normalized @ normalized.T
    off_diagonal = pairwise[~torch.eye(4, dtype=torch.bool, device=pairwise.device)]
    # svdvals on the raw [4, layers*H*dk*hidden] matrix overflows cuSOLVER's
    # 32-bit workspace sizing at campaign width (~4x38M); the four singular
    # values are exactly the root eigenvalues of the 4x4 Gram matrix.
    gram = (q_rows @ q_rows.T).double()
    singular = torch.linalg.eigvalsh(gram).clamp_min(0.0).sqrt().float()
    effective_rank = singular.sum().square() / singular.square().sum().clamp_min(1e-12)
    phase_weights = torch.cat([
        module.components.phase_proj.weight.detach().float().reshape(-1) for module in modules
    ])
    phase_biases = torch.cat([
        module.components.phase_proj.bias.detach().float().reshape(-1) for module in modules
    ])
    if not isinstance(live, Mapping) or not isinstance(live.get("time_braid"), Mapping):
        raise QwenRuntimeConfigurationError(
            "hybrid_live_diagnostics_unavailable", "live decay-timescale diagnostics are required"
        )
    time_braid = live["time_braid"]
    sink = torch.cat([module.hola.sink_logit.detach().float().reshape(-1, 4) for module in modules])
    read = sink.softmax(-1)
    read_entropy = -(read * read.clamp_min(1e-12).log()).sum(-1).mean()
    gates = torch.cat([
        torch.cat(((1.0 - module.components.trapezoid_proj.bias.detach().sigmoid()).reshape(-1),
                   module.components.cache_gate_amplitude.detach().reshape(-1))).float()
        for module in modules
    ])
    measured = {
        "rank_update_norms": list(trainer.last_rank_update_norms),
        "rank_similarity": float(off_diagonal.abs().mean().cpu()),
        "effective_rank": float(effective_rank.cpu()),
        "phase_magnitude": float(phase_biases.abs().mean().cpu()),
        "phase_frequency": float(phase_weights.abs().mean().cpu()),
        "effective_decay_horizons": list(time_braid["effective_horizons"]),
        "decay_horizon_ratios": list(time_braid["horizon_ratios"]),
        "all_lanes_update_each_token": bool(time_braid["all_lanes_update_each_token"]),
        "state_router_active": bool(time_braid["state_router_active"]),
        "cache_admissions": int(live["cache_admissions"]),
        "cache_opportunities": int(live["cache_opportunities"]),
        "cache_schema": str(live["cache_schema"]),
        "layer_count": int(live["layer_count"]),
        "cache_admission_rate": float(live["cache_admission_rate"]),
        "cache_occupancy": float(live["cache_occupancy"]),
        "cache_mean_age": float(live["cache_mean_age"]),
        "cache_read_entropy": float(read_entropy.cpu()),
        "cache_gate_mean": float(torch.cat([m.components.cache_gate_amplitude.detach().float().reshape(-1) for m in modules]).mean().cpu()),
        "state_norm": float(live["state_norm"]),
        "gate_mean": float(gates.mean().cpu()), "nonfinite_count": trainer.skipped_steps,
        "flops_per_token": float(flops_per_token), "tokens_per_second": float(tokens_per_second),
        "peak_memory_bytes": int(peak_memory_bytes), "capacity_confounded": capacity_confounded,
    }
    from .results import validate_hybrid_diagnostics
    return validate_hybrid_diagnostics(measured)


def execute_job(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object],
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute one paired job inside a fully restoring stochastic transaction."""
    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    seed = job.get("seed")
    if type(seed) is not int or seed < 0:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.seed must be a nonnegative integer"
        )
    with _scoped_paired_rng(seed, runtime):
        return _execute_job_seeded(job, runtime=runtime, dependencies=dependencies)


def _execute_job_seeded(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object],
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute one bound Qwen heal job and return runner-ready diagnostics."""
    from dataclasses import replace

    from .config import CacheConfig
    from .qwen_backend import (
        LoadedQwenArm,
        PairingContractError,
        QwenArmLoadSpec,
    )
    from .qwen_checkpoint import (
        QwenCheckpointMetadata,
        QwenResumeExpectation,
    )

    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    config = _job_config(job)
    from .qwen_variants import validate_maximum_control_config
    maximum_contract = validate_maximum_control_config(dict(config))
    data_parallel = _validate_parallel_and_packing(config)
    architecture_contract = _architecture_dispatch_contract(job, config)
    training = _training_config(config)
    validate_teacher_requirement(
        training,
        teacher_present=runtime.get("teacher_model") is not None,
        phase="preflight",
    )
    runtime_values, assets = _runtime_assets(
        runtime, teacher_required=training.objective != "synthetic_only"
    )
    dependencies_value = _resolve_dependencies(dependencies)
    started = dependencies_value.monotonic()

    data = dependencies_value.load_data(
        asset=assets["data"], job=job, runtime=runtime_values
    )
    example_ids, expected_windows = _validate_job_data(data, config=config)
    if data.data_identity.get("sha256") != assets["data"].sha256:
        raise QwenRuntimeConfigurationError(
            "data_identity_mismatch",
            "loaded data identity does not match the measured runtime asset",
        )
    pairing = derive_three_arm_pairing(
        job,
        example_ids=example_ids,
        pre_replacement_checkpoint_sha256=assets["checkpoint"].sha256,
        data_sha256=assets["data"].sha256,
    )
    if job.get("pairing_id") != pairing.pairing_id:
        raise PairingContractError(
            "pairing_id_mismatch", "job pairing_id does not match three-arm contract"
        )
    paired_starts = {arm.arm: assets["checkpoint"].sha256 for arm in pairing.arms}
    if len(set(paired_starts.values())) != 1:
        raise PairingContractError(
            "checkpoint_identity_mismatch", "three Qwen arms do not share one checkpoint"
        )

    arm = _selected_arm(job)
    if architecture_contract is not None:
        # HOLA's read-normalization, sink, and mixing logit use the dedicated
        # cache learning rate and zero weight decay even for architecture jobs.
        cache_names = tuple(
            name for name in architecture_contract.trainable_names
            if ".hola." in name or name.endswith(".components.cache_gate_logit")
        )
        cache_set = set(cache_names)
        memory_names = tuple(
            name for name in architecture_contract.trainable_names if name not in cache_set
        )
    else:
        memory_names, cache_names = _training_parameter_names(config, arm)
    architecture = config.get("architecture")
    architecture_arm_id = None
    architecture_registry_sha256 = None
    if isinstance(architecture, Mapping):
        architecture_arm_id = architecture.get("arm_id")
        architecture_registry_sha256 = architecture.get("registry_sha256")
        if job.get("arm_id") != architecture_arm_id:
            raise QwenRuntimeConfigurationError(
                "architecture_arm_mismatch",
                "job arm_id does not match canonical_config.architecture.arm_id",
            )
    if architecture_contract is not None:
        architecture_arm_id = architecture_contract.architecture_arm_id
        architecture_registry_sha256 = architecture_contract.registry_sha256
    cache_mapping = _required_mapping(config.get("cache"), "canonical_config.cache")
    cache_config = CacheConfig(**dict(cache_mapping)) if arm != "native" else None
    if arm == "recency":
        assert cache_config is not None
        cache_config = replace(cache_config, score="recency")
    dtype = torch.float32 if runtime_values["dtype"] == "float32" else torch.bfloat16
    spec = QwenArmLoadSpec(
        arm=arm,
        job_id=job["job_id"],
        model_asset=_asset_spec(assets["model"]),
        native_checkpoint=_asset_spec(assets["checkpoint"]),
        data_asset=_asset_spec(assets["data"]),
        cache_resume=None,
        trainable_names=memory_names + cache_names,
        pre_replacement_checkpoint_sha256=assets["checkpoint"].sha256,
        model_loader_kwargs={"torch_dtype": dtype, "low_cpu_mem_usage": True},
        architecture_arm_id=architecture_arm_id,
        architecture_registry_sha256=architecture_registry_sha256,
        diagnostic_training=(architecture_contract.diagnostic_training
                             if architecture_contract is not None else False),
        maximum_control_id=(maximum_contract.control_id if maximum_contract is not None else None),
    )
    loaded = dependencies_value.load_arm(
        spec, model_config=None, cache_config=cache_config
    )
    if not isinstance(loaded, LoadedQwenArm) or loaded.arm != arm:
        raise TypeError("Qwen arm loader returned an incompatible result")
    loaded.model.to(runtime_values["student_device"])
    if (training.gradient_checkpointing and architecture_contract is not None
            and architecture_contract.architecture_arm_id.startswith("gdn2-mimo-r4-braid-")):
        enable_checkpointing = getattr(loaded.model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise QwenRuntimeConfigurationError(
                "activation_checkpointing_unavailable",
                "configured activation checkpointing is not implemented by the loaded model",
            )
        enable_checkpointing()
        loaded.model._kmd2_checkpointing_enabled = True
    amplitude_initial = _cache_amplitudes(loaded.model)

    if maximum_contract is not None and not maximum_contract.replacement:
        if arm != "native" or loaded.upgraded_indices:
            raise QwenRuntimeConfigurationError(
                "stock_replacement_forbidden", "stock Qwen evaluation must retain the untouched source model"
            )
        for parameter in loaded.model.parameters():
            parameter.requires_grad_(False)
        dependencies_value.reset_peak_vram(runtime_values["student_device"])
        evaluation = dependencies_value.evaluate(
            loaded_arm=loaded, data=data, job=job, runtime=runtime_values,
            amplitude_initial=amplitude_initial,
            generate_answers=dependencies_value.generate_answers,
            generate_state_values=dependencies_value.generate_state_values,
            tokenizer_asset=assets.get("tokenizer"),
        )
        if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("metrics"), Mapping):
            raise QwenRuntimeConfigurationError(
                "evaluation_invalid", "stock Qwen evaluator metrics must be present"
            )
        finished = dependencies_value.monotonic()
        wall_time = finished - started
        if not math.isfinite(wall_time) or wall_time < 0.0:
            raise QwenRuntimeConfigurationError("clock_invalid", "stock evaluation duration is invalid")
        duration = max(wall_time, 1.0e-12)
        evaluated_tokens = sum(_batch_token_count(batch) for batch in data.eval_microbatches)
        recurrent_state = evaluation.get("recurrent_state", {"elements": 0, "bytes": 0})
        if not isinstance(recurrent_state, Mapping) or set(recurrent_state) != {"elements", "bytes"}:
            raise QwenRuntimeConfigurationError("evaluation_invalid", "stock recurrent state is incomplete")
        total_parameters = sum(parameter.numel() for parameter in loaded.model.parameters())
        payload = {
            "stock_evaluation": True,
            "optimizer_created": False,
            "architecture_replaced": False,
            "metrics": dict(evaluation["metrics"]),
            "loss_curves": {"train": [], "validation": []},
            "counts": {"nonfinite_loss": 0, "nonfinite_gradient": 0, "skipped_steps": 0},
            "parameters": {"trainable": 0, "total": total_parameters},
            "recurrent_state": dict(recurrent_state),
            "performance": {
                "wall_time_seconds": wall_time,
                "examples_per_second": len(data.eval_microbatches) / duration,
                "tokens_per_second": evaluated_tokens / duration,
                "peak_vram_bytes": dependencies_value.peak_vram_bytes(runtime_values["student_device"]),
            },
            "identities": {
                "model": _identity_record(assets["model"]),
                "checkpoint": _identity_record(assets["checkpoint"]),
                "data": _identity_record(assets["data"]),
                "implementation": "stock_qwen_source_evaluator",
                "stock": maximum_contract.identity_sha256,
            },
        }
        if "evaluation_execution" in evaluation:
            payload["evaluation_execution"] = copy.deepcopy(
                evaluation["evaluation_execution"]
            )
        # The per-episode RULER rows are the paired-comparison payload; the
        # heal path forwards them and the stock reference must as well or the
        # teacher side of every teacher-vs-arm pairing is aggregate-only.
        if "evaluations" in evaluation:
            payload["evaluations"] = copy.deepcopy(evaluation["evaluations"])
        return payload

    teacher = None
    if training.objective != "synthetic_only":
        teacher = dependencies_value.load_teacher(
            asset=assets["teacher_model"], job=job, runtime=runtime_values
        )
        if not isinstance(teacher, torch.nn.Module):
            raise TypeError("teacher loader must return a torch.nn.Module")

    optimizer_mapping = _required_mapping(
        config.get("optimizer"), "canonical_config.optimizer"
    )
    if optimizer_mapping.get("name") != "adamw":
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "Qwen heal requires AdamW"
        )
    optimizer = dependencies_value.build_optimizer(
        loaded.model,
        memory_parameter_names=memory_names,
        cache_parameter_names=cache_names,
        learning_rate=optimizer_mapping.get("learning_rate"),
        lr_cache=cache_mapping.get("lr_cache"),
        betas=tuple(optimizer_mapping.get("betas", ())),
        eps=optimizer_mapping.get("eps"),
        weight_decay=optimizer_mapping.get("weight_decay"),
    )
    scheduler = dependencies_value.build_scheduler(
        optimizer=optimizer, config=config, job=job
    )
    scaler_builder = dependencies_value.build_grad_scaler or _default_build_grad_scaler
    grad_scaler = scaler_builder(
        device=runtime_values["student_device"],
        dtype=next(loaded.model.parameters()).dtype,
    )
    moved_train = tuple(
        _move_batch(batch, runtime_values["student_device"])
        for batch in data.train_microbatches
    )
    if data_parallel > 1:
        moved_train, expected_windows = _shard_training_windows(
            moved_train, rank=torch.distributed.get_rank(), world_size=data_parallel
        )
    trainer_model: torch.nn.Module = loaded.model
    if data_parallel > 1:
        device = torch.device(runtime_values["student_device"])
        device_ids = [device.index] if device.type == "cuda" else None
        trainer_model = torch.nn.parallel.DistributedDataParallel(
            loaded.model, device_ids=device_ids, broadcast_buffers=True,
        )
    trainer = QwenHealTrainer(
        model=trainer_model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training,
        job_id=job["job_id"],
        pairing_id=pairing.pairing_id,
        arm=arm,
        expected_example_windows=expected_windows,
        teacher_device=(
            runtime_values.get("teacher_device") if teacher is not None else None
        ),
        distributed=data_parallel > 1,
        grad_scaler=grad_scaler,
    )
    target_module_names = tuple(
        sorted(f"model.layers.{index}.linear_attn" for index in loaded.upgraded_indices)
    )
    source_hashes = _source_hashes()
    source_hashes.update(
        {f"asset:{name}": asset.sha256 for name, asset in sorted(assets.items())}
    )
    promotion = config.get("promotion")
    if not isinstance(promotion, Mapping) or not promotion:
        promotion = cache_mapping
    promotion = dict(promotion)
    if architecture_contract is not None:
        promotion["architecture_diagnostic_training"] = architecture_contract.diagnostic_training
    from .architecture import registry_sha256
    checkpoint_architecture_arm_id = (
        loaded.architecture_arm_id
        or ("kmd2-r1" if arm == "native" else f"exact-cache-{arm}-r1")
    )
    checkpoint_architecture_registry_sha256 = (
        loaded.architecture_registry_sha256 or registry_sha256()
    )
    auxiliary_identity: Mapping[str, object] = {}
    if training.specialization_updates > 0:
        try:
            _, auxiliary_identity = package_b_auxiliary_loss(
                loaded.model, lambda_spec=training.lambda_spec, lambda_gate=training.lambda_gate,
                successful_updates=training.specialization_updates,
                specialization_updates=training.specialization_updates,
            )
        except ValueError as error:
            # Non-Package-B arms (e.g. gdn2-r1) legitimately train with no
            # auxiliary loss; mirror the training-loop guard at the metadata
            # replay site rather than failing the whole diagnostics pass.
            if "requires four-state" not in str(error):
                raise
    metadata_kwargs = {
        "job_id": job["job_id"],
        "pairing_id": pairing.pairing_id,
        "arm": arm,
        "source_hashes": source_hashes,
        "data_identity": data.data_identity,
        "example_ids": example_ids,
        "promotion_config": promotion,
        "architecture_arm_id": checkpoint_architecture_arm_id,
        "architecture_registry_sha256": checkpoint_architecture_registry_sha256,
        "auxiliary_identity": auxiliary_identity,
    }
    checkpoint_path = (
        runtime_values["output"] / "checkpoints" / job["job_id"] / "latest.pt"
    )
    if runtime_values["resume"] and checkpoint_path.is_file():
        expectation = QwenResumeExpectation(**metadata_kwargs)
        resumed = dependencies_value.load_checkpoint(
            checkpoint_path,
            model=loaded.model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=expectation,
            target_module_names=target_module_names,
            grad_scaler=grad_scaler,
        )
        expected_resume_identity = {
            "job_id": job["job_id"],
            "pairing_id": pairing.pairing_id,
            "arm": arm,
        }
        if any(
            getattr(resumed, name, None) != expected
            for name, expected in expected_resume_identity.items()
        ):
            raise QwenRuntimeConfigurationError(
                "resume_identity_mismatch",
                "checkpoint loader returned inconsistent job/pair/arm identity",
            )
        if (
            type(getattr(resumed, "step", None)) is not int
            or type(getattr(resumed, "tokens_seen", None)) is not int
            or resumed.step < 0
            or resumed.tokens_seen < 0
        ):
            raise QwenRuntimeConfigurationError(
                "resume_progress_invalid", "checkpoint progress is malformed"
            )
        if resumed.step > training.max_updates:
            raise QwenRuntimeConfigurationError(
                "resume_progress_invalid", "resume step exceeds configured update budget"
            )
        cursor = getattr(resumed, "example_cursor", resumed.step * training.accumulation_steps)
        if cursor != resumed.step * training.accumulation_steps:
            raise QwenRuntimeConfigurationError(
                "resume_progress_invalid", "resume sampler cursor does not match successful updates"
            )
        prefix_tokens = sum(
            _batch_token_count(batch) for batch in moved_train[:cursor]
        )
        if data_parallel > 1:
            prefix_total = torch.tensor(prefix_tokens, dtype=torch.int64,
                                        device=runtime_values["student_device"])
            torch.distributed.all_reduce(prefix_total, op=torch.distributed.ReduceOp.SUM)
            prefix_tokens = int(prefix_total.item())
        if resumed.tokens_seen != prefix_tokens:
            raise QwenRuntimeConfigurationError(
                "resume_progress_invalid", "resume token progress does not match data windows"
            )
        trainer.step = resumed.step
        trainer.tokens_seen = resumed.tokens_seen
        trainer.example_cursor = cursor

    # Loading, optional resume, and setup are outside the measured peak window;
    # the reset baseline still includes every resident runtime tensor.
    dependencies_value.reset_peak_vram(runtime_values["student_device"])

    logs: list[HealStepLog] = []
    checkpoint_writer = data_parallel == 1 or torch.distributed.get_rank() == 0
    checkpoint_every = runtime_values["checkpoint_every"]
    live_metrics_path = (
        runtime_values["output"] / "live" / f"{job['job_id']}.jsonl"
        if checkpoint_writer and os.environ.get("GDNX_LIVE_METRICS") == "1"
        else None
    )
    while trainer.step < training.max_updates:
        start = trainer.example_cursor
        stop = start + training.accumulation_steps
        log = trainer.train_update(moved_train[start:stop])
        logs.append(log)
        if live_metrics_path is not None:
            _emit_live_metrics(live_metrics_path, log, training.max_updates)
        if checkpoint_writer and (trainer.step % checkpoint_every == 0 or trainer.step == training.max_updates):
            metadata = QwenCheckpointMetadata(
                step=trainer.step,
                tokens_seen=trainer.tokens_seen,
                example_cursor=trainer.example_cursor,
                **metadata_kwargs,
            )
            dependencies_value.save_checkpoint(
                checkpoint_path,
                model=loaded.model,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata=metadata,
                target_module_names=target_module_names,
                grad_scaler=grad_scaler,
            )

    evaluation = dependencies_value.evaluate(
        loaded_arm=loaded,
        data=data,
        job=job,
        runtime=runtime_values,
        amplitude_initial=amplitude_initial,
        generate_answers=dependencies_value.generate_answers,
        generate_state_values=dependencies_value.generate_state_values,
        tokenizer_asset=assets.get("tokenizer"),
    )
    if not isinstance(evaluation, Mapping):
        raise TypeError("Qwen evaluator must return a mapping")
    metrics = evaluation.get("metrics")
    recurrent_state = evaluation.get("recurrent_state")
    if not isinstance(metrics, Mapping) or not metrics:
        raise QwenRuntimeConfigurationError(
            "evaluation_invalid", "Qwen evaluator metrics must be nonempty"
        )
    if not isinstance(recurrent_state, Mapping) or set(recurrent_state) != {
        "elements",
        "bytes",
    }:
        raise QwenRuntimeConfigurationError(
            "evaluation_invalid", "Qwen evaluator recurrent_state is incomplete"
        )
    finished = dependencies_value.monotonic()
    wall_time = finished - started
    if not math.isfinite(wall_time) or wall_time < 0.0:
        raise QwenRuntimeConfigurationError(
            "clock_invalid", "monotonic execution duration is invalid"
        )
    duration = max(wall_time, 1.0e-12)
    loss_curves = {
        name: [float(log.losses[name]) for log in logs]
        for name in ("total", "ce", "kl", "layerwise")
    }
    trainable = sum(
        parameter.numel() for parameter in loaded.model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in loaded.model.parameters())
    payload: dict[str, object] = {
        "metrics": dict(metrics),
        "loss_curves": loss_curves,
        "counts": {
            "nonfinite_loss": 0,
            "nonfinite_gradient": 0,
            "skipped_steps": trainer.skipped_steps,
        },
        "parameters": {"trainable": trainable, "total": total_parameters},
        "recurrent_state": dict(recurrent_state),
        "performance": {
            "wall_time_seconds": wall_time,
            "examples_per_second": len(example_ids) / duration,
            "tokens_per_second": trainer.tokens_seen / duration,
            "peak_vram_bytes": dependencies_value.peak_vram_bytes(
                runtime_values["student_device"]
            ),
        },
        "identities": {
            "model": _identity_record(assets["model"]),
            "checkpoint": _identity_record(assets["checkpoint"]),
            "data": _identity_record(assets["data"]),
            "paired_starts": paired_starts,
            **(
                {"teacher_model": _identity_record(assets["teacher_model"])}
                if teacher is not None
                else {}
            ),
            "architecture_arm_id": checkpoint_architecture_arm_id,
            "architecture_registry_sha256": checkpoint_architecture_registry_sha256,
            "architecture_classification": loaded.architecture_classification,
            "architecture_identity_passed": loaded.architecture_identity_passed,
            "architecture_implementation": loaded.architecture_implementation,
            "architecture_tensor_manifest": (
                None
                if loaded.architecture_tensor_manifest is None
                else dict(loaded.architecture_tensor_manifest)
            ),
        },
    }
    if loaded.architecture_arm_id is not None:
        payload.update({
            "architecture_arm_id": loaded.architecture_arm_id,
            "architecture_registry_sha256": loaded.architecture_registry_sha256,
            "architecture_classification": loaded.architecture_classification,
            "architecture_identity_passed": loaded.architecture_identity_passed,
            "architecture_implementation": loaded.architecture_implementation,
            "architecture_tensor_manifest": dict(loaded.architecture_tensor_manifest or {}),
        })
        if loaded.architecture_arm_id.startswith("gdn2-mimo-r4-braid-"):
            from .qwen_hybrid_math import DEFERRED_FUSION_WARNING, REFERENCE_IMPLEMENTATION
            payload["performance"].update({
                "execution": REFERENCE_IMPLEMENTATION,
                "warning": DEFERRED_FUSION_WARNING,
            })
        architecture_modules = tuple(
            loaded.model.model.layers[index].linear_attn
            for index in loaded.upgraded_indices
        )
        # Incremental arms keep the stock conv at module.conv1d; the maximum
        # hybrids copy it into their HybridComponents submodule.
        convolution_parameters = sum(
            (module.conv1d if hasattr(module, "conv1d")
             else module.components.conv1d).weight.numel()
            for module in architecture_modules
        )
        transformed_parameters = sum(
            module.erase_proj.weight.numel()
            + module.write_proj.weight.numel()
            + module.write_offset.numel()
            for module in architecture_modules
            if hasattr(module, "erase_proj")
        )
        payload["resources"] = {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable,
            "recurrent_state_elements": recurrent_state["elements"],
            "recurrent_state_bytes": recurrent_state["bytes"],
            "convolution_parameters": convolution_parameters,
            "transformed_parameters": transformed_parameters,
            "reference_implementation": loaded.architecture_implementation,
        }
        payload["resources"].update(_architecture_new_state_resources(architecture_modules))
        if loaded.architecture_arm_id.startswith("gdn2-mimo-r4-braid-"):
            input_shapes = tuple(
                batch["input_ids"].shape
                for batch in (*data.train_microbatches, *data.eval_microbatches)
            )
            batch_size = max(int(shape[0]) for shape in input_shapes)
            sequence_length = max(int(shape[1]) for shape in input_shapes)
            payload["resources"]["hybrid"] = _hybrid_module_resources(
                architecture_modules, optimizer=optimizer, batch_size=batch_size,
                sequence_length=sequence_length,
                checkpointing=training.gradient_checkpointing,
                resident_model=loaded.model,
                data_parallel=data_parallel,
            )
        rank = getattr(architecture_modules[0], "rank", 1)
        output_width = getattr(architecture_modules[0], "output_width", 1)
        if loaded.architecture_arm_id == "rout-4":
            if any(type(getattr(module, "output_width", None)) is not int
                   for module in architecture_modules):
                raise QwenRuntimeConfigurationError(
                    "architecture_tensor_manifest_malformed_width",
                    "widening production layers must declare integer output_width",
                )
            if any(getattr(module, "output_width") != output_width
                   for module in architecture_modules):
                raise QwenRuntimeConfigurationError(
                    "architecture_tensor_manifest_heterogeneous_width",
                    "widening production layers do not have one homogeneous width",
                )
            from .architecture import architecture_record
            record = architecture_record(loaded.architecture_arm_id)
            input_shapes = tuple(batch["input_ids"].shape for batch in
                                 (*data.train_microbatches, *data.eval_microbatches))
            batch_size, max_tokens = _widening_resource_shape_maxima(tuple(
                (int(shape[0]), int(shape[1])) for shape in input_shapes
            ))
            payload["resources"].update(_shared_query_widening_resources(
                width=output_width, layers=len(architecture_modules),
                batch_size=batch_size, sequence_length=1, max_tokens=max_tokens,
                heads=record.num_heads, key_dim=record.state_key_dim,
                value_dim=record.state_value_dim,
            ))
        if rank > 1:
            if any(getattr(module, "rank", None) != rank for module in architecture_modules):
                raise QwenRuntimeConfigurationError(
                    "architecture_tensor_manifest_heterogeneous_rank",
                    "true MIMO production layers do not have one homogeneous rank",
                )
            from .architecture import architecture_record
            record = architecture_record(loaded.architecture_arm_id)
            input_shapes = tuple(
                batch["input_ids"].shape
                for batch in (*data.train_microbatches, *data.eval_microbatches)
            )
            batch_size, sequence_length = max(
                input_shapes, key=lambda shape: int(shape[0]) * int(shape[1])
            )
            payload["resources"].update(_true_mimo_resources(
                rank=rank,
                layers=len(architecture_modules),
                batch_size=int(batch_size),
                sequence_length=int(sequence_length),
                heads=record.num_heads,
                key_dim=record.state_key_dim,
                value_dim=record.state_value_dim,
                native_conv_parameters=convolution_parameters,
            ))
    if "evaluations" in evaluation:
        payload["evaluations"] = evaluation["evaluations"]
        hybrid_rows = [
            row for row in evaluation["evaluations"]
            if isinstance(row, Mapping) and row.get("task") == "ruler"
            and row.get("context_length") == 32768
        ]
        if loaded.architecture_arm_id is not None and hybrid_rows:
            teacher_rows = [row for row in hybrid_rows if row.get("evaluation_mode") == "teacher_forced"]
            generated_rows = [row for row in hybrid_rows if row.get("evaluation_mode") == "free_generation"]
            if teacher_rows and generated_rows:
                task_params = _required_mapping(
                    _required_mapping(config.get("task"), "canonical_config.task").get("params"),
                    "canonical_config.task.params",
                )
                payload["hybrid_promotion_observation"] = {
                    "arm_id": job["arm_id"], "seed": job["seed"],
                    "pairing_id": pairing.pairing_id,
                    "baseline_arm_id": str(task_params.get("promotion_baseline_arm_id", "gdn2-channel-r1")),
                    "checkpoint_sha256": assets["checkpoint"].sha256,
                    "data_sha256": assets["data"].sha256,
                    "cells": sorted({str(row["cell_id"]) for row in hybrid_rows}),
                    "recall": sum(int(row["numerator"]) for row in teacher_rows)
                    / sum(int(row["denominator"]) for row in teacher_rows),
                    "free_generation": sum(int(row["numerator"]) for row in generated_rows)
                    / sum(int(row["denominator"]) for row in generated_rows),
                    "lm_loss": float(metrics["eval_loss"]), "nonfinite_count": trainer.skipped_steps,
                    "capacity_confounded": bool(task_params.get("capacity_confounded", False)),
                }
    if "evaluation_execution" in evaluation:
        payload["evaluation_execution"] = copy.deepcopy(
            evaluation["evaluation_execution"]
        )
    if (loaded.architecture_arm_id is not None
            and loaded.architecture_arm_id.startswith("gdn2-mimo-r4-braid-")):
        resources = payload["resources"]["hybrid"]
        assert isinstance(resources, Mapping)
        estimated_flops = 2.0 * float(resources["parameter_bytes"]) / torch.empty((), dtype=dtype).element_size()
        task_params = _required_mapping(
            _required_mapping(config.get("task"), "canonical_config.task").get("params"),
            "canonical_config.task.params",
        )
        capacity_confounded = task_params.get("capacity_confounded", False)
        if type(capacity_confounded) is not bool:
            raise QwenRuntimeConfigurationError(
                "job_configuration_invalid", "capacity_confounded must be bool"
            )
        if not data.eval_microbatches:
            raise QwenRuntimeConfigurationError(
                "hybrid_diagnostic_batch_unavailable", "hybrid run has no declared evaluation batch"
            )
        diagnostic_batch = _move_batch(
            data.eval_microbatches[0], runtime_values["student_device"]
        )
        live_diagnostics = _run_live_hybrid_diagnostic_pass(
            model=loaded.model, modules=architecture_modules,
            batch=diagnostic_batch, optimizer=optimizer, scheduler=scheduler,
            scaler=grad_scaler, trainer=trainer, metrics=metrics,
        )
        payload["hybrid_diagnostics"] = _collect_hybrid_diagnostics(
            architecture_modules, trainer=trainer,
            tokens_per_second=float(payload["performance"]["tokens_per_second"]),
            peak_memory_bytes=int(payload["performance"]["peak_vram_bytes"]),
            flops_per_token=estimated_flops,
            capacity_confounded=capacity_confounded,
            live=live_diagnostics,
        )
        if "hybrid_promotion_observation" in payload:
            seed_evidence = payload["hybrid_promotion_observation"]
            assert isinstance(seed_evidence, dict)
            seed_evidence["effective_rank"] = payload["hybrid_diagnostics"]["effective_rank"]
            seed_evidence["decay_horizon_ratios"] = payload["hybrid_diagnostics"]["decay_horizon_ratios"]
            thresholds = task_params.get("hybrid_promotion_thresholds", {})
            if not isinstance(thresholds, Mapping):
                raise QwenRuntimeConfigurationError(
                    "job_configuration_invalid", "hybrid_promotion_thresholds must be a mapping"
                )
            seed_evidence["thresholds"] = dict(thresholds)
    if arm != "native":
        exact_cache = evaluation.get("exact_cache")
        if not isinstance(exact_cache, Mapping):
            raise QwenRuntimeConfigurationError(
                "evaluation_invalid", "cache evaluator omitted exact_cache diagnostics"
            )
        payload["exact_cache"] = dict(exact_cache)
    return payload


def run_job(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object] | None = None,
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Runner-discoverable entry point; runtime paths must be explicitly bound."""
    if runtime is None:
        raise QwenRuntimeConfigurationError(
            "runtime_required",
            "Qwen run_job requires build_job_dispatcher(runtime, dependencies)",
        )
    return execute_job(job, runtime=runtime, dependencies=dependencies)


def build_job_dispatcher(
    runtime: Mapping[str, object],
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    """Bind non-semantic runtime state into the runner's one-argument protocol."""
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    frozen_runtime = dict(runtime)
    resolved_dependencies = _resolve_dependencies(dependencies)

    def dispatch(job: Mapping[str, object]) -> Mapping[str, object]:
        return run_job(
            job,
            runtime=frozen_runtime,
            dependencies=resolved_dependencies,
        )

    dispatch.__name__ = "run_bound_qwen_job"
    return dispatch


__all__ = [
    "HealLossBreakdown",
    "HealStepLog",
    "QwenExecutionDependencies",
    "QwenHealTrainer",
    "QwenHealTrainingConfig",
    "QwenJobData",
    "QwenRuntimeConfigurationError",
    "QwenTrainingError",
    "TeacherRequiredError",
    "build_job_dispatcher",
    "build_qwen_heal_optimizer",
    "causal_cross_entropy",
    "compute_heal_loss",
    "distillation_kl",
    "derive_three_arm_pairing",
    "execute_job",
    "layerwise_alignment_loss",
    "project_cache_amplitudes_",
    "project_hybrid_constraints_",
    "run_qwen_arm",
    "run_job",
    "validate_teacher_requirement",
]
