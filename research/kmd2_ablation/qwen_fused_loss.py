"""Memory-bounded joint Qwen heal CE and full-vocabulary KL."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


_ROW_CHUNK = 8


def _causal_targets(labels: torch.Tensor) -> torch.Tensor:
    targets = labels.new_full(labels.shape, -100)
    targets[:, :-1] = labels[:, 1:]
    return targets.reshape(-1)


class _ChunkedHealCEKL(torch.autograd.Function):
    """Recompute token-row chunks instead of retaining full-vocabulary graphs."""

    @staticmethod
    def forward(
        ctx: Any,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, vocabulary = student_logits.shape
        rows = batch * tokens
        student = student_logits.reshape(rows, vocabulary)
        teacher = teacher_logits.reshape(rows, vocabulary)
        valid_count = labels[:, 1:].ne(-100).sum()
        # The custom Function forward runs without an autograd graph, so the
        # canonical BF16 CE workspace is released immediately.  Recomputing
        # CE by row chunks would change its BF16 reduction and logged scalar.
        ce = F.cross_entropy(
            student_logits[:, :-1, :].reshape(-1, vocabulary),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        kl_partials: list[torch.Tensor] = []

        for start in range(0, rows, _ROW_CHUNK):
            stop = min(rows, start + _ROW_CHUNK)
            student_float = student[start:stop].float()
            # Keep the frozen teacher logits on the teacher GPU.  A complete
            # T=4096 BF16 vocabulary tensor is about 1.9 GiB; transferring
            # only the rows being reduced preserves the exact BF16 input to
            # the objective while bounding student-device residency.
            teacher_float = teacher[start:stop].to(
                device=student_logits.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            ).float()
            student_log = F.log_softmax(student_float / temperature, dim=-1)
            teacher_log = F.log_softmax(teacher_float / temperature, dim=-1)
            kl_partials.append(
                F.kl_div(
                    student_log,
                    teacher_log,
                    reduction="sum",
                    log_target=True,
                )
            )

        kl = (
            torch.stack(kl_partials).sum()
            * (temperature * temperature)
            / rows
        )
        ctx.save_for_backward(
            student_logits, teacher_logits, labels, valid_count
        )
        ctx.set_materialize_grads(False)
        ctx.temperature = temperature
        ctx.rows = rows
        return ce, kl

    @staticmethod
    def backward(
        ctx: Any,
        grad_ce: torch.Tensor | None,
        grad_kl: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None, None]:
        student_logits, teacher_logits, labels, valid_count = ctx.saved_tensors
        if grad_ce is None and grad_kl is None:
            return None, None, None, None

        batch, tokens, vocabulary = student_logits.shape
        student = student_logits.reshape(ctx.rows, vocabulary)
        teacher = teacher_logits.reshape(ctx.rows, vocabulary)
        targets = _causal_targets(labels)
        gradient = torch.empty_like(student)

        for start in range(0, ctx.rows, _ROW_CHUNK):
            stop = min(ctx.rows, start + _ROW_CHUNK)
            chunk_gradient: torch.Tensor | None = None
            if grad_kl is not None:
                student_probability = F.softmax(
                    student[start:stop].float() / ctx.temperature,
                    dim=-1,
                )
                teacher_probability = F.softmax(
                    teacher[start:stop].to(
                        device=student_logits.device,
                        dtype=torch.bfloat16,
                        non_blocking=True,
                    ).float() / ctx.temperature,
                    dim=-1,
                )
                chunk_gradient = student_probability.sub_(
                    teacher_probability
                ).mul_(
                    grad_kl.float() * ctx.temperature / ctx.rows
                ).to(student_logits.dtype)

            if grad_ce is not None:
                with torch.enable_grad():
                    # Use the native BF16 CE derivative per chunk to preserve
                    # its branch quantization while bounding saved workspace.
                    ce_input = (
                        student[start:stop].detach().requires_grad_(True)
                    )
                    ce_sum = F.cross_entropy(
                        ce_input,
                        targets[start:stop],
                        ignore_index=-100,
                        reduction="sum",
                    )
                    (ce_gradient,) = torch.autograd.grad(
                        ce_sum,
                        ce_input,
                        grad_outputs=(
                            grad_ce.to(ce_sum.dtype) / valid_count
                        ),
                    )
                chunk_gradient = (
                    ce_gradient
                    if chunk_gradient is None
                    else chunk_gradient + ce_gradient
                )

            assert chunk_gradient is not None
            gradient[start:stop].copy_(chunk_gradient)

        return gradient.reshape_as(student_logits), None, None, None


def _reference_heal_ce_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(
        student_logits[:, :-1, :].reshape(-1, student_logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(
        teacher_logits.detach().to(
            device=student_logits.device, dtype=torch.float32
        )
        / temperature,
        dim=-1,
    )
    kl = F.kl_div(
        student_log,
        teacher_log,
        reduction="batchmean",
        log_target=True,
    ) * (temperature * temperature) / student_logits.shape[1]
    return ce, kl


def fused_heal_ce_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate CE/KL values, using bounded recomputation when safe.

    The caller retains the public loss validation contract.  CUDA BF16 is the
    only optimized dispatch; every other dtype/device follows the canonical
    PyTorch expressions exactly.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("labels must match logits batch/time dimensions")
    if type(temperature) not in (int, float):
        raise TypeError("temperature must be a real number")
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be positive")

    optimized = (
        student_logits.is_cuda
        and student_logits.dtype == torch.bfloat16
        and teacher_logits.is_cuda
        and teacher_logits.dtype in {torch.bfloat16, torch.float32}
        and labels.device == student_logits.device
        and labels.dtype == torch.long
    )
    if optimized:
        return _ChunkedHealCEKL.apply(
            student_logits,
            teacher_logits.detach(),
            labels,
            scale,
        )
    return _reference_heal_ce_kl(
        student_logits, teacher_logits, labels, scale
    )


__all__ = ["fused_heal_ce_kl"]
