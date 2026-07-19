from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _config():
    from research.kmd2_ablation.qwen_training import QwenHealTrainingConfig

    return QwenHealTrainingConfig(
        objective="language_model_heal",
        ce_weight=0.1,
        kl_weight=1.0,
        layerwise_weight=0.0,
        temperature=1.7,
        accumulation_steps=1,
        max_updates=1,
        max_tokens=64,
        gradient_checkpointing=False,
    )


def test_joint_heal_loss_cpu_fallback_matches_public_losses_and_gradient() -> None:
    from research.kmd2_ablation.qwen_fused_loss import fused_heal_ce_kl
    from research.kmd2_ablation.qwen_training import (
        causal_cross_entropy,
        distillation_kl,
    )

    generator = torch.Generator().manual_seed(4817)
    values = torch.randn(2, 5, 11, dtype=torch.float64, generator=generator)
    teacher = torch.randn(2, 5, 11, dtype=torch.float64, generator=generator)
    labels = torch.randint(0, 11, (2, 5), generator=generator)
    labels[0, 2] = -100

    reference_student = values.clone().requires_grad_(True)
    reference_teacher = teacher.clone().requires_grad_(True)
    expected_ce = causal_cross_entropy(reference_student, labels)
    expected_kl = distillation_kl(
        reference_student, reference_teacher, temperature=1.7
    )
    (0.1 * expected_ce + expected_kl).backward()

    actual_student = values.clone().requires_grad_(True)
    actual_teacher = teacher.clone().requires_grad_(True)
    actual_ce, actual_kl = fused_heal_ce_kl(
        actual_student,
        actual_teacher,
        labels,
        temperature=1.7,
    )
    (0.1 * actual_ce + actual_kl).backward()

    torch.testing.assert_close(actual_ce, expected_ce, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_kl, expected_kl, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_student.grad, reference_student.grad, rtol=0.0, atol=0.0
    )
    assert actual_teacher.grad is None
    assert reference_teacher.grad is None


def test_compute_heal_loss_routes_joint_ce_kl_through_fused_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.kmd2_ablation.qwen_training as training

    student_logits = torch.randn(1, 4, 7, requires_grad=True)
    teacher_logits = torch.randn(1, 4, 7)
    labels = torch.tensor([[1, 3, -100, 2]])
    observed: dict[str, object] = {}

    def probe(
        student: torch.Tensor,
        teacher: torch.Tensor,
        target: torch.Tensor,
        *,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed.update(
            student=student,
            teacher=teacher,
            labels=target,
            temperature=temperature,
        )
        anchor = student.sum() * 0.0
        return anchor + 2.0, anchor + 3.0

    monkeypatch.setattr(training, "fused_heal_ce_kl", probe)
    breakdown = training.compute_heal_loss(
        SimpleNamespace(logits=student_logits),
        SimpleNamespace(logits=teacher_logits),
        labels,
        _config(),
    )

    assert observed["student"] is student_logits
    assert isinstance(observed["teacher"], torch.Tensor)
    assert observed["teacher"].data_ptr() == teacher_logits.data_ptr()
    assert observed["teacher"].requires_grad is False
    assert observed["labels"] is labels
    assert observed["temperature"] == 1.7
    torch.testing.assert_close(breakdown.ce, torch.tensor(2.0))
    torch.testing.assert_close(breakdown.kl, torch.tensor(3.0))
    torch.testing.assert_close(breakdown.total, torch.tensor(3.2))


def _cuda_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_joint_heal_loss_cuda_float32_uses_reference_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.kmd2_ablation.qwen_fused_loss as fused_module
    from research.kmd2_ablation.qwen_training import (
        causal_cross_entropy,
        distillation_kl,
    )

    device = torch.device("cuda:0")
    student = torch.randn(2, 5, 11, device=device)
    teacher = torch.randn(2, 5, 11, device=device)
    labels = torch.randint(0, 11, (2, 5), device=device)

    def forbidden_dispatch(*args: object) -> None:
        raise AssertionError("CUDA FP32 must not use the BF16 custom autograd")

    monkeypatch.setattr(
        fused_module._ChunkedHealCEKL,
        "apply",
        staticmethod(forbidden_dispatch),
    )
    actual_ce, actual_kl = fused_module.fused_heal_ce_kl(
        student, teacher, labels, temperature=1.7
    )
    expected_ce = causal_cross_entropy(student, labels)
    expected_kl = distillation_kl(student, teacher, temperature=1.7)
    torch.testing.assert_close(actual_ce, expected_ce, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_kl, expected_kl, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not _cuda_bf16_available(), reason="CUDA BF16 is unavailable"
)
def test_joint_heal_loss_cuda_bf16_matches_reference_and_transfers_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.kmd2_ablation.qwen_training as training

    device = torch.device("cuda:0")
    teacher_device = (
        torch.device("cuda:1")
        if torch.cuda.device_count() > 1
        else device
    )
    generator = torch.Generator(device=device).manual_seed(8123)
    values = torch.randn(
        2, 17, 4099, device=device, dtype=torch.bfloat16, generator=generator
    )
    teacher_values = torch.randn(
        2, 17, 4099, device=device, dtype=torch.float32, generator=generator
    ).to(teacher_device)
    labels = torch.randint(0, 4099, (2, 17), device=device, generator=generator)
    labels[:, ::7] = -100

    reference_student = values.clone().requires_grad_(True)
    transferred_teacher = teacher_values.to(
        device=device, dtype=torch.bfloat16
    )
    expected_ce = training.causal_cross_entropy(reference_student, labels)
    expected_kl = training.distillation_kl(
        reference_student, transferred_teacher, temperature=1.7
    )
    (0.1 * expected_ce + expected_kl).backward()

    actual_student = values.clone().requires_grad_(True)
    actual_teacher = teacher_values.detach().requires_grad_(True)
    implementation = training.fused_heal_ce_kl
    observed: dict[str, object] = {}

    def probe(
        student: torch.Tensor,
        teacher: torch.Tensor,
        target: torch.Tensor,
        *,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed.update(
            teacher_dtype=teacher.dtype,
            teacher_device=teacher.device,
            teacher_requires_grad=teacher.requires_grad,
        )
        return implementation(
            student, teacher, target, temperature=temperature
        )

    monkeypatch.setattr(training, "fused_heal_ce_kl", probe)
    actual = training.compute_heal_loss(
        SimpleNamespace(logits=actual_student),
        SimpleNamespace(logits=actual_teacher),
        labels,
        _config(),
    )
    actual.total.backward()

    assert observed == {
        "teacher_dtype": teacher_values.dtype,
        "teacher_device": teacher_device,
        "teacher_requires_grad": False,
    }
    torch.testing.assert_close(actual.ce, expected_ce, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.kl, expected_kl, rtol=2.0e-5, atol=1.0e-6)
    torch.testing.assert_close(
        actual_student.grad,
        reference_student.grad,
        rtol=5.0e-3,
        atol=5.0e-7,
    )
    exact_fraction = (
        actual_student.grad == reference_student.grad
    ).float().mean()
    assert float(exact_fraction) > 0.99
    assert actual_teacher.grad is None


@pytest.mark.skipif(
    not _cuda_bf16_available(), reason="CUDA BF16 is unavailable"
)
def test_joint_heal_loss_cuda_bf16_ce_is_canonical_across_shapes() -> None:
    from research.kmd2_ablation.qwen_fused_loss import fused_heal_ce_kl
    from research.kmd2_ablation.qwen_training import causal_cross_entropy

    device = torch.device("cuda:0")
    cases = (
        (17, 1, 9, 7),
        (29, 2, 12, 4099),
        (41, 3, 16, 7),
        (53, 2, 17, 257),
    )
    for seed, batch, tokens, vocabulary in cases:
        generator = torch.Generator(device=device).manual_seed(seed)
        student = torch.randn(
            batch,
            tokens,
            vocabulary,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        teacher = torch.randn(
            student.shape,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        labels = torch.randint(
            0,
            vocabulary,
            (batch, tokens),
            device=device,
            generator=generator,
        )
        labels[:, ::5] = -100
        expected = causal_cross_entropy(student, labels)
        actual, _ = fused_heal_ce_kl(
            student, teacher, labels, temperature=1.3
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not _cuda_bf16_available(), reason="CUDA BF16 is unavailable"
)
def test_joint_heal_loss_cuda_peak_memory_is_bounded() -> None:
    from research.kmd2_ablation.qwen_fused_loss import fused_heal_ce_kl
    from research.kmd2_ablation.qwen_training import (
        causal_cross_entropy,
        distillation_kl,
    )

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(119)
    values = torch.randn(
        1, 64, 65536, device=device, dtype=torch.bfloat16, generator=generator
    )
    teacher = torch.randn(
        1, 64, 65536, device=device, dtype=torch.bfloat16, generator=generator
    )
    labels = torch.randint(0, 65536, (1, 64), device=device, generator=generator)

    def peak_increment(*, fused: bool) -> int:
        student = values.clone().requires_grad_(True)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        if fused:
            ce, kl = fused_heal_ce_kl(
                student, teacher, labels, temperature=2.0
            )
        else:
            ce = causal_cross_entropy(student, labels)
            kl = distillation_kl(student, teacher, temperature=2.0)
        (0.1 * ce + kl).backward()
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) - baseline

    reference_peak = peak_increment(fused=False)
    fused_peak = peak_increment(fused=True)
    assert fused_peak < 0.6 * reference_peak
