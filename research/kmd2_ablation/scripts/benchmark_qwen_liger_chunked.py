"""Benchmark the default chunked Triton training path against its fallback."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from research.kmd2_ablation.qwen_hybrid_liger_chunked import (
    set_liger_chunked_training,
)
from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise as shared


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser


def _run_step(module, source, cotangent):
    module.zero_grad(set_to_none=True)
    hidden = source.detach().clone().requires_grad_(True)
    output, _cache = module.scan(hidden)
    (output.float() * cotangent).mean().backward()
    return output.detach(), hidden.grad.detach()


def _run_forward(module, source):
    module.zero_grad(set_to_none=True)
    hidden = source.detach().clone().requires_grad_(True)
    output, _cache = module.scan(hidden)
    return output


def _gradient_error(candidate, baseline) -> tuple[float, float]:
    numerator = torch.zeros((), device="cuda", dtype=torch.float64)
    denominator = torch.zeros((), device="cuda", dtype=torch.float64)
    maximum = 0.0
    baseline_parameters = dict(baseline.named_parameters())
    for name, parameter in candidate.named_parameters():
        reference = baseline_parameters[name]
        if parameter.grad is None or reference.grad is None:
            if parameter.grad is not None or reference.grad is not None:
                return float("inf"), float("inf")
            continue
        delta = parameter.grad.double() - reference.grad.double()
        numerator += delta.square().sum()
        denominator += reference.grad.double().square().sum()
        maximum = max(maximum, float(delta.abs().max()))
    relative_mse = float(numerator / denominator.clamp_min(1e-30))
    return maximum, relative_mse


def _measure(module, source, cotangent, repeats: int) -> dict[str, float | int]:
    module.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _run_step(module, source, cotangent)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / repeats
    peak = max(0, torch.cuda.max_memory_allocated() - resident)
    tokens = source.shape[0] * source.shape[1]
    return {
        "milliseconds": elapsed_ms,
        "tokens_per_second": 1000.0 * tokens / elapsed_ms,
        "incremental_peak_bytes": peak,
    }


def _measure_forward(module, source, repeats: int) -> float:
    module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = _run_forward(module, source)
        # Drop this graph before constructing the next one.  Assignment alone
        # evaluates the next forward while the previous output is still live.
        del output
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    if options.tokens < 16 or options.tokens % 64 not in (0, *range(16, 64)):
        raise SystemExit("tokens must decompose into eligible 16-64 token segments")
    if options.batch_size < 1:
        raise SystemExit("batch size must be positive")
    if options.warmup < 1 or options.repeats < 1:
        raise SystemExit("warmup and repeats must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if options.dtype == "bfloat16" else torch.float32
    torch.manual_seed(20260920)
    baseline = shared._build_package_b(device, dtype)
    candidate = copy.deepcopy(baseline)
    baseline.train()
    candidate.train()
    if set_liger_chunked_training(baseline, False) != 1:
        raise RuntimeError("Package-B baseline layer was not found")
    if set_liger_chunked_training(candidate) != 1:
        raise RuntimeError("Package-B layer was not found")
    generator = shared._generator(device, 20260921)
    source = 0.1 * torch.randn(
        options.batch_size,
        options.tokens,
        shared.CAMPAIGN_HIDDEN,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    cotangent = torch.randn(
        source.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    baseline_output, baseline_input_grad = _run_step(
        baseline, source, cotangent
    )
    candidate_output, candidate_input_grad = _run_step(
        candidate, source, cotangent
    )
    parameter_max, parameter_relative_mse = _gradient_error(candidate, baseline)
    correctness = {
        "output_max_abs": float(
            (candidate_output.float() - baseline_output.float()).abs().max()
        ),
        "input_grad_max_abs": float(
            (candidate_input_grad.float() - baseline_input_grad.float()).abs().max()
        ),
        "parameter_grad_max_abs": parameter_max,
        "parameter_grad_relative_mse": parameter_relative_mse,
    }
    # Correctness artifacts retain complete autograd graphs and would otherwise
    # contaminate both the timing and peak-memory measurements below.
    del (
        baseline_output,
        baseline_input_grad,
        candidate_output,
        candidate_input_grad,
    )
    baseline.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    for _ in range(options.warmup):
        _run_step(baseline, source, cotangent)
        _run_step(candidate, source, cotangent)
    baseline_forward_ms = _measure_forward(baseline, source, options.repeats)
    candidate_forward_ms = _measure_forward(candidate, source, options.repeats)
    baseline_metrics = _measure(baseline, source, cotangent, options.repeats)
    candidate_metrics = _measure(candidate, source, cotangent, options.repeats)
    report = {
        "batch_size": options.batch_size,
        "tokens": options.tokens,
        "dtype": options.dtype,
        "warmup": options.warmup,
        "repeats": options.repeats,
        "baseline": baseline_metrics,
        "liger_chunked": candidate_metrics,
        "forward": {
            "baseline_milliseconds": baseline_forward_ms,
            "liger_chunked_milliseconds": candidate_forward_ms,
            "speedup": baseline_forward_ms / candidate_forward_ms,
        },
        "speedup": (
            baseline_metrics["milliseconds"] / candidate_metrics["milliseconds"]
        ),
        "peak_memory_ratio": (
            candidate_metrics["incremental_peak_bytes"]
            / max(1, baseline_metrics["incremental_peak_bytes"])
        ),
        "correctness": correctness,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
