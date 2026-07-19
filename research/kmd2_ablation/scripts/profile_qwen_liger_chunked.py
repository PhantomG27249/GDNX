"""Profile one all-Triton Package-B training step by CUDA kernel."""

from __future__ import annotations

import argparse

import torch

from research.kmd2_ablation.qwen_hybrid_liger_chunked import (
    set_liger_chunked_training,
)
from research.kmd2_ablation.scripts import benchmark_qwen_chunkwise as shared
from research.kmd2_ablation.scripts.benchmark_qwen_liger_chunked import _run_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    options = parser.parse_args()
    if options.tokens < 16 or options.tokens > 64:
        raise SystemExit("tokens must be in [16,64]")
    if options.batch_size < 1:
        raise SystemExit("batch size must be positive")
    device = torch.device("cuda")
    module = shared._build_package_b(device, torch.bfloat16)
    module.train()
    set_liger_chunked_training(module)
    generator = shared._generator(device, 20261201)
    source = 0.1 * torch.randn(
        options.batch_size,
        options.tokens,
        shared.CAMPAIGN_HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cotangent = torch.randn(
        source.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    _run_step(module, source, cotangent)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
    ) as profile:
        _run_step(module, source, cotangent)
    torch.cuda.synchronize()
    print(
        profile.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
