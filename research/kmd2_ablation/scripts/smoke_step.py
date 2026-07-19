"""One-update GDN-X smoke test: forward+backward timing for the flagship layer.

Builds `package-b-hola-w64` hybrid layers from a synthetic native layer at the
campaign's frozen Qwen3.5-0.8B dimensions (18 linear-attention layers, 16 heads, 128/128
key/value head widths), runs exactly one optimizer update at the campaign
microbatch shape (B=1, T=4096 by default), and reports measured times, peak
memory, and the wall-clock extrapolation for the full 18-layer x 64-update
heal budget.  No model assets are required; projections are randomly
initialized, which does not change the arithmetic being timed.

Usage:
    python -m research.kmd2_ablation.scripts.smoke_step \
        [--tokens 4096] [--layers 1] [--hidden 1024] [--heads 16] [--device cuda:0] \
        [--dtype bfloat16] [--segment 64] [--outer-checkpoint] \
        [--target-layers 18] [--updates 64]
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch


CAMPAIGN_TARGET_LAYERS = 18
CAMPAIGN_UPDATES = 64
CAMPAIGN_TOKENS_PER_MICROBATCH = 4096
CAMPAIGN_HIDDEN_SIZE = 1024
CAMPAIGN_HEADS = 16


def build_flagship_layer(hidden: int, device: torch.device, dtype: torch.dtype,
                         heads: int = CAMPAIGN_HEADS):
    """One converted flagship layer with the campaign feature flags installed."""
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_architecture import build_maximum_control_architecture

    config = SimpleNamespace(
        hidden_size=hidden, linear_num_value_heads=heads, linear_num_key_heads=heads,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
    )
    native = KMD2NativeAttn(config, layer_idx=0)
    with torch.no_grad():
        for parameter in native.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, std=0.02)
        native.rot_proj.weight.zero_()
        native.rot_proj.bias.fill_(-9.0)
    native = native.to(device=device, dtype=dtype)
    prior_rout = native.r_out
    native.r_out = 1  # canonical R1 source, as the production installer requires
    if hasattr(native, "q_slot_scale"):
        del native.q_slot_scale, native.out_mix
    layer = build_maximum_control_architecture(native, "package-b-hola-w64")
    native.r_out = prior_rout
    return layer


def one_update(layers, hidden_states, optimizer, *, aux_loss,
               outer_checkpoint: bool = False) -> dict[str, float]:
    """Forward through the stacked hybrid layers, loss, backward, step."""
    device = hidden_states.device
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    activations = hidden_states
    for layer in layers:
        if outer_checkpoint:
            from torch.utils.checkpoint import checkpoint
            update = checkpoint(layer, activations, use_reentrant=False)
        else:
            update = layer(activations)
        activations = activations + update  # residual, like the backbone
    loss = activations.float().square().mean()
    if aux_loss is not None:
        loss = loss + aux_loss()
    torch.cuda.synchronize(device)
    forward_done = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_done = time.perf_counter()
    from research.kmd2_ablation.qwen_training import _move_optimizer_state_
    _move_optimizer_state_(optimizer, to_parameter_devices=True)
    try:
        optimizer.step()
    finally:
        _move_optimizer_state_(optimizer, to_parameter_devices=False)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    step_done = time.perf_counter()
    return {
        "forward_seconds": forward_done - started,
        "backward_seconds": backward_done - forward_done,
        "optimizer_seconds": step_done - backward_done,
        "total_seconds": step_done - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "loss": float(loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=CAMPAIGN_TOKENS_PER_MICROBATCH)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=1,
                        help="hybrid layers to actually build and time")
    parser.add_argument("--hidden", type=int, default=CAMPAIGN_HIDDEN_SIZE)
    parser.add_argument("--heads", type=int, default=CAMPAIGN_HEADS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--segment", type=int, default=64,
                        help="checkpoint_segment_tokens (0 disables within-layer BPTT chunking)")
    parser.add_argument("--target-layers", type=int, default=CAMPAIGN_TARGET_LAYERS)
    parser.add_argument("--updates", type=int, default=CAMPAIGN_UPDATES)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed warmup updates before the measured one")
    parser.add_argument(
        "--outer-checkpoint", action="store_true",
        help="wrap every hybrid layer like the campaign's decoder checkpoint",
    )
    parser.add_argument(
        "--optimizer-state-offload", action="store_true",
        help="keep Adam moments on CPU outside optimizer.step",
    )
    parser.add_argument("--json", action="store_true", help="emit a single JSON document")
    options = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("smoke_step requires CUDA")
    device = torch.device(options.device)
    dtype = torch.bfloat16 if options.dtype == "bfloat16" else torch.float32
    torch.manual_seed(20260715)

    layers = torch.nn.ModuleList(
        build_flagship_layer(options.hidden, device, dtype, options.heads)
        for _ in range(options.layers)
    )
    for layer in layers:
        layer.checkpoint_segment_tokens = options.segment if options.segment > 0 else None

    from research.kmd2_ablation.qwen_training import package_b_auxiliary_loss

    def aux_loss():
        value, _ = package_b_auxiliary_loss(
            layers, lambda_spec=0.001, lambda_gate=0.001,
            successful_updates=0, specialization_updates=8,
        )
        return value

    trainable = [p for p in layers.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, fused=True
    )
    optimizer._gdnx_cpu_state_offload = options.optimizer_state_offload
    hidden_states = torch.randn(options.batch, options.tokens, options.hidden,
                                device=device, dtype=dtype) * 0.5

    for _ in range(options.warmup):
        one_update(
            layers, hidden_states, optimizer, aux_loss=aux_loss,
            outer_checkpoint=options.outer_checkpoint,
        )
    measured = one_update(
        layers, hidden_states, optimizer, aux_loss=aux_loss,
        outer_checkpoint=options.outer_checkpoint,
    )

    tokens = options.batch * options.tokens
    per_layer = measured["total_seconds"] / options.layers
    update_estimate = per_layer * options.target_layers
    report = {
        "schema_version": "1.0.0",
        "shape": {"batch": options.batch, "tokens": options.tokens,
                  "hidden": options.hidden, "heads": options.heads,
                  "dk": 128, "dv": 128,
                  "timed_layers": options.layers, "dtype": options.dtype,
                  "checkpoint_segment_tokens": options.segment or None,
                  "outer_checkpoint": options.outer_checkpoint,
                  "optimizer": ("fused_adamw_cpu_moment_offload"
                                if options.optimizer_state_offload
                                else "fused_adamw")},
        "measured": measured,
        "throughput": {
            "tokens_per_second_per_layer": tokens / per_layer,
            "tokens_per_second_full_model_estimate": tokens / update_estimate,
        },
        "extrapolation": {
            "assumes": f"{options.target_layers} hybrid layers dominate; backbone/teacher excluded",
            "seconds_per_update_estimate": update_estimate,
            "minutes_per_update_estimate": update_estimate / 60,
            "hours_for_campaign_estimate": update_estimate * options.updates / 3600,
            "updates": options.updates,
        },
        "trainable_parameters": sum(p.numel() for p in trainable),
    }
    if options.json:
        print(json.dumps(report, sort_keys=True))
        return
    print(f"shape: B={options.batch} T={options.tokens} hidden={options.hidden} "
          f"heads={options.heads} "
          f"dtype={options.dtype} segment={options.segment or 'off'} "
          f"timed_layers={options.layers}")
    print(f"forward   {measured['forward_seconds']:8.2f} s")
    print(f"backward  {measured['backward_seconds']:8.2f} s")
    print(f"optimizer {measured['optimizer_seconds']:8.2f} s")
    print(f"total     {measured['total_seconds']:8.2f} s   peak {measured['peak_memory_gib']:.2f} GiB")
    print(f"per-layer {per_layer:8.2f} s   -> {options.target_layers}-layer update ~ "
          f"{update_estimate/60:.1f} min")
    print(f"campaign  {options.updates} updates ~ "
          f"{update_estimate*options.updates/3600:.1f} h (hybrid layers only)")


if __name__ == "__main__":
    main()
