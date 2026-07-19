"""Export the canonical GDN-X native warm-start checkpoint from stock Qwen3.5.

The native checkpoint is fully determined by the stock model: the nine native
GatedDeltaNet tensors are copied warm and the KMD-2 degrees of freedom are
their documented identity inits (rot_proj zeros/-9, decay_chan zeros, bw_off
zeros).  Uses the real GDN3UpgradeManager + KMD2NativeAttn construction so the
exported tensors match the production installer's strict 13-suffix contract
(q_slot_scale/out_mix are runtime-constructed and deliberately not exported).

Usage:
    GDN3_KMD2_NATIVE=1 GDN3_KMD2_ROUT=4 python -m \
        research.kmd2_ablation.scripts.export_native_checkpoint \
        --model /path/to/qwen3_5_0_8b --out native_checkpoint.pt [--dtype bfloat16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

EXPORT_SUFFIXES = (
    "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
    "in_proj_a.weight", "conv1d.weight", "dt_bias", "A_log",
    "norm.weight", "out_proj.weight", "rot_proj.weight",
    "rot_proj.bias", "decay_chan", "bw_off",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    options = parser.parse_args()

    if os.environ.get("GDN3_KMD2_NATIVE") != "1":
        raise SystemExit("set GDN3_KMD2_NATIVE=1 (and GDN3_KMD2_ROUT) for a canonical export")

    from transformers import AutoModelForCausalLM

    from gdn3.gdn3_upgrade import GDN3UpgradeManager
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.architecture import TARGET_LAYERS

    dtype = torch.bfloat16 if options.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(options.model), dtype=dtype)
    config = getattr(model.config, "text_config", model.config)
    manager = GDN3UpgradeManager(model, config=config)
    upgraded = tuple(manager.apply_upgrade())
    if upgraded != TARGET_LAYERS:
        raise SystemExit(f"upgraded layers {upgraded} != canonical {TARGET_LAYERS}")

    checkpoint: dict[str, torch.Tensor] = {}
    for index in upgraded:
        module = model.model.layers[index].linear_attn
        if type(module) is not KMD2NativeAttn:
            raise SystemExit(f"layer {index} is not KMD2NativeAttn")
        state = module.state_dict()
        for suffix in EXPORT_SUFFIXES:
            checkpoint[f"model.layers.{index}.linear_attn.{suffix}"] = (
                state[suffix].to(dtype).cpu().clone()
            )

    options.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = options.out.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(options.out)
    print(json.dumps({
        "tensors": len(checkpoint),
        "layers": list(upgraded),
        "dtype": options.dtype,
        "sha256": hashlib.sha256(options.out.read_bytes()).hexdigest(),
        "bytes": options.out.stat().st_size,
    }, indent=1))


if __name__ == "__main__":
    main()
