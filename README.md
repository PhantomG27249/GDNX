# Package B GDN2 + HOLA + Mamba-3-Inspired Decay-Braided Flagship

This archive is the runnable Package B Qwen3.5-0.8B maximum-hybrid package. It
contains four separable recurrent contributions aligned exactly with the four
MIMO ranks. Their continuous decay horizons are the learned native GDN horizon
multiplied by 1:16:64:256, and their clocked GDN update periods use the same
1:16:64:256 tuple. Every lane decays and remains readable on every valid
token, while erase/write/trapezoid updates occur only on that lane's clock.
There is no cross-rank state router or additional CMS state axis. On shared
ticks the layer has four owned GDN-2 writes and homogeneous transitions, and
every token retains sixteen cross-rank reads.
The full stack also retains true MIMO R4, cumulative Mamba-3 complex rotation,
token-dependent exponential-trapezoid input mixing, unit-key-preserving
directional Q/K adaptation, HOLA exact-outer W64 with
C256 processing, and stock convolution.

Unlike aggregate-state Mamba-3 MIMO, distinct decay histories require these
rank contributions to remain separable, so recurrent state memory is four times
the single-state baseline. This package does not claim Mamba-3's state-size
preservation result.

## Performance contract

The FP32 PyTorch recurrence remains the numerical oracle. Canonical
boundary-free CUDA segments dispatch a strict custom Triton forward/backward
kernel, with vectorized HOLA and a joined checkpoint-gradient path around it;
unsupported inputs fail closed to the oracle.

On two RTX 5060 Ti 16-GiB cards, the complete official Qwen trainer measured
30.98 seconds per steady update, including teacher distillation, every declared
loss, backward, finite checks, transactional snapshots, and the CPU-offloaded
fused-AdamW phase. This is 116.2 job steps/hour, or 58.1 steps/GPU-hour when
both occupied cards are charged. Peak allocation was 12.93 GiB student and
3.55 GiB teacher.

## Prepare the environment

Create and activate a Python 3.11 environment, then install `requirements.txt`
manually. The launcher does not install packages or modify the environment.

Edit `config.toml` before running. In particular, confirm the package root,
cache, model, tokenizer, native checkpoint, data, teacher model, and output
paths for the remote machine.

## Run

From any directory, launch the configured full campaign with the root interface:

```bash
bash /home/shadeform/phantom/testb/run.sh
```

Validate configuration and print the planned execution without training:

```bash
bash /home/shadeform/phantom/testb/run.sh --dry-run
```

Run outputs are written beneath the `paths.output` directory selected in
`config.toml`. The archive itself contains no model weights, datasets,
checkpoints, credentials, or generated outputs.
