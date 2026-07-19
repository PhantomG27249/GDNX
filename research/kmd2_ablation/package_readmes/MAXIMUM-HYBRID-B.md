# Maximum Hybrid Package B

This archive runs the four MIMO-rank-contribution GDN-2/MIMO-R4 flagship. Package B
uses the four MIMO ranks themselves as four separable decay histories. The
learned native GDN log-decay, including `A_log`, `dt_bias`, the content
projection, and pair-tied channel residual, is scaled by fixed factors
`1/s_r` for native-relative horizon multipliers `s=(1,16,64,256)`. Lane zero
therefore retains the native GDN decay law, while the other lanes have 16x,
64x, and 256x its instantaneous horizon. The same tuple independently defines
per-rank CMS clocks: all ranks decay
and remain readable every valid token, while GDN erase/write/trapezoid updates
occur every 1, 16, 64, or 256 valid tokens. There is no extra CMS state axis and
no 4x4 state router. On shared ticks the layer performs four owned GDN-2 writes
and rankwise homogeneous transitions; every token retains sixteen cross-rank reads.

The decay is tied inside adjacent complex coordinate pairs, so it commutes with
the cumulative Mamba-3-inspired rotations. The state input uses a token-dependent
exponential-trapezoid coefficient and transports the previous endpoint through
the same current decay-plus-erase GDN transition as the recurrent state.
Its Option-A initialization uses trapezoid logit +4
(`lambda≈0.982`), keeping conversion near the native current-endpoint update
while retaining a live gradient for the previous endpoint.
Lookahead is deliberately absent. HOLA uses checkpoint schema v2 with a
`cache_gate_logit` initialized to -4; legacy direct-amplitude gate checkpoints
fail closed.

Keeping distinct decay histories requires four separable rank contributions and
therefore four times the recurrent-state memory of the aggregate-state baseline.
This adaptation does not claim Mamba-3 MIMO's single-state size preservation.
Its selected config is `campaigns/maximum_hybrid/09-package-b-hola-w64.json`.

The forced PyTorch FP32 path remains the numerical oracle. Canonical all-valid,
boundary-free CUDA segments use the strict custom
`research.kmd2_ablation.qwen_hybrid_triton` forward/backward recurrence;
unsupported shapes or masks fail closed to the oracle. Projection,
normalization, global read mixing, and exact HOLA are vectorized around it.

Measured on two RTX 5060 Ti 16-GiB cards, the complete official-model trainer
runs a steady update in 30.98 seconds: 116.2 job steps/hour, or 58.1
steps/GPU-hour after charging both the student and teacher cards. Peak allocated
memory is 12.93 GiB student / 3.55 GiB teacher. The measurement includes every
declared loss, backward, finite checks, rollback snapshots, and the
CPU-offloaded fused-AdamW optimizer phase.

For the supported remote interface, start with the archive-root `README.md`,
edit root `config.toml`, install root `requirements.txt` manually, and run:

```bash
bash /home/shadeform/phantom/testb/run.sh
```

Use `bash /home/shadeform/phantom/testb/run.sh --dry-run` to validate the
configuration without training. The lower-level research launchers are bundled
for provenance and internal diagnostics, not as the preferred package entry point.

Stock convolution is retained and trainable. The archive intentionally contains no models, weights, datasets, checkpoints,
credentials, run outputs, or generated artifacts.
