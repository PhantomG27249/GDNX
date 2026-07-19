"""Package B: four rank-aligned states with a full 4x4 routed read."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen_hybrid_components import HybridComponents, TIMESCALES
from .qwen_hybrid_math import RANK, REFERENCE_IMPLEMENTATION, RUNTIME_BACKEND


def _can_use_torch_chunk_with_cache(cache_active: bool) -> bool:
    """Allow generalized-WY only while constructing an autograd graph."""
    del cache_active
    return torch.is_grad_enabled()


def gdn_homogeneous_transition(state: Tensor, key: Tensor, erase: Tensor,
                               gamma: Tensor) -> Tensor:
    """Apply A_t(X)=Gamma_t X-k_t(beta_e k_t^T Gamma_t X)."""
    decayed = gamma[..., None] * state
    memory = torch.einsum("bhrk,bhrkv->bhrv", erase * key, decayed)
    return decayed - key[..., None] * memory[..., None, :]


@dataclass(frozen=True)
class FourStateHybridCache:
    states: Tensor
    phase: Tensor
    previous_key: Tensor
    previous_value: Tensor
    conv_tail: Tensor
    has_history: Tensor
    update_count: Tensor
    hola_state: object | None = None


class QwenFourStateHybrid(nn.Module):
    """Reference FP32 four-state GDN-2/R4 hybrid with exact chunk carry."""

    rank = RANK
    timescales = TIMESCALES
    update_periods = (1, 16, 64, 256)
    scan_implementation = REFERENCE_IMPLEMENTATION

    @staticmethod
    def actual_implementation_identity() -> str:
        return REFERENCE_IMPLEMENTATION

    def __init__(self, native: nn.Module) -> None:
        super().__init__()
        self.H, self.dv = native.H, native.dv
        if native.dk % (2 * RANK) != 0:
            raise ValueError(
                "Package B native key width must partition into four compact "
                "complex-pair bands"
            )
        # Cache-slot identity for transformers' model-level Cache bookkeeping.
        self.layer_idx = getattr(native, "layer_idx", None)
        self.r_out = RANK
        self.components = HybridComponents.from_native(native, package="four_state")
        self.dk = self.components.dk
        self.key_dim, self.value_dim, self.conv_k = self.H * self.dk, native.value_dim, native.conv_k
        self.rot_proj = copy.deepcopy(native.rot_proj)
        from .qwen_hybrid_hola import HybridHOLACache
        self.hola = HybridHOLACache(width=64, block_size=256, heads=self.H, rank_in=RANK,
                                    key_dim=self.dk, value_dim=self.dv)
        self.hola.to(device=self.components.q_weight.device, dtype=self.components.q_weight.dtype)
        # Tokens per checkpointed BPTT segment inside _scan; None disables the
        # within-layer chunking and retains the whole token-loop graph.
        self.checkpoint_segment_tokens: int | None = 64
        # Runtime-only research switch.  False preserves the validated PyTorch
        # production path and state-dict schema; True attempts the local
        # Liger-style chunked Triton training operator and fails closed to the
        # existing dispatch chain whenever its narrow envelope is not met.
        # Default training path.  Eligibility remains fail-closed, so missing
        # Triton/CUDA, no-grad execution, and unsupported shapes continue to
        # fall through to the established PyTorch/persistent authorities.
        self.use_liger_chunked_kernel: bool = True

    def _active(self, name: str, default=True):
        return getattr(self, "active_feature_flags", {}).get(name, default)

    @classmethod
    def from_native(cls, native: nn.Module) -> "QwenFourStateHybrid":
        return cls(native)

    def recurrent_state_bytes(self, *, batch_size: int = 1) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return batch_size * RANK * self.H * self.dk * self.dv * 4

    def resource_report(self, *, batch_size: int = 1) -> dict[str, object]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        hidden = self.components.hidden
        conv_element = self.components.q_weight.element_size()
        persistent = {
            "states": batch_size * RANK * self.H * self.dk * self.dv * 4,
            "phase_history": batch_size * self.H * RANK * (self.dk // 2) * 4,
            "trapezoid_previous_key": batch_size * RANK * self.H * self.dk * 4,
            "trapezoid_previous_value": batch_size * RANK * self.H * self.dv * 4,
            "convolution_history": batch_size * (self.conv_k - 1) * hidden * conv_element,
            "history_flags": batch_size * RANK,
            "update_counter": batch_size * 8,
        }
        mixer = self.components.output_mixer
        parameters = {
            "output_mixer": mixer.numel() * mixer.element_size(),
            "trapezoid_projection": sum(
                p.numel() * p.element_size() for p in self.components.trapezoid_proj.parameters()
            ),
        }
        return {
            "persistent": persistent, "persistent_bytes": sum(persistent.values()),
            "parameters": parameters, "parameter_bytes": sum(parameters.values()),
            "time_braid_mode": "native_gdn_relative_horizon_multipliers_with_clocked_gdn_updates",
            "timescale_axis": "mimo_rank",
            "horizon_multipliers": TIMESCALES,
            "native_decay_lane": 0,
            "cadence_or_cms_updates": True,
            "update_periods": self.update_periods,
            "off_tick_behavior": "passive_decay_and_read_without_erase_or_write",
            "state_router": False,
            "trapezoid_projection_flops_per_token": 2 * hidden * self.H * RANK,
            "transition_coefficients_per_token": RANK,
            "output_cross_reads_per_token": RANK * RANK,
            "read_compute_native_equivalents": RANK,
            "compact_key_band_width": self.dk,
            "lazy_off_tick_decay": True,
            "trapezoid_history": "exact_key_value_outer_product_factors",
        }

    def transformation_manifest(self) -> dict[str, object]:
        result = self.components.transformation_manifest()
        result["parameters"] = {
            name: {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in self.state_dict().items()
        }
        result.update({
            "state_shape": ("B", self.H, RANK, self.dk, self.dv),
            "state_count": RANK, "write_paths": RANK, "transition_paths": RANK,
            "read_paths": RANK * RANK,
            "timescales": TIMESCALES,
            "update_periods": self.update_periods,
            "state_bytes_per_batch": self.recurrent_state_bytes(),
            "implementation": self.actual_implementation_identity(),
            "scan_implementation": self.scan_implementation,
            # Separate machine-readable fields: `implementation` remains the
            # semantic oracle every path must match; `runtime_backend` names
            # what production dispatch actually executes on eligible segments.
            "runtime_backend": RUNTIME_BACKEND,
            "resources": self.resource_report(),
            "hola_resources": (self.hola.resource_report()
                               if self._active("cache_policy", "hola_exact_outer_w64") != "none" else None),
            "hola_implementation": (self.hola.implementation_reference
                                    if self._active("cache_policy", "hola_exact_outer_w64") != "none" else None),
        })
        return result

    def architecture_tensor_manifest(self) -> dict[str, tuple]:
        return {"copied": (), "transformed": (),
                "new": tuple(name for name, _ in self.named_parameters())}

    @staticmethod
    def _adjacent_complex_layout(value: Tensor) -> Tensor:
        half = value.shape[-1] // 2
        return torch.stack((value[..., :half], value[..., half:]), -1).flatten(-2)

    @staticmethod
    def history_active(boundary: Tensor, valid: Tensor, has_history: Tensor) -> Tensor:
        return valid[:, None] & ~boundary[:, None] & has_history

    def lane_update_mask(self, update_count: Tensor, valid: Tensor) -> Tensor:
        """Return the rank-lane CMS ticks for each sequence row."""
        if update_count.ndim != 1 or valid.shape != update_count.shape or valid.dtype != torch.bool:
            raise ValueError("update_count and valid must be matching [B] integer/bool tensors")
        if update_count.dtype != torch.int64 or update_count.device != valid.device:
            raise ValueError("update_count must be int64 and share valid's device")
        periods = update_count.new_tensor(self.update_periods)
        return valid[:, None] & update_count[:, None].remainder(periods[None]).eq(0)

    def _initial_cache(self, hidden: Tensor) -> FourStateHybridCache:
        batch = hidden.shape[0]
        states = torch.zeros(batch, self.H, RANK, self.dk, self.dv, device=hidden.device)
        return FourStateHybridCache(
            states=states,
            phase=torch.zeros(batch, self.H, RANK, self.dk // 2, device=hidden.device),
            previous_key=torch.zeros(batch, self.H, RANK, self.dk, device=hidden.device),
            previous_value=torch.zeros(batch, self.H, RANK, self.dv, device=hidden.device),
            conv_tail=hidden.new_zeros(batch, self.conv_k - 1, hidden.shape[-1]),
            has_history=torch.zeros(batch, RANK, dtype=torch.bool, device=hidden.device),
            update_count=torch.zeros(batch, dtype=torch.int64, device=hidden.device),
            hola_state=None,
        )

    def _validate_inputs(self, hidden: Tensor, boundary: Tensor, valid: Tensor) -> None:
        if hidden.ndim != 3 or hidden.shape[-1] != self.components.hidden:
            raise ValueError("hidden states must be [B,T,hidden]")
        if hidden.dtype != self.components.q_weight.dtype or hidden.device != self.components.q_weight.device:
            raise ValueError("hidden states must match module dtype/device")
        expected = hidden.shape[:2]
        if (boundary.shape != expected or valid.shape != expected or boundary.dtype != torch.bool
                or valid.dtype != torch.bool or boundary.device != hidden.device or valid.device != hidden.device):
            raise ValueError("boundary and valid must be boolean [B,T] on the input device")

    def _validate_cache(self, cache: FourStateHybridCache, hidden: Tensor) -> None:
        if type(cache) is not FourStateHybridCache:
            raise TypeError("initial_cache must be FourStateHybridCache")
        B = hidden.shape[0]
        expected = {
            "states": ((B, self.H, RANK, self.dk, self.dv), torch.float32),
            "phase": ((B, self.H, RANK, self.dk // 2), torch.float32),
            "previous_key": ((B, self.H, RANK, self.dk), torch.float32),
            "previous_value": ((B, self.H, RANK, self.dv), torch.float32),
            "conv_tail": ((B, self.conv_k - 1, self.components.hidden), hidden.dtype),
            "has_history": ((B, RANK), torch.bool),
            "update_count": ((B,), torch.int64),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(cache, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape or value.dtype != dtype or value.device != hidden.device:
                raise ValueError(f"initial_cache {name} shape/dtype/device mismatch")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"initial_cache {name} must be finite")
        if bool((cache.update_count < 0).any()):
            raise ValueError("initial_cache update_count must be nonnegative")

    def _project_convolved(self, hidden: Tensor, tail: Tensor, boundary: Tensor,
                           valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B, T = hidden.shape[:2]
        conv_dim = 2 * self.key_dim + self.value_dim
        conv_weight = self._compact_conv_weight().reshape(RANK, conv_dim, 1, self.conv_k)
        lanes: list[list[tuple[Tensor, Tensor, Tensor]]] = [[] for _ in range(RANK)]
        for token in range(T):
            reset = (boundary[:, token] & valid[:, token])[:, None, None]
            tail = torch.where(reset, torch.zeros_like(tail), tail)
            window = torch.cat((tail, hidden[:, token:token + 1]), 1)
            q, k, v, _, _, _ = self.components.project_inputs(window)
            for rank in range(RANK):
                qkv = torch.cat((q[:, :, :, rank].flatten(2), k[:, :, :, rank].flatten(2),
                                 v[:, :, :, rank].flatten(2)), -1)
                mixed = F.silu(F.conv1d(qkv.transpose(1, 2), conv_weight[rank],
                                        groups=qkv.shape[-1])).transpose(1, 2)[:, 0]
                lanes[rank].append(torch.split(mixed, (self.key_dim, self.key_dim, self.value_dim), -1))
            shifted = torch.cat((tail[:, 1:], hidden[:, token:token + 1]), 1)
            tail = torch.where(valid[:, token, None, None], shifted, tail)
        q = torch.stack([torch.stack([x[0] for x in lane], 1).reshape(B, T, self.H, self.dk)
                         for lane in lanes], 3)
        k = torch.stack([torch.stack([x[1] for x in lane], 1).reshape(B, T, self.H, self.dk)
                         for lane in lanes], 3)
        v = torch.stack([torch.stack([x[2] for x in lane], 1).reshape(B, T, self.H, self.dv)
                         for lane in lanes], 3)
        return q, k, v, tail

    def _compact_conv_weight(self) -> Tensor:
        indices = self.components.conv_channel_indices.reshape(-1)
        return self.components.conv1d.weight.index_select(0, indices)

    @staticmethod
    def _flatten_hola_state(state: object) -> tuple[Tensor, ...]:
        import dataclasses
        return tuple(getattr(state, field.name) for field in dataclasses.fields(state))

    @staticmethod
    def _rotate_pairs(value: Tensor, phase: Tensor) -> Tensor:
        """apply_complex_rotation without per-call validation (hot loop)."""
        real, imag = value[..., 0::2], value[..., 1::2]
        cosine, sine = torch.cos(phase), torch.sin(phase)
        return torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), dim=-1).flatten(-2)

    def _project_convolved_fast(self, hidden: Tensor, tail: Tensor
                                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Boundary-free, all-valid vectorized equivalent of _project_convolved.

        The projection is per-token linear and the convolution is depthwise
        causal, so projecting cat(tail, hidden) once and convolving each rank's
        channel stack over the whole sequence computes exactly the reference's
        per-token windows.
        """
        B, T = hidden.shape[:2]
        extended = torch.cat((tail, hidden), 1)
        q, k, v, _, _, _ = self.components.project_inputs(extended)

        def channels(x: Tensor, width: int) -> Tensor:
            return x.permute(0, 3, 2, 4, 1).reshape(B, RANK, self.H * width, -1)

        qkv = torch.cat((channels(q, self.dk), channels(k, self.dk),
                         channels(v, self.dv)), 2)
        conv_dim = qkv.shape[2]
        mixed = F.silu(F.conv1d(qkv.reshape(B, RANK * conv_dim, -1),
                                self._compact_conv_weight(), groups=RANK * conv_dim))
        mixed = mixed.reshape(B, RANK, conv_dim, T).permute(0, 3, 1, 2)
        qm, km, vm = torch.split(mixed, (self.key_dim, self.key_dim, self.value_dim), -1)

        def heads(x: Tensor, width: int) -> Tensor:
            return x.reshape(B, T, RANK, self.H, width).permute(0, 1, 3, 2, 4)

        new_tail = extended[:, extended.shape[1] - (self.conv_k - 1):]
        return heads(qm, self.dk), heads(km, self.dk), heads(vm, self.dv), new_tail

    def _scan(self, hidden: Tensor, boundary: Tensor, valid: Tensor,
              initial_cache: FourStateHybridCache | None) -> tuple[Tensor, FourStateHybridCache]:
        B, T = hidden.shape[:2]
        self._validate_inputs(hidden, boundary, valid)
        dtype = hidden.dtype
        cache = self._initial_cache(hidden) if initial_cache is None else initial_cache
        self._validate_cache(cache, hidden)
        states, phase = cache.states, cache.phase
        previous_key, previous_value = cache.previous_key, cache.previous_value
        has_history = cache.has_history
        update_count = cache.update_count
        hola_state = cache.hola_state
        # One host sync per scan buys the boundary-free/all-valid fast path:
        # per-token reset/valid masking, defensive finiteness syncs, and the
        # HOLA promotion sync are all skippable when nothing is masked.
        flags = torch.stack(((boundary & valid).any(), valid.all())).cpu()
        fast = ((not bool(flags[0])) and bool(flags[1])
                and not getattr(self, "force_reference_path", False))
        if fast:
            q, k, v, new_tail = self._project_convolved_fast(hidden, cache.conv_tail)
        else:
            q, k, v, new_tail = self._project_convolved(hidden, cache.conv_tail, boundary, valid)
        _, _, _, erase, write, z = self.components.project_inputs(hidden)
        # GDN-compatible adaptation of Mamba-3 BC/QK bias: normalize the base
        # directions, apply diagonal/additive directional adaptation, then
        # renormalize so recurrent keys remain exactly unit norm.
        q = F.normalize(self._adjacent_complex_layout(q).float(), dim=-1, eps=1e-6)
        k = F.normalize(self._adjacent_complex_layout(k).float(), dim=-1, eps=1e-6)
        if self._active("affine_qk"):
            q, k = self.components.affine_qk(q, k)
        q = F.normalize(q, dim=-1, eps=1e-6) * self.dk ** -0.5
        k = F.normalize(k, dim=-1, eps=1e-6)
        v, erase, write = v.float(), self._adjacent_complex_layout(erase.float().sigmoid()), write.float().sigmoid()
        gamma = self._adjacent_complex_layout(self.components.decay_gamma(hidden).float())
        if self._active("rotation"):
            native_theta = F.softplus(self.rot_proj(hidden)).float()
            if self.components.compact_key_bands:
                theta = native_theta.reshape(B, T, self.H, RANK, self.dk // 2)
            else:
                theta = native_theta.reshape(B, T, self.H, 1, self.dk // 2).expand(
                    B, T, self.H, RANK, self.dk // 2
                )
            theta = theta + self.components.phase_logits(hidden).float()
        else:
            # Inference-time rotation reliance control: freeze phase at its
            # carried value so q/k stay in a fixed frame.  Training keeps the
            # default-on flag; only explicit evaluation ablations flip it.
            theta = hidden.new_zeros(B, T, self.H, RANK, self.dk // 2, dtype=torch.float32)
        trap_lambda = self.components.trapezoid_lambda(hidden)
        cache_active = self._active("cache_policy", "hola_exact_outer_w64") != "none"
        if cache_active:
            hola_state = self.hola._empty(B, hidden.device) if hola_state is None else hola_state
            self.hola._validate_inputs(q, k, v, torch.zeros(B, T, self.H, device=hidden.device),
                                       None, valid, boundary)
            self.hola._validate_state(hola_state, B, hidden.device)
        initial_block_fill = None
        if cache_active and fast:
            # Check uniformity once per full scan.  Every all-valid segment
            # advances occupancy deterministically, so subsequent checkpoint
            # segments derive their fill from ``start`` without synchronizing.
            fill_range = torch.stack((hola_state.block_count.amin(),
                                      hola_state.block_count.amax())).cpu()
            if int(fill_range[0]) == int(fill_range[1]):
                initial_block_fill = int(fill_range[0])
        hola_flat = self._flatten_hola_state(hola_state) if cache_active else ()
        carry = (states, phase, previous_key, previous_value, has_history, update_count)
        # Within-layer chunked BPTT: the reference token loop retains every
        # per-token [B,H,4,dk,dv] intermediate, which is unaffordable at long T
        # even under per-layer activation checkpointing.  Segmented
        # non-reentrant checkpointing caps the live graph at one segment. Its
        # forward/cache values match the plain loop exactly; backward remains
        # within the pinned FP32 parity envelope despite accumulation order.
        segment = getattr(self, "checkpoint_segment_tokens", 64)
        segmented = type(segment) is int and segment > 0
        use_checkpoint = (
            segmented and torch.is_grad_enabled()
            and (hidden.requires_grad or any(p.requires_grad for p in self.parameters()))
        )
        # Keep the same bounded computational segment in inference and outer
        # checkpoint recomputation.  Only the autograd wrapper is conditional;
        # this also gives fused backends a stable <=segment token contract.
        step = segment if segmented else T
        # The 4x4xVxV mixer is a large parameter.  Applying it independently
        # in every 64-token checkpoint segment makes autograd add 64 full-size
        # parameter gradients per layer.  On the canonical outer-checkpointed
        # training path, retain the bounded normalized reads and apply both
        # mixer contractions once across T.  This changes neither the
        # per-token equation nor HOLA visibility, only gradient accumulation
        # granularity.  The fallback remains available for memory diagnostics.
        # Under no_grad there is no parameter-gradient accumulation to save,
        # and retaining every [B,T,H,4,4,V] read until the final cat scales
        # the inference transient linearly with T (~1.7x peak at 4K, tens of
        # GiB per layer at 32K) — so global mixing requires a live grad mode.
        global_mix = (
            fast
            and initial_block_fill is not None
            and torch.is_grad_enabled()
            and not getattr(self, "force_segment_mixing", False)
        )
        # The true chunked backend owns the full recurrent sequence.  Its
        # DPLR carry kernel scans fixed-size boundaries internally and its
        # intra-chunk kernels expose all token reads in parallel.  Keeping it
        # outside this Python checkpoint loop is essential: wrapping one
        # custom op per segment would rebuild WY factors three times and erase
        # the point of a chunked training kernel.
        true_chunked_sequence = False
        if (
            fast
            and torch.is_grad_enabled()
            and getattr(self, "use_liger_chunked_kernel", False)
            and not getattr(self, "force_torch_recurrence", False)
            and not torch.are_deterministic_algorithms_enabled()
            and T >= 16
            and q.is_cuda
            and q.shape[-2:] == (4, 32)
            and v.shape[-1] == 128
            and (not cache_active or initial_block_fill is not None)
        ):
            from .qwen_hybrid_liger_dplr import true_chunked_dplr_available

            true_chunked_sequence = true_chunked_dplr_available()
        if true_chunked_sequence:
            from .qwen_hybrid_liger_dplr import (
                true_chunked_dplr_four_state_sequence,
            )

            periods = update_count.new_tensor(self.update_periods)
            token_counts = update_count[:, None] + torch.arange(
                T, device=update_count.device, dtype=torch.int64
            )[None]
            fast_ticks = token_counts[:, :, None].remainder(periods).eq(0)
            phase_sequence = torch.cat((phase[:, None], theta), 1).cumsum(1)[:, 1:]
            fast_queries = self._rotate_pairs(q, phase_sequence)
            fast_keys = self._rotate_pairs(k, phase_sequence)
            triton_lambda = (
                trap_lambda.float()
                if self._active("trapezoid")
                else torch.ones_like(trap_lambda, dtype=torch.float32)
            )
            (
                reads,
                states,
                previous_key,
                previous_value,
                innovation_sq,
                has_history,
                update_count,
            ) = true_chunked_dplr_four_state_sequence(
                fast_queries,
                fast_keys,
                v,
                erase,
                write,
                gamma,
                triton_lambda,
                states,
                previous_key,
                previous_value,
                has_history,
                update_count,
            )
            phase = phase_sequence[:, -1]
            pieces = []
            hola_pieces = []
            post_inputs = tuple(
                torch.split(value, step, dim=1)
                for value in (
                    reads,
                    fast_queries,
                    fast_keys,
                    v,
                    z,
                    innovation_sq,
                    fast_ticks,
                )
            )
            start = 0
            for chunks in zip(*post_inputs):
                segment_block_fill = (
                    (initial_block_fill + start) % self.hola.block_size
                    if initial_block_fill is not None
                    else -1
                )
                args = (
                    segment_block_fill,
                    global_mix,
                    *chunks,
                    *hola_flat,
                )
                if use_checkpoint:
                    from torch.utils.checkpoint import checkpoint

                    results = checkpoint(
                        self._post_recurrence_segment,
                        *args,
                        use_reentrant=False,
                    )
                else:
                    results = self._post_recurrence_segment(*args)
                pieces.append(results[0])
                if global_mix and cache_active:
                    hola_pieces.append(results[1])
                hola_flat = tuple(results[2:])
                start += int(chunks[0].shape[1])
            if global_mix:
                normalized = torch.cat(pieces, 1)
                mixed_sequence = torch.einsum(
                    "hijvw,bthijw->bthv",
                    self.components.output_mixer,
                    normalized,
                )
                if cache_active:
                    hola_reads = torch.cat(hola_pieces, 1)
                    cache_mixed = torch.einsum(
                        "hijvw,bthijw->bthv",
                        self.components.output_mixer.float(),
                        hola_reads,
                    )
                    cache_gate = self.components.cache_gate_amplitude.float()[
                        None, None, :, None
                    ]
                    mixed_sequence = mixed_sequence + cache_gate * cache_mixed
                out = self.components.out_proj(
                    mixed_sequence.flatten(2).to(dtype)
                )
            else:
                out = torch.cat(pieces, 1)
            if cache_active:
                from .qwen_hybrid_hola import HybridHOLAState

                hola_state = HybridHOLAState(*hola_flat)
            return out, FourStateHybridCache(
                states=states,
                phase=phase,
                previous_key=previous_key,
                previous_value=previous_value,
                conv_tail=new_tail,
                has_history=has_history,
                update_count=update_count,
                hola_state=hola_state,
            )
        pieces = []
        hola_pieces = []
        # Use one split node per sequence tensor.  Independent ``x[:, a:b]``
        # views create one SliceBackward per segment; each of those allocates
        # and accumulates a full-[T] gradient, turning a 64-segment scan into
        # hundreds of redundant multi-megabyte adds.  SplitBackward joins the
        # segment gradients once without changing any segment payload.
        segment_inputs = tuple(
            torch.split(value, step, dim=1)
            for value in (
                q, k, v, erase, write, z, gamma, theta, trap_lambda,
                boundary, valid,
            )
        )
        start = 0
        for chunks in zip(*segment_inputs):
            stop = start + int(chunks[0].shape[1])
            segment_block_fill = (
                (initial_block_fill + start) % self.hola.block_size
                if initial_block_fill is not None else -1
            )
            args = (fast, segment_block_fill, global_mix,
                    *chunks, *carry, *hola_flat)
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                results = checkpoint(self._hybrid_segment, *args, use_reentrant=False)
            else:
                results = self._hybrid_segment(*args)
            pieces.append(results[0])
            if global_mix and cache_active:
                hola_pieces.append(results[1])
            carry, hola_flat = tuple(results[2:8]), tuple(results[8:])
            start = stop
        if global_mix:
            normalized = torch.cat(pieces, 1)
            mixed_sequence = torch.einsum(
                "hijvw,bthijw->bthv",
                self.components.output_mixer,
                normalized,
            )
            if cache_active:
                hola_reads = torch.cat(hola_pieces, 1)
                cache_mixed = torch.einsum(
                    "hijvw,bthijw->bthv",
                    self.components.output_mixer.float(),
                    hola_reads,
                )
                cache_gate = self.components.cache_gate_amplitude.float()[
                    None, None, :, None
                ]
                mixed_sequence = mixed_sequence + cache_gate * cache_mixed
            out = self.components.out_proj(
                mixed_sequence.flatten(2).to(dtype)
            )
        else:
            out = (
                torch.cat(pieces, 1)
                if pieces else hidden.new_empty(B, 0, self.components.hidden)
            )
        states, phase, previous_key, previous_value, has_history, update_count = carry
        if cache_active:
            from .qwen_hybrid_hola import HybridHOLAState
            hola_state = HybridHOLAState(*hola_flat)
        return out, FourStateHybridCache(
            states=states, phase=phase, previous_key=previous_key,
            previous_value=previous_value,
            conv_tail=new_tail, has_history=has_history,
            update_count=update_count, hola_state=hola_state,
        )

    def _post_recurrence_segment(
        self,
        block_fill_value: int,
        defer_global_mix: bool,
        reads: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        z: Tensor,
        innovation_sq: Tensor,
        ticks: Tensor,
        *hola_flat: Tensor,
    ) -> tuple[Tensor, ...]:
        """Checkpointable HOLA/norm/mixer tail after a full chunk scan."""
        from .qwen_hybrid_hola import HybridHOLAState

        cache_active = bool(hola_flat)
        hola_state = HybridHOLAState(*hola_flat) if cache_active else None
        B, T = reads.shape[:2]
        dtype = z.dtype
        gates = z.to(dtype)[:, :, :, :, None, :].expand(
            B, T, self.H, RANK, RANK, self.dv
        )
        normalized = self.components.norm(
            reads.reshape(-1, self.dv).to(dtype),
            gates.reshape(-1, self.dv),
        ).reshape(B, T, self.H, RANK, RANK, self.dv)
        hola_read = query.new_empty(0)
        if cache_active:
            assert hola_state is not None
            tick_count = ticks.sum(-1).to(innovation_sq.dtype).clamp_min(1.0)
            score = torch.sqrt(
                innovation_sq.sum(-1) / tick_count[:, :, None]
            )
            hola_read, hola_state, _ = self.hola.scan_fast(
                hola_state,
                query,
                key,
                value,
                score,
                block_fill_value,
            )
        if defer_global_mix:
            result = (normalized, hola_read)
        else:
            mixed = torch.einsum(
                "hijvw,bthijw->bthv",
                self.components.output_mixer,
                normalized,
            )
            if cache_active:
                cache_mixed = torch.einsum(
                    "hijvw,bthijw->bthv",
                    self.components.output_mixer.float(),
                    hola_read,
                )
                cache_gate = self.components.cache_gate_amplitude.float()[
                    None, None, :, None
                ]
                mixed = mixed + cache_gate * cache_mixed
            result = (
                self.components.out_proj(mixed.flatten(2).to(dtype)),
                query.new_empty(0),
            )
        if cache_active:
            assert hola_state is not None
            result = result + self._flatten_hola_state(hola_state)
        return result

    def _hybrid_segment(self, fast: bool, block_fill_value: int,
                        defer_global_mix: bool,
                        q: Tensor, k: Tensor, v: Tensor, erase: Tensor,
                        write: Tensor, z: Tensor, gamma: Tensor, theta: Tensor,
                        trap_lambda: Tensor, boundary: Tensor, valid: Tensor, states: Tensor,
                        phase: Tensor, previous_key: Tensor, previous_value: Tensor,
                        has_history: Tensor,
                        update_count: Tensor, *hola_flat: Tensor) -> tuple[Tensor, ...]:
        from .qwen_hybrid_hola import HybridHOLAState, four_state_normalized_update_score
        cache_active = bool(hola_flat)
        hola_state = HybridHOLAState(*hola_flat) if cache_active else None
        dtype = z.dtype
        B, T = valid.shape
        # Hoisted per-segment constants for the reference fallback.
        periods = update_count.new_tensor(self.update_periods)
        trapezoid_active = self._active("trapezoid")
        norm = self.components.norm
        output_mixer = self.components.output_mixer
        output_mixer_float = (
            output_mixer.float()
            if cache_active and not defer_global_mix else None
        )
        cache_gate = (
            self.components.cache_gate_amplitude.float()[None, :, None]
            if cache_active and not defer_global_mix else None
        )
        out_proj = self.components.out_proj
        zero = states.new_zeros(())
        one = states.new_ones(())
        block_fill = block_fill_value if block_fill_value >= 0 else None
        batched_hola = cache_active and fast and block_fill is not None
        defer_output_projection = batched_hola
        initial_has_history = has_history
        initial_update_count = update_count
        if fast:
            token_counts = update_count[:, None] + torch.arange(
                T, device=update_count.device, dtype=torch.int64
            )[None]
            fast_ticks = token_counts[:, :, None].remainder(periods).eq(0)
            seen_before = (
                fast_ticks.to(torch.int32).cumsum(1)
                - fast_ticks.to(torch.int32)
            ).gt(0)
            fast_history = has_history[:, None] | seen_before
            phase_sequence = torch.cat((phase[:, None], theta), 1).cumsum(1)[:, 1:]
            fast_queries = self._rotate_pairs(q, phase_sequence)
            fast_keys = self._rotate_pairs(k, phase_sequence)
            phase = phase_sequence[:, -1]
            has_history = has_history | fast_ticks.any(1)
            update_count = update_count + T
        liger_chunked_recurrence = False
        triton_recurrence = False
        decode_recurrence = False
        torch_chunk_recurrence = False
        rematerialized_recurrence = False
        if (
            fast
            and (not cache_active or batched_hola)
            and not getattr(self, "force_torch_recurrence", False)
        ):
            from .qwen_hybrid_triton import (
                can_use_triton_four_state_segment,
                triton_four_state_segment,
            )
            triton_lambda = (
                trap_lambda.float() if trapezoid_active
                else torch.ones_like(trap_lambda, dtype=torch.float32)
            )
            if getattr(self, "use_liger_chunked_kernel", False):
                from .qwen_hybrid_liger_chunked import (
                    can_use_liger_chunked_four_state,
                    liger_chunked_four_state_segment,
                )
                liger_chunked_recurrence = (
                    can_use_liger_chunked_four_state(
                        fast_queries, fast_keys, v, erase, write, gamma,
                        triton_lambda, states, previous_key, previous_value,
                        initial_has_history, initial_update_count,
                    )
                )
            if not liger_chunked_recurrence and B == 1:
                # Projection/view layouts are intentionally optimized for the
                # batched PyTorch path; the persistent kernel needs token-major
                # contiguous V/write vectors.  Do not make these copies for the
                # batch>1 shape that Triton rejects and the WY path accepts.
                triton_v = v if v.is_contiguous() else v.contiguous()
                triton_write = (
                    write if write.is_contiguous() else write.contiguous()
                )
                triton_recurrence = can_use_triton_four_state_segment(
                    fast_queries, fast_keys, triton_v, erase, triton_write,
                    gamma, triton_lambda, states, previous_key, previous_value,
                    initial_has_history, initial_update_count,
                )
            # The direct cached-decode step earned production eligibility for
            # B2 only.  Keep it behind the B1 Triton attempt and inside the
            # uniform-HOLA fast envelope established above.
            if (
                not liger_chunked_recurrence
                and not triton_recurrence
                and cache_active
                and B == 2
            ):
                from .qwen_hybrid_chunkwise import (
                    _can_use_package_b_decode_step,
                )
                decode_recurrence = _can_use_package_b_decode_step(
                    fast_queries, fast_keys, v, erase, write, gamma,
                    triton_lambda, states, previous_key, previous_value,
                    initial_has_history, initial_update_count,
                )
            # Generalized-WY changes the FP32 reduction order slightly.  It is
            # production-eligible only for autograd training, where its full
            # forward/VJP/optimizer envelope is tested against the token-loop
            # authority.  Grad-disabled HOLA evaluation stays exact because a
            # near-tied score can otherwise change discrete survivor ordering.
            if (
                not liger_chunked_recurrence
                and not triton_recurrence
                and not decode_recurrence
                and _can_use_torch_chunk_with_cache(cache_active)
            ):
                if torch.is_grad_enabled():
                    from .qwen_hybrid_chunkwise import (
                        can_use_rematerialized_torch_chunk_four_state_segment,
                    )
                    rematerialized_recurrence = (
                        can_use_rematerialized_torch_chunk_four_state_segment(
                            fast_queries, fast_keys, v, erase, write, gamma,
                            triton_lambda, states, previous_key, previous_value,
                            initial_has_history, initial_update_count,
                        )
                    )
                else:
                    # The production seam rejects no-grad execution.  Keeping
                    # this branch permits the serialized research benchmark to
                    # reopen and measure the eager WY prototype explicitly.
                    from .qwen_hybrid_chunkwise import (
                        can_use_torch_chunk_four_state_segment,
                    )
                    torch_chunk_recurrence = (
                        can_use_torch_chunk_four_state_segment(
                            fast_queries, fast_keys, v, erase, write, gamma,
                            triton_lambda, states, previous_key, previous_value,
                            initial_has_history, initial_update_count,
                        )
                    )
        if liger_chunked_recurrence:
            (
                reads,
                _state_trace,
                states,
                previous_key,
                previous_value,
                innovation_sq,
                has_history,
                update_count,
            ) = liger_chunked_four_state_segment(
                fast_queries, fast_keys, v, erase, write, gamma,
                triton_lambda, states, previous_key, previous_value,
                initial_has_history, initial_update_count,
            )
            del _state_trace
        elif triton_recurrence:
            reads, states, previous_key, previous_value, innovation_sq = (
                triton_four_state_segment(
                    fast_queries, fast_keys, triton_v, erase, triton_write, gamma,
                    triton_lambda, states, previous_key, previous_value,
                    initial_has_history, initial_update_count,
                )
                )
        elif decode_recurrence:
            from .qwen_hybrid_chunkwise import _torch_four_state_decode_step
            (
                reads,
                states,
                previous_key,
                previous_value,
                innovation_sq,
                has_history,
                update_count,
            ) = _torch_four_state_decode_step(
                fast_queries, fast_keys, v, erase, write, gamma,
                triton_lambda, states, previous_key, previous_value,
                initial_has_history, initial_update_count,
            )
        elif rematerialized_recurrence:
            from .qwen_hybrid_chunkwise import (
                rematerialized_torch_chunk_four_state_segment,
            )
            (
                reads,
                _state_trace,
                states,
                previous_key,
                previous_value,
                innovation_sq,
                has_history,
                update_count,
            ) = rematerialized_torch_chunk_four_state_segment(
                fast_queries, fast_keys, v, erase, write, gamma,
                triton_lambda, states, previous_key, previous_value,
                initial_has_history, initial_update_count,
            )
            del _state_trace
        elif torch_chunk_recurrence:
            from .qwen_hybrid_chunkwise import torch_chunk_four_state_segment
            (
                reads,
                _state_trace,
                states,
                previous_key,
                previous_value,
                innovation_sq,
                has_history,
                update_count,
            ) = torch_chunk_four_state_segment(
                fast_queries, fast_keys, v, erase, write, gamma,
                triton_lambda, states, previous_key, previous_value,
                initial_has_history, initial_update_count,
            )
            del _state_trace
        if (
            liger_chunked_recurrence
            or triton_recurrence
            or decode_recurrence
            or torch_chunk_recurrence
            or rematerialized_recurrence
        ):
            gates = z.to(dtype)[:, :, :, :, None, :].expand(
                B, T, self.H, RANK, RANK, self.dv
            )
            normalized = norm(
                reads.reshape(-1, self.dv).to(dtype),
                gates.reshape(-1, self.dv),
            ).reshape(B, T, self.H, RANK, RANK, self.dv)
            if cache_active:
                tick_count = fast_ticks.sum(-1).to(innovation_sq.dtype).clamp_min(1.0)
                score = torch.sqrt(
                    innovation_sq.sum(-1) / tick_count[:, :, None]
                )
                hola_read, hola_state, block_fill = self.hola.scan_fast(
                    hola_state, fast_queries, fast_keys, v, score, block_fill
                )
            if defer_global_mix:
                result = (
                    normalized,
                    hola_read if cache_active else q.new_empty(0),
                    states,
                    phase,
                    previous_key,
                    previous_value,
                    has_history,
                    update_count,
                )
            else:
                mixed_sequence = torch.einsum(
                    "hijvw,bthijw->bthv", output_mixer, normalized
                )
                if cache_active:
                    cache_mixed = torch.einsum(
                        "hijvw,bthijw->bthv", output_mixer_float, hola_read
                    )
                    mixed_sequence = (
                        mixed_sequence + cache_gate[:, None] * cache_mixed
                    )
                out = out_proj(mixed_sequence.flatten(2).to(dtype))
                result = (
                    out,
                    q.new_empty(0),
                    states,
                    phase,
                    previous_key,
                    previous_value,
                    has_history,
                    update_count,
                )
            if cache_active:
                result = result + self._flatten_hola_state(hola_state)
            return result
        outputs = []
        state_outputs = []
        hola_queries = []
        hola_keys = []
        hola_values = []
        hola_scores = []
        # Fast all-valid segments keep slow CMS lanes in a factored lazy-decay
        # representation.  Off-tick tokens update only K scale scalars, while reads
        # materialize scale*state.  Segment outputs are materialized once so
        # the public/cache carry remains an ordinary physical state.
        lazy_scale = states.new_ones(B, self.H, RANK, self.dk) if fast else None
        base_states = states
        base_previous_key = previous_key
        base_previous_value = previous_value
        for token in range(T):
            if not fast:
                reset_rows = boundary[:, token] & valid[:, token]
                reset = reset_rows[:, None, None, None]
                states = torch.where(reset[..., None], torch.zeros_like(states), states)
                phase = torch.where(reset, torch.zeros_like(phase), phase)
                previous_key = torch.where(reset, torch.zeros_like(previous_key), previous_key)
                previous_value = torch.where(reset, torch.zeros_like(previous_value), previous_value)
                has_history = torch.where(reset_rows[:, None], torch.zeros_like(has_history), has_history)
                update_count = torch.where(reset_rows, torch.zeros_like(update_count), update_count)
                active = valid[:, token, None, None, None]
                history = self.history_active(boundary[:, token], valid[:, token], has_history)
                tick_lanes = valid[:, token, None] & update_count[:, None].remainder(periods[None]).eq(0)
            else:
                history = fast_history[:, token]
                tick_lanes = fast_ticks[:, token]
            tick = tick_lanes[:, None, :, None, None]
            if not fast:
                phase = torch.where(active, phase + theta[:, token], phase)
                cosine, sine = torch.cos(phase), torch.sin(phase)
                qt, kt = q[:, token], k[:, token]
                qr = torch.stack((qt[..., 0::2] * cosine - qt[..., 1::2] * sine,
                                  qt[..., 0::2] * sine + qt[..., 1::2] * cosine), dim=-1).flatten(-2)
                kr = torch.stack((kt[..., 0::2] * cosine - kt[..., 1::2] * sine,
                                  kt[..., 0::2] * sine + kt[..., 1::2] * cosine), dim=-1).flatten(-2)
            else:
                qr, kr = fast_queries[:, token], fast_keys[:, token]
            vv = v[:, token]
            gamma_t = gamma[:, token, ..., None]
            erased_key = erase[:, token] * kr
            if fast:
                assert lazy_scale is not None
                physical_scale = lazy_scale * gamma[:, token]
                decayed = physical_scale[..., None] * base_states
                previous_decayed_key = physical_scale * base_previous_key
                previous_endpoint_value = base_previous_value
            else:
                decayed = gamma_t * states
                previous_decayed_key = gamma[:, token] * previous_key
                previous_endpoint_value = previous_value
            # A_t(X) = Gamma_t X - k_t (beta_e . k_t)^T Gamma_t X, with the
            # decayed operand computed once and shared with the score below.
            # Recurrent state math is FP32 even when the surrounding model is
            # under AMP.  Elementwise operations above preserve their FP32
            # input dtype; explicitly fence the autocast-eligible reductions.
            with torch.autocast(qr.device.type, enabled=False):
                memory = torch.einsum(
                    "bhrk,bhrkv->bhrv", erased_key, decayed
                )
            full_homogeneous = decayed - kr[..., None] * memory[..., None, :]
            homogeneous = torch.where(tick, full_homogeneous, decayed)
            current_value = write[:, token] * vv
            current_write = kr[..., None] * current_value[..., None, :]
            # Mamba-3 exponential-trapezoid adapted to the GDN homogeneous
            # transition: S_t=A_t(S_{t-1})+(1-lambda_t)A_t(W_{t-1})+lambda_t W_t.
            # The current erase belongs in A_t and is therefore applied to the
            # previous endpoint with exactly the same transition as the state.
            with torch.autocast(qr.device.type, enabled=False):
                previous_memory = torch.einsum(
                    "bhrk,bhrk->bhr", erased_key, previous_decayed_key
                )
            previous_transported_key = (
                previous_decayed_key - kr * previous_memory[..., None]
            )
            previous_transported = (
                previous_transported_key[..., None]
                * previous_endpoint_value[..., None, :]
            )
            lam = trap_lambda[:, token, ..., None, None] if trapezoid_active else torch.ones_like(trap_lambda[:, token, ..., None, None])
            lam = torch.where(history[:, None, :, None, None], lam, one)
            tick_update = (1.0 - lam) * previous_transported + lam * current_write
            input_update = torch.where(tick, tick_update, zero)
            next_states = homogeneous + input_update
            if fast:
                tick_key = tick_lanes[:, None, :, None]
                tick_value = tick_lanes[:, None, :, None]
                base_states = torch.where(tick, next_states, base_states)
                base_previous_key = torch.where(tick_key, kr, base_previous_key)
                base_previous_value = torch.where(
                    tick_value, current_value, base_previous_value
                )
                lazy_scale = torch.where(tick_key, one, physical_scale)
                states = next_states
            else:
                states = torch.where(active[..., None], next_states, states)
                tick_key = tick_lanes[:, None, :, None]
                previous_key_candidate = torch.where(
                    tick_key, kr, previous_decayed_key
                )
                previous_value_candidate = torch.where(
                    tick_key, current_value, previous_value
                )
                previous_key = torch.where(active, previous_key_candidate, previous_key)
                previous_value = torch.where(active, previous_value_candidate, previous_value)
                update_count = update_count + valid[:, token].to(torch.int64)
                has_history = has_history | tick_lanes
            if defer_output_projection:
                state_outputs.append(states)
            else:
                with torch.autocast(qr.device.type, enabled=False):
                    reads = torch.einsum(
                        "bhik,bhjkv->bhijv", qr, states
                    )
                gates = z[:, token].to(dtype)[:, :, :, None, :].expand(
                    B, self.H, RANK, RANK, self.dv
                )
                normalized = norm(
                    reads.reshape(-1, self.dv).to(dtype),
                    gates.reshape(-1, self.dv),
                ).reshape(B, self.H, RANK, RANK, self.dv)
                mixed = torch.einsum(
                    "hijvw,bhijw->bhv", output_mixer, normalized
                )
            if cache_active:
                # HOLA generalization: committed state change beyond passive
                # decay, RMS-normalized over ticking CMS lanes so periodic
                # multi-lane ticks do not inflate admission scores.
                score = four_state_normalized_update_score(next_states - decayed, tick_lanes)
                if batched_hola:
                    hola_queries.append(qr)
                    hola_keys.append(kr)
                    hola_values.append(vv)
                    hola_scores.append(score)
                else:
                    hola_read, hola_state = self.hola.step_unchecked(
                        hola_state, qr, kr, vv, score, hola_state.next_position,
                        valid[:, token], boundary[:, token])
                    cache_mixed = torch.einsum(
                        "hijvw,bhijw->bhv", output_mixer_float, hola_read
                    )
                    mixed = mixed + cache_gate * cache_mixed
            if not defer_output_projection:
                out = out_proj(mixed.flatten(1).to(dtype))
                outputs.append(torch.where(valid[:, token, None], out, torch.zeros_like(out)))
        if fast:
            assert lazy_scale is not None
            states = lazy_scale[..., None] * base_states
            previous_key = lazy_scale * base_previous_key
            previous_value = base_previous_value
        if defer_output_projection:
            state_sequence = torch.stack(state_outputs, 1)
            reads = torch.einsum(
                "bthik,bthjkv->bthijv", fast_queries, state_sequence
            )
            gates = z.to(dtype)[:, :, :, :, None, :].expand(
                B, T, self.H, RANK, RANK, self.dv
            )
            normalized = norm(
                reads.reshape(-1, self.dv).to(dtype),
                gates.reshape(-1, self.dv),
            ).reshape(B, T, self.H, RANK, RANK, self.dv)
            hola_read, hola_state, block_fill = self.hola.scan_fast(
                hola_state,
                torch.stack(hola_queries, 1),
                torch.stack(hola_keys, 1),
                torch.stack(hola_values, 1),
                torch.stack(hola_scores, 1),
                block_fill,
            )
            if defer_global_mix:
                result = (
                    normalized,
                    hola_read,
                    states,
                    phase,
                    previous_key,
                    previous_value,
                    has_history,
                    update_count,
                )
                return result + self._flatten_hola_state(hola_state)
            mixed_sequence = torch.einsum(
                "hijvw,bthijw->bthv", output_mixer, normalized
            )
            cache_mixed = torch.einsum(
                "hijvw,bthijw->bthv", output_mixer_float, hola_read
            )
            mixed_sequence = mixed_sequence + cache_gate[:, None] * cache_mixed
            out = out_proj(mixed_sequence.flatten(2).to(dtype))
        else:
            out = torch.stack(outputs, 1)
        result = (
            out,
            q.new_empty(0),
            states,
            phase,
            previous_key,
            previous_value,
            has_history,
            update_count,
        )
        if cache_active:
            result = result + self._flatten_hola_state(hola_state)
        return result

    def scan(self, hidden_states: Tensor, *, boundary: Tensor | None = None, valid: Tensor | None = None,
             initial_cache: FourStateHybridCache | None = None) -> tuple[Tensor, FourStateHybridCache]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden states must be [B,T,hidden]")
        B, T = hidden_states.shape[:2]
        boundary = torch.zeros(B, T, dtype=torch.bool, device=hidden_states.device) if boundary is None else boundary
        valid = torch.ones(B, T, dtype=torch.bool, device=hidden_states.device) if valid is None else valid
        self._validate_inputs(hidden_states, boundary, valid)
        cache = self._initial_cache(hidden_states) if initial_cache is None else initial_cache
        self._validate_cache(cache, hidden_states)
        if T == 0:
            return hidden_states.new_empty(B, 0, self.components.hidden), cache
        return self._scan(hidden_states, boundary, valid, cache)

    def reference(self, hidden_states: Tensor, *, boundary: Tensor, valid: Tensor) -> Tensor:
        return self._scan(hidden_states, boundary, valid, None)[0]

    def project_coefficients_(self) -> None:
        self.components.project_coefficients_()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None, *,
                boundary: Tensor | None = None, valid: Tensor | None = None, **kwargs) -> Tensor:
        use_cache = kwargs.pop("use_cache", False)
        initial_cache = kwargs.pop("past_key_value", kwargs.pop("past_key_values", None))
        kwargs.pop("cache_position", None)
        cache_params = kwargs.pop("cache_params", None)
        # Transformers forwards this model-level reporting flag through every
        # decoder layer; the attention module has no separate hidden-state
        # payload to return.
        kwargs.pop("output_hidden_states", None)
        if cache_params is not None:
            # Model-level use_cache=True makes transformers hand every linear
            # layer its Cache object.  An empty slot is inert bookkeeping (the
            # hybrid keeps its own FourStateHybridCache via use_cache below);
            # only actual precomputed state would change semantics, so only
            # that fails closed.
            has_previous = getattr(cache_params, "has_previous_state", None)
            layer_idx = getattr(self, "layer_idx", None)
            if (not callable(has_previous) or layer_idx is None
                    or has_previous(layer_idx)):
                raise ValueError("unsupported incremental arguments: cache_params")
        if kwargs:
            raise ValueError("unsupported incremental arguments: " + ", ".join(sorted(kwargs)))
        if attention_mask is not None:
            mask = attention_mask if attention_mask.ndim == 2 else attention_mask.squeeze(-1)
            valid = mask.bool() if valid is None else valid & mask.bool()
        out, cache = self.scan(hidden_states, boundary=boundary, valid=valid, initial_cache=initial_cache)
        if use_cache:
            self.last_recurrent_cache = cache
        return out


__all__ = ["FourStateHybridCache", "QwenFourStateHybrid", "gdn_homogeneous_transition"]
