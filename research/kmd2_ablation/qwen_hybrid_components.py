"""Identity-preserving shared components for the dual Qwen GDN-2 hybrids."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen_hybrid_math import RANK, braided_decay, identity_output_mixer

# Package-B time braiding rescales the learned native GDN horizon; these are
# relative horizon multipliers, not absolute token horizons.
TIMESCALES = (1.0, 16.0, 64.0, 256.0)
SHARED_TIMESCALES = (64.0, 512.0, 4096.0, 32768.0)


class HybridComponents(nn.Module):
    """Rank-stacked projections and neutral controls shared by Packages A/B."""

    def __init__(self, *, hidden: int, heads: int, key_width: int, value_width: int,
                 package: str, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if package not in {"shared", "four_state"}:
            raise ValueError("package must be 'shared' or 'four_state'")
        self.hidden, self.H, self.dk, self.dv = hidden, heads, key_width, value_width
        self.package = package
        self.phase_topology = "shared" if package == "shared" else "per_rank"
        factory = {"dtype": dtype, "device": device}
        self.q_weight = nn.Parameter(torch.zeros(RANK, heads * key_width, hidden, **factory))
        self.k_weight = nn.Parameter(torch.zeros_like(self.q_weight))
        self.v_weight = nn.Parameter(torch.zeros(RANK, heads * value_width, hidden, **factory))
        self.erase_weight = nn.Parameter(torch.zeros(RANK, heads * key_width, hidden, **factory))
        self.write_weight = nn.Parameter(torch.zeros(RANK, heads * value_width, hidden, **factory))
        self.z_weight = nn.Parameter(torch.zeros(RANK, heads * value_width, hidden, **factory))
        self.write_offset = nn.Parameter(torch.zeros(RANK, heads, **factory))
        # Package B has one content-dependent base rate.  Its four MIMO-rank
        # contributions receive fixed log-rate multipliers in decay_gamma(); it
        # must not learn four unrelated decays or add a second timescale axis.
        self.native_decay_weight = nn.Parameter(torch.zeros(heads, hidden, **factory))
        self.native_A_log = nn.Parameter(torch.zeros(heads, **factory))
        self.native_dt_bias = nn.Parameter(torch.zeros(heads, **factory))
        # Shared-state Package A preserves the native per-coordinate decay.
        # Package B requires one coefficient per complex coordinate pair; store
        # that identifiable parameter directly instead of learning two values
        # whose antisymmetric component is discarded by averaging.
        self.native_decay_chan = (
            nn.Parameter(torch.zeros(heads, key_width, **factory))
            if package == "shared" else None
        )
        self.native_decay_pair = (
            nn.Parameter(torch.zeros(heads, RANK, key_width // 2, **factory))
            if package == "four_state" else None
        )
        self.native_decay_topology = "single_shared_content_rate"
        phase_ranks = 1 if package == "shared" else RANK
        self.phase_proj = nn.Linear(hidden, phase_ranks * heads * (key_width // 2),
                                    bias=True, **factory)
        self.output_mixer = nn.Parameter(identity_output_mixer(
            package, heads, value_width, dtype=dtype, device=device,
        ))
        if package == "shared":
            self.c_logits = nn.Parameter(torch.zeros(heads, RANK, **factory))
            self.d_raw = nn.Parameter(torch.full((heads, RANK), 1 / RANK, **factory))
            self.braid_router = nn.Linear(hidden, heads * key_width * RANK, bias=True, **factory)
        else:
            # Data-dependent Mamba-3-style trapezoid coefficient lambda_t.
            # Bias +4 initializes lambda=sigmoid(4)~=0.982: writes start ~98%
            # current-endpoint, keeping the warm start near the native GDN
            # update while the coefficient stays learnable with a live
            # gradient ("Option A", 2026-07-15; zero bias / lambda=.5 was the
            # repealed capacity-first init).  The first token per lane still
            # forces lambda=1.  Meta construction and explicit filling
            # preserve the source RNG.
            self.trapezoid_proj = nn.Linear(
                hidden, heads * RANK, bias=True, dtype=dtype, device="meta"
            ).to_empty(device=device)
            with torch.no_grad():
                self.trapezoid_proj.weight.zero_()
                self.trapezoid_proj.bias.fill_(4.0)
        residual_shape = (heads, key_width, RANK) if package == "shared" else (heads, RANK, key_width)
        self.braid_residual = nn.Parameter(torch.zeros(*residual_shape, **factory)) if package == "shared" else None
        self.decay_residual_topology = "braided_channels" if package == "shared" else "fixed_log_rate_multipliers"
        self.trapezoid_gate = nn.Parameter(torch.zeros(heads, RANK, **factory)) if package == "shared" else None
        self.lookahead_gate = nn.Parameter(torch.zeros(heads, RANK, **factory)) if package == "shared" else None
        diagonal_basis = torch.ones(heads, RANK, key_width, **factory)
        coordinate_basis = torch.linspace(1.0 / key_width, 1.0, key_width, **factory)
        coordinate_basis = coordinate_basis.expand(heads, RANK, -1).clone()
        self.d_q = nn.Parameter(diagonal_basis.clone())
        self.d_k = nn.Parameter(diagonal_basis.clone())
        self.b_q = nn.Parameter(coordinate_basis.clone())
        self.b_k = nn.Parameter(coordinate_basis.flip(-1).clone())
        self.alpha_q = nn.Parameter(torch.zeros(heads, RANK, **factory))
        self.beta_q = nn.Parameter(torch.zeros(heads, RANK, **factory))
        self.alpha_k = nn.Parameter(torch.zeros(heads, RANK, **factory))
        self.beta_k = nn.Parameter(torch.zeros(heads, RANK, **factory))
        # HOLA uses a sigmoid cache gate with logit -4 initialization.  Keep
        # the checkpoint key stable, but store the unconstrained logit rather
        # than a directly-clamped amplitude.
        # Versioned name: an old direct-amplitude ``cache_gate`` checkpoint must
        # fail closed rather than silently reinterpret 0 as sigmoid(0)=0.5.
        self.cache_gate_logit = nn.Parameter(torch.full((heads,), -4.0, **factory))
        # Shape-derived and RNG-independent: every DDP replica and resume receives
        # byte-identical persistent specialization identity.
        def deterministic_probe(rows: int) -> Tensor:
            index = torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)
            value = torch.sin((index + 1.0) * 0.7548776662466927)
            return value * torch.rsqrt(value.square().mean())
        probe = deterministic_probe(heads * key_width)
        value_probe = deterministic_probe(heads * value_width)
        coefficients = torch.tensor((-3.0, -1.0, 1.0, 3.0), dtype=torch.float32) / (20.0 ** .5)
        self.register_buffer("specialization_probe", probe.to(device=device), persistent=True)
        self.register_buffer("specialization_value_probe", value_probe.to(device=device), persistent=True)
        self.register_buffer("specialization_coefficients", coefficients.to(device=device), persistent=True)
        self.specialization_probe_sha256 = hashlib.sha256(
            probe.contiguous().numpy().tobytes()
        ).hexdigest()
        self.specialization_value_probe_sha256 = hashlib.sha256(
            value_probe.contiguous().numpy().tobytes()
        ).hexdigest()
        self.register_buffer("_braid_entropy_sum", torch.zeros((), dtype=dtype, device=device), persistent=False)
        self.register_buffer("_braid_occupancy_sum", torch.zeros((), dtype=dtype, device=device), persistent=False)
        self.register_buffer("_braid_sample_count", torch.zeros((), dtype=torch.int64, device=device), persistent=False)

    def reset_runtime_braid_statistics(self) -> None:
        self._braid_entropy_sum.zero_(); self._braid_occupancy_sum.zero_(); self._braid_sample_count.zero_()

    def record_runtime_braid_probabilities(self, probabilities: Tensor, valid: Tensor) -> None:
        detached = probabilities.detach().float()
        if detached.ndim < 2 or valid.dtype != torch.bool or detached.shape[:2] != valid.shape:
            raise ValueError("runtime braid probabilities/valid shape mismatch")
        rows = detached[valid]
        if rows.numel() == 0:
            return
        entropy = -(rows * rows.clamp_min(1e-12).log()).sum(-1)
        occupancy = (rows > (1.0 / rows.shape[-1]) * 0.25).float().sum(-1)
        self._braid_entropy_sum.add_(entropy.to(self._braid_entropy_sum.dtype).sum())
        self._braid_occupancy_sum.add_(occupancy.to(self._braid_occupancy_sum.dtype).sum())
        self._braid_sample_count.add_(entropy.numel())

    def runtime_braid_statistics(self) -> dict[str, float | int]:
        count = int(self._braid_sample_count.item())
        return {"entropy_sum": float(self._braid_entropy_sum.item()),
                "occupancy_sum": float(self._braid_occupancy_sum.item()), "count": count}

    @classmethod
    def from_native(cls, native: Any, *, package: str) -> "HybridComponents":
        """Transactionally copy a canonical native R1 projection set four ways."""
        required = ("H", "dk", "dv", "key_dim", "value_dim", "conv_k", "r_out",
                    "in_proj_qkv", "in_proj_b", "in_proj_z", "in_proj_a", "conv1d",
                    "dt_bias", "A_log", "norm", "out_proj", "rot_proj", "decay_chan", "bw_off")
        missing = tuple(name for name in required if not hasattr(native, name))
        if missing:
            raise TypeError(f"native component source is missing {missing}")
        if native.key_dim != native.H * native.dk or native.value_dim != native.H * native.dv:
            raise ValueError("native projection dimensions are inconsistent")
        if not isinstance(native, nn.Module) or native.r_out != 1:
            raise TypeError("native component source must be an exact R1 torch module")
        if native.H <= 0 or native.dk <= 0 or native.dk % 2 or native.dv <= 0:
            raise ValueError("native heads/value width must be positive and key width positive even")
        expected_keys = {
            "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
            "conv1d.weight", "dt_bias", "A_log", "norm.weight", "out_proj.weight",
            "rot_proj.weight", "rot_proj.bias", "decay_chan", "bw_off",
        }
        actual_keys = set(native.state_dict())
        if actual_keys != expected_keys:
            raise ValueError(f"native state keys mismatch: missing={sorted(expected_keys-actual_keys)}, unexpected={sorted(actual_keys-expected_keys)}")
        tensors = tuple(native.state_dict().values())
        if any(not isinstance(value, Tensor) or not value.is_floating_point() for value in tensors):
            raise TypeError("native projection tensors must be floating point")
        if len({value.dtype for value in tensors}) != 1 or len({value.device for value in tensors}) != 1:
            raise ValueError("native projection tensors must share dtype and device")
        hidden = native.in_proj_qkv.in_features
        expected = (2 * native.key_dim + native.value_dim, hidden)
        if tuple(native.in_proj_qkv.weight.shape) != expected:
            raise ValueError(f"in_proj_qkv.weight must have shape {expected}")
        expected_shapes = {
            "in_proj_b.weight": (native.H, hidden),
            "in_proj_z.weight": (native.value_dim, hidden),
            "in_proj_a.weight": (native.H, hidden),
            "out_proj.weight": (hidden, native.value_dim),
            "decay_chan": (native.H, native.dk),
            "dt_bias": (native.H,), "A_log": (native.H,), "bw_off": (native.H,),
            "rot_proj.weight": (native.H * (native.dk // 2), hidden),
            "rot_proj.bias": (native.H * (native.dk // 2),),
            "norm.weight": (native.dv,),
        }
        for name, expected_shape in expected_shapes.items():
            tensor = native.state_dict()[name]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
        conv_dim = 2 * native.key_dim + native.value_dim
        if (native.conv1d.in_channels, native.conv1d.out_channels, native.conv1d.groups,
                native.conv1d.kernel_size, native.conv1d.padding, native.conv1d.bias) != (
                conv_dim, conv_dim, conv_dim, (native.conv_k,), (native.conv_k - 1,), None):
            raise ValueError("conv1d configuration is incompatible with stock depthwise convolution")
        # Package B partitions the native complex-pair key basis across its
        # four existing MIMO lanes.  Four K/4 states therefore have exactly
        # the recurrent matrix capacity of one native K state while retaining
        # four independent write paths and the existing 4x4 read/mixer graph.
        compact_bands = package == "four_state" and native.dk % (2 * RANK) == 0
        key_width = native.dk if package == "shared" or not compact_bands else native.dk // RANK
        result = cls(hidden=hidden, heads=native.H, key_width=key_width,
                     value_width=native.dv, package=package, dtype=tensors[0].dtype,
                     device=tensors[0].device)
        q, k, v = torch.split(native.in_proj_qkv.weight.detach(),
                              [native.key_dim, native.key_dim, native.value_dim], dim=0)
        with torch.no_grad():
            if package == "shared" or not compact_bands:
                for target, source in ((result.q_weight, q), (result.k_weight, k),
                                       (result.v_weight, v),
                                       (result.z_weight, native.in_proj_z.weight.detach())):
                    target.copy_(source.unsqueeze(0).expand_as(target))
            else:
                half = native.dk // 2
                band_half = half // RANK
                band_rows = []
                for rank in range(RANK):
                    per_head = []
                    for head in range(native.H):
                        base = head * native.dk
                        per_head.extend(range(base + rank * band_half,
                                              base + (rank + 1) * band_half))
                        per_head.extend(range(base + half + rank * band_half,
                                              base + half + (rank + 1) * band_half))
                    band_rows.append(per_head)
                band_index = torch.tensor(band_rows, dtype=torch.long, device=q.device)
                result.q_weight.copy_(q[band_index])
                result.k_weight.copy_(k[band_index])
                result.v_weight.copy_(v.unsqueeze(0).expand_as(result.v_weight))
                result.z_weight.copy_(native.in_proj_z.weight.detach().unsqueeze(0).expand_as(result.z_weight))
            erase_source = native.in_proj_b.weight.detach()[:, None, :].expand(-1, native.dk, -1).reshape(native.key_dim, hidden)
            write_source = native.in_proj_b.weight.detach()[:, None, :].expand(-1, native.dv, -1).reshape(native.value_dim, hidden)
            if package == "shared" or not compact_bands:
                result.erase_weight.copy_(erase_source.unsqueeze(0).expand_as(result.erase_weight))
            else:
                result.erase_weight.copy_(erase_source[band_index])
            for target, source in ((result.write_weight, write_source),
                                   (result.write_offset, native.bw_off.detach())):
                target.copy_(source.unsqueeze(0).expand_as(target))
            for target, source in ((result.native_decay_weight, native.in_proj_a.weight.detach()),
                                   (result.native_A_log, native.A_log.detach()),
                                   (result.native_dt_bias, native.dt_bias.detach())):
                target.copy_(source)
            if package == "shared":
                assert result.native_decay_chan is not None
                result.native_decay_chan.copy_(native.decay_chan.detach())
            else:
                # Rotation pairs the first and second key-width halves.  Use
                # the identifiable, nearest pair-tied native log-decay so the
                # first braid lane retains the native GDN decay law without
                # introducing a non-commuting channel scale.
                assert result.native_decay_pair is not None
                native_channel = native.decay_chan.detach()
                tied = 0.5 * (native_channel[:, : native.dk // 2]
                              + native_channel[:, native.dk // 2 :])
                result.native_decay_pair.copy_(
                    tied.reshape(native.H, RANK, key_width // 2)
                    if compact_bands else tied[:, None].expand(-1, RANK, -1)
                )
            result.phase_proj.weight.zero_()
            result.phase_proj.bias.zero_()
            if package == "shared":
                result.braid_router.weight.zero_()
                result.braid_router.bias.zero_()
        result.conv1d = copy.deepcopy(native.conv1d)
        if package == "four_state":
            # The Q/K bands use their matching native depthwise kernels.  V
            # remains full-width in every lane and intentionally shares the
            # one native V kernel bank; index_select keeps gradients tied.
            if not compact_bands:
                band_index = torch.arange(native.key_dim, device=q.device)[None].expand(RANK, -1)
            q_indices = band_index
            k_indices = band_index + native.key_dim
            v_indices = torch.arange(native.value_dim, device=q.device)[None].expand(RANK, -1)
            result.register_buffer(
                "conv_channel_indices",
                torch.cat((q_indices, k_indices, v_indices + 2 * native.key_dim), -1),
                persistent=False,
            )
        result.compact_key_bands = compact_bands
        result.norm = copy.deepcopy(native.norm)
        result.out_proj = copy.deepcopy(native.out_proj)
        return result

    def _rank_project(self, x: Tensor, weight: Tensor, width: int) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.hidden or x.dtype != weight.dtype or x.device != weight.device:
            raise ValueError(f"input must be [B,T,{self.hidden}] with component dtype/device")
        projected = torch.einsum("btd,rod->btro", x, weight)
        return projected.reshape(*x.shape[:2], RANK, self.H, width).permute(0, 1, 3, 2, 4)

    def project_inputs(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return Q/K/V, erase/write, and output-query factors as [B,T,H,4,W]."""
        q = self._rank_project(hidden_states, self.q_weight, self.dk)
        k = self._rank_project(hidden_states, self.k_weight, self.dk)
        v = self._rank_project(hidden_states, self.v_weight, self.dv)
        erase = self._rank_project(hidden_states, self.erase_weight, self.dk)
        write = self._rank_project(hidden_states, self.write_weight, self.dv)
        write = write + self.write_offset.transpose(0, 1)[None, None, :, :, None]
        z = self._rank_project(hidden_states, self.z_weight, self.dv)
        return q, k, v, erase, write, z

    def phase_logits(self, hidden_states: Tensor) -> Tensor:
        projected = self.phase_proj(hidden_states)
        base = (*hidden_states.shape[:2], self.H)
        return projected.reshape(*base, self.dk // 2) if self.package == "shared" else projected.reshape(*base, RANK, self.dk // 2)

    def braid_probabilities(self, hidden_states: Tensor) -> Tensor:
        if self.package != "shared":
            raise ValueError("four-state lanes have fixed rank-aligned timescales")
        logits = self.braid_router(hidden_states).reshape(*hidden_states.shape[:2], self.H, self.dk, RANK)
        return logits.softmax(-1)

    def affine_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        expected = (*q.shape[:3], RANK, self.dk)
        if q.shape != k.shape or q.ndim != 5 or tuple(q.shape) != expected:
            raise ValueError("q and k must have shape [B,T,H,4,K]")
        alpha_q = self.alpha_q[None, None, :, :, None]
        beta_q = self.beta_q[None, None, :, :, None]
        alpha_k = self.alpha_k[None, None, :, :, None]
        beta_k = self.beta_k[None, None, :, :, None]
        return ((1 + alpha_q * self.d_q) * q + beta_q * self.b_q,
                (1 + alpha_k * self.d_k) * k + beta_k * self.b_k)

    def native_decay(self, hidden_states: Tensor, *, rankwise: bool = False) -> Tensor:
        if self.package == "shared":
            if rankwise:
                raise ValueError("shared native decay has no rankwise parameter set")
            assert self.native_decay_chan is not None
            a = torch.einsum("btd,hd->bth", hidden_states, self.native_decay_weight)
            g_head = -self.native_A_log.exp()[None, None] * F.softplus(a + self.native_dt_bias[None, None])
            return (g_head[..., None] + self.native_decay_chan[None, None]).exp().clamp(min=2.0**-24, max=1.0)
        if rankwise:
            raise ValueError("Package B rankwise decay is derived by decay_gamma, not independently projected")
        assert self.native_decay_pair is not None
        a = torch.einsum("btd,hd->bth", hidden_states, self.native_decay_weight)
        g_head = -self.native_A_log.exp()[None, None] * F.softplus(a + self.native_dt_bias[None, None])
        channel = torch.cat((self.native_decay_pair, self.native_decay_pair), -1)
        return (g_head[..., None, None] + channel[None, None]).exp().clamp(min=2.0**-24, max=1.0)

    def decay_gamma(self, hidden_states: Tensor) -> Tensor:
        """Apply the neutral shared braid or the four owned lane residuals."""
        if self.package == "shared":
            tau = self.native_decay_weight.new_tensor(SHARED_TIMESCALES)
            native = self.native_decay(hidden_states)
            probabilities = self.braid_probabilities(hidden_states)
            residual = self.braid_residual[None, None].expand_as(probabilities)
            return braided_decay(native, probabilities, residual, tau)
        horizon_multiplier = self.native_decay_weight.new_tensor(TIMESCALES)
        # Preserve the native GDN content-dependent log decay and braid its
        # horizon across the four existing MIMO ranks:
        #   log gamma_native = -exp(A_log) softplus(a_t + dt_bias) + channel
        #   gamma_{t,r} = exp(log gamma_native / s_r)
        #                = gamma_native ** (1 / s_r)
        # for s=(1,16,64,256).  Thus rank zero is the pair-tied native decay;
        # the remaining ranks have 16x/64x/256x its instantaneous horizon.
        accumulation_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
        x = hidden_states.to(accumulation_dtype)
        weight = self.native_decay_weight.to(accumulation_dtype)
        a = torch.einsum("btd,hd->bth", x, weight)
        A = self.native_A_log.to(accumulation_dtype).exp()[None, None]
        bias = self.native_dt_bias.to(accumulation_dtype)[None, None]
        native_log_gamma = -A * F.softplus(a + bias)
        assert self.native_decay_pair is not None
        pair_log_gamma = native_log_gamma[..., None, None] + self.native_decay_pair.to(
            accumulation_dtype
        )[None, None]
        # Bound in log space BEFORE the horizon division.  The unconstrained
        # additive residual can push pre-clamp log gamma below log(2^-24);
        # the previous post-division [2^-24, 1] floor then saturated ONLY the
        # fast lanes, which destroyed the exact 1:16:64:256 rate ratios (a
        # floored lane 0 no longer relates to lane 1 by 16x).  Clamping the
        # shared log rate once keeps every lane on the same rate, so the
        # ratios hold identically at and beyond the floor.  The 0.0 ceiling
        # is bit-identical to the previous per-lane gamma<=1 clamp (all lanes
        # reach gamma=1 together).  Off the bounds this changes nothing.
        # Known limitation (documented, monitored): a hard-saturated channel
        # still has a dead decay gradient; lane 0 fidelity to the native
        # decay law rules out a strictly-interior reparameterization here.
        pair_log_gamma = pair_log_gamma.clamp(min=-16.6355323, max=0.0)
        pair_gamma = (
            pair_log_gamma
            / horizon_multiplier.to(accumulation_dtype)[None, None, None, :, None]
        ).exp()
        gamma = torch.cat((pair_gamma, pair_gamma), -1)
        return gamma.clamp(min=2.0 ** -24, max=1.0)

    def trapezoid_lambda(self, hidden_states: Tensor) -> Tensor:
        """Return token-dependent current-endpoint weights [B,T,H,4]."""
        if self.package != "four_state":
            raise ValueError("rankwise trapezoid lambda exists only in Package B")
        return self.trapezoid_proj(hidden_states).reshape(
            *hidden_states.shape[:2], self.H, RANK
        ).float().sigmoid()

    def mix_reads(self, reads: Tensor) -> Tensor:
        """Apply the exact Package-A four-read or Package-B sixteen-read mixer."""
        if self.package == "shared":
            if reads.ndim != 5 or tuple(reads.shape[2:]) != (self.H, RANK, self.dv):
                raise ValueError("shared reads must have shape [B,T,H,4,V]")
            return torch.einsum("hrvw,bthrw->bthv", self.output_mixer, reads)
        if reads.ndim != 6 or tuple(reads.shape[2:]) != (self.H, RANK, RANK, self.dv):
            raise ValueError("four-state reads must have shape [B,T,H,4,4,V]")
        return torch.einsum("hijvw,bthijw->bthv", self.output_mixer, reads)

    def project_output(self, mixed: Tensor) -> Tensor:
        if mixed.ndim != 4 or tuple(mixed.shape[2:]) != (self.H, self.dv):
            raise ValueError("mixed reads must have shape [B,T,H,V]")
        return self.out_proj(mixed.flatten(2))

    def transformation_manifest(self) -> dict[str, object]:
        compact = self.package == "four_state" and self.compact_key_bands
        return {
            "rank": RANK, "package": self.package,
            "replicated": (("v", "write", "z") if compact
                           else ("q", "k", "v", "erase", "write", "z")),
            "band_partitioned": (("q", "k", "erase") if compact else ()),
            "key_band_width": self.dk,
            "conv1d": "copied_once",
            "conv1d_execution": ("pair_banded_qk_shared_v" if compact else "native_depthwise"),
            "norm": "copied_once", "out_proj": "copied_once",
            "phase_topology": self.phase_topology,
            "decay_residual_topology": self.decay_residual_topology,
            "native_decay_topology": self.native_decay_topology,
            "parameters": {
                name: {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in self.state_dict().items()
            },
        }

    def project_coefficients_(self) -> None:
        """Project every explicitly bounded raw coefficient after optimizer steps."""
        with torch.no_grad():
            if self.package == "shared":
                self.d_raw.clamp_(0.0, 1.0)
            if self.trapezoid_gate is not None:
                self.trapezoid_gate.clamp_(0.0, 1.0)
            if self.lookahead_gate is not None:
                self.lookahead_gate.clamp_(0.0, 1.0)

    @property
    def cache_gate_amplitude(self) -> Tensor:
        """Paper HOLA cache-read mixing coefficient in the open interval (0, 1)."""
        return self.cache_gate_logit.sigmoid()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        old_key = prefix + "cache_gate"
        if old_key in state_dict:
            error_msgs.append(
                f"{old_key} uses removed HOLA amplitude schema v1; migrate explicitly "
                "to cache_gate_logit=logit(clamp(amplitude, eps, 1-eps)) or reject the checkpoint"
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @property
    def c(self) -> Tensor:
        """Uniform erase coefficients (named as in the approved equations)."""
        if self.package != "shared":
            raise AttributeError("four-state updates do not use shared erase coefficients")
        return self.c_logits.softmax(-1)

    @property
    def d(self) -> Tensor:
        """Uniform write coefficients (named as in the approved equations)."""
        if self.package != "shared":
            raise AttributeError("four-state updates do not use shared write coefficients")
        return self.d_raw.clamp(0.0, 1.0)


__all__ = ["HybridComponents"]
