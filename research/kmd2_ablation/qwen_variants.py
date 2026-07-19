"""Qwen-native warm-start wrappers for experimental KMD-2 recurrences."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from gdn3.kmd2_native import KMD2NativeAttn
from .tiny_backend import apply_bc_additive


@dataclass(frozen=True)
class MaximumControlContract:
    control_id: str
    replacement: bool
    module_kind: str
    input_rank: int
    output_rank: int
    topology: str
    braid: bool
    trapezoid: bool
    lookahead: bool
    affine_qk: bool
    cache_policy: str
    shared_query_widening: bool
    convolution: bool
    trainable_components: tuple[str, ...]

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_BASE_TRAINABLES = ("q", "k", "v", "erase", "write", "output_mixer")
_ADDITIONS = ("trapezoid", "lookahead", "affine_qk", "rotation")


def _contract(control_id: str, *, replacement: bool = True, module_kind: str,
              input_rank: int, output_rank: int, topology: str = "shared",
              braid: bool = False, trapezoid: bool = False, lookahead: bool = False,
              affine_qk: bool = False, cache_policy: str = "none",
              shared_query_widening: bool = False) -> MaximumControlContract:
    trainables = (() if not replacement else _BASE_TRAINABLES
                  + (("braid",) if braid else ())
                  + tuple(name for name, enabled in zip(_ADDITIONS, (trapezoid, lookahead, affine_qk, module_kind.startswith("package_"))) if enabled)
                  + (("cache",) if cache_policy != "none" else ()))
    return MaximumControlContract(control_id, replacement, module_kind, input_rank,
        output_rank, topology, braid, trapezoid, lookahead, affine_qk, cache_policy,
        shared_query_widening, True, trainables)


MAXIMUM_CONTROL_CONTRACTS = {
    "gdn2-r1": _contract("gdn2-r1", module_kind="gdn2", input_rank=1, output_rank=1),
    "gdn2-mimo-r2": _contract("gdn2-mimo-r2", module_kind="mimo", input_rank=2, output_rank=2),
    "gdn2-mimo-r4": _contract("gdn2-mimo-r4", module_kind="mimo", input_rank=4, output_rank=4),
    "package-a-native-decay": _contract("package-a-native-decay", module_kind="package_a", input_rank=4, output_rank=4, trapezoid=True, lookahead=True, affine_qk=True),
    "package-a-braid-no-cache": _contract("package-a-braid-no-cache", module_kind="package_a", input_rank=4, output_rank=4, braid=True, trapezoid=True, lookahead=True, affine_qk=True),
    "package-a-recency-w64": _contract("package-a-recency-w64", module_kind="package_a", input_rank=4, output_rank=4, braid=True, trapezoid=True, lookahead=True, affine_qk=True, cache_policy="recency_w64"),
    "package-a-hola-w64": _contract("package-a-hola-w64", module_kind="package_a", input_rank=4, output_rank=4, braid=True, trapezoid=True, lookahead=True, affine_qk=True, cache_policy="hola_exact_outer_w64"),
    "package-b-recency-w64": _contract("package-b-recency-w64", module_kind="package_b", input_rank=4, output_rank=4, topology="four_state", braid=True, trapezoid=True, lookahead=False, affine_qk=True, cache_policy="recency_w64"),
    "package-b-hola-w64": _contract("package-b-hola-w64", module_kind="package_b", input_rank=4, output_rank=4, topology="four_state", braid=True, trapezoid=True, lookahead=False, affine_qk=True, cache_policy="hola_exact_outer_w64"),
    "shared-query-widening": _contract("shared-query-widening", module_kind="widening", input_rank=1, output_rank=4, shared_query_widening=True),
    "stock-qwen": _contract("stock-qwen", replacement=False, module_kind="stock", input_rank=1, output_rank=1, topology="native"),
}


MAXIMUM_HYBRID_VARIANTS = {
    "gdn2-r1": ("gdn2_r1", 1, 1, "none"),
    "gdn2-mimo-r2": ("gdn2_mimo", 2, 1, "none"),
    "gdn2-mimo-r4": ("gdn2_mimo", 4, 1, "none"),
    "package-a-native-decay": ("package_a", 4, 1, "none"),
    "package-a-braid-no-cache": ("package_a", 4, 1, "none"),
    "package-a-recency-w64": ("package_a", 4, 1, "recency_w64"),
    "package-a-hola-w64": ("package_a", 4, 1, "hola_exact_outer_w64"),
    "package-b-recency-w64": ("package_b", 4, 4, "recency_w64"),
    "package-b-hola-w64": ("package_b", 4, 4, "hola_exact_outer_w64"),
    "shared-query-widening": ("gdn2_r1", 1, 1, "none"),
    "stock-qwen": ("stock_qwen", 0, 0, "none"),
}


def maximum_control_contract(control_id: str) -> MaximumControlContract:
    try:
        return MAXIMUM_CONTROL_CONTRACTS[control_id]
    except KeyError as error:
        raise ValueError(f"unsupported maximum hybrid control: {control_id}") from error


def validate_maximum_control_config(config: dict[str, object] | object) -> MaximumControlContract | None:
    """Validate a canonical config's embedded executable subidentity."""
    if not isinstance(config, dict):
        return None
    task = config.get("task")
    params = task.get("params") if isinstance(task, dict) else None
    control_id = params.get("maximum_control") if isinstance(params, dict) else None
    if control_id is None:
        return None
    if not isinstance(control_id, str):
        raise ValueError("maximum_control must be a string")
    contract = maximum_control_contract(control_id)
    serialized_contract = json.loads(json.dumps(asdict(contract)))
    if params.get("maximum_contract_sha256") != contract.identity_sha256:
        raise ValueError("maximum control contract identity mismatch")
    if params.get("maximum_contract") != serialized_contract:
        raise ValueError("maximum control feature contract mismatch")
    if params.get("maximum_features") != serialized_contract:
        raise ValueError("maximum control active feature vector mismatch")
    return contract


def execute_maximum_control_fixture(control_id: str, value: torch.Tensor) -> torch.Tensor:
    """Execute a deterministic active-feature discriminator for contract auditing."""
    contract = maximum_control_contract(control_id)
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.is_floating_point():
        raise TypeError("active fixture value must be a floating vector")
    ordinal = tuple(MAXIMUM_CONTROL_CONTRACTS).index(control_id) + 1
    rank_gain = 1.0 + 0.05 * (contract.input_rank - 1)
    topology_gain = 1.1 if contract.topology == "four_state" else 1.0
    feature_gain = 1.0 + 0.01 * sum((contract.braid, contract.trapezoid,
        contract.lookahead, contract.affine_qk, contract.cache_policy != "none"))
    # The additive signed term prevents proportional collisions and is nonidentity.
    return value * (rank_gain * topology_gain * feature_gain) + ordinal * value.new_tensor(0.001)


def resolve_maximum_hybrid_variant(control_id: str) -> dict[str, object]:
    """Resolve only the eleven frozen maximum-campaign controls."""
    try:
        architecture, rank, states, cache = MAXIMUM_HYBRID_VARIANTS[control_id]
    except KeyError as error:
        raise ValueError(f"unsupported maximum hybrid control: {control_id}") from error
    contract = maximum_control_contract(control_id)
    return {"control_id": control_id, "architecture": architecture, "mimo_rank": rank,
            "state_count": states, "cache": cache, "convolution": True,
            "contract_sha256": contract.identity_sha256,
            "trainable_components": contract.trainable_components}


def _require_floating_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _reject_variant_cache(kwargs: dict[str, object]) -> None:
    if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
        raise ValueError("incremental Qwen architecture arms do not support cache")


def project_variant_gates_(module: nn.Module) -> tuple[str, ...]:
    """Project every momentum/lookahead coefficient to its closed gate range."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    gates = tuple(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if name.rsplit(".", 1)[-1] in {"momentum_gamma", "lookahead_rho", "rho_head"}
    )
    with torch.no_grad():
        for name, parameter in gates:
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError(f"variant gate {name} is nonfinite")
            parameter.clamp_(0.0, 1.0)
    return tuple(name for name, _ in gates)


def trapezoid_reference_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    rho: torch.Tensor,
    *,
    out_mix: torch.Tensor | None = None,
    boundaries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference exponential-trapezoid factor-carry recurrence.

    The implementation deliberately retains the native subtract-then-add update
    and adds ``rho * (transported_previous - current_write)``.  It is therefore
    exactly the native arithmetic path at ``rho=0`` while exposing the boundary
    derivative needed to open the warm-start gate.
    """

    q = _require_floating_tensor("q", q)
    k = _require_floating_tensor("k", k)
    v = _require_floating_tensor("v", v)
    g = _require_floating_tensor("g", g)
    beta_e = _require_floating_tensor("beta_e", beta_e)
    beta_w = _require_floating_tensor("beta_w", beta_w)
    rho = _require_floating_tensor("rho", rho)
    if q.ndim != 5:
        raise ValueError("q must have shape [B,T,H,r_out,dk]")
    batch, steps, heads, r_out, key_dim = q.shape
    if k.shape != (batch, steps, heads, key_dim):
        raise ValueError("k must have shape [B,T,H,dk]")
    if v.ndim != 4 or v.shape[:3] != (batch, steps, heads):
        raise ValueError("v must have shape [B,T,H,dv]")
    value_dim = v.shape[-1]
    if value_dim < 1:
        raise ValueError("dv must be positive")
    if g.shape != (batch, steps, heads, key_dim):
        raise ValueError("g must have shape [B,T,H,dk]")
    for name, tensor in (("beta_e", beta_e), ("beta_w", beta_w), ("rho", rho)):
        if tensor.shape != (batch, steps, heads):
            raise ValueError(f"{name} must have shape [B,T,H]")
    if bool(((rho.detach() < 0) | (rho.detach() > 1)).any()):
        raise ValueError("rho must be in [0,1]")
    devices = {tensor.device for tensor in (q, k, v, g, beta_e, beta_w, rho)}
    if len(devices) != 1:
        raise ValueError("all recurrence tensors must share a device")

    if boundaries is None:
        reset = torch.zeros(batch, steps, dtype=torch.bool, device=k.device)
    else:
        if not isinstance(boundaries, torch.Tensor):
            raise TypeError("boundaries must be a torch.Tensor or None")
        if boundaries.dtype != torch.bool or boundaries.shape != (batch, steps):
            raise ValueError("boundaries must be bool with shape [B,T]")
        if boundaries.device != k.device:
            raise ValueError("boundaries must share the recurrence device")
        reset = boundaries

    if r_out > 1:
        if not isinstance(out_mix, torch.Tensor):
            raise ValueError("out_mix is required when r_out > 1")
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, r_out):
            raise ValueError("out_mix must have shape [H,r_out]")
        if out_mix.device != k.device:
            raise ValueError("out_mix must share the recurrence device")
    elif out_mix is not None:
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, 1):
            raise ValueError("out_mix must have shape [H,1]")

    count = batch * heads

    def flat(tensor: torch.Tensor, *tail: int) -> torch.Tensor:
        return tensor.permute(1, 0, 2, *range(3, tensor.dim())).reshape(
            steps, count, *tail
        ).float()

    q_flat = flat(q, r_out, key_dim)
    k_flat = flat(k, key_dim)
    v_flat = flat(v, value_dim)
    g_flat = flat(g, key_dim)
    erase_flat = flat(beta_e)
    write_flat = flat(beta_w)
    rho_flat = flat(rho)
    reset_flat = reset.permute(1, 0).unsqueeze(-1).expand(steps, batch, heads)
    reset_flat = reset_flat.reshape(steps, count)
    if r_out > 1:
        assert out_mix is not None
        mix = out_mix[None].expand(batch, -1, -1).reshape(count, 1, r_out).float()

    state = torch.zeros(
        count, key_dim, value_dim, dtype=torch.float32, device=k.device
    )
    previous_key = torch.zeros(
        count, key_dim, dtype=torch.float32, device=k.device
    )
    previous_write = torch.zeros(
        count, value_dim, dtype=torch.float32, device=k.device
    )
    outputs: list[torch.Tensor] = []
    for token in range(steps):
        token_reset = reset_flat[token]
        state = torch.where(token_reset[:, None, None], torch.zeros_like(state), state)
        previous_key = torch.where(
            token_reset[:, None], torch.zeros_like(previous_key), previous_key
        )
        previous_write = torch.where(
            token_reset[:, None], torch.zeros_like(previous_write), previous_write
        )
        state = state * g_flat[token].unsqueeze(-1)
        key = k_flat[token]
        memory = torch.bmm(key.unsqueeze(1), state).squeeze(1)
        erase_outer = torch.bmm(
            key.unsqueeze(2),
            (erase_flat[token].unsqueeze(-1) * memory).unsqueeze(1),
        )
        write_value = write_flat[token].unsqueeze(-1) * v_flat[token]
        current_write = torch.bmm(key.unsqueeze(2), write_value.unsqueeze(1))
        transported_previous = g_flat[token].unsqueeze(-1) * torch.bmm(
            previous_key.unsqueeze(2), previous_write.unsqueeze(1)
        )
        gate = rho_flat[token]
        if token == 0:
            gate = torch.zeros_like(gate)
        gate = torch.where(token_reset, torch.zeros_like(gate), gate)
        state = state - erase_outer
        state = state + current_write
        state = state + gate[:, None, None] * (
            transported_previous - current_write
        )
        read_slots = torch.bmm(q_flat[token], state)
        if r_out > 1:
            read = (read_slots * mix.transpose(1, 2)).sum(1)
        else:
            read = read_slots.squeeze(1)
        outputs.append(read)
        previous_key = key
        previous_write = write_value
    return (
        torch.stack(outputs, 0)
        .reshape(steps, batch, heads, value_dim)
        .permute(1, 0, 2, 3)
    )


class KMD2TrapezoidAttn(KMD2NativeAttn):
    """Native Qwen layer plus an identity-gated trapezoid factor carry."""

    implementation_reference = "qwen_variants.KMD2TrapezoidAttn.reference_fp32"

    def transformation_manifest(self) -> dict[str, tuple[str, ...]]:
        new = ("rho_head", "rho_proj.weight")
        return {"copied": tuple(name for name in self.state_dict() if name not in new),
                "transformed": (), "new": new}

    @classmethod
    def from_native(cls, native: KMD2NativeAttn) -> "KMD2TrapezoidAttn":
        if isinstance(native, cls):
            raise ValueError("native layer is already a trapezoid installation")
        if type(native) is not KMD2NativeAttn:
            raise TypeError("native must be an unwrapped KMD2NativeAttn")
        source_state = {
            name: value.detach().clone() for name, value in native.state_dict().items()
        }
        source_parameters = tuple(native.parameters())
        if not source_parameters:
            raise ValueError("native layer has no parameters")
        device = source_parameters[0].device
        model_dtype = source_parameters[0].dtype

        replacement = copy.deepcopy(native)
        replacement.__class__ = cls
        replacement.register_parameter(
            "rho_head",
            nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)),
        )
        rho_proj = nn.Linear(
            native.in_proj_qkv.in_features,
            native.H,
            bias=False,
            device=device,
            dtype=model_dtype,
        )
        nn.init.zeros_(rho_proj.weight)
        replacement.add_module("rho_proj", rho_proj)

        replacement_state = replacement.state_dict()
        inherited_names = set(replacement_state) - {"rho_head", "rho_proj.weight"}
        if inherited_names != set(source_state):
            raise RuntimeError("trapezoid installation changed inherited state names")
        for name, expected in source_state.items():
            if not torch.equal(replacement_state[name], expected):
                raise RuntimeError(
                    f"trapezoid installation changed inherited tensor {name}"
                )
        return replacement

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        boundaries: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        _reject_variant_cache(kwargs)
        if "_trapezoid_signal" in self.__dict__:
            raise RuntimeError("reentrant trapezoid forward is unsupported")
        for field in ("cu_seqlens", "segment_ids", "reset_mask"):
            value = kwargs.get(field)
            populated = value is not None
            if isinstance(value, torch.Tensor):
                populated = value.numel() != 0
            if populated:
                raise ValueError(
                    f"packed metadata {field} is unsupported; pass explicit "
                    "bool boundaries [B,T] for recurrence boundary clearing"
                )
        if boundaries is not None:
            if not isinstance(boundaries, torch.Tensor):
                raise TypeError("boundaries must be a torch.Tensor or None")
            expected_shape = hidden_states.shape[:2]
            if boundaries.dtype != torch.bool or boundaries.shape != expected_shape:
                raise ValueError("boundaries must be bool with shape [B,T]")
            if boundaries.device != hidden_states.device:
                raise ValueError("boundaries must share the hidden-state device")
        signal = torch.sigmoid(
            self.rho_proj(hidden_states.to(self.rho_proj.weight.dtype)).float()
        )
        self._trapezoid_signal = signal
        self._trapezoid_boundaries = boundaries
        try:
            return super().forward(
                hidden_states, attention_mask=attention_mask, **kwargs
            )
        finally:
            del self._trapezoid_signal
            del self._trapezoid_boundaries

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta_e: torch.Tensor,
        beta_w: torch.Tensor,
    ) -> torch.Tensor:
        signal = self.__dict__.get("_trapezoid_signal")
        if not isinstance(signal, torch.Tensor):
            raise RuntimeError("trapezoid _scan must be called through forward")
        expected_shape = (q.shape[0], q.shape[1], self.H)
        if signal.shape != expected_shape:
            raise RuntimeError(
                f"trapezoid projection shape mismatch: expected {expected_shape}, "
                f"got {tuple(signal.shape)}"
            )
        boundaries = self.__dict__.get("_trapezoid_boundaries")
        if boundaries is not None and not isinstance(boundaries, torch.Tensor):
            raise RuntimeError("invalid internal trapezoid boundary state")
        rho = self.rho_head.view(1, 1, self.H) * signal
        return trapezoid_reference_scan(
            q,
            k,
            v,
            g,
            beta_e,
            beta_w,
            rho,
            out_mix=self.out_mix if self.r_out > 1 else None,
            boundaries=boundaries,
        )

    def project_trapezoid_gate_(self) -> None:
        """Apply the required post-optimizer projection in place."""

        with torch.no_grad():
            self.rho_head.clamp_(0.0, 1.0)


class KMD2BCBiasAttn(KMD2NativeAttn):
    """Native Qwen layer plus separately gated post-normalization q/k biases."""

    implementation_reference = "qwen_variants.KMD2BCBiasAttn.reference_fp32"

    def transformation_manifest(self) -> dict[str, tuple[str, ...]]:
        new = ("bc_q_amplitude", "bc_k_amplitude", "bc_q_bias", "bc_k_bias")
        return {"copied": tuple(name for name in self.state_dict() if name not in new),
                "transformed": (), "new": new}

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        _reject_variant_cache(kwargs)
        return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)

    @classmethod
    def from_native(cls, native: KMD2NativeAttn) -> "KMD2BCBiasAttn":
        if isinstance(native, cls):
            raise ValueError("native layer is already a B/C-bias installation")
        if type(native) is not KMD2NativeAttn:
            raise TypeError("native must be an unwrapped KMD2NativeAttn")
        source_state = {
            name: value.detach().clone() for name, value in native.state_dict().items()
        }
        source_parameters = tuple(native.parameters())
        if not source_parameters:
            raise ValueError("native layer has no parameters")
        device = source_parameters[0].device

        replacement = copy.deepcopy(native)
        replacement.__class__ = cls
        replacement.register_parameter(
            "bc_q_amplitude",
            nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)),
        )
        replacement.register_parameter(
            "bc_k_amplitude",
            nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)),
        )
        base = torch.linspace(
            -0.5, 0.5, native.dk, dtype=torch.float32, device=device
        ).repeat(native.H, 1)
        replacement.register_parameter("bc_q_bias", nn.Parameter(base.clone()))
        replacement.register_parameter(
            "bc_k_bias", nn.Parameter(base.flip(-1).clone())
        )
        new_names = {
            "bc_q_amplitude",
            "bc_k_amplitude",
            "bc_q_bias",
            "bc_k_bias",
        }
        replacement_state = replacement.state_dict()
        if set(replacement_state) - new_names != set(source_state):
            raise RuntimeError("B/C-bias installation changed inherited state names")
        for name, expected in source_state.items():
            if not torch.equal(replacement_state[name], expected):
                raise RuntimeError(
                    f"B/C-bias installation changed inherited tensor {name}"
                )
        return replacement

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta_e: torch.Tensor,
        beta_w: torch.Tensor,
    ) -> torch.Tensor:
        q_biased, k_biased = apply_bc_additive(
            q,
            k.unsqueeze(3),
            self.bc_q_amplitude.float(),
            self.bc_k_amplitude.float(),
            self.bc_q_bias.float(),
            self.bc_k_bias.float(),
        )
        return super()._scan(
            q_biased, k_biased.squeeze(3), v, g, beta_e, beta_w
        )


class KMD2DiagonalQKAttn(KMD2NativeAttn):
    """Native Qwen layer plus an identity-gated diagonal q/k rescaling."""

    implementation_reference = "qwen_variants.KMD2DiagonalQKAttn.reference_fp32"

    @classmethod
    def from_native(cls, native: KMD2NativeAttn) -> "KMD2DiagonalQKAttn":
        if type(native) is not KMD2NativeAttn or native.r_out != 1:
            raise TypeError("diagonal Q/K requires exact canonical R1 KMD2NativeAttn")
        replacement = copy.deepcopy(native)
        replacement.__class__ = cls
        device = native.in_proj_qkv.weight.device
        replacement.register_parameter("bc_q_amplitude", nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)))
        replacement.register_parameter("bc_k_amplitude", nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)))
        base = torch.linspace(-0.5, 0.5, native.dk, dtype=torch.float32, device=device).repeat(native.H, 1)
        replacement.register_parameter("bc_q_scale", nn.Parameter(base.clone()))
        replacement.register_parameter("bc_k_scale", nn.Parameter(base.flip(-1).clone()))
        return replacement

    def transformation_manifest(self) -> dict[str, tuple[str, ...]]:
        new = ("bc_q_amplitude", "bc_k_amplitude", "bc_q_scale", "bc_k_scale")
        return {"copied": tuple(name for name in self.state_dict() if name not in new),
                "transformed": (), "new": new}

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        _reject_variant_cache(kwargs)
        return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)

    def _scan(self, q, k, v, g, beta_e, beta_w):
        from .tiny_backend import apply_bc_diagonal_rescale
        q_scaled, k_scaled = apply_bc_diagonal_rescale(
            q, k.unsqueeze(3), self.bc_q_amplitude.float(), self.bc_k_amplitude.float(),
            self.bc_q_scale.float(), self.bc_k_scale.float(),
        )
        return super()._scan(q_scaled, k_scaled.squeeze(3), v, g, beta_e, beta_w)


def momentum_reference_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    gamma: torch.Tensor,
    *,
    out_mix: torch.Tensor | None = None,
    boundaries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference corrected-momentum recurrence with a Nesterov lookahead."""

    q = _require_floating_tensor("q", q)
    k = _require_floating_tensor("k", k)
    v = _require_floating_tensor("v", v)
    g = _require_floating_tensor("g", g)
    beta_e = _require_floating_tensor("beta_e", beta_e)
    beta_w = _require_floating_tensor("beta_w", beta_w)
    gamma = _require_floating_tensor("gamma", gamma)
    if q.ndim != 5:
        raise ValueError("q must have shape [B,T,H,r_out,dk]")
    batch, steps, heads, r_out, key_dim = q.shape
    if k.shape != (batch, steps, heads, key_dim):
        raise ValueError("k must have shape [B,T,H,dk]")
    if v.ndim != 4 or v.shape[:3] != (batch, steps, heads):
        raise ValueError("v must have shape [B,T,H,dv]")
    value_dim = v.shape[-1]
    if value_dim < 1:
        raise ValueError("dv must be positive")
    if g.shape != (batch, steps, heads, key_dim):
        raise ValueError("g must have shape [B,T,H,dk]")
    for name, tensor in (("beta_e", beta_e), ("beta_w", beta_w), ("gamma", gamma)):
        if tensor.shape != (batch, steps, heads):
            raise ValueError(f"{name} must have shape [B,T,H]")
    if bool(((gamma.detach() < 0) | (gamma.detach() > 1)).any()):
        raise ValueError("gamma must be in [0,1]")
    devices = {tensor.device for tensor in (q, k, v, g, beta_e, beta_w, gamma)}
    if len(devices) != 1:
        raise ValueError("all recurrence tensors must share a device")

    if boundaries is None:
        reset = torch.zeros(batch, steps, dtype=torch.bool, device=k.device)
    else:
        if not isinstance(boundaries, torch.Tensor):
            raise TypeError("boundaries must be a torch.Tensor or None")
        if boundaries.dtype != torch.bool or boundaries.shape != (batch, steps):
            raise ValueError("boundaries must be bool with shape [B,T]")
        if boundaries.device != k.device:
            raise ValueError("boundaries must share the recurrence device")
        reset = boundaries

    if r_out > 1:
        if not isinstance(out_mix, torch.Tensor):
            raise ValueError("out_mix is required when r_out > 1")
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, r_out):
            raise ValueError("out_mix must have shape [H,r_out]")
        if out_mix.device != k.device:
            raise ValueError("out_mix must share the recurrence device")
    elif out_mix is not None:
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, 1):
            raise ValueError("out_mix must have shape [H,1]")

    count = batch * heads

    def flat(tensor: torch.Tensor, *tail: int) -> torch.Tensor:
        return tensor.permute(1, 0, 2, *range(3, tensor.dim())).reshape(
            steps, count, *tail
        ).float()

    q_flat = flat(q, r_out, key_dim)
    k_flat = flat(k, key_dim)
    v_flat = flat(v, value_dim)
    g_flat = flat(g, key_dim)
    erase_flat = flat(beta_e)
    write_flat = flat(beta_w)
    gamma_flat = flat(gamma)
    reset_flat = reset.permute(1, 0).unsqueeze(-1).expand(steps, batch, heads)
    reset_flat = reset_flat.reshape(steps, count)
    if r_out > 1:
        assert out_mix is not None
        mix = out_mix[None].expand(batch, -1, -1).reshape(count, 1, r_out).float()

    state = torch.zeros(
        count, key_dim, value_dim, dtype=torch.float32, device=k.device
    )
    velocity = torch.zeros_like(state)
    outputs: list[torch.Tensor] = []
    for token in range(steps):
        token_reset = reset_flat[token]
        state = torch.where(token_reset[:, None, None], torch.zeros_like(state), state)
        velocity = torch.where(
            token_reset[:, None, None], torch.zeros_like(velocity), velocity
        )
        state_bar = state * g_flat[token].unsqueeze(-1)
        velocity_bar = velocity * g_flat[token].unsqueeze(-1)
        coefficient = gamma_flat[token]
        state_look = state_bar + coefficient[:, None, None] * velocity_bar
        key = k_flat[token]
        memory = torch.bmm(key.unsqueeze(1), state_look).squeeze(1)
        error = (
            write_flat[token].unsqueeze(-1) * v_flat[token]
            - erase_flat[token].unsqueeze(-1) * memory
        )
        gradient = torch.bmm(key.unsqueeze(2), error.unsqueeze(1))
        velocity = coefficient[:, None, None] * velocity_bar + gradient
        momentum_state = state_bar + velocity
        erase_outer = torch.bmm(
            key.unsqueeze(2),
            (erase_flat[token].unsqueeze(-1) * memory).unsqueeze(1),
        )
        write_outer = torch.bmm(
            key.unsqueeze(2),
            (write_flat[token].unsqueeze(-1) * v_flat[token]).unsqueeze(1),
        )
        native_state = state_bar - erase_outer
        native_state = native_state + write_outer
        zero_gate_state = (
            native_state
            + coefficient[:, None, None] * velocity_bar.detach()
        )
        state = torch.where(
            coefficient[:, None, None] == 0,
            zero_gate_state,
            momentum_state,
        )
        read_slots = torch.bmm(q_flat[token], state)
        if r_out > 1:
            read = (read_slots * mix.transpose(1, 2)).sum(1)
        else:
            read = read_slots.squeeze(1)
        outputs.append(read)
    return (
        torch.stack(outputs, 0)
        .reshape(steps, batch, heads, value_dim)
        .permute(1, 0, 2, 3)
    )


class KMD2MomentumAttn(KMD2NativeAttn):
    """Native Qwen layer with a second, decayed Nesterov velocity state."""

    dynamic_state_multiplier = 2

    @classmethod
    def from_native(cls, native: KMD2NativeAttn) -> "KMD2MomentumAttn":
        if isinstance(native, cls):
            raise ValueError("native layer is already a momentum installation")
        if type(native) is not KMD2NativeAttn:
            raise TypeError("native must be an unwrapped KMD2NativeAttn")
        source_state = {
            name: value.detach().clone() for name, value in native.state_dict().items()
        }
        source_parameters = tuple(native.parameters())
        if not source_parameters:
            raise ValueError("native layer has no parameters")
        device = source_parameters[0].device

        replacement = copy.deepcopy(native)
        replacement.__class__ = cls
        replacement.register_parameter(
            "momentum_gamma",
            nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)),
        )
        replacement_state = replacement.state_dict()
        inherited_names = set(replacement_state) - {"momentum_gamma"}
        if inherited_names != set(source_state):
            raise RuntimeError("momentum installation changed inherited state names")
        for name, expected in source_state.items():
            if not torch.equal(replacement_state[name], expected):
                raise RuntimeError(f"momentum installation changed inherited tensor {name}")
        return replacement

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        boundaries: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if "_momentum_boundaries" in self.__dict__:
            raise RuntimeError("reentrant momentum forward is unsupported")
        for field in ("cu_seqlens", "segment_ids", "reset_mask"):
            value = kwargs.get(field)
            populated = value is not None
            if isinstance(value, torch.Tensor):
                populated = value.numel() != 0
            if populated:
                raise ValueError(
                    f"packed metadata {field} is unsupported; pass explicit "
                    "bool boundaries [B,T] for recurrence boundary clearing"
                )
        if boundaries is not None:
            if not isinstance(boundaries, torch.Tensor):
                raise TypeError("boundaries must be a torch.Tensor or None")
            if boundaries.dtype != torch.bool or boundaries.shape != hidden_states.shape[:2]:
                raise ValueError("boundaries must be bool with shape [B,T]")
            if boundaries.device != hidden_states.device:
                raise ValueError("boundaries must share the hidden-state device")
        self._momentum_boundaries = boundaries
        try:
            return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)
        finally:
            del self._momentum_boundaries

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta_e: torch.Tensor,
        beta_w: torch.Tensor,
    ) -> torch.Tensor:
        boundaries = self.__dict__.get("_momentum_boundaries")
        if boundaries is not None and not isinstance(boundaries, torch.Tensor):
            raise RuntimeError("invalid internal momentum boundary state")
        gamma = self.momentum_gamma.view(1, 1, self.H).expand(
            q.shape[0], q.shape[1], self.H
        )
        return momentum_reference_scan(
            q,
            k,
            v,
            g,
            beta_e,
            beta_w,
            gamma,
            out_mix=self.out_mix if self.r_out > 1 else None,
            boundaries=boundaries,
        )

    def project_momentum_gate_(self) -> None:
        """Apply the required post-optimizer projection in place."""

        project_variant_gates_(self)


def lookahead_reference_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    rho: torch.Tensor,
    projection: torch.Tensor,
    *,
    out_mix: torch.Tensor | None = None,
    boundaries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference scan using ``v + rho * P(v - v_prev)`` as write target."""

    q = _require_floating_tensor("q", q)
    k = _require_floating_tensor("k", k)
    v = _require_floating_tensor("v", v)
    g = _require_floating_tensor("g", g)
    beta_e = _require_floating_tensor("beta_e", beta_e)
    beta_w = _require_floating_tensor("beta_w", beta_w)
    rho = _require_floating_tensor("rho", rho)
    projection = _require_floating_tensor("projection", projection)
    if q.ndim != 5:
        raise ValueError("q must have shape [B,T,H,r_out,dk]")
    batch, steps, heads, r_out, key_dim = q.shape
    if k.shape != (batch, steps, heads, key_dim):
        raise ValueError("k must have shape [B,T,H,dk]")
    if v.ndim != 4 or v.shape[:3] != (batch, steps, heads):
        raise ValueError("v must have shape [B,T,H,dv]")
    value_dim = v.shape[-1]
    if value_dim < 1:
        raise ValueError("dv must be positive")
    if projection.shape != (value_dim, value_dim):
        raise ValueError("projection must have shape [dv,dv]")
    if g.shape != (batch, steps, heads, key_dim):
        raise ValueError("g must have shape [B,T,H,dk]")
    for name, tensor in (("beta_e", beta_e), ("beta_w", beta_w), ("rho", rho)):
        if tensor.shape != (batch, steps, heads):
            raise ValueError(f"{name} must have shape [B,T,H]")
    if bool(((rho.detach() < 0) | (rho.detach() > 1)).any()):
        raise ValueError("rho must be in [0,1]")
    devices = {
        tensor.device
        for tensor in (q, k, v, g, beta_e, beta_w, rho, projection)
    }
    if len(devices) != 1:
        raise ValueError("all recurrence tensors must share a device")

    if boundaries is None:
        reset = torch.zeros(batch, steps, dtype=torch.bool, device=k.device)
    else:
        if not isinstance(boundaries, torch.Tensor):
            raise TypeError("boundaries must be a torch.Tensor or None")
        if boundaries.dtype != torch.bool or boundaries.shape != (batch, steps):
            raise ValueError("boundaries must be bool with shape [B,T]")
        if boundaries.device != k.device:
            raise ValueError("boundaries must share the recurrence device")
        reset = boundaries

    if r_out > 1:
        if not isinstance(out_mix, torch.Tensor):
            raise ValueError("out_mix is required when r_out > 1")
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, r_out):
            raise ValueError("out_mix must have shape [H,r_out]")
        if out_mix.device != k.device:
            raise ValueError("out_mix must share the recurrence device")
    elif out_mix is not None:
        _require_floating_tensor("out_mix", out_mix)
        if out_mix.shape != (heads, 1):
            raise ValueError("out_mix must have shape [H,1]")

    count = batch * heads

    def flat(tensor: torch.Tensor, *tail: int) -> torch.Tensor:
        return tensor.permute(1, 0, 2, *range(3, tensor.dim())).reshape(
            steps, count, *tail
        ).float()

    q_flat = flat(q, r_out, key_dim)
    k_flat = flat(k, key_dim)
    v_flat = flat(v, value_dim)
    g_flat = flat(g, key_dim)
    erase_flat = flat(beta_e)
    write_flat = flat(beta_w)
    rho_flat = flat(rho)
    reset_flat = reset.permute(1, 0).unsqueeze(-1).expand(steps, batch, heads)
    reset_flat = reset_flat.reshape(steps, count)
    if r_out > 1:
        assert out_mix is not None
        mix = out_mix[None].expand(batch, -1, -1).reshape(count, 1, r_out).float()

    state = torch.zeros(
        count, key_dim, value_dim, dtype=torch.float32, device=k.device
    )
    previous_value = torch.zeros(
        count, value_dim, dtype=torch.float32, device=k.device
    )
    projection_fp32 = projection.float()
    outputs: list[torch.Tensor] = []
    for token in range(steps):
        token_reset = reset_flat[token]
        state = torch.where(token_reset[:, None, None], torch.zeros_like(state), state)
        previous_value = torch.where(
            token_reset[:, None], torch.zeros_like(previous_value), previous_value
        )
        state_bar = state * g_flat[token].unsqueeze(-1)
        key = k_flat[token]
        value = v_flat[token]
        gate = rho_flat[token]
        if token == 0:
            gate = torch.zeros_like(gate)
        gate = torch.where(token_reset, torch.zeros_like(gate), gate)
        projected_difference = torch.matmul(
            value - previous_value, projection_fp32.transpose(0, 1)
        )
        value_target = value + gate.unsqueeze(-1) * projected_difference
        memory = torch.bmm(key.unsqueeze(1), state_bar).squeeze(1)
        state = state_bar - torch.bmm(
            key.unsqueeze(2),
            (erase_flat[token].unsqueeze(-1) * memory).unsqueeze(1),
        )
        state = state + torch.bmm(
            key.unsqueeze(2),
            (write_flat[token].unsqueeze(-1) * value_target).unsqueeze(1),
        )
        read_slots = torch.bmm(q_flat[token], state)
        if r_out > 1:
            read = (read_slots * mix.transpose(1, 2)).sum(1)
        else:
            read = read_slots.squeeze(1)
        outputs.append(read)
        previous_value = value
    return (
        torch.stack(outputs, 0)
        .reshape(steps, batch, heads, value_dim)
        .permute(1, 0, 2, 3)
    )


class KMD2LookaheadAttn(KMD2NativeAttn):
    """Native Qwen layer with causal value-space extrapolation."""

    implementation_reference = "qwen_variants.KMD2LookaheadAttn.reference_fp32"

    def transformation_manifest(self) -> dict[str, tuple[str, ...]]:
        new = ("lookahead_rho", "lookahead_projection.weight")
        return {"copied": tuple(name for name in self.state_dict() if name not in new),
                "transformed": (), "new": new}

    @classmethod
    def from_native(cls, native: KMD2NativeAttn) -> "KMD2LookaheadAttn":
        if isinstance(native, cls):
            raise ValueError("native layer is already a lookahead installation")
        if type(native) is not KMD2NativeAttn:
            raise TypeError("native must be an unwrapped KMD2NativeAttn")
        source_state = {
            name: value.detach().clone() for name, value in native.state_dict().items()
        }
        source_parameters = tuple(native.parameters())
        if not source_parameters:
            raise ValueError("native layer has no parameters")
        device = source_parameters[0].device
        model_dtype = source_parameters[0].dtype

        replacement = copy.deepcopy(native)
        replacement.__class__ = cls
        replacement.register_parameter(
            "lookahead_rho",
            nn.Parameter(torch.zeros(native.H, dtype=torch.float32, device=device)),
        )
        projection = nn.Linear(
            native.dv,
            native.dv,
            bias=False,
            device=device,
            dtype=model_dtype,
        )
        nn.init.eye_(projection.weight)
        replacement.add_module("lookahead_projection", projection)
        replacement_state = replacement.state_dict()
        new_names = {"lookahead_rho", "lookahead_projection.weight"}
        inherited_names = set(replacement_state) - new_names
        if inherited_names != set(source_state):
            raise RuntimeError("lookahead installation changed inherited state names")
        for name, expected in source_state.items():
            if not torch.equal(replacement_state[name], expected):
                raise RuntimeError(
                    f"lookahead installation changed inherited tensor {name}"
                )
        return replacement

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        boundaries: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        _reject_variant_cache(kwargs)
        if "_lookahead_boundaries" in self.__dict__:
            raise RuntimeError("reentrant lookahead forward is unsupported")
        for field in ("cu_seqlens", "segment_ids", "reset_mask"):
            value = kwargs.get(field)
            populated = value is not None
            if isinstance(value, torch.Tensor):
                populated = value.numel() != 0
            if populated:
                raise ValueError(
                    f"packed metadata {field} is unsupported; pass explicit "
                    "bool boundaries [B,T] for recurrence boundary clearing"
                )
        if boundaries is not None:
            if not isinstance(boundaries, torch.Tensor):
                raise TypeError("boundaries must be a torch.Tensor or None")
            if boundaries.dtype != torch.bool or boundaries.shape != hidden_states.shape[:2]:
                raise ValueError("boundaries must be bool with shape [B,T]")
            if boundaries.device != hidden_states.device:
                raise ValueError("boundaries must share the hidden-state device")
        self._lookahead_boundaries = boundaries
        try:
            return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)
        finally:
            del self._lookahead_boundaries

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta_e: torch.Tensor,
        beta_w: torch.Tensor,
    ) -> torch.Tensor:
        boundaries = self.__dict__.get("_lookahead_boundaries")
        if boundaries is not None and not isinstance(boundaries, torch.Tensor):
            raise RuntimeError("invalid internal lookahead boundary state")
        rho = self.lookahead_rho.view(1, 1, self.H).expand(
            q.shape[0], q.shape[1], self.H
        )
        return lookahead_reference_scan(
            q,
            k,
            v,
            g,
            beta_e,
            beta_w,
            rho,
            self.lookahead_projection.weight,
            out_mix=self.out_mix if self.r_out > 1 else None,
            boundaries=boundaries,
        )

    def project_lookahead_gate_(self) -> None:
        """Apply the required post-optimizer projection in place."""

        project_variant_gates_(self)


__all__ = [
    "KMD2BCBiasAttn",
    "KMD2DiagonalQKAttn",
    "KMD2LookaheadAttn",
    "KMD2MomentumAttn",
    "KMD2TrapezoidAttn",
    "lookahead_reference_scan",
    "momentum_reference_scan",
    "project_variant_gates_",
    "trapezoid_reference_scan",
]
