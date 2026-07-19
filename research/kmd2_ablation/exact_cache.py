"""Independent pure-PyTorch math used by the KMD-2 ablation suite."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from .config import CacheConfig


_ADMISSION_POLICIES = frozenset(
    {
        "exact_outer",
        "coupled_paper",
        "residual_only",
        "write_value",
        "recency",
        "reservoir",
        "future_query_oracle",
    }
)


class AdmissionScoreError(ValueError):
    """A stable typed admission-score contract violation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fp32_vector_norm(value: torch.Tensor) -> torch.Tensor:
    """Return the ordinary fp32 norm with an fp32-only underflow fallback."""

    norm = torch.linalg.vector_norm(value, dim=-1)
    scale = value.abs().amax(dim=-1)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scaled_norm = scale * torch.linalg.vector_norm(
        value / safe_scale.unsqueeze(-1), dim=-1
    )
    return torch.where((norm == 0) & (scale > 0), scaled_norm, norm)


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def admission_scores(
    *,
    policy: str,
    key: torch.Tensor,
    value: torch.Tensor,
    memory: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    positions: torch.Tensor,
    valid: torch.Tensor,
    selector_seed: int | None,
    future_relevance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return detached fp32 per-token/head priorities for one declared policy."""

    if type(policy) is not str:
        raise AdmissionScoreError("type_invalid", "policy must be a string")
    if policy not in _ADMISSION_POLICIES:
        raise AdmissionScoreError(
            "policy_invalid",
            "policy must be one of: " + ", ".join(sorted(_ADMISSION_POLICIES)),
        )
    named = {"key": key, "value": value, "memory": memory}
    if any(not isinstance(tensor, torch.Tensor) for tensor in named.values()):
        raise AdmissionScoreError(
            "type_invalid", "key, value, and memory must be tensors"
        )
    if key.ndim != 4 or value.ndim != 4 or memory.shape != value.shape:
        raise AdmissionScoreError(
            "shape_invalid", "key/value/memory must have shape [B,T,H,D]"
        )
    batch, steps, heads, key_dim = key.shape
    if min(batch, steps, heads, key_dim, value.shape[-1]) < 1:
        raise AdmissionScoreError("shape_invalid", "tensor dimensions must be positive")
    if value.shape[:3] != (batch, steps, heads):
        raise AdmissionScoreError(
            "shape_invalid", "key/value/memory leading dimensions must match"
        )
    if not isinstance(beta_e, torch.Tensor) or not isinstance(beta_w, torch.Tensor):
        raise AdmissionScoreError("type_invalid", "beta_e and beta_w must be tensors")
    if beta_e.shape != (batch, steps, heads) or beta_w.shape != beta_e.shape:
        raise AdmissionScoreError(
            "shape_invalid", "beta_e and beta_w must have shape [B,T,H]"
        )
    if (
        not isinstance(positions, torch.Tensor)
        or positions.dtype != torch.int64
        or positions.shape != (batch, steps)
        or not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or valid.shape != (batch, steps)
    ):
        raise AdmissionScoreError(
            "shape_invalid",
            "positions must be int64 and valid bool with shape [B,T]",
        )
    tensors = (key, value, memory, beta_e, beta_w, positions, valid)
    if len({tensor.device for tensor in tensors}) != 1:
        raise AdmissionScoreError("device_invalid", "all operands must share a device")
    if any(not tensor.is_floating_point() for tensor in (key, value, memory, beta_e, beta_w)):
        raise AdmissionScoreError("dtype_invalid", "score operands must be floating point")
    expanded_valid = valid.unsqueeze(-1).expand(batch, steps, heads)
    for name, tensor in (
        ("key", key),
        ("value", value),
        ("memory", memory),
        ("beta_e", beta_e),
        ("beta_w", beta_w),
    ):
        mask = expanded_valid.unsqueeze(-1).expand_as(tensor) if tensor.ndim == 4 else expanded_valid
        if not bool(torch.isfinite(tensor.detach()[mask]).all()):
            raise AdmissionScoreError(
                "nonfinite_input", f"{name} must be finite at valid positions"
            )
    for name, gate in (("beta_e", beta_e), ("beta_w", beta_w)):
        selected = gate.detach()[expanded_valid]
        if bool(((selected < 0) | (selected > 1)).any()):
            raise AdmissionScoreError(
                "gate_invalid", f"{name} must lie in [0,1] at valid positions"
            )
    if bool((positions[valid] < 0).any()) or bool((positions[~valid] != -1).any()):
        raise AdmissionScoreError(
            "position_invalid", "positions must be nonnegative when valid and -1 otherwise"
        )
    if selector_seed is not None and type(selector_seed) is not int:
        raise AdmissionScoreError("selector_seed_invalid", "selector_seed must be an int")
    if policy == "reservoir" and selector_seed is None:
        raise AdmissionScoreError(
            "selector_seed_required", "reservoir policy requires selector_seed"
        )
    if policy == "future_query_oracle" and future_relevance is None:
        raise AdmissionScoreError(
            "future_relevance_required",
            "future-query oracle requires explicit relevance annotations",
        )

    key_fp32 = key.float()
    value_fp32 = value.float()
    memory_fp32 = memory.float()
    beta_e_fp32 = beta_e.float()
    beta_w_fp32 = beta_w.float()
    zero = torch.zeros((), dtype=torch.float32, device=key.device)
    safe_key = torch.where(
        expanded_valid.unsqueeze(-1), key_fp32, zero
    )
    safe_value = torch.where(
        expanded_valid.unsqueeze(-1), value_fp32, zero
    )
    safe_memory = torch.where(
        expanded_valid.unsqueeze(-1), memory_fp32, zero
    )
    safe_beta_e = torch.where(expanded_valid, beta_e_fp32, zero)
    safe_beta_w = torch.where(expanded_valid, beta_w_fp32, zero)
    key_norm = _fp32_vector_norm(safe_key)
    if policy == "exact_outer":
        update = safe_beta_w.unsqueeze(-1) * safe_value - safe_beta_e.unsqueeze(-1) * safe_memory
        result = key_norm * _fp32_vector_norm(update)
    elif policy == "coupled_paper":
        result = (
            key_norm
            * safe_beta_w
            * _fp32_vector_norm(safe_value - safe_memory)
        )
    elif policy == "residual_only":
        result = key_norm * _fp32_vector_norm(safe_value - safe_memory)
    elif policy == "write_value":
        result = key_norm * safe_beta_w * _fp32_vector_norm(safe_value)
    elif policy == "recency":
        result = (
            positions.float()
            .clamp_min(0)
            .unsqueeze(-1)
            .expand(batch, steps, heads)
        )
    elif policy == "reservoir":
        assert selector_seed is not None
        values = torch.zeros(
            batch, steps, heads, dtype=torch.float32, device=key.device
        )
        for batch_index in range(batch):
            for token_index in range(steps):
                for head_index in range(heads):
                    if not bool(valid[batch_index, token_index]):
                        continue
                    material = (
                        f"{selector_seed}:{batch_index}:{token_index}:{head_index}:"
                        f"{int(positions[batch_index, token_index])}"
                    ).encode("ascii")
                    integer = int.from_bytes(hashlib.sha256(material).digest()[:3], "big")
                    values[batch_index, token_index, head_index] = integer / 16777216.0
        result = values
    else:
        assert future_relevance is not None
        if not isinstance(future_relevance, torch.Tensor):
            raise AdmissionScoreError(
                "type_invalid", "future_relevance must be a tensor"
            )
        if future_relevance.shape == (batch, steps):
            relevance = future_relevance.unsqueeze(-1).expand(batch, steps, heads)
        elif future_relevance.shape == (batch, steps, heads):
            relevance = future_relevance
        else:
            raise AdmissionScoreError(
                "shape_invalid",
                "future_relevance must have shape [B,T] or [B,T,H]",
            )
        if relevance.device != key.device:
            raise AdmissionScoreError(
                "device_invalid", "future_relevance must share the operand device"
            )
        if not (relevance.dtype == torch.bool or relevance.is_floating_point()):
            raise AdmissionScoreError(
                "dtype_invalid", "future_relevance must have bool or floating dtype"
            )
        selected = relevance.detach()[expanded_valid]
        if not bool(torch.isfinite(selected.float()).all()) or bool((selected < 0).any()):
            raise AdmissionScoreError(
                "future_relevance_invalid",
                "future_relevance must be finite and nonnegative at valid positions",
            )
        result = relevance.float()
    return torch.where(expanded_valid, result, torch.zeros_like(result)).detach()


@dataclass(frozen=True)
class CacheReadParameters:
    """Trainable parameters for normalized exact-cache reads."""

    gamma_q: torch.nn.Parameter
    gamma_k: torch.nn.Parameter
    sink_logit: torch.nn.Parameter
    amplitude: torch.nn.Parameter


def initialize_cache_read_parameters(
    key_dim: int,
    heads: int,
    device: torch.device | str | None = None,
) -> CacheReadParameters:
    """Return the declared fp32 cache-read initialization."""
    for name, value in (("key_dim", key_dim), ("heads", heads)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact int")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    return CacheReadParameters(
        gamma_q=torch.nn.Parameter(
            torch.ones(key_dim, dtype=torch.float32, device=device)
        ),
        gamma_k=torch.nn.Parameter(
            torch.ones(key_dim, dtype=torch.float32, device=device)
        ),
        sink_logit=torch.nn.Parameter(
            torch.zeros(heads, dtype=torch.float32, device=device)
        ),
        amplitude=torch.nn.Parameter(
            torch.zeros(heads, dtype=torch.float32, device=device)
        ),
    )


@dataclass(frozen=True)
class CacheReadDiagnostics:
    """Observable causal candidates, probabilities, and workspace accounting."""

    persistent_selected_positions: torch.Tensor
    hit_ready_positions: torch.Tensor
    candidate_valid: torch.Tensor
    attention_weights: torch.Tensor
    top1_positions: torch.Tensor
    attention_entropy: torch.Tensor
    top1_mass: torch.Tensor
    sink_mass: torch.Tensor
    persistent_bytes: int
    block_bytes: int

@dataclass(frozen=True)
class ExactCacheState:
    """Persistent exact-cache tensors in canonical ``[B, H, W, ...]`` form."""

    keys: torch.Tensor
    values: torch.Tensor
    scores: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {
            "keys": self.keys,
            "values": self.values,
            "scores": self.scores,
            "positions": self.positions,
            "valid": self.valid,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if self.keys.ndim != 4:
            raise ValueError(
                "keys must have 4 dimensions with shape [B, H, W, dk]; "
                f"got {tuple(self.keys.shape)}"
            )
        batch, heads, width, key_dim = self.keys.shape
        if batch <= 0 or heads <= 0 or key_dim <= 0:
            raise ValueError("keys dimensions B, H, and dk must be positive")
        if (
            self.values.ndim != 4
            or self.values.shape[:3] != (batch, heads, width)
            or self.values.shape[-1] <= 0
        ):
            raise ValueError(
                "values shape must be [B, H, W, dv] matching keys with positive dv; "
                f"got {tuple(self.values.shape)}"
            )
        metadata_shape = (batch, heads, width)
        for name, tensor in (
            ("scores", self.scores),
            ("positions", self.positions),
            ("valid", self.valid),
        ):
            if tensor.shape != metadata_shape:
                raise ValueError(
                    f"{name} shape must be [B, H, W] matching keys; "
                    f"expected {metadata_shape}, got {tuple(tensor.shape)}"
                )

        storage_dtypes = (torch.float32, torch.bfloat16)
        if self.keys.dtype not in storage_dtypes:
            raise TypeError("keys dtype must be float32 or bfloat16")
        if self.values.dtype not in storage_dtypes:
            raise TypeError("values dtype must be float32 or bfloat16")
        if self.keys.dtype != self.values.dtype:
            raise TypeError("keys and values must use the same dtype")
        if self.scores.dtype != torch.float32:
            raise TypeError("scores dtype must be float32")
        if self.positions.dtype != torch.int64:
            raise TypeError("positions dtype must be int64")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid dtype must be bool")

        if len({tensor.device for tensor in tensors.values()}) != 1:
            raise ValueError("all ExactCacheState tensors must share a device")
        if self.scores.requires_grad or self.scores.grad_fn is not None:
            raise ValueError("scores must be detached")
        detached_valid = self.valid.detach()
        if bool((self.positions.detach()[detached_valid] < 0).any()):
            raise ValueError("positions must be nonnegative at every valid cache slot")
        for name, tensor in (("keys", self.keys), ("values", self.values)):
            if not bool(torch.isfinite(tensor.detach()[detached_valid]).all()):
                raise ValueError(f"{name} must be finite at every valid cache slot")
        if not bool(torch.isfinite(self.scores[self.valid]).all()):
            raise ValueError("scores must be finite at every valid cache slot")

    @property
    def nbytes(self) -> int:
        """Return the bytes occupied by the five state tensors."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.keys,
                self.values,
                self.scores,
                self.positions,
                self.valid,
            )
        )


def deterministic_topw(
    scores: torch.Tensor,
    positions: torch.Tensor,
    valid: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Return deterministic top-width indices for each ``[B, H]`` row.

    Valid entries rank by descending score, then descending absolute position.
    Rows with fewer than ``width`` valid entries are padded with ``-1``.
    """
    if type(width) is not int:
        raise TypeError("width must be an exact int")
    if width < 0:
        raise ValueError("width must be nonnegative")
    if (
        scores.ndim != 3
        or positions.shape != scores.shape
        or valid.shape != scores.shape
    ):
        raise ValueError(
            "scores, positions, and valid must have the same shape [B, H, N]"
        )
    if not torch.is_floating_point(scores):
        raise TypeError("scores must have a floating-point dtype")
    try:
        torch.iinfo(positions.dtype)
    except TypeError as error:
        raise TypeError("positions must have an integer dtype") from error
    if valid.dtype != torch.bool:
        raise TypeError("valid must have a bool dtype")

    with torch.no_grad():
        detached_scores = scores.detach()
        detached_positions = positions.detach()
        detached_valid = valid.detach()
        if not bool(torch.isfinite(detached_scores[detached_valid]).all()):
            raise ValueError("scores must be finite at every valid position")

        batch, heads, candidates = scores.shape
        if width == 0:
            return torch.empty(
                batch,
                heads,
                0,
                dtype=torch.int64,
                device=scores.device,
            )

        position_order = torch.argsort(
            detached_positions,
            dim=-1,
            descending=True,
            stable=True,
        )
        masked_scores = torch.where(
            detached_valid,
            detached_scores,
            torch.full_like(detached_scores, -torch.inf),
        )
        scores_by_position = torch.gather(masked_scores, -1, position_order)
        score_order = torch.argsort(
            scores_by_position,
            dim=-1,
            descending=True,
            stable=True,
        )
        ranked_indices = torch.gather(position_order, -1, score_order)
        ranked_valid = torch.gather(detached_valid, -1, ranked_indices)

        selected_count = min(width, candidates)
        selected = ranked_indices[..., :selected_count]
        selected = torch.where(
            ranked_valid[..., :selected_count],
            selected,
            torch.full_like(selected, -1),
        ).to(dtype=torch.int64)
        if selected_count < width:
            padding = torch.full(
                (batch, heads, width - selected_count),
                -1,
                dtype=torch.int64,
                device=scores.device,
            )
            selected = torch.cat((selected, padding), dim=-1)
        return selected.detach()


def _validate_merge_inputs(
    state: ExactCacheState | None,
    block_k: torch.Tensor,
    block_v: torch.Tensor,
    block_scores: torch.Tensor,
    block_positions: torch.Tensor,
    block_valid: torch.Tensor,
    width: int,
    storage_dtype: torch.dtype,
) -> tuple[int, int, int, int, int]:
    if state is not None and not isinstance(state, ExactCacheState):
        raise TypeError("state must be an ExactCacheState or None")
    if type(width) is not int:
        raise TypeError("width must be an exact int")
    if width < 0:
        raise ValueError("width must be nonnegative")
    if storage_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("storage_dtype must be torch.float32 or torch.bfloat16")

    block_tensors = {
        "block_k": block_k,
        "block_v": block_v,
        "block_scores": block_scores,
        "block_positions": block_positions,
        "block_valid": block_valid,
    }
    for name, tensor in block_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if block_k.ndim != 4:
        raise ValueError(
            "block_k must have 4 dimensions with shape [B, L, H, dk]; "
            f"got {tuple(block_k.shape)}"
        )
    batch, block_length, heads, key_dim = block_k.shape
    if batch <= 0 or heads <= 0 or key_dim <= 0:
        raise ValueError("block_k dimensions B, H, and dk must be positive")
    if (
        block_v.ndim != 4
        or block_v.shape[:3] != (batch, block_length, heads)
        or block_v.shape[-1] <= 0
    ):
        raise ValueError(
            "block_v shape must be [B, L, H, dv] matching block_k with positive dv; "
            f"got {tuple(block_v.shape)}"
        )
    value_dim = block_v.shape[-1]
    if block_scores.shape != (batch, block_length, heads):
        raise ValueError(
            "block_scores shape must be [B, L, H] matching block_k; "
            f"expected {(batch, block_length, heads)}, got {tuple(block_scores.shape)}"
        )
    token_shape = (batch, block_length)
    if block_positions.shape != token_shape:
        raise ValueError(
            "block_positions shape must be [B, L] matching block_k; "
            f"expected {token_shape}, got {tuple(block_positions.shape)}"
        )
    if block_valid.shape != token_shape:
        raise ValueError(
            "block_valid shape must be [B, L] matching block_k; "
            f"expected {token_shape}, got {tuple(block_valid.shape)}"
        )

    for name, tensor in (
        ("block_k", block_k),
        ("block_v", block_v),
        ("block_scores", block_scores),
    ):
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must have a floating-point dtype")
    try:
        torch.iinfo(block_positions.dtype)
    except TypeError as error:
        raise TypeError("block_positions must have an integer dtype") from error
    if block_valid.dtype != torch.bool:
        raise TypeError("block_valid must have a bool dtype")

    devices = {tensor.device for tensor in block_tensors.values()}
    if len(devices) != 1:
        raise ValueError("all completed-block tensors must share a device")
    block_device = block_k.device

    if state is not None:
        if state.keys.shape[0] != batch:
            raise ValueError("state batch dimension must match block_k")
        if state.keys.shape[1] != heads:
            raise ValueError("state head dimension must match block_k")
        if state.keys.shape[2] != width:
            raise ValueError("state width dimension must match requested width")
        if state.keys.shape[-1] != key_dim:
            raise ValueError("state key dimension must match block_k")
        if state.values.shape[-1] != value_dim:
            raise ValueError("state value dimension must match block_v")
        if state.keys.device != block_device:
            raise ValueError("state and completed-block tensors must share a device")

    detached_valid = block_valid.detach()
    if bool((block_positions.detach()[detached_valid] < 0).any()):
        raise ValueError(
            "block_positions must be nonnegative at every valid block position"
        )
    for name, tensor in (("block_k", block_k), ("block_v", block_v)):
        if not bool(torch.isfinite(tensor.detach()[detached_valid]).all()):
            raise ValueError(f"{name} must be finite at every valid block position")
    valid_scores = block_scores.detach()[detached_valid]
    if not bool(torch.isfinite(valid_scores).all()):
        raise ValueError("block_scores must be finite at every valid block position")

    return batch, block_length, heads, key_dim, value_dim


def merge_persistent_cache(
    state: ExactCacheState | None,
    block_k: torch.Tensor,
    block_v: torch.Tensor,
    block_scores: torch.Tensor,
    block_positions: torch.Tensor,
    block_valid: torch.Tensor,
    width: int,
    storage_dtype: torch.dtype,
) -> ExactCacheState:
    """Merge completed-block candidates into a bounded persistent cache."""
    batch, block_length, heads, key_dim, value_dim = _validate_merge_inputs(
        state,
        block_k,
        block_v,
        block_scores,
        block_positions,
        block_valid,
        width,
        storage_dtype,
    )

    candidate_keys = block_k.permute(0, 2, 1, 3)
    candidate_values = block_v.permute(0, 2, 1, 3)
    candidate_scores = block_scores.detach().to(dtype=torch.float32).permute(0, 2, 1)
    candidate_positions = (
        block_positions.detach()
        .to(dtype=torch.int64)
        .unsqueeze(1)
        .expand(batch, heads, block_length)
    )
    candidate_valid = (
        block_valid.detach().unsqueeze(1).expand(batch, heads, block_length)
    )

    if state is not None:
        candidate_keys = torch.cat((state.keys, candidate_keys), dim=2)
        candidate_values = torch.cat((state.values, candidate_values), dim=2)
        candidate_scores = torch.cat((state.scores, candidate_scores), dim=2)
        candidate_positions = torch.cat((state.positions, candidate_positions), dim=2)
        candidate_valid = torch.cat((state.valid, candidate_valid), dim=2)

    selected = deterministic_topw(
        candidate_scores,
        candidate_positions,
        candidate_valid,
        width,
    ).detach()
    selected_valid = selected >= 0
    candidate_count = candidate_scores.shape[-1]

    if candidate_count == 0:
        selected_keys = torch.zeros(
            batch,
            heads,
            width,
            key_dim,
            dtype=storage_dtype,
            device=block_k.device,
        )
        selected_values = torch.zeros(
            batch,
            heads,
            width,
            value_dim,
            dtype=storage_dtype,
            device=block_v.device,
        )
        selected_scores = torch.zeros(
            batch,
            heads,
            width,
            dtype=torch.float32,
            device=block_scores.device,
        )
        selected_positions = torch.full(
            (batch, heads, width),
            -1,
            dtype=torch.int64,
            device=block_positions.device,
        )
    else:
        safe_selected = selected.clamp_min(0)
        selected_keys = torch.gather(
            candidate_keys,
            2,
            safe_selected.unsqueeze(-1).expand(-1, -1, -1, key_dim),
        )
        selected_values = torch.gather(
            candidate_values,
            2,
            safe_selected.unsqueeze(-1).expand(-1, -1, -1, value_dim),
        )
        selected_scores = torch.gather(candidate_scores, 2, safe_selected)
        selected_positions = torch.gather(candidate_positions, 2, safe_selected)

        selected_keys = torch.where(
            selected_valid.unsqueeze(-1),
            selected_keys,
            torch.zeros((), dtype=selected_keys.dtype, device=selected_keys.device),
        ).to(dtype=storage_dtype)
        selected_values = torch.where(
            selected_valid.unsqueeze(-1),
            selected_values,
            torch.zeros((), dtype=selected_values.dtype, device=selected_values.device),
        ).to(dtype=storage_dtype)
        selected_scores = torch.where(
            selected_valid,
            selected_scores,
            torch.zeros((), dtype=torch.float32, device=selected_scores.device),
        )
        selected_positions = torch.where(
            selected_valid,
            selected_positions,
            torch.full((), -1, dtype=torch.int64, device=selected_positions.device),
        )

    return ExactCacheState(
        keys=selected_keys,
        values=selected_values,
        scores=selected_scores.detach(),
        positions=selected_positions.detach(),
        valid=selected_valid.detach(),
    )


def _validate_cache_read_inputs(
    q_eff: torch.Tensor,
    query_positions: torch.Tensor,
    state: ExactCacheState | None,
    block_k: torch.Tensor,
    block_v: torch.Tensor,
    block_scores: torch.Tensor,
    block_positions: torch.Tensor,
    block_valid: torch.Tensor,
    config: CacheConfig,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    sink_logit: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if not isinstance(config, CacheConfig):
        raise TypeError("config must be a CacheConfig")
    if config.read not in {"unit_l2", "fixed_temperature", "rmsnorm"}:
        raise ValueError(f"unsupported cache read policy: {config.read}")
    if config.storage_dtype not in {"fp32", "bf16"}:
        raise ValueError(
            f"unsupported cache storage dtype: {config.storage_dtype}"
        )
    if config.compute_dtype != "fp32":
        raise ValueError("cache reads require config.compute_dtype=fp32")
    if config.inclusive is not True:
        raise ValueError("cache reads require config.inclusive=true")
    if state is not None and not isinstance(state, ExactCacheState):
        raise TypeError("state must be an ExactCacheState or None")

    named_tensors = {
        "q_eff": q_eff,
        "query_positions": query_positions,
        "block_k": block_k,
        "block_v": block_v,
        "block_scores": block_scores,
        "block_positions": block_positions,
        "block_valid": block_valid,
        "gamma_q": gamma_q,
        "gamma_k": gamma_k,
        "sink_logit": sink_logit,
    }
    for name, tensor in named_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if q_eff.ndim != 4:
        raise ValueError(
            "q_eff must have 4 dimensions with shape [B, T, H, dk]; "
            f"got {tuple(q_eff.shape)}"
        )
    batch, steps, heads, key_dim = q_eff.shape
    if min(batch, steps, heads, key_dim) <= 0:
        raise ValueError("q_eff dimensions B, T, H, and dk must be positive")
    if query_positions.shape != (batch, steps):
        raise ValueError(
            "query_positions shape must be [B, T] matching q_eff; "
            f"expected {(batch, steps)}, got {tuple(query_positions.shape)}"
        )
    if block_k.ndim != 4:
        raise ValueError(
            "block_k must have 4 dimensions with shape [B, L, H, dk]; "
            f"got {tuple(block_k.shape)}"
        )
    block_length = block_k.shape[1]
    if block_k.shape != (batch, block_length, heads, key_dim):
        raise ValueError(
            "block_k shape must be [B, L, H, dk] matching q_eff; "
            f"got {tuple(block_k.shape)}"
        )
    if (
        block_v.ndim != 4
        or block_v.shape[:3] != (batch, block_length, heads)
        or block_v.shape[-1] <= 0
    ):
        raise ValueError(
            "block_v shape must be [B, L, H, dv] matching block_k with positive dv; "
            f"got {tuple(block_v.shape)}"
        )
    value_dim = block_v.shape[-1]
    if block_scores.shape != (batch, block_length, heads):
        raise ValueError(
            "block_scores shape must be [B, L, H] matching block_k; "
            f"expected {(batch, block_length, heads)}, got {tuple(block_scores.shape)}"
        )
    token_shape = (batch, block_length)
    if block_positions.shape != token_shape:
        raise ValueError(
            "block_positions shape must be [B, L] matching block_k; "
            f"expected {token_shape}, got {tuple(block_positions.shape)}"
        )
    if block_valid.shape != token_shape:
        raise ValueError(
            "block_valid shape must be [B, L] matching block_k; "
            f"expected {token_shape}, got {tuple(block_valid.shape)}"
        )
    if gamma_q.shape != (key_dim,):
        raise ValueError(
            f"gamma_q shape must be [dk]; expected {(key_dim,)}, got {tuple(gamma_q.shape)}"
        )
    if gamma_k.shape != (key_dim,):
        raise ValueError(
            f"gamma_k shape must be [dk]; expected {(key_dim,)}, got {tuple(gamma_k.shape)}"
        )
    if sink_logit.shape != (heads,):
        raise ValueError(
            f"sink_logit shape must be [H]; expected {(heads,)}, got {tuple(sink_logit.shape)}"
        )

    for name, tensor in (
        ("q_eff", q_eff),
        ("block_k", block_k),
        ("block_v", block_v),
        ("block_scores", block_scores),
        ("gamma_q", gamma_q),
        ("gamma_k", gamma_k),
        ("sink_logit", sink_logit),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} dtype must be float32")
    for name, tensor in (
        ("query_positions", query_positions),
        ("block_positions", block_positions),
    ):
        if tensor.dtype != torch.int64:
            raise TypeError(f"{name} dtype must be int64")
    if block_valid.dtype != torch.bool:
        raise TypeError("block_valid must have a bool dtype")

    devices = {tensor.device for tensor in named_tensors.values()}
    if len(devices) != 1:
        raise ValueError("all cache-read tensors must share a device")
    device = q_eff.device
    detached_query_positions = query_positions.detach()
    detached_block_positions = block_positions.detach()
    detached_block_valid = block_valid.detach()
    if bool((detached_query_positions < 0).any()):
        raise ValueError("query_positions must be nonnegative")
    if bool((detached_block_positions[detached_block_valid] < 0).any()):
        raise ValueError(
            "block_positions must be nonnegative at every valid block position"
        )

    if block_length > 0:
        if block_length != steps:
            raise ValueError(
                "current block length must equal query steps when the block is nonempty"
            )
        if not torch.equal(
            detached_block_positions[detached_block_valid],
            detached_query_positions[detached_block_valid],
        ):
            raise ValueError(
                "block_positions must exactly equal query_positions for a nonempty current block"
            )

    if state is not None:
        if state.keys.shape[0] != batch:
            raise ValueError("state batch dimension must match q_eff")
        if state.keys.shape[1] != heads:
            raise ValueError("state head dimension must match q_eff")
        if state.keys.shape[2] != config.width:
            raise ValueError("state width dimension must match config.width")
        if state.keys.shape[-1] != key_dim:
            raise ValueError("state key dimension must match q_eff")
        if state.values.shape[-1] != value_dim:
            raise ValueError("state value dimension must match block_v")
        if state.keys.device != device:
            raise ValueError("state and cache-read tensors must share a device")
        expected_storage_dtype = (
            torch.float32 if config.storage_dtype == "fp32" else torch.bfloat16
        )
        if state.keys.dtype != expected_storage_dtype:
            raise TypeError("state key/value dtype must match config.storage_dtype")

    if not bool(torch.isfinite(q_eff.detach()).all()):
        raise ValueError("q_eff must contain only finite values")
    for name, tensor in (
        ("gamma_q", gamma_q),
        ("gamma_k", gamma_k),
        ("sink_logit", sink_logit),
    ):
        if not bool(torch.isfinite(tensor.detach()).all()):
            raise ValueError(f"{name} must contain only finite values")

    detached_block_valid = block_valid.detach()
    block_head_valid = detached_block_valid.unsqueeze(-1).expand(
        batch, block_length, heads
    )
    for name, tensor in (
        ("block_k", block_k),
        ("block_v", block_v),
    ):
        expanded_valid = block_head_valid.unsqueeze(-1).expand_as(tensor)
        if not bool(torch.isfinite(tensor.detach()[expanded_valid]).all()):
            raise ValueError(f"{name} must be finite at every valid block position")
    if not bool(torch.isfinite(block_scores.detach()[block_head_valid]).all()):
        raise ValueError("block_scores must be finite at every valid block position")

    if state is not None:
        for name, tensor in (("state.keys", state.keys), ("state.values", state.values)):
            expanded_valid = state.valid.detach().unsqueeze(-1).expand_as(tensor)
            if not bool(torch.isfinite(tensor.detach()[expanded_valid]).all()):
                raise ValueError(f"{name} must be finite at every valid cache slot")

    return batch, steps, heads, key_dim, value_dim, block_length


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def cache_read_blocks(
    q_eff: torch.Tensor,
    query_positions: torch.Tensor,
    state: ExactCacheState | None,
    block_k: torch.Tensor,
    block_v: torch.Tensor,
    block_scores: torch.Tensor,
    block_positions: torch.Tensor,
    block_valid: torch.Tensor,
    config: CacheConfig,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    sink_logit: torch.Tensor,
) -> tuple[torch.Tensor, CacheReadDiagnostics]:
    """Read persistent and causally visible current-block exact-cache entries."""
    batch, steps, heads, key_dim, value_dim, block_length = (
        _validate_cache_read_inputs(
            q_eff,
            query_positions,
            state,
            block_k,
            block_v,
            block_scores,
            block_positions,
            block_valid,
            config,
            gamma_q,
            gamma_k,
            sink_logit,
        )
    )
    storage_dtype = (
        torch.float32 if config.storage_dtype == "fp32" else torch.bfloat16
    )
    device = q_eff.device

    block_keys = block_k.to(dtype=storage_dtype).to(dtype=torch.float32).permute(0, 2, 1, 3)
    block_values = (
        block_v.to(dtype=storage_dtype)
        .to(dtype=torch.float32)
        .permute(0, 2, 1, 3)
    )
    block_candidate_positions = (
        block_positions.detach()
        .to(dtype=torch.int64)
        .unsqueeze(1)
        .expand(batch, heads, block_length)
    )
    block_candidate_valid = (
        block_valid.detach().unsqueeze(1).expand(batch, heads, block_length)
    )

    if state is None:
        persistent_width = 0
        persistent_keys = torch.empty(
            batch, heads, 0, key_dim, dtype=torch.float32, device=device
        )
        persistent_values = torch.empty(
            batch, heads, 0, value_dim, dtype=torch.float32, device=device
        )
        persistent_positions = torch.empty(
            batch, heads, 0, dtype=torch.int64, device=device
        )
        persistent_valid = torch.empty(
            batch, heads, 0, dtype=torch.bool, device=device
        )
    else:
        persistent_width = state.keys.shape[2]
        persistent_keys = state.keys.to(dtype=torch.float32)
        persistent_values = state.values.to(dtype=torch.float32)
        persistent_positions = state.positions.detach()
        persistent_valid = state.valid.detach()

    candidate_keys = torch.cat((persistent_keys, block_keys), dim=2)
    candidate_values = torch.cat((persistent_values, block_values), dim=2)
    candidate_positions = torch.cat(
        (persistent_positions, block_candidate_positions), dim=2
    )
    base_candidate_valid = torch.cat(
        (persistent_valid, block_candidate_valid), dim=2
    )
    candidate_count = persistent_width + block_length

    safe_candidate_keys = torch.where(
        base_candidate_valid.unsqueeze(-1),
        candidate_keys,
        torch.zeros((), dtype=torch.float32, device=device),
    )
    safe_candidate_values = torch.where(
        base_candidate_valid.unsqueeze(-1),
        candidate_values,
        torch.zeros((), dtype=torch.float32, device=device),
    )
    q = q_eff.to(dtype=torch.float32)
    gamma_q_fp32 = gamma_q.to(dtype=torch.float32)
    gamma_k_fp32 = gamma_k.to(dtype=torch.float32)

    if config.read in {"unit_l2", "fixed_temperature"}:
        q_norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(
            config.eps_cache
        )
        key_norm = torch.linalg.vector_norm(
            safe_candidate_keys, dim=-1, keepdim=True
        ).clamp_min(config.eps_cache)
        cosine = torch.einsum(
            "bthd,bhnd->bthn",
            q / q_norm,
            safe_candidate_keys / key_norm,
        )
        if config.read == "unit_l2":
            candidate_logits = cosine / math.sqrt(key_dim)
        else:
            candidate_logits = cosine * math.sqrt(key_dim)
    else:
        q_rms = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + config.eps_cache)
        key_rms = safe_candidate_keys * torch.rsqrt(
            safe_candidate_keys.square().mean(dim=-1, keepdim=True)
            + config.eps_cache
        )
        q_rms = q_rms * gamma_q_fp32
        key_rms = key_rms * gamma_k_fp32
        candidate_logits = torch.einsum(
            "bthd,bhnd->bthn", q_rms, key_rms
        ) / math.sqrt(key_dim)

    query_position_view = query_positions.detach().to(dtype=torch.int64).view(
        batch, steps, 1, 1
    )
    expanded_positions = candidate_positions.unsqueeze(1).expand(
        batch, steps, heads, candidate_count
    )
    persistent_index_visible = torch.ones(
        batch,
        steps,
        heads,
        persistent_width,
        dtype=torch.bool,
        device=device,
    )
    block_index_visible = (
        torch.arange(block_length, device=device).view(1, 1, 1, block_length)
        <= torch.arange(steps, device=device).view(1, steps, 1, 1)
    ).expand(batch, steps, heads, block_length)
    causal_index_visible = torch.cat(
        (persistent_index_visible, block_index_visible), dim=-1
    )
    candidate_valid = base_candidate_valid.unsqueeze(1).expand(
        batch, steps, heads, candidate_count
    ) & (expanded_positions <= query_position_view) & causal_index_visible
    if not bool(torch.isfinite(candidate_logits.detach()[candidate_valid]).all()):
        raise ValueError("cache read produced a nonfinite valid candidate logit")

    masked_logits = torch.where(
        candidate_valid,
        candidate_logits,
        torch.full((), -torch.inf, dtype=torch.float32, device=device),
    )
    sink_logits = sink_logit.to(dtype=torch.float32).view(1, 1, heads, 1).expand(
        batch, steps, heads, 1
    )
    all_logits = torch.cat((masked_logits, sink_logits), dim=-1)
    attention_weights = torch.softmax(all_logits, dim=-1)
    if not bool(torch.isfinite(attention_weights.detach()).all()):
        raise ValueError("cache read produced nonfinite attention weights")

    y_cache = torch.einsum(
        "bthn,bhnd->bthd",
        attention_weights[..., :candidate_count],
        safe_candidate_values,
    )
    hit_ready_positions = torch.where(
        candidate_valid,
        expanded_positions,
        torch.full((), -1, dtype=torch.int64, device=device),
    )
    if state is None:
        persistent_selected_positions = persistent_positions
    else:
        persistent_selected_positions = torch.where(
            persistent_valid,
            persistent_positions,
            torch.full((), -1, dtype=torch.int64, device=device),
        )

    top1_indices = attention_weights.argmax(dim=-1)
    if candidate_count == 0:
        top1_positions = torch.full(
            (batch, steps, heads), -1, dtype=torch.int64, device=device
        )
    else:
        gathered_positions = torch.gather(
            hit_ready_positions,
            -1,
            top1_indices.clamp_max(candidate_count - 1).unsqueeze(-1),
        ).squeeze(-1)
        top1_positions = torch.where(
            top1_indices == candidate_count,
            torch.full((), -1, dtype=torch.int64, device=device),
            gathered_positions,
        )

    entropy_floor = torch.finfo(torch.float32).tiny
    attention_entropy = -(
        attention_weights * attention_weights.clamp_min(entropy_floor).log()
    ).sum(dim=-1)
    top1_mass = attention_weights.max(dim=-1).values
    sink_mass = attention_weights[..., -1]
    storage_element_size = 4 if storage_dtype == torch.float32 else 2
    block_bytes = batch * heads * block_length * (
        key_dim * storage_element_size
        + value_dim * storage_element_size
        + 4
        + 8
        + 1
    )

    diagnostics = CacheReadDiagnostics(
        persistent_selected_positions=persistent_selected_positions.detach(),
        hit_ready_positions=hit_ready_positions.detach(),
        candidate_valid=candidate_valid.detach(),
        attention_weights=attention_weights,
        top1_positions=top1_positions.detach(),
        attention_entropy=attention_entropy,
        top1_mass=top1_mass,
        sink_mass=sink_mass,
        persistent_bytes=0 if state is None else state.nbytes,
        block_bytes=block_bytes,
    )
    return y_cache, diagnostics


def _validate_reference_scan_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    out_mix: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int]:
    named_inputs = {
        "q": q,
        "k": k,
        "v": v,
        "decay": decay,
        "beta_e": beta_e,
        "beta_w": beta_w,
    }
    if out_mix is not None:
        named_inputs["out_mix"] = out_mix

    for name, tensor in named_inputs.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must have a floating-point dtype")

    if q.ndim != 5:
        raise ValueError(
            "q must have 5 dimensions with shape [B, T, H, R, dk]; "
            f"got {tuple(q.shape)}"
        )
    batch, steps, heads, slots, key_dim = q.shape
    if min(batch, steps, heads, slots, key_dim) <= 0:
        raise ValueError("q dimensions B, T, H, R, and dk must all be positive")
    if k.shape != (batch, steps, heads, key_dim):
        raise ValueError(
            "k shape must be [B, T, H, dk] matching q; "
            f"expected {(batch, steps, heads, key_dim)}, got {tuple(k.shape)}"
        )
    if v.ndim != 4 or v.shape[:3] != (batch, steps, heads) or v.shape[-1] <= 0:
        raise ValueError(
            "v shape must be [B, T, H, dv] matching q with positive dv; "
            f"got {tuple(v.shape)}"
        )
    value_dim = v.shape[-1]
    if decay.shape != (batch, steps, heads, key_dim):
        raise ValueError(
            "decay shape must be [B, T, H, dk] matching k; "
            f"expected {(batch, steps, heads, key_dim)}, got {tuple(decay.shape)}"
        )
    gate_shape = (batch, steps, heads)
    if beta_e.shape != gate_shape:
        raise ValueError(
            "beta_e shape must be [B, T, H]; "
            f"expected {gate_shape}, got {tuple(beta_e.shape)}"
        )
    if beta_w.shape != gate_shape:
        raise ValueError(
            "beta_w shape must be [B, T, H]; "
            f"expected {gate_shape}, got {tuple(beta_w.shape)}"
        )

    if slots == 1:
        if out_mix is not None:
            raise ValueError("out_mix must be None when R is 1")
    elif out_mix is None:
        raise ValueError("out_mix is required when R is greater than 1")
    elif out_mix.shape != (heads, slots):
        raise ValueError(
            "out_mix shape must be [H, R]; "
            f"expected {(heads, slots)}, got {tuple(out_mix.shape)}"
        )

    devices = {tensor.device for tensor in named_inputs.values()}
    if len(devices) != 1:
        raise ValueError("q, k, v, decay, beta_e, beta_w, and out_mix must share a device")
    for name, tensor in named_inputs.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"all inputs must be finite; {name} contains NaN or infinity")

    return batch, steps, heads, slots, key_dim, value_dim


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def reference_scan_with_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta_e: torch.Tensor,
    beta_w: torch.Tensor,
    out_mix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return post-update state reads and detached exact outer-update scores.

    The state starts at zero and is updated token by token as
    ``S = decay * S + k * (beta_w * v - beta_e * (k^T S))^T``. Float64
    inputs retain float64 reference precision; all other floating dtypes use
    float32 compute.
    """
    batch, steps, heads, slots, key_dim, value_dim = _validate_reference_scan_inputs(
        q, k, v, decay, beta_e, beta_w, out_mix
    )
    tensors = (q, k, v, decay, beta_e, beta_w)
    if out_mix is not None:
        tensors = (*tensors, out_mix)
    compute_dtype = (
        torch.float64
        if any(tensor.dtype == torch.float64 for tensor in tensors)
        else torch.float32
    )
    q_c, k_c, v_c, decay_c, beta_e_c, beta_w_c = (
        tensor.to(dtype=compute_dtype) for tensor in (q, k, v, decay, beta_e, beta_w)
    )
    mix_c = None if out_mix is None else out_mix.to(dtype=compute_dtype)

    state = torch.zeros(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=compute_dtype,
        device=q.device,
    )
    outputs: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    for token in range(steps):
        state_bar = decay_c[:, token].unsqueeze(-1) * state
        key = k_c[:, token]
        memory = torch.matmul(key.unsqueeze(-2), state_bar).squeeze(-2)
        update = (
            beta_w_c[:, token].unsqueeze(-1) * v_c[:, token]
            - beta_e_c[:, token].unsqueeze(-1) * memory
        )
        state = state_bar + key.unsqueeze(-1) * update.unsqueeze(-2)

        slot_reads = torch.matmul(q_c[:, token], state)
        if slots == 1:
            output = slot_reads.squeeze(-2)
        else:
            assert mix_c is not None
            output = (slot_reads * mix_c.unsqueeze(0).unsqueeze(-1)).sum(dim=-2)
        outputs.append(output)
        scores.append(
            torch.linalg.vector_norm(key, dim=-1)
            * torch.linalg.vector_norm(update, dim=-1)
        )

    return torch.stack(outputs, dim=1), torch.stack(scores, dim=1).detach()
