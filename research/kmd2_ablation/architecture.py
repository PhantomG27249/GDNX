"""Shared, dependency-light state-update equations for KMD-2 ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


def _validate_common(
    state: Tensor,
    key: Tensor,
    value: Tensor,
    erase: Tensor,
    write: Tensor,
) -> tuple[int, int, int, int, int]:
    operands = (state, key, value, erase, write)
    if not all(isinstance(operand, Tensor) for operand in operands):
        raise TypeError("architecture operands must be torch tensors")
    if any(not operand.is_floating_point() for operand in operands):
        raise TypeError("architecture operands must be floating point")
    if len({operand.device for operand in operands}) != 1:
        raise ValueError("architecture operands must share one device")
    if len({operand.dtype for operand in operands}) != 1:
        raise ValueError("architecture operands must share one dtype")
    if any(not bool(torch.isfinite(operand.detach()).all()) for operand in operands):
        raise ValueError("architecture operands must contain only finite values")
    if bool((erase.detach() < 0).any()) or bool((write.detach() < 0).any()):
        raise ValueError("architecture gates must be nonnegative")
    if state.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("architecture state/key/value ranks are invalid")
    batch, heads, key_dim, value_dim = state.shape
    if key.shape[:2] != (batch, heads) or key.shape[-1] != key_dim:
        raise ValueError("architecture key must have shape [B,H,R,dk]")
    rank = key.shape[2]
    if rank < 1 or value.shape != (batch, heads, rank, value_dim):
        raise ValueError("architecture value must have shape [B,H,R,dv]")
    return batch, heads, rank, key_dim, value_dim


def true_mimo_update(
    state: Tensor,
    key: Tensor,
    value: Tensor,
    beta_e: Tensor,
    beta_w: Tensor,
) -> Tensor:
    """Apply the simultaneous scalar-gated rank-R update."""

    batch, heads, rank, _, _ = _validate_common(
        state, key, value, beta_e, beta_w
    )
    if beta_e.shape != (batch, heads, rank) or beta_w.shape != (
        batch,
        heads,
        rank,
    ):
        raise ValueError("true-MIMO gates must have shape [B,H,R]")
    return _true_mimo_update_unchecked(state, key, value, beta_e, beta_w)


def _true_mimo_update_unchecked(
    state: Tensor,
    key: Tensor,
    value: Tensor,
    beta_e: Tensor,
    beta_w: Tensor,
) -> Tensor:
    """Apply scalar-gated arithmetic to operands validated by the caller."""

    rank = key.shape[2]
    if rank == 1:
        k = key[:, :, 0]
        v = value[:, :, 0]
        memory = torch.matmul(k.unsqueeze(-2), state).squeeze(-2)
        update = beta_w[:, :, 0, None] * v - beta_e[:, :, 0, None] * memory
        return state + k.unsqueeze(-1) * update.unsqueeze(-2)
    memory = torch.matmul(key, state)
    erase = torch.einsum(
        "bhrd,bhrv->bhdv", key, (beta_e / rank).unsqueeze(-1) * memory
    )
    write = torch.einsum(
        "bhrd,bhrv->bhdv", key, beta_w.unsqueeze(-1) * value
    )
    return state - erase + write


def channelwise_gdn2_update(
    state: Tensor,
    key: Tensor,
    value: Tensor,
    erase: Tensor,
    write: Tensor,
) -> Tensor:
    """Apply the channelwise Gated DeltaNet-2 rank-one update."""

    batch, heads, rank, key_dim, value_dim = _validate_common(
        state, key, value, erase, write
    )
    if rank != 1:
        raise ValueError("channelwise Gated DeltaNet-2 gates require rank R=1")
    if erase.shape != (batch, heads, 1, key_dim) or write.shape != (
        batch,
        heads,
        1,
        value_dim,
    ):
        raise ValueError(
            "channelwise gates must have shapes [B,H,1,dk] and [B,H,1,dv]"
        )
    return _channelwise_gdn2_update_unchecked(state, key, value, erase, write)


def _channelwise_gdn2_update_unchecked(
    state: Tensor,
    key: Tensor,
    value: Tensor,
    erase: Tensor,
    write: Tensor,
) -> Tensor:
    """Apply channel-gated arithmetic to operands validated by the caller."""

    k = key[:, :, 0]
    v = value[:, :, 0]
    memory = torch.matmul((erase[:, :, 0] * k).unsqueeze(-2), state).squeeze(-2)
    update = write[:, :, 0] * v - memory
    return state + k.unsqueeze(-1) * update.unsqueeze(-2)


TARGET_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)
SEEDS = (11, 29, 47)
IDENTITY_FIELDS = ("model_tree_sha256", "ordered_examples_sha256", "pretraining_checkpoint_sha256")
_REGISTRY_INITIALIZING = True
STAGE1_IDS = (
    "stock", "kmd2-r1", "rot-off", "rot-constant", "rot-noncumulative",
    "rot-fixed-rope", "rot-moving-frame-oracle", "conv-off", "trapezoid",
    "qk-bc-additive", "qk-diagonal", "qk-constant-coordinate-oracle",
    "lookahead", "state-64x128", "state-256x128", "rout-4", "mimo-r2",
    "mimo-r4", "gdn2-channel-r1", "cache-block-only-w0",
    "cache-surprise-w64", "cache-recency-w64", "cache-coupled-w64",
    "cache-residual-w64", "cache-write-value-w64", "cache-reservoir-w64",
    "cache-per-slot-read-w64", "cache-unbounded-oracle",
    "gdn2-mimo-r4-braid-shared-hola-w64",
    "gdn2-mimo-r4-braid-four-state-hola-w64",
)


@dataclass(frozen=True)
class CacheArchitecture:
    enabled: bool = False
    width: int | None = 64
    block_size: int = 256
    score: str = "none"
    read: str = "block"
    read_init: str = "zero"
    coordinate_frame: str = "rotated_recurrence"
    storage_dtype: str = "bfloat16"
    compute_dtype: str = "float32"
    inclusive: bool = True
    tie_policy: str = "score-desc-position-desc"
    selector_seed: int = 0
    amplitude_init: float = 0.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.inclusive) is not bool:
            raise TypeError("cache booleans must be bool")
        if self.width is not None and (type(self.width) is not int or self.width < 0):
            raise ValueError("cache width must be a nonnegative int or null for unbounded")
        if type(self.block_size) is not int or self.block_size != 256:
            raise ValueError("cache block_size must be 256")
        allowed_scores = {"none", "block_only", "||k||_2 * ||beta_w*v - beta_e*m||_2", "||k||_2 * beta_w * ||v-m||_2", "||k||_2 * ||v-m||_2", "||k||_2 * beta_w * ||v||_2", "token_position", "SHA256(seed:batch:token:head:position)[0:3]_big_endian / 16777216", "per_slot_exact_outer", "active_update_frobenius", "unbounded_oracle"}
        if type(self.score) is not str or self.score not in allowed_scores:
            raise ValueError("unknown cache score semantics")
        if self.read not in {"block", "unit_l2", "per_slot_unit_l2", "rmsnorm_rank_aware"}:
            raise ValueError("unknown cache read semantics")
        if self.read_init not in {"zero", "gamma_one_sink_zero_amplitude_zero", "hola_gate_logit_v2_minus4"}:
            raise ValueError("unknown cache read initialization")
        if self.coordinate_frame != "rotated_recurrence" or self.storage_dtype != "bfloat16" or self.compute_dtype != "float32":
            raise ValueError("cache frame and dtype semantics are frozen")
        if self.tie_policy != "score-desc-position-desc":
            raise ValueError("cache tie policy is frozen")
        if type(self.selector_seed) is not int or self.selector_seed < 0:
            raise ValueError("selector_seed must be a nonnegative int")
        if type(self.amplitude_init) is not float or self.amplitude_init != 0.0:
            raise ValueError("amplitude_init must be exactly 0.0")
        if self.width is None and self.score != "unbounded_oracle":
            raise ValueError("only the unbounded oracle may have null width")
        if self.enabled and self.width == 0:
            raise ValueError("enabled cache must have bounded positive or unbounded width")
        if not self.enabled and self.width not in {0, 64}:
            raise ValueError("disabled cache width must be the explicit default or W0 control")
        reservoir_score = "SHA256(seed:batch:token:head:position)[0:3]_big_endian / 16777216"
        if self.score == reservoir_score and self.selector_seed != 11:
            raise ValueError("reservoir selector_seed is frozen to 11")
        if self.score != reservoir_score and self.selector_seed != 0:
            raise ValueError("only reservoir selection has a nonzero selector seed")


@dataclass(frozen=True)
class Compatibility:
    forbidden_arm_ids: tuple[str, ...] = ()
    forbidden_families: tuple[str, ...] = ()
    requires_semantics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("forbidden_arm_ids", "forbidden_families", "requires_semantics"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not str or not item for item in value):
                raise TypeError(f"{name} must be a tuple of nonempty strings")
            if len(set(value)) != len(value):
                raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class QKMode:
    family: str
    q_amplitude: float
    k_amplitude: float
    coordinate_init: tuple[float, float]

    def __post_init__(self) -> None:
        if self.family != "diagonal": raise ValueError("QK family must be diagonal")
        if type(self.q_amplitude) is not float or type(self.k_amplitude) is not float:
            raise TypeError("QK amplitudes must be floats")
        if self.q_amplitude != 0.0 or self.k_amplitude != 0.0:
            raise ValueError("QK diagonal amplitudes must initialize to zero")
        if type(self.coordinate_init) is not tuple:
            raise TypeError("QK coordinate_init must be an immutable tuple")
        if self.coordinate_init != (-0.5, 0.5):
            raise ValueError("QK coordinate initialization is frozen")


@dataclass(frozen=True)
class ArchitectureRecord:
    arm_id: str
    baseline_id: str
    classification: str
    promotable: bool
    stage2_eligible: bool
    target_layers: tuple[int, ...] = TARGET_LAYERS
    seeds: tuple[int, ...] = SEEDS
    # 2026-07-15: heal budget raised 64 -> 1536 updates (bible footnote 17).
    # Option A starts ~1.8% from the teacher, but opening the zero-initialized
    # capacity (mixer lanes 1-3, trapezoid, cache gate) needs optimizer steps;
    # 64 was a pipeline-validation budget, 1536 matches retrofit experience.
    updates: int = 1536
    tokens: int = 6291456
    curriculum_lengths: tuple[int, ...] = (512, 2048, 4096, 8192)
    extrapolation_lengths: tuple[int, ...] = (16384, 32768)
    # Qwen3.5-0.8B exposes sixteen 128-wide Gated DeltaNet heads.  Its eight
    # full-attention heads are a separate dimension and remain untouched.
    num_heads: int = 16
    state_key_dim: int = 128
    state_value_dim: int = 128
    rotation_mode: str = "current"
    convolution_on: bool = True
    state_input_mode: str = "native"
    qk_mode: object = "none"
    lookahead: bool = False
    output_width: int = 1
    mimo_rank: int = 1
    gate_mode: str = "scalar"
    cache: CacheArchitecture = CacheArchitecture()
    identity_fields: tuple[str, ...] = IDENTITY_FIELDS
    compatibility: Compatibility = Compatibility()
    core_family: str = "legacy_scalar"
    timescales: tuple[int, ...] = ()
    state_topology: str = "shared"
    cache_scope: str = "per_head"
    update_paths: int = 1
    phase_topology: str = "none"
    input_rank: int = 1
    output_rank: int = 1
    read_topology: str = "legacy"
    read_paths: int = 1
    transition_paths: int | None = None
    qk_contract: str = "none"

    def __post_init__(self) -> None:
        for name in ("arm_id", "baseline_id", "classification", "rotation_mode", "state_input_mode", "gate_mode", "core_family", "state_topology", "cache_scope", "phase_topology", "read_topology", "qk_contract"):
            if type(getattr(self, name)) is not str or not getattr(self, name): raise TypeError(f"{name} must be a nonempty str")
        for name in ("promotable", "stage2_eligible", "convolution_on", "lookahead"):
            if type(getattr(self, name)) is not bool: raise TypeError(f"{name} must be bool")
        for name in ("target_layers", "seeds", "curriculum_lengths", "extrapolation_lengths", "identity_fields", "timescales"):
            value = getattr(self, name)
            if type(value) is not tuple: raise TypeError(f"{name} must be an immutable tuple")
        if self.arm_id not in STAGE1_IDS: raise ValueError("unknown architecture arm_id")
        if self.baseline_id not in {"stock", "kmd2-r1"}: raise ValueError("unknown baseline_id")
        if self.classification not in {"source", "converted_control", "reliance", "alternative", "diagnostic", "addition", "cold_redesign", "control", "replacement"}:
            raise ValueError("unknown architecture classification")
        if self.target_layers != TARGET_LAYERS or self.seeds != SEEDS:
            raise ValueError("target layers and seeds are frozen")
        if self.curriculum_lengths != (512, 2048, 4096, 8192) or self.extrapolation_lengths != (16384, 32768):
            raise ValueError("curriculum and extrapolation lengths are frozen")
        for name in ("updates", "tokens", "num_heads", "state_key_dim", "state_value_dim", "output_width", "mimo_rank", "update_paths", "input_rank", "output_rank", "read_paths"):
            value = getattr(self, name)
            if type(value) is not int or value < 1: raise ValueError(f"{name} must be a positive int")
        if self.transition_paths is not None and (type(self.transition_paths) is not int or self.transition_paths < 1):
            raise ValueError("transition_paths must be a positive int or None")
        if self.updates != 1536 or self.tokens != 6291456 or self.num_heads != 16:
            raise ValueError("architecture budget and head count are frozen")
        if self.state_key_dim not in {64, 128, 256} or self.state_value_dim != 128:
            raise ValueError("unknown recurrent state dimensions")
        if self.output_width not in {1, 4} or self.mimo_rank not in {1, 2, 4}:
            raise ValueError("unknown output width or MIMO rank")
        if self.rotation_mode not in {"current", "off", "constant", "noncumulative", "fixed-rope", "moving-frame-oracle", "mamba3_complex_input_dependent_cumulative"}:
            raise ValueError("unknown rotation mode")
        if self.state_input_mode not in {"native", "trapezoid"} or self.gate_mode not in {"scalar", "channelwise"}:
            raise ValueError("unknown state input or gate mode")
        if not isinstance(self.cache, CacheArchitecture) or not isinstance(self.compatibility, Compatibility):
            raise TypeError("cache and compatibility must be frozen value objects")
        if not (type(self.qk_mode) is str or isinstance(self.qk_mode, QKMode)):
            raise TypeError("qk_mode must be a frozen QK mode or string choice")
        if type(self.qk_mode) is str and self.qk_mode not in {"none", "bc_additive", "constant_coordinate_oracle"}:
            raise ValueError("unknown qk_mode")
        if self.identity_fields != IDENTITY_FIELDS: raise ValueError("identity_fields are frozen")
        hybrid = self.arm_id.startswith("gdn2-mimo-r4-braid-")
        if self.mimo_rank > 1 and not hybrid and (self.output_width != 1 or self.gate_mode != "scalar" or self.cache.enabled):
            raise ValueError("true MIMO requires scalar gates, R_out=1, and disabled cache")
        if self.gate_mode == "channelwise" and not hybrid and (self.mimo_rank != 1 or self.cache.enabled):
            raise ValueError("channelwise GDN-2 requires R=1 and disabled cache")
        if self.output_width > 1 and self.mimo_rank > 1 and not hybrid:
            raise ValueError("shared-query widening is incompatible with true MIMO")
        if hybrid:
            expected_topology = "mimo_rank_contributions" if "four-state" in self.arm_id else "shared"
            expected_phase = "per_rank" if expected_topology == "mimo_rank_contributions" else "shared"
            expected_read_topology = "full_cross_state" if expected_topology == "mimo_rank_contributions" else "shared_state_outputs"
            expected_read_paths = 16 if expected_topology == "mimo_rank_contributions" else 4
            if (self.core_family != "gdn2_channelwise" or self.mimo_rank != 4 or
                    self.input_rank != 4 or self.output_rank != 4 or
                    self.output_width != 4 or self.gate_mode != "channelwise" or
                    self.timescales != ((1, 16, 64, 256) if expected_topology == "mimo_rank_contributions"
                                        else (64, 512, 4096, 32768)) or
                    self.state_topology != expected_topology or
                    self.rotation_mode != "mamba3_complex_input_dependent_cumulative" or
                    self.phase_topology != expected_phase or
                    self.read_topology != expected_read_topology or
                    self.read_paths != expected_read_paths or
                    self.transition_paths != (4 if expected_topology == "mimo_rank_contributions" else None) or
                    self.cache_scope != "per_head_shared_across_ranks" or
                    self.update_paths != 4 or not self.convolution_on or
                    self.state_input_mode != "trapezoid" or
                    self.lookahead != (expected_topology == "shared") or
                    self.qk_contract != ("gdn_unit_directional_affine_postrenorm_v1"
                                         if expected_topology == "mimo_rank_contributions"
                                         else "affine_diagonal_plus_additive_identity_init") or
                    not isinstance(self.qk_mode, QKMode) or not self.cache.enabled or
                    self.cache.width != 64 or self.cache.block_size != 256 or
                    self.cache.score != "active_update_frobenius" or
                    self.cache.read != "rmsnorm_rank_aware"):
                raise ValueError("maximum hybrid topology differs from its fail-closed contract")
        if not _REGISTRY_INITIALIZING and self.arm_id in _REGISTRY and self != _REGISTRY[self.arm_id]:
            raise ValueError(f"{self.arm_id} differs from its canonical frozen record")

    @property
    def true_mimo(self) -> bool:
        return self.mimo_rank > 1

    def compatible_with(self, other_arm_id: str) -> bool:
        other = architecture_record(other_arm_id)
        if other_arm_id in self.compatibility.forbidden_arm_ids or self.arm_id in other.compatibility.forbidden_arm_ids:
            return False
        if {self.arm_id, other.arm_id} & {"gdn2-channel-r1"} and (self.mimo_rank > 1 or other.mimo_rank > 1):
            return False
        if (self.cache.enabled and other.mimo_rank > 1) or (other.cache.enabled and self.mimo_rank > 1):
            return False
        if (self.mimo_rank > 1 and other.output_width > 1) or (other.mimo_rank > 1 and self.output_width > 1):
            return False
        if (self.gate_mode == "channelwise" and other.cache.enabled) or (other.gate_mode == "channelwise" and self.cache.enabled):
            return False
        return True


def _record(arm_id: str) -> ArchitectureRecord:
    classifications = {
        "stock": ("source", False), "kmd2-r1": ("converted_control", False),
        "rot-off": ("reliance", False), "rot-constant": ("alternative", True),
        "rot-noncumulative": ("alternative", True), "rot-fixed-rope": ("alternative", True),
        "rot-moving-frame-oracle": ("diagnostic", False), "conv-off": ("reliance", False),
        "trapezoid": ("addition", True), "qk-bc-additive": ("addition", True),
        "qk-diagonal": ("alternative", True), "qk-constant-coordinate-oracle": ("diagnostic", False),
        "lookahead": ("addition", True), "state-64x128": ("cold_redesign", True),
        "state-256x128": ("cold_redesign", True), "rout-4": ("control", False),
        "mimo-r2": ("cold_redesign", True), "mimo-r4": ("cold_redesign", True),
        "gdn2-channel-r1": ("replacement", True), "cache-block-only-w0": ("control", False),
        "cache-surprise-w64": ("addition", True), "cache-recency-w64": ("alternative", True),
        "cache-coupled-w64": ("alternative", True), "cache-residual-w64": ("alternative", True),
        "cache-write-value-w64": ("alternative", True), "cache-reservoir-w64": ("alternative", True),
        "cache-per-slot-read-w64": ("diagnostic", False), "cache-unbounded-oracle": ("diagnostic", False),
        "gdn2-mimo-r4-braid-shared-hola-w64": ("replacement", True),
        "gdn2-mimo-r4-braid-four-state-hola-w64": ("replacement", True),
    }
    ineligible = {"stock", "kmd2-r1", "rot-moving-frame-oracle", "conv-off", "qk-constant-coordinate-oracle", "rout-4", "cache-block-only-w0", "cache-per-slot-read-w64", "cache-unbounded-oracle"}
    classification, promotable = classifications[arm_id]
    kw: dict[str, Any] = {}
    if arm_id.startswith("rot-"): kw["rotation_mode"] = arm_id.removeprefix("rot-")
    if arm_id == "conv-off": kw["convolution_on"] = False
    if arm_id == "trapezoid": kw["state_input_mode"] = "trapezoid"
    if arm_id == "qk-bc-additive": kw["qk_mode"] = "bc_additive"
    if arm_id == "qk-diagonal": kw["qk_mode"] = QKMode("diagonal", 0.0, 0.0, (-0.5, 0.5))
    if arm_id == "qk-constant-coordinate-oracle": kw["qk_mode"] = "constant_coordinate_oracle"
    if arm_id == "lookahead": kw["lookahead"] = True
    if arm_id == "state-64x128": kw.update(state_key_dim=64, state_value_dim=128)
    if arm_id == "state-256x128": kw.update(state_key_dim=256, state_value_dim=128)
    if arm_id == "rout-4": kw["output_width"] = 4
    if arm_id.startswith("mimo-r"): kw["mimo_rank"] = int(arm_id[-1])
    if arm_id == "gdn2-channel-r1": kw["gate_mode"] = "channelwise"
    if arm_id.startswith("mimo-r"): kw["compatibility"] = Compatibility(("gdn2-channel-r1", "rout-4"), ("cache", "shared_query_widening"), ("rankwise_true_mimo",))
    if arm_id == "gdn2-channel-r1": kw["compatibility"] = Compatibility(("mimo-r2", "mimo-r4"), ("cache", "mimo"), ("rank_r1", "channelwise_gates"))
    if arm_id == "rout-4": kw["compatibility"] = Compatibility(("mimo-r2", "mimo-r4"), ("mimo",), ("shared_query_widening",))
    if arm_id.startswith("cache-"):
        semantics = {
            "cache-block-only-w0": ("block_only", "block", 0, 0),
            "cache-surprise-w64": ("||k||_2 * ||beta_w*v - beta_e*m||_2", "unit_l2", 0, 64),
            "cache-recency-w64": ("token_position", "unit_l2", 0, 64),
            "cache-coupled-w64": ("||k||_2 * beta_w * ||v-m||_2", "unit_l2", 0, 64),
            "cache-residual-w64": ("||k||_2 * ||v-m||_2", "unit_l2", 0, 64),
            "cache-write-value-w64": ("||k||_2 * beta_w * ||v||_2", "unit_l2", 0, 64),
            "cache-reservoir-w64": ("SHA256(seed:batch:token:head:position)[0:3]_big_endian / 16777216", "unit_l2", 11, 64),
            "cache-per-slot-read-w64": ("per_slot_exact_outer", "per_slot_unit_l2", 0, 64),
            "cache-unbounded-oracle": ("unbounded_oracle", "unit_l2", 0, None),
        }
        enabled = arm_id != "cache-block-only-w0"
        score, read, selector_seed, width = semantics[arm_id]
        kw["cache"] = CacheArchitecture(enabled=enabled, width=width, score=score, read=read, read_init="gamma_one_sink_zero_amplitude_zero", selector_seed=selector_seed)
        if enabled:
            kw["compatibility"] = Compatibility(("mimo-r2", "mimo-r4", "gdn2-channel-r1"), ("mimo", "channelwise_gdn2"), ("causal_inclusive_read",))
    if arm_id.startswith("gdn2-mimo-r4-braid-"):
        kw.update(
            core_family="gdn2_channelwise", mimo_rank=4, output_width=4,
            gate_mode="channelwise", timescales=((1, 16, 64, 256) if "four-state" in arm_id
                                                  else (64, 512, 4096, 32768)),
            state_topology="mimo_rank_contributions" if "four-state" in arm_id else "shared",
            rotation_mode="mamba3_complex_input_dependent_cumulative",
            phase_topology="per_rank" if "four-state" in arm_id else "shared",
            input_rank=4, output_rank=4,
            read_topology="full_cross_state" if "four-state" in arm_id else "shared_state_outputs",
            read_paths=16 if "four-state" in arm_id else 4,
            **({"transition_paths": 4} if "four-state" in arm_id else {}),
            cache_scope="per_head_shared_across_ranks", update_paths=4,
            state_input_mode="trapezoid", lookahead="four-state" not in arm_id,
            qk_contract=("gdn_unit_directional_affine_postrenorm_v1" if "four-state" in arm_id
                         else "affine_diagonal_plus_additive_identity_init"),
            qk_mode=QKMode("diagonal", 0.0, 0.0, (-0.5, 0.5)),
            cache=CacheArchitecture(enabled=True, width=64, score="active_update_frobenius", read="rmsnorm_rank_aware", read_init="hola_gate_logit_v2_minus4"),
            compatibility=Compatibility(
                (),
                (("convolution_off", "per_state_cache", "cross_rank_state_router", "lookahead")
                 if "four-state" in arm_id else
                 ("convolution_off", "per_state_cache", "cross_rank_state_router", "cms_periodic_updates", "lookahead")),
                (("genuine_gdn2", "rankwise_true_mimo_contributions", "continuous_decay_timescales",
                  "native_gdn_relative_horizon_multipliers",
                  "cms_periodic_updates", "clocked_gdn_updates", "single_per_head_cache",
                  "state_memory_4x_vs_mamba3_aggregate")
                 if "four-state" in arm_id else
                 ("genuine_gdn2", "rankwise_true_mimo_contributions", "continuous_decay_timescales",
                  "single_per_head_cache", "state_memory_4x_vs_mamba3_aggregate")),
            ),
        )
    baseline = "stock" if arm_id in {"stock", "kmd2-r1"} else "kmd2-r1"
    return ArchitectureRecord(arm_id, baseline, classification, promotable, arm_id not in ineligible, **kw)


_REGISTRY = {arm_id: _record(arm_id) for arm_id in STAGE1_IDS}
_REGISTRY_INITIALIZING = False


def architecture_record(arm_id: str) -> ArchitectureRecord:
    try: return _REGISTRY[arm_id]
    except KeyError: raise KeyError(f"unknown architecture arm: {arm_id!r}") from None


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple): return [_jsonable(item) for item in value]
    if isinstance(value, list): return [_jsonable(item) for item in value]
    if isinstance(value, dict): return {key: _jsonable(item) for key, item in value.items()}
    return value


def architecture_document() -> dict[str, Any]:
    records = []
    for item in STAGE1_IDS:
        record = asdict(_REGISTRY[item])
        if record.get("transition_paths") is None:
            record.pop("transition_paths")
        records.append(_jsonable(record))
    return {"schema_version": "1", "registry_version": "qwen08b-stage1-v1", "records": records}


def canonical_registry_json() -> bytes:
    return (json.dumps(architecture_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def registry_sha256() -> str:
    return hashlib.sha256(canonical_registry_json()).hexdigest()


@dataclass(frozen=True)
class LoadedArchitectureDocument:
    schema_version: str
    registry_version: str
    records: tuple[ArchitectureRecord, ...]


def load_architecture_document(value: bytes | str | Mapping[str, Any]) -> LoadedArchitectureDocument:
    serialized: bytes | None = value if isinstance(value, bytes) else (value.encode("utf-8") if isinstance(value, str) else None)
    if isinstance(value, bytes): value = value.decode("utf-8")
    if isinstance(value, str):
        def reject_duplicate(pairs):
            result = {}
            for key, item in pairs:
                if key in result: raise ValueError(f"duplicate key: {key}")
                result[key] = item
            return result
        parsed = json.loads(value, object_pairs_hook=reject_duplicate, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"noncanonical value: {token}")))
    else: parsed = dict(value)
    if parsed != architecture_document(): raise ValueError("architecture document has schema/runtime or semantic drift")
    if serialized is not None and serialized != canonical_registry_json():
        raise ValueError("architecture document is not canonical registry JSON")
    records = []
    for raw in parsed["records"]:
        values = dict(raw)
        values["target_layers"] = tuple(values["target_layers"])
        values["seeds"] = tuple(values["seeds"])
        values["curriculum_lengths"] = tuple(values["curriculum_lengths"])
        values["extrapolation_lengths"] = tuple(values["extrapolation_lengths"])
        values["identity_fields"] = tuple(values["identity_fields"])
        values["timescales"] = tuple(values["timescales"])
        values["cache"] = CacheArchitecture(**values["cache"])
        compatibility = values["compatibility"]
        values["compatibility"] = Compatibility(
            tuple(compatibility["forbidden_arm_ids"]),
            tuple(compatibility["forbidden_families"]),
            tuple(compatibility["requires_semantics"]),
        )
        if isinstance(values["qk_mode"], dict):
            qk = values["qk_mode"]
            values["qk_mode"] = QKMode(qk["family"], qk["q_amplitude"], qk["k_amplitude"], tuple(qk["coordinate_init"]))
        records.append(ArchitectureRecord(**values))
    return LoadedArchitectureDocument(parsed["schema_version"], parsed["registry_version"], tuple(records))


PAIR_IDS = (
    "mimo-r2-x-conv", "mimo-r4-x-conv",
    *tuple(f"mimo-r{rank}-x-state-{size}x128" for rank in (2,4) for size in (64,256)),
    *tuple(f"mimo-r{rank}-x-{qk}" for rank in (2,4) for qk in ("qk-bc-additive","qk-diagonal","qk-constant-coordinate-oracle")),
    "gdn2-channel-r1-x-conv", "gdn2-channel-r1-x-trapezoid", "trapezoid-x-conv", "lookahead-x-conv", "lookahead-x-trapezoid",
    "qk-bc-additive-x-trapezoid", "qk-diagonal-x-trapezoid", "qk-constant-coordinate-oracle-x-trapezoid",
    *tuple(f"cache-{cache}-w64-x-rot-{rot}" for cache in ("surprise","recency","coupled","residual","write-value","reservoir") for rot in ("off","constant","noncumulative","fixed-rope")),
    *tuple(f"cache-{cache}-w64-x-rout-4" for cache in ("surprise","recency","coupled","residual","write-value","reservoir")),
    *tuple(f"cache-{cache}-w64-x-mimo-r{rank}" for cache in ("surprise","recency","coupled","residual","write-value","reservoir") for rank in (2,4)),
)


@dataclass(frozen=True)
class FactorialCell:
    pair_id: str; cell: str; left_arm_id: str; right_arm_id: str
    enabled_arm_ids: tuple[str, ...]; baseline_id: str; status: str
    incompatible_reason: str | None; config_sha256: str

    def __post_init__(self) -> None:
        if type(self.enabled_arm_ids) is not tuple:
            raise TypeError("enabled_arm_ids must be an immutable tuple")
        if self.pair_id not in PAIR_IDS or self.cell not in {"00", "10", "01", "11"}:
            raise ValueError("unknown factorial pair or cell")
        if self.left_arm_id not in STAGE1_IDS or self.right_arm_id not in {*STAGE1_IDS, "conv"}:
            raise ValueError("factorial arms must be registered or conceptual conv")
        if self.baseline_id != "kmd2-r1": raise ValueError("factorial baseline must be kmd2-r1")
        expected_enabled = {"00": (), "10": (self.left_arm_id,), "01": (self.right_arm_id,), "11": (self.left_arm_id, self.right_arm_id)}[self.cell]
        if self.enabled_arm_ids != expected_enabled: raise ValueError("enabled arms do not match factorial cell bits")
        expected_reason = "rankwise_cache_qk_undefined" if self.left_arm_id.startswith("cache-") and self.right_arm_id.startswith("mimo-") else ("oracle_nonpromotable" if "qk-constant-coordinate-oracle" in self.pair_id else None)
        if self.incompatible_reason != expected_reason: raise ValueError("incompatibility reason is inconsistent")
        expected_status = "incompatible" if expected_reason else "preregistered"
        if self.status != expected_status: raise ValueError("factorial status is inconsistent")
        if type(self.config_sha256) is not str or len(self.config_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.config_sha256):
            raise ValueError("config_sha256 must be lowercase 64hex")
        expected_hash = factorial_config_sha256(pair_id=self.pair_id, baseline_id=self.baseline_id, left_arm_id=self.left_arm_id, right_arm_id=self.right_arm_id, cell=self.cell, enabled_arm_ids=self.enabled_arm_ids, incompatible_reason=self.incompatible_reason)
        if self.config_sha256 != expected_hash: raise ValueError("config_sha256 does not match factorial identity")


def factorial_config_sha256(
    *, pair_id: str, baseline_id: str, left_arm_id: str, right_arm_id: str,
    cell: str, enabled_arm_ids: tuple[str, ...], incompatible_reason: str | None,
) -> str:
    """Hash immutable scientific identity; execution status is deliberately absent."""
    def config(arm_id: str) -> Any:
        if arm_id == "conv":
            return {"arm_id": "conv", "convolution_on": True}
        return _jsonable(asdict(architecture_record(arm_id)))
    payload = {
        "pair_id": pair_id, "baseline_id": baseline_id,
        "left_arm_id": left_arm_id, "right_arm_id": right_arm_id,
        "cell": cell, "enabled_arm_ids": enabled_arm_ids,
        "architecture_config": {
            "baseline": config(baseline_id), "left": config(left_arm_id),
            "right": config(right_arm_id),
        },
        "incompatible_reason": incompatible_reason,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _materialize_stage2() -> tuple[FactorialCell, ...]:
    result = []
    for pair_id in PAIR_IDS:
        left, right = pair_id.split("-x-", 1)
        reason = "rankwise_cache_qk_undefined" if left.startswith("cache-") and right.startswith("mimo-") else ("oracle_nonpromotable" if "qk-constant-coordinate-oracle" in pair_id else None)
        for cell, enabled in (("00",()), ("10",(left,)), ("01",(right,)), ("11",(left,right))):
            status = "incompatible" if reason else "preregistered"
            digest = factorial_config_sha256(pair_id=pair_id, baseline_id="kmd2-r1", left_arm_id=left, right_arm_id=right, cell=cell, enabled_arm_ids=enabled, incompatible_reason=reason)
            result.append(FactorialCell(pair_id, cell, left, right, enabled, "kmd2-r1", status, reason, digest))
    return tuple(result)


STAGE2_CELLS = _materialize_stage2()


def expand_stage2() -> tuple[FactorialCell, ...]:
    """Return the preregistered, pre-hashed Stage 2 factorial cells."""
    return STAGE2_CELLS
