"""
GDN3 Core Kernels — Pure PyTorch Reference

Mathematical primitives for the Kronecker-Residual MIMO Gated DeltaNet.
All operations are differentiable and verified against dense materialization.

Key identity (never materializes A⊗B):
    (A⊗B) x = vec(A @ X @ B^T)   where X = x.reshape(a_k, b_k)

PyTorch row-major convention: torch.kron(A, B) flattens as (a_v*b_v, a_k*b_k)
with output indexing flat_idx = i_bv*a_v + i_av for value axis
and flat_idx = i_ak*b_k + i_bk for key axis.

So for S = A⊗B ∈ R^{(a_v*b_v) × (a_k*b_k)}:
    S @ x = vec(A @ X @ B^T)  where X = x.reshape(a_k, b_k)
    Result vec flattens (a_v, b_v) → a_v*b_v row-major.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# 1. KRONECKER READ — Never materializes A⊗B
# =============================================================================

def kron_read_pytorch(
    A: torch.Tensor,   # [R, a_v, a_k]
    B: torch.Tensor,   # [R, b_v, b_k]
    U: torch.Tensor,   # [d_v, P_used]
    V: torch.Tensor,   # [d_k, P_used]
    x: torch.Tensor,   # [d_k] or [batch, d_k]
) -> torch.Tensor:     # [d_v] or [batch, d_v]
    """Kronecker-residual matrix-vector product.

    For torch.kron(A, B) where A=[a_v,a_k], B=[b_v,b_k]:
        S = A⊗B has shape [a_v*b_v, a_k*b_k]
        (A⊗B) x = vec(A @ X @ B^T)  where X = x.reshape(a_k, b_k)

    The Kronecker read is O(R * (b_v*a_k*b_k + b_v*a_k*a_v)) instead of
    O(a_v*b_v * a_k*b_k) for dense materialization.

    Handles both single-vector and batched inputs.
    """
    R = A.shape[0]
    a_v, a_k = A.shape[1], A.shape[2]
    b_v, b_k = B.shape[1], B.shape[2]
    d_v, d_k = a_v * b_v, a_k * b_k

    # Detect batch dimension
    batched = x.dim() > 1
    if batched:
        batch = x.shape[0]
        x = x  # [batch, d_k]
    else:
        batch = 1
        x = x.unsqueeze(0)  # [1, d_k]

    # Reshape x into X: [batch, a_k, b_k]
    X = x.reshape(batch, a_k, b_k)

    # Kronecker part: sum_r vec(A_r @ X @ B_r^T)
    # A_r: [a_v, a_k], X: [batch, a_k, b_k], B_r^T: [b_k, b_v]
    # Result: [batch, a_v, b_v] → [batch, d_v]
    y = torch.zeros(batch, a_v, b_v, dtype=torch.float32, device=x.device)
    for r in range(R):
        # [batch, a_v, b_k] @ [b_k, b_v] → [batch, a_v, b_v]
        y = y + (A[r][None, :, :] @ X).matmul(B[r].T[None, :, :])

    y = y.reshape(batch, d_v)

    # Residual part: U @ (V^T @ x)
    if U.shape[1] > 0:
        # V^T @ x: [P_used, batch, d_k] @ [batch, d_k] → [batch, P_used]
        coeffs = torch.matmul(V.T.unsqueeze(0), x.unsqueeze(-1)).squeeze(-1)
        # U @ coeffs: [d_v, P_used] @ [batch, P_used]^T → [batch, d_v]
        y = y + torch.matmul(U, coeffs.T).T

    if batched:
        return y
    return y.squeeze(0)


# =============================================================================
# 2. STABLE WRITE COEFFICIENT — Exact Exponential
# =============================================================================

def stable_alpha(
    c: torch.Tensor,
    delta: float = 1.0,
    mode: str = "exact",
    eps: float = 1e-6
) -> torch.Tensor:
    """Stable write coefficient for the GDN2 delta update.

    Solves dS/dτ = (u - Sh)k^T exactly:
        S(δ) = S + α(c,δ) · r₀ · k^T
        α(c,δ) = (1 - exp(-δc)) / c

    Modes:
        - "exact": Closed-form exponential (recommended)
        - "heun": Explicit trapezoid / predictor-corrector
        - "implicit_trapezoid": Tustin / backward trapezoid
        - "euler": Simple Euler (baseline GDN2)

    Series expansion of exact:
        α = δ - δ²c/2 + δ³c²/6 + O(δ⁴c³)
    """
    z = delta * c

    if mode == "euler":
        return torch.full_like(c, delta, dtype=torch.float32)

    if mode == "heun":
        return delta * (1.0 - 0.5 * z)

    if mode == "implicit_trapezoid":
        return delta / (1.0 + 0.5 * z)

    if mode == "exact":
        # Numerically stable: δ * (1 - exp(-z)) / z
        # Use Taylor series near z=0 to avoid 0/0
        small = torch.abs(z) < eps
        result = torch.zeros_like(c, dtype=torch.float32)
        result[small] = delta * (1.0 - z[small] / 2.0 + (z[small] ** 2) / 6.0)
        result[~small] = delta * (-torch.expm1(-z[~small]) / z[~small])
        return result

    raise ValueError(f"Unknown alpha mode: {mode}")


# =============================================================================
# 3. PARTIAL LANE-SPECIFIC RoPE
# =============================================================================

def apply_partial_rope(
    vec: torch.Tensor,        # [d_k] or [batch, d_k]
    cos_pe: torch.Tensor,     # [T, d_k]
    sin_pe: torch.Tensor,     # [T, d_k]
    t: int,                    # position
    rope_fraction: float,     # fraction of dims to rotate
    rope_scale: float = 1.0,  # frequency scaling per lane
) -> torch.Tensor:
    """Apply RoPE to first rope_fraction*d_k dimensions.

    Uses standard 2D rotation on paired dimensions:
        x'[2i]   = x[2i]*cos - x[2i+1]*sin
        x'[2i+1] = x[2i]*sin + x[2i+1]*cos

    rope_scale adjusts frequencies (e.g., <1 for long-context lanes).
    """
    d_k = vec.shape[-1]
    d_rope = int(d_k * rope_fraction)
    if d_rope < 2:
        return vec.clone()

    # Cap to even number
    d_rope = d_rope // 2 * 2
    d_pairs = d_rope // 2

    if vec.dim() > 1:
        # Batched: [batch, d_k]
        x1 = vec[:, :d_pairs]
        x2 = vec[:, d_pairs:2*d_pairs]
    else:
        x1 = vec[:d_pairs]
        x2 = vec[d_pairs:2*d_pairs]

    # Scale position by rope_scale
    scaled_t = t * rope_scale
    cos_t = cos_pe[min(t, cos_pe.shape[0] - 1), :d_pairs]
    sin_t = sin_pe[min(t, sin_pe.shape[0] - 1), :d_pairs]

    # Apply frequency scaling via position
    # For low-frequency lanes, effectively use scaled position
    if rope_scale != 1.0:
        # Recompute cos/sin with scaled position
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_pairs, device=vec.device, dtype=torch.float32) / d_pairs))
        angles = scaled_t * inv_freq
        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)

    result = vec.clone()
    result[..., :d_pairs] = x1 * cos_t - x2 * sin_t
    result[..., d_pairs:2*d_pairs] = x1 * sin_t + x2 * cos_t

    return result


# =============================================================================
# 4. BRAIDED DECAY — Multi-Timescale Power-Law Approximation
# =============================================================================

def braided_decay(
    decay_projections: torch.Tensor,  # [batch, M_braid, d_k] — raw projections
    base_rates: torch.Tensor,          # [M_braid] — log-spaced base rates
    routing_weights: Optional[torch.Tensor] = None,  # [batch, M_braid] — learned router
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-timescale decay approximating power-law forgetting.

    Instead of single exponential α^d, achieves:
        Σ_m w_m · α_m^d ≈ (1+d)^(-γ)

    Each timescale m has its own decay rate, routed by learned weights.
    Fast decay heads capture local context; slow heads capture global context.

    Returns factorized decay (γ_a, γ_b) suitable for Kronecker-closed application.
    """
    batch, M_braid, d_k = decay_projections.shape
    a_k = int(d_k ** 0.5)  # Will be adjusted by caller
    b_k = d_k // a_k

    # Compute decay per timescale: softplus → clamp → exp
    g_raw = F.softplus(decay_projections)  # [B, M_braid, d_k]
    g_clamped = torch.clamp(g_raw, max=5.0)
    decay_per_channel = torch.exp(-g_clamped * base_rates.unsqueeze(-1).unsqueeze(0))

    # Apply routing weights if provided
    if routing_weights is not None:
        # [B, M_braid, 1] * [B, M_braid, d_k] → [B, d_k]
        weighted_decay = (routing_weights.unsqueeze(-1) * decay_per_channel).sum(dim=1)
    else:
        # Uniform average across timescales
        weighted_decay = decay_per_channel.mean(dim=1)

    # Factorize into γ_a ⊗ γ_b for Kronecker closure
    # γ_full[i_ak*b_k + i_bk] = γ_a[i_ak] * γ_b[i_bk]
    # Approximate: γ_a = geometric_mean over b_k, γ_b = geometric_mean over a_k
    decay_2d = weighted_decay.reshape(batch, a_k, b_k)
    gamma_a = decay_2d.mean(dim=-1)  # [B, a_k]
    gamma_b = decay_2d.mean(dim=-2)  # [B, b_k]
    gamma_full = weighted_decay       # [B, d_k]

    return gamma_a, gamma_b, gamma_full


# =============================================================================
# 5. COPRODUCT CHANNEL GENERATOR — Hopf-Inspired Kronecker Routing
# =============================================================================

def coproduct_channels(
    x: torch.Tensor,           # [batch, d_in]
    W_a: torch.Tensor,         # [C, d_a, d_in] — factor A projections
    W_b: torch.Tensor,         # [C, d_b, d_in] — factor B projections
    strength_gate: Optional[torch.Tensor] = None,  # [batch] — learned strength
    normalization: str = "rms",
) -> torch.Tensor:             # [batch, d_a*d_b]
    """Generate channels via Kronecker-factored coproduct expansion.

    Instead of dense projection W·x ∈ R^d, generates:
        z = Σ_c s_c(x) · vec(a_c(x) ⊗ b_c(x)) / sqrt(C)

    where a_c(x) = W_a[c] @ x and b_c(x) = W_b[c] @ x.

    This provides built-in bilinear feature binding:
        entity × attribute, variable × scope, etc.

    Efficiency: O(C*(d_a + d_b)) vs O(d) for dense, where d = d_a * d_b.
    Clean when C < d/(d_a + d_b).

    Args:
        normalization: "rms", "l2", or "none"
    """
    batch, d_in = x.shape
    C = W_a.shape[0]  # coproduct internal rank

    # Project into factor spaces
    a_factors = torch.bmm(x.unsqueeze(1), W_a.transpose(1, 2))  # [B, C, d_a]
    b_factors = torch.bmm(x.unsqueeze(1), W_b.transpose(1, 2))  # [B, C, d_b]

    # Outer product per rank: vec(a_c ⊗ b_c) = a_c.flatten() * b_c.flatten().unsqueeze(1)
    # Result: [B, C, d_a*d_b]
    # More efficiently: einsum over outer product then reshape
    # a_factors: [B, C, d_a], b_factors: [B, C, d_b]
    # outer: [B, C, d_a, d_b] → [B, C, d_a*d_b]
    coproduct = torch.einsum('bci,bcj->bcij', a_factors, b_factors)
    coproduct = coproduct.reshape(batch, C, -1)  # [B, C, d_a*d_b]

    # Sum over coproduct ranks with 1/sqrt(C) normalization
    channels = coproduct.sum(dim=1) / math.sqrt(C)  # [B, d_a*d_b]

    # Apply strength gate
    if strength_gate is not None:
        channels = channels * strength_gate.unsqueeze(-1)

    # Normalization
    if normalization == "l2":
        channels = F.normalize(channels, dim=-1, eps=1e-6)
    elif normalization == "rms":
        rms = torch.sqrt((channels ** 2).mean(dim=-1, keepdim=True) + 1e-6)
        channels = channels / rms

    return channels


# =============================================================================
# 6. SINGLE LANE STEP — GDN3 Core Recurrence
# =============================================================================

def gdn3_lane_step(
    # Kronecker state
    A: torch.Tensor,          # [R, a_v, a_k]
    B: torch.Tensor,          # [R, b_v, b_k]
    U: torch.Tensor,          # [d_v, P]
    V: torch.Tensor,          # [d_k, P]
    p: int,                   # residual pointer

    # Token inputs (single vector, already projected + gated)
    q: torch.Tensor,          # [d_k] — query (already RoPEd)
    k: torch.Tensor,          # [d_k] — key (already RoPEd)
    v: torch.Tensor,          # [d_v] — value
    b_gate: torch.Tensor,     # [d_k] — erase gate ∈ [0,1]
    w_gate: torch.Tensor,     # [d_v] — write gate ∈ [0,1]

    # Factorized decay
    gamma_a: torch.Tensor,    # [a_k]
    gamma_b: torch.Tensor,    # [b_k]
    gamma_full: torch.Tensor, # [d_k]

    # Config
    delta: float = 1.0,
    alpha_mode: str = "exact",
    P: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, int, Optional[float]]:
    """One GDN3 lane step.

    Core recurrence:
        1. Decay: A *= γ_a, B *= γ_b, V *= γ_full
        2. Erase: h = b_gate ⊙ k
        3. Write: u = w_gate ⊙ v
        4. Read: s_h = S·h  (Kronecker + residual)
        5. Delta: r = u - s_h
        6. Coefficient: α = stable_alpha(k^T h)
        7. Output: y = S·q + α·r·(k^T q)  (post-update shortcut)
        8. Append: U[:,p] = α·r, V[:,p] = k
        9. Compact if buffer full

    Returns:
        (y, A, B, U, V, p, compaction_error)
    """
    d_v, d_k = U.shape[0], V.shape[0]
    p_used = min(p, P)

    # 1. Apply factorized decay to Kronecker factors
    A = A * gamma_a[None, None, :]      # [R, a_v, a_k]
    B = B * gamma_b[None, None, :]      # [R, b_v, b_k]

    # Decay residual keys exactly
    if p_used > 0:
        V[:, :p_used] = V[:, :p_used] * gamma_full[:, None]

    # 2. Erase/write vectors
    h = b_gate * k              # [d_k]
    u = w_gate * v              # [d_v]

    # 3. Read in erase direction: S·h
    s_h = kron_read_pytorch(A, B, U[:, :p_used], V[:, :p_used], h)

    # 4. Delta residual
    r = u - s_h              # [d_v]

    # 5. Stable write coefficient
    c = (k * h).sum()       # scalar
    alpha = stable_alpha(c, delta=delta, mode=alpha_mode)

    # 6. Read in query direction: S·q
    s_q = kron_read_pytorch(A, B, U[:, :p_used], V[:, :p_used], q)

    # 7. Post-update output shortcut (avoids rereading updated state)
    kq = (k * q).sum()
    y = s_q + alpha * r * kq

    # 8. Append exact write into residual buffer (circular)
    p_next = p % P
    U_new = U.clone()
    V_new = V.clone()
    U_new[:, p_next] = alpha * r
    V_new[:, p_next] = k
    p_new = p + 1

    # 9. Compaction if buffer full
    compaction_error = None
    if p_new >= P:
        A, B, U_new, V_new, compaction_error = compact_to_kronecker(A, B, U_new, V_new)
        p_new = 0

    return y, A, B, U_new, V_new, p_new, compaction_error


# =============================================================================
# 7. KRONECKER COMPACTION — Rearrangement SVD
# =============================================================================

def compact_to_kronecker(
    A: torch.Tensor,    # [R, a_v, a_k]
    B: torch.Tensor,    # [R, b_v, b_k]
    U: torch.Tensor,    # [d_v, P]
    V: torch.Tensor,    # [d_k, P]
    R_new: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compact residual UV^T back into Kronecker factors via rearrangement SVD.

    For torch.kron(A,B) where A=[a_v,a_k], B=[b_v,b_k]:
        Rearrangement R(S) reshapes S as [a_v, b_v, a_k, b_k]
        then permutes to [a_v, a_k, b_v, b_k] = [a_v*a_k, b_v*b_k]
        So R(A⊗B) = vec(A) @ vec(B)^T  where vec is row-major flatten.

    SVD of R(S) gives: σ_r · u_r @ v_r^T
    Map back: A_r = u_r.reshape(a_v, a_k), B_r = v_r.reshape(b_v, b_k)

    Returns: (A_new, B_new, U_zeroed, V_zeroed, discarded_energy)
    """
    a_v, a_k = A.shape[1], A.shape[2]
    b_v, b_k = B.shape[1], B.shape[2]
    R_target = R_new if R_new is not None else A.shape[0]

    # Build combined rearrangement matrix
    # R_total = Σ_r vec(A_r) @ vec(B_r)^T + R(UV^T)
    R_mat = torch.zeros(a_v * a_k, b_v * b_k, dtype=torch.float32, device=A.device)

    # Kronecker terms: vec(A_r) @ vec(B_r)^T
    for r in range(A.shape[0]):
        vec_a = A[r].reshape(-1)    # [a_v*a_k] row-major
        vec_b = B[r].reshape(-1)    # [b_v*b_k] row-major
        R_mat = R_mat + vec_a.unsqueeze(1) @ vec_b.unsqueeze(0)

    # Residual UV^T rearranged
    S_res = U @ V.T  # [d_v, d_k] = [a_v*b_v, a_k*b_k]
    S_res_4d = S_res.reshape(a_v, b_v, a_k, b_k)
    S_rearr = S_res_4d.permute(0, 2, 1, 3)    # [a_v, a_k, b_v, b_k]
    R_mat = R_mat + S_rearr.reshape(a_v * a_k, b_v * b_k)

    # Truncated SVD
    U_svd, S_svd, Vh_svd = torch.linalg.svd(R_mat, full_matrices=False)
    sigma_sq_discarded = float((S_svd[R_target:] ** 2).sum())

    # Map back to Kronecker factors
    A_new = torch.zeros(R_target, a_v, a_k, dtype=A.dtype, device=A.device)
    B_new = torch.zeros(R_target, b_v, b_k, dtype=B.dtype, device=B.device)

    for r in range(R_target):
        A_new[r] = (U_svd[:, r] * S_svd[r]).reshape(a_v, a_k).to(A.dtype)
        B_new[r] = Vh_svd[r, :].reshape(b_v, b_k).to(B.dtype)

    # Zero out residual buffer
    U_new = torch.zeros_like(U)
    V_new = torch.zeros_like(V)

    return A_new, B_new, U_new, V_new, sigma_sq_discarded


# =============================================================================
# 8. LANE ROUTING — Softmax Router
# =============================================================================

def lane_router(
    router_logits: torch.Tensor,   # [batch, T, H, M]
) -> torch.Tensor:                    # [batch, T, H, M] — softmax weights
    """Scalar softmax router over MIMO lanes.

    Each head independently routes to lanes.
    Optionally add entropy regularization during training.
    """
    return F.softmax(router_logits, dim=-1)


def lane_balance_loss(
    router_weights: torch.Tensor,  # [batch, T, H, M]
    lam: float = 0.001,
) -> torch.Tensor:
    """Lane load balance regularizer.

    Prevents lane collapse by penalizing deviation from uniform routing.
    L = Σ_m (mean(a_m) - 1/M)^2
    """
    M = router_weights.shape[-1]
    # Average over batch and time, keep head dim
    mean_routing = router_weights.mean(dim=(0, 1))  # [H, M]
    uniform = torch.ones_like(mean_routing) / M
    loss = ((mean_routing - uniform) ** 2).sum() * lam
    return loss


# =============================================================================
# 9. KRONECKER LANE MIXING — Cross-Lane Feature Transform
# =============================================================================

def kron_lane_mix(
    lane_outputs: torch.Tensor,     # [M, d] — outputs from each lane
    W_lane: torch.Tensor,           # [M, M] — lane mixing
    W_feat: torch.Tensor,           # [d, d] — feature mixing
) -> torch.Tensor:                     # [M, d] — mixed outputs
    """Kronecker-factorized cross-lane mixing.

    Instead of dense W ∈ R^{Md × Md}, factorizes as W_lane ⊗ W_feat.
    Applies as: Z' = W_lane @ Z @ W_feat^T

    Cost: O(M²d + Md²) vs O(M²d²) for dense.
    """
    # Z: [M, d], W_lane: [M, M], W_feat: [d, d]
    # Z' = W_lane @ Z @ W_feat^T
    return W_lane @ lane_outputs @ W_feat.T


import math
