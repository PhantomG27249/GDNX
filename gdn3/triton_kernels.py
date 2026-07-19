from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional acceleration path
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _build_compaction_rmat_kernel(
        A, Bk, U, Vb, Out,
        stride_an: tl.constexpr, stride_ar: tl.constexpr, stride_aav: tl.constexpr, stride_aak: tl.constexpr,
        stride_bn: tl.constexpr, stride_br: tl.constexpr, stride_bbv: tl.constexpr, stride_bbk: tl.constexpr,
        stride_un: tl.constexpr, stride_uv: tl.constexpr, stride_up: tl.constexpr,
        stride_vn: tl.constexpr, stride_vk: tl.constexpr, stride_vp: tl.constexpr,
        stride_on: tl.constexpr, stride_om: tl.constexpr, stride_ok: tl.constexpr,
        R: tl.constexpr, P: tl.constexpr,
        A_V: tl.constexpr, A_K: tl.constexpr, B_V: tl.constexpr, B_K: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        n_id = tl.program_id(0)
        m_block = tl.program_id(1)

        rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (rows < A_V * A_K)[:, None] & (cols < B_V * B_K)[None, :]

        av = rows // A_K
        ak = rows - av * A_K
        bv = cols // B_K
        bk = cols - bv * B_K
        v_idx = av[:, None] * B_V + bv[None, :]
        k_idx = ak[:, None] * B_K + bk[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for r in range(0, R):
            a = tl.load(
                A + n_id * stride_an + r * stride_ar + av * stride_aav + ak * stride_aak,
                mask=rows < A_V * A_K,
                other=0.0,
            )
            b = tl.load(
                Bk + n_id * stride_bn + r * stride_br + bv * stride_bbv + bk * stride_bbk,
                mask=cols < B_V * B_K,
                other=0.0,
            )
            acc += a[:, None] * b[None, :]

        for p in range(0, P):
            u = tl.load(
                U + n_id * stride_un + v_idx * stride_uv + p * stride_up,
                mask=mask,
                other=0.0,
            )
            v = tl.load(
                Vb + n_id * stride_vn + k_idx * stride_vk + p * stride_vp,
                mask=mask,
                other=0.0,
            )
            acc += u * v

        tl.store(Out + n_id * stride_on + rows[:, None] * stride_om + cols[None, :] * stride_ok, acc, mask=mask)


def build_compaction_rmat_triton(
    A: torch.Tensor,
    Bk: torch.Tensor,
    U: torch.Tensor,
    Vb: torch.Tensor,
    *,
    a_v: int,
    a_k: int,
    b_v: int,
    b_k: int,
) -> torch.Tensor:
    """Build rearranged compaction matrix [N, a_v*a_k, b_v*b_k].

    Formula matches the PyTorch compaction path:
      sum_r vec(A_r) vec(B_r)^T + rearrange(U @ Vb^T).
    """
    if triton is None or not A.is_cuda:
        raise RuntimeError("Triton compaction kernel is unavailable")

    N, R = A.shape[:2]
    P = U.shape[-1]
    out = torch.empty((N, a_v * a_k, b_v * b_k), dtype=torch.float32, device=A.device)
    grid = (N, triton.cdiv(a_v * a_k, 16))
    with torch.cuda.device(A.device):
      _build_compaction_rmat_kernel[grid](
        A, Bk, U, Vb, out,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        Bk.stride(0), Bk.stride(1), Bk.stride(2), Bk.stride(3),
        U.stride(0), U.stride(1), U.stride(2),
        Vb.stride(0), Vb.stride(1), Vb.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        R, P, a_v, a_k, b_v, b_k,
        BLOCK_M=16, BLOCK_N=64,
    )
    return out


if triton is not None:
    @triton.jit
    def _kron_read_chunk_kernel(
        A, Bk, X, Out,
        stride_an: tl.constexpr, stride_ar: tl.constexpr, stride_aav: tl.constexpr, stride_aak: tl.constexpr,
        stride_bn: tl.constexpr, stride_br: tl.constexpr, stride_bbv: tl.constexpr, stride_bbk: tl.constexpr,
        stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xk: tl.constexpr,
        stride_on: tl.constexpr, stride_oc: tl.constexpr, stride_ov: tl.constexpr,
        R: tl.constexpr, C: tl.constexpr,
        A_V: tl.constexpr, A_K: tl.constexpr, B_V: tl.constexpr, B_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        n_id = tl.program_id(0)
        c_id = tl.program_id(1)
        offs_v = tl.arange(0, BLOCK_V)
        mask_v = offs_v < A_V * B_V
        av = offs_v // B_V
        bv = offs_v - av * B_V

        acc = tl.zeros((BLOCK_V,), tl.float32)
        for r in range(0, R):
            for ak in range(0, A_K):
                a = tl.load(
                    A + n_id * stride_an + r * stride_ar + av * stride_aav + ak * stride_aak,
                    mask=mask_v,
                    other=0.0,
                )
                for bk in range(0, B_K):
                    x = tl.load(X + n_id * stride_xn + c_id * stride_xc + (ak * B_K + bk) * stride_xk)
                    b = tl.load(
                        Bk + n_id * stride_bn + r * stride_br + bv * stride_bbv + bk * stride_bbk,
                        mask=mask_v,
                        other=0.0,
                    )
                    acc += a * x * b

        tl.store(Out + n_id * stride_on + c_id * stride_oc + offs_v * stride_ov, acc, mask=mask_v)


def kron_read_chunk_triton(
    A: torch.Tensor,
    Bk: torch.Tensor,
    x_chunk: torch.Tensor,
    *,
    a_v: int,
    a_k: int,
    b_v: int,
    b_k: int,
) -> torch.Tensor:
    """Read carried Kronecker state for x_chunk [N, C, K] -> [N, C, V]."""
    if triton is None or not A.is_cuda:
        raise RuntimeError("Triton Kronecker-read kernel is unavailable")
    N, R = A.shape[:2]
    C = x_chunk.shape[1]
    out = torch.empty((N, C, a_v * b_v), dtype=torch.float32, device=A.device)
    grid = (N, C)
    # Triton launches on the *current* CUDA device; guard so multi-GPU placement
    # (e.g. student on cuda:1) targets the tensors' device, not default cuda:0.
    with torch.cuda.device(A.device):
        _kron_read_chunk_kernel[grid](
            A, Bk, x_chunk, out,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            Bk.stride(0), Bk.stride(1), Bk.stride(2), Bk.stride(3),
            x_chunk.stride(0), x_chunk.stride(1), x_chunk.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            R, C, a_v, a_k, b_v, b_k,
            BLOCK_V=128,
        )
    return out


def _kron_read_chunk_torch(A: torch.Tensor, Bk: torch.Tensor, x_chunk: torch.Tensor,
                           a_v: int, a_k: int, b_v: int, b_k: int) -> torch.Tensor:
    N, C = x_chunk.shape[:2]
    X = x_chunk.reshape(N, C, a_k, b_k)
    AX = torch.einsum('nrau,ncub->ncrab', A, X)
    AXB = torch.einsum('ncrab,nrdb->ncrad', AX, Bk)
    return AXB.sum(2).reshape(N, C, a_v * b_v)


class _KronReadChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, Bk, x_chunk, a_v: int, a_k: int, b_v: int, b_k: int):
        ctx.save_for_backward(A, Bk)
        ctx.dims = (a_v, a_k, b_v, b_k)
        if triton is not None and A.is_cuda:
            return kron_read_chunk_triton(A, Bk, x_chunk, a_v=a_v, a_k=a_k, b_v=b_v, b_k=b_k)
        return _kron_read_chunk_torch(A, Bk, x_chunk, a_v, a_k, b_v, b_k)

    @staticmethod
    def backward(ctx, grad_out):
        A, Bk = ctx.saved_tensors
        a_v, a_k, b_v, b_k = ctx.dims
        N, C = grad_out.shape[:2]
        grad_v = grad_out.reshape(N, C, a_v, b_v)
        grad_x = torch.einsum('ncvw,nrvu,nrwb->ncub', grad_v, A, Bk).reshape(N, C, a_k * b_k)
        return None, None, grad_x, None, None, None, None


def kron_read_chunk_autograd(
    A: torch.Tensor,
    Bk: torch.Tensor,
    x_chunk: torch.Tensor,
    *,
    a_v: int,
    a_k: int,
    b_v: int,
    b_k: int,
) -> torch.Tensor:
    """Autograd wrapper for the Triton carried-state Kronecker read.

    The carried Kronecker state is produced by zero initialization or no-grad
    compaction, so training only needs gradients with respect to x_chunk.
    """
    return _KronReadChunk.apply(A, Bk, x_chunk, a_v, a_k, b_v, b_k)
