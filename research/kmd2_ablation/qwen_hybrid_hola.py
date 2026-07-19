"""Rank-aware, block-promoted HOLA cache for the dual Qwen hybrids."""
from __future__ import annotations

from dataclasses import dataclass, replace
import torch
from torch import Tensor, nn


def shared_exact_update_score(post_trap_update: Tensor) -> Tensor:
    """Norm of the additive innovation actually committed to Package A's state.

    For paper GDN this matrix is beta * k * e^T and, because ||k||=1, this
    reduces exactly to beta*||e||.  Independent gates and trapezoidal history
    can make the extended update higher-rank, so its Frobenius norm is the
    faithful basis-independent generalization rather than either gate alone.
    """
    return torch.linalg.vector_norm(post_trap_update.float(), dim=(-2, -1)).detach()


def four_state_exact_update_score(lane_updates: Tensor) -> Tensor:
    """Committed-innovation norm in Package B's four-state direct-sum space."""
    return torch.sqrt(lane_updates.float().square().sum((-3, -2, -1))).detach()


def four_state_normalized_update_score(lane_updates: Tensor, tick_lanes: Tensor) -> Tensor:
    """Tick-cadence-normalized committed-innovation score (RMS over ticking lanes).

    The raw direct-sum Frobenius norm grows with how many CMS lanes tick at a
    token, so 16/64/256-aligned positions receive inflated admission scores for
    clock reasons rather than content surprise.  Off-tick lanes commit exactly
    zero, so dividing the squared norm by the ticking-lane count yields the
    per-lane RMS innovation and removes the periodic bias.
    """
    if lane_updates.ndim < 4:
        raise ValueError("lane updates must be [B,H,R,K,V]")
    if tick_lanes.dtype != torch.bool or tick_lanes.shape != (lane_updates.shape[0], lane_updates.shape[2]):
        raise ValueError("tick_lanes must be bool [B,R]")
    squared = lane_updates.float().square().sum((-2, -1)).sum(-1)
    count = tick_lanes.sum(-1).to(squared.dtype).clamp_min(1.0)
    return torch.sqrt(squared / count[:, None]).detach()


@dataclass(frozen=True)
class HybridHOLAState:
    keys: Tensor
    values: Tensor
    scores: Tensor
    positions: Tensor
    valid: Tensor
    epochs: Tensor
    block_keys: Tensor
    block_values: Tensor
    block_scores: Tensor
    block_positions: Tensor
    block_valid: Tensor
    block_epochs: Tensor
    block_count: Tensor
    next_position: Tensor
    current_epoch: Tensor
    admission_count: Tensor | None = None
    age_sum: Tensor | None = None
    age_count: Tensor | None = None

    @property
    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in self.__dict__.values() if isinstance(x, Tensor))


class HybridHOLACache(nn.Module):
    """Per-head W-cache with causal C-block staging and end-of-block promotion."""
    implementation_reference = "qwen_hybrid_hola.HybridHOLACache.pytorch_reference_host_sync"
    runtime_warning = "reference promotion control performs one tensor-to-host scalar synchronization per admission"

    def __init__(self, *, width: int = 64, block_size: int = 256, heads: int,
                 rank_in: int = 4, key_dim: int, value_dim: int,
                 storage_dtype: torch.dtype = torch.bfloat16,
                 policy: str = "exact_outer") -> None:
        super().__init__()
        if type(width) is not int or width < 0 or type(block_size) is not int or not 1 <= block_size <= 256:
            raise ValueError("width must be nonnegative and block_size in [1,256]")
        if min(heads, rank_in, key_dim, value_dim) < 1:
            raise ValueError("cache dimensions must be positive")
        if storage_dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("storage_dtype must be fp32 or bf16")
        if policy not in ("exact_outer", "recency"):
            raise ValueError("policy must be exact_outer or recency")
        self.width, self.block_size, self.heads = width, block_size, heads
        self.rank_in, self.key_dim, self.value_dim = rank_in, key_dim, value_dim
        self.storage_dtype, self.policy = storage_dtype, policy
        self.gamma_q = nn.Parameter(torch.ones(heads, rank_in, key_dim))
        self.gamma_k = nn.Parameter(torch.ones(heads, rank_in, key_dim))
        self.sink_logit = nn.Parameter(torch.zeros(heads, rank_in))

    def resource_report(self, *, batch_size: int = 1) -> dict[str, object]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        B,H,W,C,R,K,V = batch_size,self.heads,self.width,self.block_size,self.rank_in,self.key_dim,self.value_dim
        e = torch.empty((), dtype=self.storage_dtype).element_size()
        persistent = {"keys":B*H*W*R*K*e,"values":B*H*W*R*V*e,"scores":B*H*W*4,
                      "positions":B*H*W*8,"valid":B*H*W,"epochs":B*H*W*8,
                      "next_position":B*8,"current_epoch":B*H*8,
                      "admission_count":B*H*8,"age_sum":B*H*8,"age_count":B*H*8}
        workspace = {"block_keys":B*H*C*R*K*e,"block_values":B*H*C*R*V*e,
                     "block_scores":B*H*C*4,"block_positions":B*H*C*8,
                     "block_valid":B*H*C,"block_epochs":B*H*C*8,"block_count":B*H*8}
        return {"persistent":persistent,"persistent_bytes":sum(persistent.values()),
                "workspace":workspace,"workspace_bytes":sum(workspace.values()),
                "allocated_bytes":sum(persistent.values())+sum(workspace.values()),
                "promotion_control":"pytorch_reference_host_sync","runtime_warning":self.runtime_warning}

    def _empty(self, B: int, device: torch.device) -> HybridHOLAState:
        H,W,C,R,K,V = self.heads,self.width,self.block_size,self.rank_in,self.key_dim,self.value_dim
        stored=lambda *s: torch.zeros(*s, device=device, dtype=self.storage_dtype)
        return HybridHOLAState(stored(B,H,W,R,K),stored(B,H,W,R,V),torch.zeros(B,H,W,device=device),
            torch.full((B,H,W),-1,device=device,dtype=torch.int64),torch.zeros(B,H,W,device=device,dtype=torch.bool),
            torch.full((B,H,W),-1,device=device,dtype=torch.int64),
            stored(B,H,C,R,K),stored(B,H,C,R,V),torch.zeros(B,H,C,device=device),
            torch.full((B,H,C),-1,device=device,dtype=torch.int64),torch.zeros(B,H,C,device=device,dtype=torch.bool),
            torch.full((B,H,C),-1,device=device,dtype=torch.int64),torch.zeros(B,H,device=device,dtype=torch.int64),
            torch.zeros(B,device=device,dtype=torch.int64),torch.zeros(B,H,device=device,dtype=torch.int64),
            torch.zeros(B,H,device=device,dtype=torch.int64),
            torch.zeros(B,H,device=device,dtype=torch.float64),
            torch.zeros(B,H,device=device,dtype=torch.int64))

    @staticmethod
    def _require(condition: Tensor, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def _validate_state(self, state: HybridHOLAState, B: int, device: torch.device) -> None:
        if type(state) is not HybridHOLAState:
            raise TypeError("initial_state must be HybridHOLAState")
        H,W,C,R,K,V=self.heads,self.width,self.block_size,self.rank_in,self.key_dim,self.value_dim
        if state.admission_count is None:
            object.__setattr__(state, "admission_count", torch.zeros(B,H,device=device,dtype=torch.int64))
            object.__setattr__(state, "age_sum", torch.zeros(B,H,device=device,dtype=torch.float64))
            object.__setattr__(state, "age_count", torch.zeros(B,H,device=device,dtype=torch.int64))
        expected={"keys":((B,H,W,R,K),self.storage_dtype),"values":((B,H,W,R,V),self.storage_dtype),
            "scores":((B,H,W),torch.float32),"positions":((B,H,W),torch.int64),"valid":((B,H,W),torch.bool),
            "epochs":((B,H,W),torch.int64),
            "block_keys":((B,H,C,R,K),self.storage_dtype),"block_values":((B,H,C,R,V),self.storage_dtype),
            "block_scores":((B,H,C),torch.float32),"block_positions":((B,H,C),torch.int64),
            "block_valid":((B,H,C),torch.bool),"block_count":((B,H),torch.int64),"next_position":((B,),torch.int64)}
        expected.update({"block_epochs":((B,H,C),torch.int64),"current_epoch":((B,H),torch.int64)})
        expected.update({"admission_count":((B,H),torch.int64), "age_sum":((B,H),torch.float64),
                         "age_count":((B,H),torch.int64)})
        for name,(shape,dtype) in expected.items():
            value=getattr(state,name)
            if not isinstance(value,Tensor): raise TypeError(f"initial_state {name} must be a tensor")
            if tuple(value.shape)!=shape: raise ValueError(f"initial_state {name} shape mismatch")
            if value.dtype!=dtype: raise TypeError(f"initial_state {name} dtype mismatch")
            if value.device!=device: raise ValueError(f"initial_state {name} device mismatch")
        for name in ("keys","values","scores","block_keys","block_values","block_scores"):
            self._require(torch.isfinite(getattr(state,name)).all(),f"initial_state {name} must be finite")
        self._require(((state.block_count>=0)&(state.block_count<=C)).all(),
                      "initial_state block_count must be in range 0..C")
        self._require((state.epochs<=state.current_epoch[...,None]).all(),
                      "initial_state future persistent epoch is forbidden")
        self._require((state.block_epochs<=state.current_epoch[...,None]).all(),
                      "initial_state future block epoch is forbidden")
        self._require((state.epochs>=-1).all(),"initial_state persistent epochs must be -1 or nonnegative")
        self._require((state.block_epochs>=-1).all(),"initial_state block epochs must be -1 or nonnegative")
        effective_block=state.block_valid & (state.block_epochs==state.current_epoch[...,None])
        expected_block=torch.arange(C,device=device)[None,None,:] < state.block_count[...,None]
        self._require(torch.equal(effective_block,expected_block),
                      "initial_state block_valid must agree with block_count")
        effective_valid=state.valid & (state.epochs==state.current_epoch[...,None])
        self._require(((state.positions>=0)|~effective_valid).all(),
                      "initial_state positions must be nonnegative exactly at valid slots")
        self._require(((state.block_positions>=0)|~effective_block).all(),
                      "initial_state block_positions must be nonnegative exactly at valid slots")
        self._require((state.next_position>=0).all(),"initial_state next_position must be nonnegative")
        self._require((state.current_epoch>=0).all(),"initial_state current_epoch must be nonnegative")
        self._require((state.admission_count>=0).all(),"initial_state admission_count must be nonnegative")
        self._require((state.age_count>=0).all(),"initial_state age_count must be nonnegative")
        self._require(torch.isfinite(state.age_sum).all() & (state.age_sum>=0).all(),
                      "initial_state age_sum must be finite and nonnegative")

    def _validate_inputs(self, query: Tensor, keys: Tensor, values: Tensor, scores: Tensor,
                         positions: Tensor|None, valid: Tensor, boundary: Tensor) -> None:
        if not all(isinstance(x,Tensor) for x in (query,keys,values,scores,valid,boundary)):
            raise TypeError("HOLA inputs must be tensors")
        B,T=query.shape[:2] if query.ndim>=2 else (-1,-1)
        if query.ndim!=5 or query.shape[:3]!=(B,T,self.heads) or query.shape[3]<1 or query.shape[4]!=self.key_dim:
            raise ValueError("query shape must be [B,T,H,Rout,K]")
        if keys.shape!=(B,T,self.heads,self.rank_in,self.key_dim): raise ValueError("keys shape must be [B,T,H,Rin,K]")
        if values.shape!=(B,T,self.heads,self.rank_in,self.value_dim): raise ValueError("values shape must be [B,T,H,Rin,V]")
        if scores.shape!=(B,T,self.heads): raise ValueError("scores shape must be [B,T,H]")
        if valid.shape!=(B,T) or valid.dtype!=torch.bool: raise TypeError("valid must be bool [B,T]")
        if boundary.shape!=(B,T) or boundary.dtype!=torch.bool: raise TypeError("boundary must be bool [B,T]")
        if positions is not None and (positions.shape!=(B,T) or positions.dtype!=torch.int64):
            raise TypeError("positions must be int64 [B,T]")
        tensors=(query,keys,values,scores,valid,boundary)+(() if positions is None else (positions,))
        if any(x.device!=query.device for x in tensors): raise ValueError("HOLA inputs must share a device")
        if not all(x.is_floating_point() for x in (query,keys,values,scores)): raise TypeError("query/keys/values/scores must be floating point")
        self._require(torch.isfinite(query).all(),"query must be finite")
        self._require(torch.isfinite(keys).all(),"keys must be finite")
        self._require(torch.isfinite(values).all(),"values must be finite")
        self._require(torch.isfinite(scores).all(),"scores must be finite")
        if positions is not None:
            self._require((((positions>=0)&valid)|((positions==-1)&~valid)).all(),
                          "positions must be nonnegative when valid and -1 otherwise")

    def _promotion_transform(self, state: HybridHOLAState, rows: Tensor) -> HybridHOLAState:
        ck=torch.cat((state.keys,state.block_keys),2); cv=torch.cat((state.values,state.block_values),2)
        rank_scores = state.block_positions.float() if self.policy == "recency" else state.block_scores
        cs=torch.cat((state.positions.float() if self.policy == "recency" else state.scores,rank_scores),2)
        raw_scores=torch.cat((state.scores,state.block_scores),2)
        cp=torch.cat((state.positions,state.block_positions),2)
        cm=torch.cat((state.valid&(state.epochs==state.current_epoch[...,None]),
                      state.block_valid&(state.block_epochs==state.current_epoch[...,None])),2)
        sel=self._select_survivors(cs,cp,cm); ok=sel>=0; safe=sel.clamp_min(0)
        def gather(x):
            idx=safe[(...,)+(None,)*(x.ndim-3)].expand(*safe.shape,*x.shape[3:])
            return torch.gather(x,2,idx)
        nk,nv=gather(ck),gather(cv); ns=torch.gather(raw_scores,2,safe); np=torch.gather(cp,2,safe)
        nk=torch.where(ok[...,None,None],nk,torch.zeros_like(nk)); nv=torch.where(ok[...,None,None],nv,torch.zeros_like(nv))
        ns=torch.where(ok,ns,torch.zeros_like(ns)).detach(); np=torch.where(ok,np,torch.full_like(np,-1))
        mask=rows[...,None]; mask5=rows[...,None,None,None]
        zero_bk=torch.zeros_like(state.block_keys); zero_bv=torch.zeros_like(state.block_values)
        zero_bs=torch.zeros_like(state.block_scores); neg_bp=torch.full_like(state.block_positions,-1); zero_bm=torch.zeros_like(state.block_valid)
        return HybridHOLAState(torch.where(mask5,nk,state.keys),torch.where(mask5,nv,state.values),
            torch.where(mask,ns,state.scores),torch.where(mask,np,state.positions),torch.where(mask,ok,state.valid),
            torch.where(mask,torch.where(ok,state.current_epoch[...,None],torch.full_like(state.epochs,-1)),state.epochs),
            torch.where(mask5,zero_bk,state.block_keys),torch.where(mask5,zero_bv,state.block_values),
            torch.where(mask,zero_bs,state.block_scores),torch.where(mask,neg_bp,state.block_positions),
            state.block_valid.clone(),torch.where(mask,torch.full_like(state.block_epochs,-1),state.block_epochs),
            torch.where(rows,torch.zeros_like(state.block_count),state.block_count),
            state.next_position.clone(),state.current_epoch.clone(),
            state.admission_count,state.age_sum,state.age_count)

    def _promote_if_complete(self, state: HybridHOLAState, rows: Tensor) -> HybridHOLAState:
        if bool(rows.any()):
            return self._promotion_transform(state,rows)
        return state

    def _select_survivors(self, cs: Tensor, cp: Tensor, cm: Tensor) -> Tensor:
        position_order=torch.argsort(cp,dim=-1,descending=True,stable=True)
        masked=torch.where(cm,cs,torch.full_like(cs,-torch.inf))
        score_order=torch.argsort(torch.gather(masked,-1,position_order),dim=-1,descending=True,stable=True)
        ranked=torch.gather(position_order,-1,score_order); ranked_valid=torch.gather(cm,-1,ranked)
        take=min(self.width,ranked.shape[-1]); sel=ranked[...,:take]
        sel=torch.where(ranked_valid[...,:take],sel,torch.full_like(sel,-1))
        if take<self.width:
            sel=torch.cat((sel,torch.full((*sel.shape[:-1],self.width-take),-1,dtype=torch.int64,device=sel.device)),-1)
        return sel

    def admit(self, state: HybridHOLAState | None, keys: Tensor, values: Tensor, scores: Tensor,
              positions: Tensor, valid: Tensor) -> HybridHOLAState:
        B,T,H,R,K=keys.shape
        if (H,R,K)!=(self.heads,self.rank_in,self.key_dim) or values.shape!=(B,T,H,R,self.value_dim):
            raise ValueError("keys/values must be [B,T,H,Rin,K/V]")
        if scores.shape!=(B,T,H) or positions.shape!=(B,T) or positions.dtype!=torch.int64 or valid.shape!=(B,T) or valid.dtype!=torch.bool:
            raise ValueError("scores [B,T,H], int64 positions and bool valid [B,T] required")
        if not all(x.is_floating_point() for x in (keys,values,scores)):
            raise TypeError("keys, values and scores must be floating point")
        if any(x.device!=keys.device for x in (values,scores,positions,valid)):
            raise ValueError("admission inputs must share a device")
        self._require(torch.isfinite(keys).all(),"keys must be finite")
        self._require(torch.isfinite(values).all(),"values must be finite")
        self._require(torch.isfinite(scores).all(),"scores must be finite")
        self._require((((positions>=0)&valid)|((positions==-1)&~valid)).all(),
                      "positions must be nonnegative when valid and -1 otherwise")
        state=self._empty(B,keys.device) if state is None else state
        self._validate_state(state,B,keys.device)
        return self._admit_unchecked(state,keys,values,scores,positions,valid)

    def _admit_unchecked(self, state: HybridHOLAState, keys: Tensor, values: Tensor, scores: Tensor,
                         positions: Tensor, valid: Tensor, *, promote_complete: bool = True
                         ) -> HybridHOLAState:
        B,T,H=keys.shape[:3]
        for t in range(T):
            active=valid[:,t]; slot=state.block_count.clamp_max(self.block_size-1)
            one=torch.nn.functional.one_hot(slot,self.block_size).to(dtype=torch.bool) & active[:,None,None]
            m=one[...,None,None]; m3=one
            # Explicit broadcast keeps each head's independent candidate entry differentiable.
            bk=torch.where(m,keys[:,t,:,None].to(self.storage_dtype),state.block_keys)
            bv=torch.where(m,values[:,t,:,None].to(self.storage_dtype),state.block_values)
            bs=torch.where(m3,scores[:,t].detach().float()[:,:,None],state.block_scores)
            bp=torch.where(m3,positions[:,t,None,None].expand(B,H,self.block_size),state.block_positions)
            bm=torch.where(m3,torch.ones_like(state.block_valid),state.block_valid)
            be=torch.where(m3,state.current_epoch[...,None],state.block_epochs)
            count=state.block_count+active[:,None].to(torch.int64)
            nxt=torch.maximum(state.next_position,torch.where(active,positions[:,t]+1,0))
            state=HybridHOLAState(state.keys,state.values,state.scores,state.positions,state.valid,state.epochs,
                bk,bv,bs,bp,bm,be,count,nxt,state.current_epoch,
                state.admission_count + active[:,None].to(torch.int64), state.age_sum, state.age_count)
            if promote_complete:
                rows=count==self.block_size
                state=self._promote_if_complete(state,rows)
        return state

    def _advance_epoch_unchecked(self, state: HybridHOLAState, reset: Tensor) -> HybridHOLAState:
        rows=reset[:,None]
        return HybridHOLAState(state.keys,state.values,state.scores,state.positions,state.valid,state.epochs,
            state.block_keys,state.block_values,state.block_scores,state.block_positions,state.block_valid,
            state.block_epochs,torch.where(rows,torch.zeros_like(state.block_count),state.block_count),
            torch.where(reset,torch.zeros_like(state.next_position),state.next_position),
            torch.where(rows,state.current_epoch+1,state.current_epoch),
            state.admission_count,state.age_sum,state.age_count)

    def step_fast(self, state: HybridHOLAState, query: Tensor, key: Tensor, value: Tensor,
                  score: Tensor, position: Tensor, block_fill: int
                  ) -> tuple[Tensor, HybridHOLAState, int]:
        """step_unchecked specialization for all-valid, boundary-free tokens.

        ``block_fill`` mirrors the (uniform) staged-block occupancy on the
        host, so block completion is decided without a per-token
        tensor-to-host synchronization. Indexed scatter also avoids building
        a full C-wide one-hot mask for each staged write. Bitwise-equivalent
        to step_unchecked(valid=all-true, boundary=all-false).
        """
        B, H = key.shape[0], self.heads
        slot = state.block_count.clamp_max(self.block_size - 1)
        slot_index = slot[..., None]
        key_index = slot_index[..., None, None].expand(B, H, 1, self.rank_in, self.key_dim)
        value_index = slot_index[..., None, None].expand(
            B, H, 1, self.rank_in, self.value_dim
        )
        bk = state.block_keys.scatter(
            2, key_index, key[:, :, None].to(self.storage_dtype)
        )
        bv = state.block_values.scatter(
            2, value_index, value[:, :, None].to(self.storage_dtype)
        )
        bs = state.block_scores.scatter(2, slot_index, score.detach().float()[:, :, None])
        bp = state.block_positions.scatter(
            2, slot_index, position[:, None, None].expand(B, H, 1)
        )
        bm = state.block_valid.scatter(
            2, slot_index, torch.ones(B, H, 1, dtype=torch.bool, device=key.device)
        )
        be = state.block_epochs.scatter(2, slot_index, state.current_epoch[..., None])
        count = state.block_count + 1
        nxt = torch.maximum(state.next_position, position + 1)
        state = HybridHOLAState(state.keys, state.values, state.scores, state.positions,
            state.valid, state.epochs, bk, bv, bs, bp, bm, be, count, nxt,
            state.current_epoch, state.admission_count + 1, state.age_sum, state.age_count)
        candidate_positions = torch.cat((state.positions, state.block_positions), 2)
        candidate_valid = torch.cat((
            state.valid & (state.epochs == state.current_epoch[..., None]),
            state.block_valid & (state.block_epochs == state.current_epoch[..., None])), 2)
        ages = (position[:, None, None] - candidate_positions).clamp_min(0)
        state = replace(state,
            age_sum=state.age_sum + (ages * candidate_valid).sum(-1).double(),
            age_count=state.age_count + candidate_valid.sum(-1).to(torch.int64))
        output = self.read(state, query)
        block_fill += 1
        if block_fill == self.block_size:
            state = self._promotion_transform(
                state, torch.ones(B, H, dtype=torch.bool, device=key.device))
            block_fill = 0
        return output, state, block_fill

    def scan_fast(self, state: HybridHOLAState, query: Tensor, key: Tensor,
                  value: Tensor, score: Tensor, block_fill: int
                  ) -> tuple[Tensor, HybridHOLAState, int]:
        """Vectorized all-valid, boundary-free HOLA segment.

        HOLA output is read-only with respect to the four recurrent states, so
        every query in a staged block can attend in one batched operation.  A
        position mask retains the exact admit-before-read causal set, including
        the C-th token before promotion.  Chunks end at promotion boundaries;
        survivor selection therefore still executes exactly once per completed
        block and selected BF16 key/value gradients follow the same gather path
        as :meth:`step_fast`.
        """
        B, T, H = query.shape[:3]
        if T == 0:
            shape = (B, 0, H, query.shape[3], self.rank_in, self.value_dim)
            return query.new_empty(shape, dtype=torch.float32), state, block_fill

        outputs = []
        start = 0
        while start < T:
            length = min(T - start, self.block_size - block_fill)
            stop = start + length
            q_chunk = query[:, start:stop]
            k_chunk = key[:, start:stop]
            v_chunk = value[:, start:stop]
            score_chunk = score[:, start:stop]
            positions = (
                state.next_position[:, None]
                + torch.arange(length, device=query.device, dtype=torch.int64)[None]
            )

            # All rows/heads have the mirrored occupancy, so one contiguous
            # differentiable replacement admits the whole segment.  Unlike a
            # per-token scatter this copies the C-wide staging tensors once.
            before = block_fill
            after = block_fill + length
            staged_keys = torch.cat((
                state.block_keys[:, :, :before],
                k_chunk.permute(0, 2, 1, 3, 4).to(self.storage_dtype),
                state.block_keys[:, :, after:],
            ), 2)
            staged_values = torch.cat((
                state.block_values[:, :, :before],
                v_chunk.permute(0, 2, 1, 3, 4).to(self.storage_dtype),
                state.block_values[:, :, after:],
            ), 2)
            staged_scores = torch.cat((
                state.block_scores[:, :, :before],
                score_chunk.detach().float().permute(0, 2, 1),
                state.block_scores[:, :, after:],
            ), 2)
            staged_positions = torch.cat((
                state.block_positions[:, :, :before],
                positions[:, None].expand(B, H, length),
                state.block_positions[:, :, after:],
            ), 2)
            staged_valid = torch.cat((
                state.block_valid[:, :, :before],
                torch.ones(B, H, length, dtype=torch.bool, device=query.device),
                state.block_valid[:, :, after:],
            ), 2)
            staged_epochs = torch.cat((
                state.block_epochs[:, :, :before],
                state.current_epoch[..., None].expand(B, H, length),
                state.block_epochs[:, :, after:],
            ), 2)
            state = HybridHOLAState(
                state.keys, state.values, state.scores, state.positions,
                state.valid, state.epochs, staged_keys, staged_values,
                staged_scores, staged_positions, staged_valid, staged_epochs,
                state.block_count + length, positions[:, -1] + 1,
                state.current_epoch, state.admission_count + length,
                state.age_sum, state.age_count,
            )

            keys = torch.cat((state.keys, state.block_keys), 2).float()
            values = torch.cat((state.values, state.block_values), 2).float()
            candidate_positions = torch.cat(
                (state.positions, state.block_positions), 2
            )
            candidate_valid = torch.cat((
                state.valid & (state.epochs == state.current_epoch[..., None]),
                state.block_valid
                & (state.block_epochs == state.current_epoch[..., None]),
            ), 2)
            visible = (
                candidate_valid[:, :, None]
                & (candidate_positions[:, :, None] <= positions[:, None, :, None])
            )

            q = q_chunk.float()
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
            q = q * self.gamma_q.float()[None, None]
            k = keys * torch.rsqrt(keys.square().mean(-1, keepdim=True) + 1e-6)
            k = k * self.gamma_k.float()[None, :, None]
            logits = (
                torch.einsum("bthok,bhnik->bthoin", q, k)
                * self.key_dim ** -0.5
            )
            mask = visible.permute(0, 2, 1, 3)[:, :, :, None, None]
            logits = logits.masked_fill(~mask, -torch.inf)
            sink = self.sink_logit.float()[None, None, :, None, :, None]
            sink = sink.expand(*logits.shape[:-1], 1)
            weights = torch.softmax(torch.cat((logits, sink), -1), -1)[..., :-1]
            outputs.append(torch.einsum("bthoin,bhniv->bthoiv", weights, values))

            ages = (positions[:, None, :, None] - candidate_positions[:, :, None]).clamp_min(0)
            state = replace(
                state,
                age_sum=state.age_sum + (ages * visible).sum((-2, -1)).double(),
                age_count=state.age_count + visible.sum((-2, -1)).to(torch.int64),
            )
            block_fill = after
            if block_fill == self.block_size:
                state = self._promotion_transform(
                    state, torch.ones(B, H, dtype=torch.bool, device=query.device)
                )
                block_fill = 0
            start = stop

        return torch.cat(outputs, 1), state, block_fill

    def step_unchecked(self, state: HybridHOLAState, query: Tensor, key: Tensor, value: Tensor,
                       score: Tensor, position: Tensor, valid: Tensor, boundary: Tensor
                       ) -> tuple[Tensor, HybridHOLAState]:
        state=self._advance_epoch_unchecked(state,boundary&valid)
        # HOLA's visible set for token t is persistent-W + every causal token in
        # the current C-block (including t) + sink.  In particular, token C must
        # read the still-staged full block before its survivors are promoted.
        state=self._admit_unchecked(state,key[:,None],value[:,None],score[:,None],
                                    position[:,None],valid[:,None],promote_complete=False)
        candidate_positions = torch.cat((state.positions, state.block_positions), 2)
        candidate_valid = torch.cat((
            state.valid & (state.epochs == state.current_epoch[...,None]),
            state.block_valid & (state.block_epochs == state.current_epoch[...,None])), 2)
        ages = (position[:,None,None] - candidate_positions).clamp_min(0)
        age_sum = (ages * candidate_valid).sum(-1).double()
        age_count = candidate_valid.sum(-1).to(torch.int64)
        active = valid[:,None]
        state = replace(state,
            age_sum=state.age_sum + torch.where(active, age_sum, torch.zeros_like(age_sum)),
            age_count=state.age_count + torch.where(active, age_count, torch.zeros_like(age_count)))
        output=self.read(state,query)
        output=torch.where(valid[:,None,None,None,None],output,torch.zeros_like(output))
        state=self._promote_if_complete(state,state.block_count==self.block_size)
        return output,state

    def read(self, state: HybridHOLAState, query: Tensor) -> Tensor:
        keys=torch.cat((state.keys,state.block_keys),2).float()
        values=torch.cat((state.values,state.block_values),2).float()
        valid=torch.cat((state.valid&(state.epochs==state.current_epoch[...,None]),
                         state.block_valid&(state.block_epochs==state.current_epoch[...,None])),2)
        # Qwen-style RMSNorm-gamma is RMSNorm(x) * gamma.  Applying gamma
        # before normalization would cancel every uniform gamma rescaling and
        # remove the paper's learned logit-sharpness control.
        q=query.float()
        q=q*torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-6)
        q=q*self.gamma_q.float()[None]
        k=keys
        k=k*torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-6)
        k=k*self.gamma_k.float()[None,:,None]
        logits=torch.einsum("bhok,bhnik->bhoin",q,k)*self.key_dim**-0.5
        logits=logits.masked_fill(~valid[:,:,None,None],-torch.inf)
        sink=self.sink_logit.float()[None,:,None,:,None].expand(*logits.shape[:-1],1)
        weights=torch.softmax(torch.cat((logits,sink),-1),-1)[...,:-1]
        return torch.einsum("bhoin,bhniv->bhoiv",weights,values)

    def _run(self, query: Tensor, keys: Tensor, values: Tensor, scores: Tensor, *, positions: Tensor|None=None,
             valid: Tensor|None=None, boundary: Tensor|None=None, initial_state: HybridHOLAState|None=None):
        B,T=query.shape[:2]; valid=torch.ones(B,T,dtype=torch.bool,device=query.device) if valid is None else valid
        boundary=torch.zeros_like(valid) if boundary is None else boundary; generated=positions is None
        self._validate_inputs(query,keys,values,scores,positions,valid,boundary)
        state=self._empty(B,query.device) if initial_state is None else initial_state
        self._validate_state(state,B,query.device)
        max_epoch=torch.iinfo(torch.int64).max
        reset_count=(boundary&valid).to(torch.int64).sum(1)[:,None]
        self._require((state.current_epoch <= max_epoch-reset_count).all(),
                      "HOLA epoch overflow at reset boundary")
        outs=[]
        for t in range(T):
            pos=state.next_position[:,None] if generated else positions[:,t:t+1]
            out,state=self.step_unchecked(state,query[:,t],keys[:,t],values[:,t],scores[:,t],
                                          pos[:,0],valid[:,t],boundary[:,t])
            outs.append(out)
        shape=(B,0,self.heads,query.shape[3],self.rank_in,self.value_dim)
        return (torch.stack(outs,1) if outs else query.new_empty(shape)),state

    def scan(self,*args,**kwargs): return self._run(*args,**kwargs)


__all__=["HybridHOLACache","HybridHOLAState","four_state_exact_update_score",
         "four_state_normalized_update_score","shared_exact_update_score"]
