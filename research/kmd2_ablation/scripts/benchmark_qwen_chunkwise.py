"""Reproducible complete-module performance gate for Package B/HOLA.

The benchmark always executes the campaign-resolved ``package-b-hola-w64``
``QwenFourStateHybrid`` module.  It never substitutes a recurrence-only
surrogate.  Candidate adapters live only in this script: missing prototypes are
reported unavailable, and no benchmark selector enters model configuration,
the public API, a cache, or a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import dataclass, fields, is_dataclass
import json
from pathlib import Path
import random
import statistics
import sys
import threading
import time
from typing import Callable, ContextManager, Iterator, Sequence

import torch
from torch import Tensor, nn


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


CONTROL_ID = "package-b-hola-w64"
MODES = ("inference", "training", "decode")
CANDIDATES = (
    "auto",
    "rematerialized",
    "decode",
    "direct-reads",
    "mixer",
    "hola",
)
ARM_NAMES = ("forced_authority", "current_auto", "candidate")
DEFAULT_SEED = 20260717
CAMPAIGN_HIDDEN = 1024
CAMPAIGN_HEADS = 16


class _CandidateUnavailableError(RuntimeError):
    """The requested private benchmark candidate has not been registered."""


@dataclass
class _CallCounter:
    count: int = 0


# Benchmark adapters replace module globals. Serialize each replacement for its
# entire lifetime so overlapping benchmark threads cannot observe or restore
# another context's hook. The lock is reentrant because composed adapters nest
# several replacements in one thread. Unrelated concurrent calls into a target
# module while it is patched remain unsupported by this benchmark-only API.
_ADAPTER_PATCH_LOCK = threading.RLock()


@contextmanager
def _no_adapter() -> Iterator[_CallCounter]:
    yield _CallCounter()


@contextmanager
def _replace_callable(
    target: tuple[object, str], replacement: Callable[..., object], *, count: bool
) -> Iterator[_CallCounter]:
    with _ADAPTER_PATCH_LOCK:
        owner, name = target
        original = getattr(owner, name)
        calls = _CallCounter()

        def wrapped(*args, **kwargs):
            if count:
                calls.count += 1
            return replacement(*args, **kwargs)

        setattr(owner, name, wrapped)
        try:
            yield calls
        finally:
            setattr(owner, name, original)


@dataclass(frozen=True)
class _CandidateAdapter:
    """Resolved benchmark-only hooks for one private candidate.

    Future candidate tasks register a factory in ``_CANDIDATE_FACTORIES`` only
    after their private prototype exists.  Until then, the named entry resolves
    to an unavailable adapter instead of guessing an implementation.
    """

    name: str
    modes: frozenset[str]
    private_target: tuple[object, str] | None = None
    private_prototype: Callable[..., object] | None = None
    private_context_factory: (
        Callable[[], ContextManager[_CallCounter]] | None
    ) = None
    arm_context_factory: (
        Callable[[str, bool], ContextManager[_CallCounter]] | None
    ) = None
    expected_calls_per_sample: Callable[[int, int, bool], int] | None = None
    production_target: tuple[object, str] | None = None
    production_fallback: Callable[..., object] | None = None
    available: bool = True
    reason: str = ""

    def arm_context(
        self, arm: str, *, integrated: bool
    ) -> ContextManager[_CallCounter]:
        if arm not in ARM_NAMES:
            raise ValueError(f"unknown benchmark arm: {arm}")
        if not self.available:
            raise RuntimeError(self.reason)
        if self.name == "auto":
            return _no_adapter()
        if self.arm_context_factory is not None:
            return self.arm_context_factory(arm, integrated)
        if integrated:
            if self.production_target is None or self.production_fallback is None:
                raise RuntimeError(
                    f"candidate {self.name!r} has no integrated production hook"
                )
            if arm == "current_auto":
                return _replace_callable(
                    self.production_target, self.production_fallback, count=False
                )
            owner, name = self.production_target
            production = getattr(owner, name)
            return _replace_callable(
                self.production_target, production, count=True
            )
        if arm == "forced_authority":
            return _no_adapter()
        if arm == "current_auto":
            return _no_adapter()
        if self.private_context_factory is not None:
            return self.private_context_factory()
        if self.private_target is None or self.private_prototype is None:
            raise RuntimeError(
                f"candidate {self.name!r} has no private benchmark adapter"
            )
        return _replace_callable(
            self.private_target, self.private_prototype, count=True
        )


_CandidateFactory = Callable[[str], _CandidateAdapter]

# These entries deliberately remain empty until the corresponding private
# prototype exists.  Candidate work extends this registry, not the model API.
_CANDIDATE_FACTORIES: dict[str, _CandidateFactory | None] = {
    name: None for name in CANDIDATES if name != "auto"
}


def _rematerialized_eight_output_compatibility(
    *args, **kwargs
) -> tuple[Tensor, ...]:
    """Adapt the private seven-output endpoint wrapper to the eager ABI."""
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    result = chunkwise._rematerialized_torch_chunk_four_state_segment(
        *args, **kwargs
    )
    diagnostic_trace = result[0].new_empty(0)
    return (result[0], diagnostic_trace, *result[1:])


def _rematerialized_training_expected_calls(
    length: int, _batch_size: int, _integrated: bool,
) -> int:
    """Forward plus nonreentrant recomputation for every 64-token segment."""
    full_segments, trailing_tokens = divmod(length, 64)
    eligible_segments = full_segments + int(trailing_tokens >= 16)
    return 2 * eligible_segments


@contextmanager
def _rematerialized_training_context() -> Iterator[_CallCounter]:
    """Count production rematerialization and reopen it for private no-grad probes."""
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from research.kmd2_ablation import qwen_hybrid_four_state as four_state
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger
    from research.kmd2_ablation import qwen_hybrid_liger_dplr as dplr

    with _ADAPTER_PATCH_LOCK:
        eligibility = chunkwise.can_use_torch_chunk_four_state_segment
        rematerialized = chunkwise._rematerialized_torch_chunk_four_state_segment

        def canonical_grad_eligibility(*args, **kwargs):
            # Reuse the production inference envelope without changing it:
            # the private arm differs only in allowing an active grad mode.
            with torch.no_grad():
                return eligibility(*args, **kwargs)

        with ExitStack() as stack:
            # Package B now selects Liger by default.  This private benchmark
            # context deliberately suppresses it so the rematerialized
            # PyTorch control remains independently measurable.
            stack.enter_context(
                _replace_callable(
                    (liger, "can_use_liger_chunked_four_state"),
                    lambda *args, **kwargs: False,
                    count=False,
                )
            )
            stack.enter_context(
                _replace_callable(
                    (dplr, "true_chunked_dplr_available"),
                    lambda: False,
                    count=False,
                )
            )
            # Production uses rematerialization only with autograd.  The
            # serialized benchmark also reopens the eager no-grad route so the
            # HOLA score-order diagnostic can compare it with exact authority.
            stack.enter_context(
                _replace_callable(
                    (four_state, "_can_use_torch_chunk_with_cache"),
                    lambda _cache_active: True,
                    count=False,
                )
            )
            stack.enter_context(
                _replace_callable(
                    (chunkwise, "can_use_torch_chunk_four_state_segment"),
                    canonical_grad_eligibility,
                    count=False,
                )
            )
            stack.enter_context(
                _replace_callable(
                    (chunkwise, "torch_chunk_four_state_segment"),
                    _rematerialized_eight_output_compatibility,
                    count=False,
                )
            )
            calls = stack.enter_context(
                _replace_callable(
                    (chunkwise, "_rematerialized_torch_chunk_four_state_segment"),
                    rematerialized,
                    count=True,
                )
            )
            yield calls


@contextmanager
def _rematerialized_arm_context(
    arm: str, integrated: bool,
) -> Iterator[_CallCounter]:
    """Keep the rematerialized control reachable after Liger became default."""
    if not integrated:
        if arm == "candidate":
            with _rematerialized_training_context() as calls:
                yield calls
        else:
            with _no_adapter() as calls:
                yield calls
        return

    from research.kmd2_ablation import qwen_hybrid_four_state as four_state
    from research.kmd2_ablation import qwen_hybrid_liger_chunked as liger
    from research.kmd2_ablation import qwen_hybrid_liger_dplr as dplr

    with _ADAPTER_PATCH_LOCK:
        with ExitStack() as stack:
            if arm == "candidate":
                stack.enter_context(
                    _replace_callable(
                        (dplr, "true_chunked_dplr_available"),
                        lambda: False,
                        count=False,
                    )
                )
                stack.enter_context(
                    _replace_callable(
                        (liger, "can_use_liger_chunked_four_state"),
                        lambda *args, **kwargs: False,
                        count=False,
                    )
                )
                production = four_state._can_use_torch_chunk_with_cache
                calls = stack.enter_context(
                    _replace_callable(
                        (four_state, "_can_use_torch_chunk_with_cache"),
                        production,
                        count=True,
                    )
                )
            else:
                calls = _CallCounter()
            yield calls


def _rematerialized_candidate_factory(_mode: str) -> _CandidateAdapter:
    from research.kmd2_ablation import qwen_hybrid_four_state as four_state

    return _CandidateAdapter(
        name="rematerialized",
        modes=frozenset({"training"}),
        private_context_factory=_rematerialized_training_context,
        arm_context_factory=_rematerialized_arm_context,
        expected_calls_per_sample=_rematerialized_training_expected_calls,
        production_target=(four_state, "_can_use_torch_chunk_with_cache"),
        production_fallback=lambda _cache_active: False,
    )


_CANDIDATE_FACTORIES["rematerialized"] = _rematerialized_candidate_factory


def _decode_eight_output_compatibility(*args, **kwargs) -> tuple[Tensor, ...]:
    """Adapt the private seven-output T=1 step to the segment ABI."""
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise

    result = chunkwise._torch_four_state_decode_step(*args, **kwargs)
    diagnostic_trace = result[0].new_empty(0)
    return (result[0], diagnostic_trace, *result[1:])


@contextmanager
def _decode_arm_context(
    arm: str, integrated: bool,
) -> Iterator[_CallCounter]:
    """Isolate one decode arm and count only direct-step executions."""
    from research.kmd2_ablation import qwen_hybrid_chunkwise as chunkwise
    from research.kmd2_ablation import qwen_hybrid_four_state as four_state
    from research.kmd2_ablation import qwen_hybrid_triton as triton

    with _ADAPTER_PATCH_LOCK:
        direct = chunkwise._torch_four_state_decode_step
        with ExitStack() as stack:
            if arm == "current_auto" or (not integrated and arm == "candidate"):
                stack.enter_context(
                    _replace_callable(
                        (chunkwise, "_can_use_package_b_decode_step"),
                        lambda *args, **kwargs: False,
                        count=False,
                    )
                )
            if not integrated and arm == "candidate":
                eligibility = chunkwise._can_use_torch_four_state_decode_step
                stack.enter_context(
                    _replace_callable(
                        (four_state, "_can_use_torch_chunk_with_cache"),
                        lambda _cache_active: True,
                        count=False,
                    )
                )
                # B1 production keeps Triton priority. Suppress it only in the
                # private candidate arm so the direct step can be measured.
                stack.enter_context(
                    _replace_callable(
                        (triton, "can_use_triton_four_state_segment"),
                        lambda *args, **kwargs: False,
                        count=False,
                    )
                )
                stack.enter_context(
                    _replace_callable(
                        (chunkwise, "can_use_torch_chunk_four_state_segment"),
                        eligibility,
                        count=False,
                    )
                )
                stack.enter_context(
                    _replace_callable(
                        (chunkwise, "torch_chunk_four_state_segment"),
                        _decode_eight_output_compatibility,
                        count=False,
                    )
                )
            calls = stack.enter_context(
                _replace_callable(
                    (chunkwise, "_torch_four_state_decode_step"),
                    direct,
                    count=True,
                )
            )
            yield calls


def _decode_expected_calls(
    _length: int, batch_size: int, integrated: bool,
) -> int:
    if batch_size not in (1, 2):
        return 0
    return int(not integrated or batch_size == 2)


def _decode_candidate_factory(_mode: str) -> _CandidateAdapter:
    return _CandidateAdapter(
        name="decode",
        modes=frozenset({"decode"}),
        arm_context_factory=_decode_arm_context,
        expected_calls_per_sample=_decode_expected_calls,
    )


_CANDIDATE_FACTORIES["decode"] = _decode_candidate_factory


def _resolve_candidate(name: str, mode: str) -> _CandidateAdapter:
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate: {name}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if name == "auto":
        return _CandidateAdapter(name="auto", modes=frozenset(MODES))
    factory = _CANDIDATE_FACTORIES[name]
    if factory is None:
        return _CandidateAdapter(
            name=name,
            modes=frozenset(),
            available=False,
            reason=f"candidate {name!r} unavailable: private prototype is missing",
        )
    adapter = factory(mode)
    if adapter.name != name:
        raise RuntimeError("candidate factory returned the wrong adapter")
    if mode not in adapter.modes:
        return _CandidateAdapter(
            name=name,
            modes=adapter.modes,
            available=False,
            reason=f"candidate {name!r} unavailable in {mode!r} mode",
        )
    return adapter


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="inference")
    parser.add_argument("--candidate", choices=CANDIDATES, default="auto")
    parser.add_argument(
        "--integrated",
        action="store_true",
        help=(
            "compare production dispatch with only the named optimization "
            "suppressed against normal production dispatch"
        ),
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=_positive_int,
        default=[32, 64, 128, 256],
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=_positive_int,
        default=None,
        help="decode supports populated-cache batches 1 and 2; other modes default to 2",
    )
    parser.add_argument("--warmup", type=_nonnegative_int, default=5)
    parser.add_argument("--iterations", type=_positive_int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _parse_options(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    options = parser.parse_args(argv)
    if options.integrated and options.candidate == "auto":
        parser.error("--integrated requires a named candidate; auto is aggregate smoke only")
    if options.batch_sizes is None:
        options.batch_sizes = [1, 2] if options.mode == "decode" else [2]
    if any(batch not in (1, 2) for batch in options.batch_sizes):
        parser.error("--batch-sizes supports only 1 and 2")
    if options.mode != "decode" and options.batch_sizes != [2]:
        parser.error("inference and training use canonical batch size 2")
    options.lengths = list(dict.fromkeys(options.lengths))
    options.batch_sizes = list(dict.fromkeys(options.batch_sizes))
    return options


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _build_package_b(device: torch.device, dtype: torch.dtype) -> nn.Module:
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.scripts.smoke_step import build_flagship_layer

    module = build_flagship_layer(
        CAMPAIGN_HIDDEN, device, dtype, heads=CAMPAIGN_HEADS
    )
    if type(module) is not QwenFourStateHybrid:
        raise RuntimeError("campaign did not resolve to exact QwenFourStateHybrid")
    contract = getattr(module, "maximum_control_contract", None)
    if contract is None or contract.control_id != CONTROL_ID:
        raise RuntimeError("benchmark module is not campaign-resolved Package B")
    if module._active("cache_policy") != "hola_exact_outer_w64":
        raise RuntimeError("Package B benchmark requires HOLA-W64 enabled")
    if module.hola.width != 64 or module.hola.policy != "exact_outer":
        raise RuntimeError("Package B benchmark requires exact-outer HOLA-W64")
    if module.dk * 4 != 128 or module.rank != 4:
        raise RuntimeError("Package B benchmark requires four native compact K/4 lanes")
    return module


@dataclass
class _ArmRuntime:
    module: nn.Module
    hidden: Tensor
    initial_cache: object | None

    def clear_gradients(self) -> None:
        self.module.zero_grad(set_to_none=True)
        self.hidden.grad = None


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _decode_cache_is_populated(cache: object) -> bool:
    """Require recurrent history and staged or promoted HOLA content per head."""
    history = getattr(cache, "has_history", None)
    update_count = getattr(cache, "update_count", None)
    hola_state = getattr(cache, "hola_state", None)
    if not isinstance(history, Tensor) or not isinstance(update_count, Tensor):
        return False
    if (
        history.dtype != torch.bool
        or history.ndim != 2
        or update_count.dtype != torch.int64
        or update_count.shape != history.shape[:1]
        or hola_state is None
    ):
        return False
    block_count = getattr(hola_state, "block_count", None)
    valid = getattr(hola_state, "valid", None)
    epochs = getattr(hola_state, "epochs", None)
    current_epoch = getattr(hola_state, "current_epoch", None)
    if not all(
        isinstance(value, Tensor)
        for value in (block_count, valid, epochs, current_epoch)
    ):
        return False
    if (
        block_count.dtype != torch.int64
        or block_count.ndim != 2
        or block_count.shape[0] != history.shape[0]
        or valid.dtype != torch.bool
        or valid.ndim != 3
        or valid.shape[:2] != block_count.shape
        or epochs.dtype != torch.int64
        or epochs.shape != valid.shape
        or current_epoch.dtype != torch.int64
        or current_epoch.shape != block_count.shape
    ):
        return False
    recurrent_populated = (update_count > 0) & history.any(-1)
    promoted_visible = valid & epochs.eq(current_epoch[..., None])
    hola_populated = block_count.gt(0) | promoted_visible.any(-1)
    return bool(recurrent_populated.all() & hola_populated.all())


def _make_runtimes(
    *,
    mode: str,
    length: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, _ArmRuntime]:
    base = _build_package_b(device, dtype)
    modules = {
        "forced_authority": base,
        "current_auto": copy.deepcopy(base),
        "candidate": copy.deepcopy(base),
    }
    modules["forced_authority"].force_torch_recurrence = True
    for module in modules.values():
        module.train(mode == "training")

    generator = _generator(device, seed + 1009 * length + 97 * batch_size)
    if mode == "decode":
        prefix = 0.1 * torch.randn(
            batch_size,
            length,
            CAMPAIGN_HIDDEN,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        token = 0.1 * torch.randn(
            batch_size,
            1,
            CAMPAIGN_HIDDEN,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        runtimes = {}
        for arm, module in modules.items():
            with torch.no_grad():
                _, cache = module.scan(prefix)
            if not _decode_cache_is_populated(cache):
                raise RuntimeError("decode benchmark requires populated recurrent/HOLA cache")
            runtimes[arm] = _ArmRuntime(module, token.detach().clone(), cache)
        return runtimes

    source = 0.1 * torch.randn(
        batch_size,
        length,
        CAMPAIGN_HIDDEN,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return {
        arm: _ArmRuntime(
            module,
            source.detach().clone().requires_grad_(mode == "training"),
            None,
        )
        for arm, module in modules.items()
    }


@dataclass
class _Execution:
    output: Tensor
    cache: object
    calls: int


def _execute_module(runtime: _ArmRuntime, *, mode: str) -> tuple[Tensor, object]:
    if mode == "training":
        output, cache = runtime.module.scan(
            runtime.hidden, initial_cache=runtime.initial_cache
        )
        output.float().square().mean().backward()
    else:
        with torch.no_grad():
            output, cache = runtime.module.scan(
                runtime.hidden, initial_cache=runtime.initial_cache
            )
    return output, cache


def _execute(
    runtime: _ArmRuntime,
    *,
    mode: str,
    arm: str,
    adapter: _CandidateAdapter,
    integrated: bool,
) -> _Execution:
    with adapter.arm_context(arm, integrated=integrated) as calls:
        output, cache = _execute_module(runtime, mode=mode)
    return _Execution(output=output, cache=cache, calls=calls.count)


def _assert_nested_close(actual: object, expected: object, path: str = "cache") -> None:
    if is_dataclass(actual) or is_dataclass(expected):
        if type(actual) is not type(expected):
            raise AssertionError(f"{path} type mismatch")
        for field in fields(actual):
            _assert_nested_close(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
        return
    if isinstance(actual, Tensor) or isinstance(expected, Tensor):
        if not isinstance(actual, Tensor) or not isinstance(expected, Tensor):
            raise AssertionError(f"{path} tensor mismatch")
        if actual.is_floating_point():
            torch.testing.assert_close(
                actual, expected, atol=2e-3, rtol=8e-3,
                msg=lambda message: f"{path}: {message}",
            )
        elif not torch.equal(actual, expected):
            raise AssertionError(f"{path} differs")
        return
    if actual != expected:
        raise AssertionError(f"{path} differs: {actual!r} != {expected!r}")


def _assert_training_gradients_close(
    actual: _ArmRuntime, expected: _ArmRuntime
) -> None:
    _assert_nested_close(actual.hidden.grad, expected.hidden.grad, "input.grad")
    actual_parameters = dict(actual.module.named_parameters())
    expected_parameters = dict(expected.module.named_parameters())
    if actual_parameters.keys() != expected_parameters.keys():
        raise AssertionError("parameter names differ")
    for name, parameter in actual_parameters.items():
        reference = expected_parameters[name]
        if parameter.grad is None or reference.grad is None:
            if parameter.grad is not reference.grad:
                raise AssertionError(f"parameter gradient presence differs: {name}")
            continue
        torch.testing.assert_close(
            parameter.grad,
            reference.grad,
            atol=3e-3,
            rtol=1e-2,
            msg=lambda message, name=name: f"{name}.grad: {message}",
        )


def _has_training_gradients(runtime: _ArmRuntime) -> bool:
    return runtime.hidden.grad is not None and any(
        parameter.grad is not None
        for parameter in runtime.module.parameters()
        if parameter.requires_grad
    )


def _correctness(
    runtimes: dict[str, _ArmRuntime],
    *,
    mode: str,
    adapter: _CandidateAdapter,
    integrated: bool,
) -> tuple[dict[str, bool], dict[str, bool], bool]:
    runtimes["forced_authority"].clear_gradients()
    authority = _execute(
        runtimes["forced_authority"],
        mode=mode,
        arm="forced_authority",
        adapter=adapter,
        integrated=integrated,
    )
    authority_backward = mode != "training" or _has_training_gradients(
        runtimes["forced_authority"]
    )
    correct = {"forced_authority": authority_backward}
    gradient_parity = {"forced_authority": authority_backward}
    backward_evidence = {"forced_authority": authority_backward}
    for arm in ("current_auto", "candidate"):
        runtimes[arm].clear_gradients()
        result = _execute(
            runtimes[arm],
            mode=mode,
            arm=arm,
            adapter=adapter,
            integrated=integrated,
        )
        try:
            _assert_nested_close(result.output, authority.output, f"{arm}.output")
            _assert_nested_close(result.cache, authority.cache, f"{arm}.cache")
        except AssertionError:
            forward_cache_correct = False
        else:
            forward_cache_correct = True
        if mode == "training":
            backward_evidence[arm] = _has_training_gradients(runtimes[arm])
            if not authority_backward or not backward_evidence[arm]:
                gradient_parity[arm] = False
            else:
                try:
                    _assert_training_gradients_close(
                        runtimes[arm], runtimes["forced_authority"]
                    )
                except AssertionError:
                    gradient_parity[arm] = False
                else:
                    gradient_parity[arm] = True
        else:
            backward_evidence[arm] = True
            gradient_parity[arm] = True
        correct[arm] = forward_cache_correct and gradient_parity[arm]
        del result
    del authority
    for runtime in runtimes.values():
        runtime.clear_gradients()
    backward_executed = mode == "training" and all(backward_evidence.values())
    return correct, gradient_parity, backward_executed


def _rotating_order(round_index: int) -> tuple[str, ...]:
    offset = round_index % len(ARM_NAMES)
    return ARM_NAMES[offset:] + ARM_NAMES[:offset]


def _sample(
    runtime: _ArmRuntime,
    *,
    mode: str,
    arm: str,
    adapter: _CandidateAdapter,
    integrated: bool,
    device: torch.device,
) -> tuple[float, int, int, int]:
    runtime.clear_gradients()
    with adapter.arm_context(arm, integrated=integrated) as calls:
        torch.cuda.synchronize(device)
        start_allocated = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        output, cache = _execute_module(runtime, mode=mode)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        incremental_peak_bytes = max(0, peak_bytes - start_allocated)
    call_count = calls.count
    del output, cache
    runtime.clear_gradients()
    return elapsed_ms, peak_bytes, incremental_peak_bytes, call_count


def _measure(
    runtimes: dict[str, _ArmRuntime],
    *,
    mode: str,
    adapter: _CandidateAdapter,
    integrated: bool,
    warmup: int,
    iterations: int,
    device: torch.device,
    tokens_per_sample: int,
    correct: dict[str, bool],
) -> dict[str, dict[str, object]]:
    for round_index in range(warmup):
        for arm in _rotating_order(round_index):
            _sample(
                runtimes[arm],
                mode=mode,
                arm=arm,
                adapter=adapter,
                integrated=integrated,
                device=device,
            )

    samples: dict[str, list[float]] = {arm: [] for arm in ARM_NAMES}
    peaks: dict[str, list[int]] = {arm: [] for arm in ARM_NAMES}
    incremental_peaks: dict[str, list[int]] = {arm: [] for arm in ARM_NAMES}
    calls = {arm: 0 for arm in ARM_NAMES}
    for iteration in range(iterations):
        for arm in _rotating_order(warmup + iteration):
            elapsed_ms, peak_bytes, incremental_peak_bytes, call_count = _sample(
                runtimes[arm],
                mode=mode,
                arm=arm,
                adapter=adapter,
                integrated=integrated,
                device=device,
            )
            samples[arm].append(elapsed_ms)
            peaks[arm].append(peak_bytes)
            incremental_peaks[arm].append(incremental_peak_bytes)
            calls[arm] += call_count

    metrics = {}
    for arm in ARM_NAMES:
        median_ms = float(statistics.median(samples[arm]))
        metrics[arm] = {
            "median_ms": median_ms,
            "tokens_per_second": float(tokens_per_sample * 1000.0 / median_ms),
            "peak_bytes": max(peaks[arm]),
            "incremental_peak_bytes": max(incremental_peaks[arm]),
            "correct": correct[arm],
            "call_count": calls[arm],
        }
    return metrics


def _gate(
    arms: dict[str, dict[str, object]], *, claimable: bool, iterations: int,
    expected_candidate_calls: int | None = None,
) -> dict[str, object]:
    baseline = arms["current_auto"]
    candidate = arms["candidate"]
    throughput_gain = (
        float(candidate["tokens_per_second"])
        / float(baseline["tokens_per_second"])
        - 1.0
    )
    baseline_peak = int(baseline["incremental_peak_bytes"])
    candidate_peak = int(candidate["incremental_peak_bytes"])
    memory_reduction = (
        1.0 - candidate_peak / baseline_peak
        if baseline_peak
        else (0.0 if candidate_peak == 0 else -1.0)
    )
    throughput_win = throughput_gain >= 0.05
    memory_win = memory_reduction >= 0.10
    other_metric_within_tolerance = (
        (throughput_win and memory_reduction >= -0.02)
        or (memory_win and throughput_gain >= -0.02)
    )
    correctness = all(bool(arms[arm]["correct"]) for arm in ARM_NAMES)
    candidate_calls = int(candidate["call_count"])
    authority_calls = int(arms["forced_authority"]["call_count"])
    baseline_calls = int(baseline["call_count"])
    expected_calls = (
        iterations
        if expected_candidate_calls is None
        else expected_candidate_calls
    )
    candidate_selected = claimable and candidate_calls == expected_calls
    other_arms_clean = authority_calls == 0 and baseline_calls == 0
    selection_proved = (not claimable) or (
        candidate_selected and other_arms_clean
    )
    selection_proof = {
        "required": claimable,
        "minimum_candidate_calls": expected_calls if claimable else 0,
        "candidate_calls": candidate_calls,
        "expected_candidate_calls": expected_calls if claimable else 0,
        "observed_candidate_calls": candidate_calls,
        "forced_authority_calls": authority_calls,
        "current_auto_calls": baseline_calls,
        "candidate_selected": candidate_selected,
        "other_arms_clean": other_arms_clean,
        "proved": selection_proved,
    }
    return {
        "throughput_gain": throughput_gain,
        "memory_reduction": memory_reduction,
        "throughput_win": throughput_win,
        "memory_win": memory_win,
        "other_metric_within_tolerance": other_metric_within_tolerance,
        "correctness": correctness,
        "claimable": claimable,
        "memory_metric": "incremental_peak_bytes",
        "selection_proof": selection_proof,
        "passed": bool(
            claimable
            and correctness
            and selection_proved
            and (throughput_win or memory_win)
            and other_metric_within_tolerance
        ),
    }


def _run_benchmark(options: argparse.Namespace) -> dict[str, object]:
    adapter = _resolve_candidate(options.candidate, options.mode)
    if not adapter.available:
        raise _CandidateUnavailableError(adapter.reason)
    if not torch.cuda.is_available():
        raise RuntimeError("complete-Package-B benchmark requires CUDA")
    device = torch.device(options.device)
    if device.type != "cuda":
        raise RuntimeError("complete-Package-B benchmark requires a CUDA device")
    dtype = torch.bfloat16 if options.dtype == "bfloat16" else torch.float32

    _seed_everything(options.seed)
    records = []
    for batch_size in options.batch_sizes:
        for length in options.lengths:
            runtimes = _make_runtimes(
                mode=options.mode,
                length=length,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                seed=options.seed,
            )
            correct, gradient_parity, backward_executed = _correctness(
                runtimes,
                mode=options.mode,
                adapter=adapter,
                integrated=options.integrated,
            )
            tokens_per_sample = batch_size if options.mode == "decode" else batch_size * length
            arms = _measure(
                runtimes,
                mode=options.mode,
                adapter=adapter,
                integrated=options.integrated,
                warmup=options.warmup,
                iterations=options.iterations,
                device=device,
                tokens_per_sample=tokens_per_sample,
                correct=correct,
            )
            named_candidate = options.candidate != "auto"
            expected_candidate_calls = (
                options.iterations
                * adapter.expected_calls_per_sample(
                    length, batch_size, options.integrated,
                )
                if named_candidate
                and adapter.expected_calls_per_sample is not None
                else None
            )
            claimable = named_candidate and (
                expected_candidate_calls is None
                or expected_candidate_calls > 0
            )
            record = {
                "length": length,
                "batch_size": batch_size,
                "complete_module": True,
                "decode_tokens_per_sample": (
                    tokens_per_sample if options.mode == "decode" else None
                ),
                "training_correctness": (
                    {
                        "backward_executed": backward_executed,
                        "gradient_parity": gradient_parity,
                    }
                    if options.mode == "training"
                    else None
                ),
                "arms": arms,
                "gate": _gate(
                    arms,
                    claimable=claimable,
                    iterations=options.iterations,
                    expected_candidate_calls=expected_candidate_calls,
                ),
            }
            records.append(record)
            del runtimes
            torch.cuda.empty_cache()

    return {
        "schema_version": "1.0.0",
        "control_id": CONTROL_ID,
        "module_type": "QwenFourStateHybrid",
        "complete_module": True,
        "mode": options.mode,
        "candidate": options.candidate,
        "integrated": options.integrated,
        "candidate_available": True,
        "aggregate_smoke_only": options.candidate == "auto",
        "seed": options.seed,
        "warmup": options.warmup,
        "iterations": options.iterations,
        "dtype": options.dtype,
        "device": str(device),
        "arm_order": "rotating_interleaved",
        "gate_thresholds": {
            "minimum_throughput_gain": 0.05,
            "minimum_memory_reduction": 0.10,
            "maximum_other_metric_regression": 0.02,
        },
        "records": records,
    }


def _print_text(report: dict[str, object]) -> None:
    print(
        f"{report['control_id']} mode={report['mode']} "
        f"candidate={report['candidate']} integrated={report['integrated']}"
    )
    for record in report["records"]:
        print(f"B={record['batch_size']} length={record['length']}")
        for arm, metrics in record["arms"].items():
            print(
                f"  {arm:18s} {metrics['median_ms']:9.3f} ms  "
                f"{metrics['tokens_per_second']:12.2f} tok/s  "
                f"peak={metrics['peak_bytes']} "
                f"incremental={metrics['incremental_peak_bytes']} "
                f"correct={metrics['correct']} "
                f"calls={metrics['call_count']}"
            )
        print(f"  gate passed={record['gate']['passed']}")


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_options(argv)
    try:
        if options.json:
            with redirect_stdout(sys.stderr):
                report = _run_benchmark(options)
        else:
            report = _run_benchmark(options)
    except _CandidateUnavailableError as error:
        candidate_available = False
        error_kind = "candidate_unavailable"
        error_message = str(error)
    except Exception as error:
        candidate_available = True
        error_kind = "execution_error"
        error_message = str(error)
    else:
        if options.json:
            print(json.dumps(report, sort_keys=True))
        else:
            _print_text(report)
        return 0
    if options.json:
        print(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "control_id": CONTROL_ID,
                    "mode": options.mode,
                    "candidate": options.candidate,
                    "candidate_available": candidate_available,
                    "error_kind": error_kind,
                    "error": error_message,
                    "records": [],
                },
                sort_keys=True,
            )
        )
    else:
        print(f"benchmark unavailable ({error_kind}): {error_message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
