"""Deterministic Qwen RULER-style multi-key retrieval examples and scorers."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


RULER_SCHEMA_VERSION = "1.0.0"
RULER_CONTEXT_LENGTHS = (512, 2048, 4096, 8192, 16384, 32768)
RULER_QUERY_COUNTS = (1, 4, 8)
RULER_DEPTH_STRATA = ("early", "middle", "late")
RULER_ARMS = ("native", "recency", "surprise")
RULER_HEAL_SEED_COUNT = 3
RULER_MIN_EPISODES_PER_CELL = 64
RULER_LONG_CELLS = ("16k_4q", "16k_8q", "32k_4q", "32k_8q")

_LENGTH_LABELS = {
    512: "512",
    2048: "2k",
    4096: "4k",
    8192: "8k",
    16384: "16k",
    32768: "32k",
}
_FILLER = (
    "The grass is green and the sky is blue. People walk along the river in "
    "the afternoon sun and talk quietly about weather and their plans."
)
_CONSONANTS = "bcdfghklmnprstv"
_VOWELS = "aeiou"


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    encoded = tokenizer(text, add_special_tokens=False)
    raw = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
    if raw is None or isinstance(raw, (str, bytes, bytearray)):
        raise TypeError("tokenizer output must expose input_ids")
    values = tuple(raw)
    if any(type(value) is not int or value < 0 for value in values):
        raise TypeError("tokenizer input_ids must be nonnegative ints")
    return values


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RulerCell:
    context_length: int
    needles: int
    queries: int

    def __post_init__(self) -> None:
        if self.context_length not in RULER_CONTEXT_LENGTHS:
            raise ValueError("RULER context_length must use the pinned evaluation grid")
        if self.needles != 16:
            raise ValueError("RULER cells require exactly 16 needles")
        if self.queries not in RULER_QUERY_COUNTS:
            raise ValueError("RULER queries must be 1, 4, or 8")
        if self.queries > self.needles:
            raise ValueError("RULER queries cannot exceed needles")

    @property
    def cell_id(self) -> str:
        return f"{_LENGTH_LABELS[self.context_length]}_{self.queries}q"


@dataclass(frozen=True)
class RulerEpisode:
    episode_id: str
    seed: int
    example_index: int
    cell: RulerCell
    input_ids: tuple[int, ...]
    prompt_end: int
    answers: tuple[str, ...]
    answer_token_ids: tuple[tuple[int, ...], ...]
    answer_spans: tuple[tuple[int, int], ...]
    source_spans: tuple[tuple[int, int], ...]
    depth_strata: tuple[str, ...]
    query_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or len(self.episode_id) != 64:
            raise ValueError("episode_id must be a SHA-256 hex digest")
        _require_int("seed", self.seed)
        _require_int("example_index", self.example_index)
        if not isinstance(self.cell, RulerCell):
            raise TypeError("cell must be a RulerCell")
        input_ids = tuple(self.input_ids)
        if any(type(value) is not int or value < 0 for value in input_ids):
            raise TypeError("input_ids must contain nonnegative ints")
        if type(self.prompt_end) is not int or not self.cell.context_length < self.prompt_end <= len(input_ids):
            raise ValueError("prompt_end must follow the exact context and precede answers")
        count = self.cell.queries
        fields = (
            self.answers,
            self.answer_token_ids,
            self.answer_spans,
            self.source_spans,
            self.depth_strata,
            self.query_keys,
        )
        if any(len(field) != count for field in fields):
            raise ValueError("RULER answer/query fields must match the cell query count")
        if len(set(self.query_keys)) != count:
            raise ValueError("RULER query keys must be unique")
        for tokens, span in zip(self.answer_token_ids, self.answer_spans):
            start, end = span
            if not tokens or not self.prompt_end <= start < end <= len(input_ids):
                raise ValueError("answer spans must be nonempty and follow the prompt")
            if tuple(input_ids[start:end]) != tuple(tokens):
                raise ValueError("answer spans must exactly identify answer tokens")
        for start, end in self.source_spans:
            if not 0 <= start < end <= self.cell.context_length:
                raise ValueError("source spans must lie in the exact context prefix")
        if any(depth not in RULER_DEPTH_STRATA for depth in self.depth_strata):
            raise ValueError("depth strata must use early, middle, or late")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "answers", tuple(self.answers))
        object.__setattr__(
            self, "answer_token_ids", tuple(tuple(tokens) for tokens in self.answer_token_ids)
        )
        object.__setattr__(self, "answer_spans", tuple(tuple(span) for span in self.answer_spans))
        object.__setattr__(self, "source_spans", tuple(tuple(span) for span in self.source_spans))
        object.__setattr__(self, "depth_strata", tuple(self.depth_strata))
        object.__setattr__(self, "query_keys", tuple(self.query_keys))


@dataclass(frozen=True)
class RulerScore:
    episode_id: str
    evaluation_mode: str
    numerator: int
    denominator: int
    episode_exact: bool
    answer_correct: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.evaluation_mode not in {"teacher_forced", "free_generation"}:
            raise ValueError("RULER evaluation mode is invalid")
        _require_int("numerator", self.numerator)
        _require_int("denominator", self.denominator, minimum=1)
        if self.numerator > self.denominator:
            raise ValueError("RULER numerator cannot exceed denominator")
        if type(self.episode_exact) is not bool:
            raise TypeError("episode_exact must be bool")
        outcomes = tuple(self.answer_correct)
        if len(outcomes) != self.denominator or any(type(value) is not bool for value in outcomes):
            raise ValueError("answer_correct must contain one bool per answer")
        if self.numerator != sum(outcomes) or self.episode_exact != all(outcomes):
            raise ValueError("RULER score totals are inconsistent")
        object.__setattr__(self, "answer_correct", outcomes)


def _pseudoword(rng: random.Random) -> str:
    return "".join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS) for _ in range(3))


def _unique_values(rng: random.Random, count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    keys: list[str] = []
    key_set: set[str] = set()
    while len(keys) < count:
        key = _pseudoword(rng)
        if key not in key_set:
            keys.append(key)
            key_set.add(key)
    answers: list[str] = []
    answer_set: set[str] = set()
    while len(answers) < count:
        answer = str(rng.randint(1_000_000, 9_999_999))
        if answer not in answer_set:
            answers.append(answer)
            answer_set.add(answer)
    return tuple(keys), tuple(answers)


def _depth(span: tuple[int, int], context_length: int) -> str:
    midpoint = (span[0] + span[1]) / 2
    fraction = midpoint / context_length
    return "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"


def build_ruler_episode(
    tokenizer: Any,
    *,
    cell: RulerCell,
    seed: int,
    example_index: int,
) -> RulerEpisode:
    """Build one arm-neutral episode with an exact-length context prefix."""

    if not isinstance(cell, RulerCell):
        raise TypeError("cell must be a RulerCell")
    _require_int("seed", seed)
    _require_int("example_index", example_index)
    identity = {
        "schema_version": RULER_SCHEMA_VERSION,
        "seed": seed,
        "example_index": example_index,
        "cell": cell.cell_id,
        "needles": cell.needles,
    }
    episode_id = _canonical_digest(identity)
    rng = random.Random(int(episode_id[:16], 16))
    keys, answers = _unique_values(rng, cell.needles)

    encoded_needles: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for key, answer in zip(keys, answers):
        prefix = _encode(tokenizer, f" One of the special magic numbers for {key} is:")
        value = _encode(tokenizer, f" {answer}")
        suffix = _encode(tokenizer, ".")
        if not value:
            raise ValueError("tokenizer must produce at least one token for each answer")
        encoded_needles.append((prefix, value, suffix))
    needle_tokens = sum(sum(len(part) for part in needle) for needle in encoded_needles)
    filler_budget = cell.context_length - needle_tokens
    if filler_budget < cell.needles + 1:
        raise ValueError("tokenized needles do not fit the requested RULER context")
    filler = _encode(tokenizer, _FILLER)
    if not filler:
        raise ValueError("tokenizer must produce filler tokens")
    gaps = [filler_budget // (cell.needles + 1)] * (cell.needles + 1)
    for index in range(filler_budget % len(gaps)):
        gaps[index] += 1
    slot_order = list(range(cell.needles))
    rng.shuffle(slot_order)
    context: list[int] = []
    spans_by_index: dict[int, tuple[int, int]] = {}
    filler_cursor = 0

    def append_filler(count: int) -> None:
        nonlocal filler_cursor
        for _ in range(count):
            context.append(filler[filler_cursor % len(filler)])
            filler_cursor += 1

    for slot, needle_index in enumerate(slot_order):
        append_filler(gaps[slot])
        prefix, value, suffix = encoded_needles[needle_index]
        context.extend(prefix)
        start = len(context)
        context.extend(value)
        spans_by_index[needle_index] = (start, len(context))
        context.extend(suffix)
    append_filler(gaps[-1])
    if len(context) != cell.context_length:
        raise AssertionError("RULER context construction did not preserve exact length")

    candidates: dict[str, list[int]] = {name: [] for name in RULER_DEPTH_STRATA}
    for needle_index, span in spans_by_index.items():
        candidates[_depth(span, cell.context_length)].append(needle_index)
    for values in candidates.values():
        rng.shuffle(values)
    selected: list[int] = []
    for query_index in range(cell.queries):
        desired = RULER_DEPTH_STRATA[query_index % len(RULER_DEPTH_STRATA)]
        pool = candidates[desired]
        if pool:
            selected.append(pool.pop())
            continue
        fallback = next((values for values in candidates.values() if values), None)
        if fallback is None:
            raise AssertionError("RULER query selection exhausted needles")
        selected.append(fallback.pop())

    query_keys = tuple(keys[index] for index in selected)
    query_answers = tuple(answers[index] for index in selected)
    query_source_spans = tuple(spans_by_index[index] for index in selected)
    depth_strata = tuple(_depth(span, cell.context_length) for span in query_source_spans)
    if cell.queries == 1:
        question = f" What is the special magic number for {query_keys[0]}? The number is:"
    else:
        question = (
            f" What are the special magic numbers for {', '.join(query_keys)}? "
            "The numbers are:"
        )
    input_ids = list(context)
    input_ids.extend(_encode(tokenizer, question))
    prompt_end = len(input_ids)
    answer_token_ids: list[tuple[int, ...]] = []
    answer_spans: list[tuple[int, int]] = []
    for index, answer in enumerate(query_answers):
        input_ids.extend(_encode(tokenizer, " " if index == 0 else ", "))
        tokens = _encode(tokenizer, answer)
        if not tokens:
            raise ValueError("tokenizer must produce answer tokens")
        start = len(input_ids)
        input_ids.extend(tokens)
        answer_token_ids.append(tokens)
        answer_spans.append((start, len(input_ids)))
    return RulerEpisode(
        episode_id=episode_id,
        seed=seed,
        example_index=example_index,
        cell=cell,
        input_ids=tuple(input_ids),
        prompt_end=prompt_end,
        answers=query_answers,
        answer_token_ids=tuple(answer_token_ids),
        answer_spans=tuple(answer_spans),
        source_spans=query_source_spans,
        depth_strata=depth_strata,
        query_keys=query_keys,
    )


def score_teacher_forced(
    episode: RulerEpisode, predicted_token_ids: Sequence[int]
) -> RulerScore:
    """Score aligned next-token predictions over the declared answer spans."""

    if not isinstance(episode, RulerEpisode):
        raise TypeError("episode must be a RulerEpisode")
    predictions = tuple(predicted_token_ids)
    if len(predictions) != len(episode.input_ids) or any(type(value) is not int for value in predictions):
        raise ValueError("teacher-forced predictions must align with episode input_ids")
    outcomes = tuple(
        predictions[start:end] == gold
        for (start, end), gold in zip(episode.answer_spans, episode.answer_token_ids)
    )
    return RulerScore(
        episode.episode_id,
        "teacher_forced",
        sum(outcomes),
        len(outcomes),
        all(outcomes),
        outcomes,
    )


def _normalize_generated(value: str) -> str:
    if type(value) is not str:
        raise TypeError("generated answers must be strings")
    return " ".join(value.strip().rstrip(".,").split()).casefold()


def score_free_generation(
    episode: RulerEpisode, generated_answers: Sequence[str]
) -> RulerScore:
    """Score a deterministic generation subset without relabelling it teacher-forced."""

    if not isinstance(episode, RulerEpisode):
        raise TypeError("episode must be a RulerEpisode")
    generated = tuple(generated_answers)
    if len(generated) != episode.cell.queries:
        raise ValueError("free-generation output must contain one answer per query")
    outcomes = tuple(
        _normalize_generated(actual) == _normalize_generated(expected)
        for actual, expected in zip(generated, episode.answers)
    )
    return RulerScore(
        episode.episode_id,
        "free_generation",
        sum(outcomes),
        len(outcomes),
        all(outcomes),
        outcomes,
    )


def select_free_generation_subset(
    episodes: Iterable[RulerEpisode], *, count: int, seed: int
) -> tuple[RulerEpisode, ...]:
    """Select an input-order-invariant subset by stable hash rank."""

    _require_int("count", count, minimum=1)
    _require_int("seed", seed)
    values = tuple(episodes)
    if any(not isinstance(episode, RulerEpisode) for episode in values):
        raise TypeError("episodes must contain RulerEpisode records")
    identities = [episode.episode_id for episode in values]
    if len(set(identities)) != len(identities):
        raise ValueError("episodes must have unique identities")
    if count > len(values):
        raise ValueError("free-generation subset count exceeds available episodes")
    ranked = sorted(
        values,
        key=lambda episode: (
            hashlib.sha256(f"{seed}:{episode.episode_id}".encode("ascii")).digest(),
            episode.episode_id,
        ),
    )[:count]
    return tuple(sorted(ranked, key=lambda episode: episode.episode_id))


def ruler_evidence_scope(
    *,
    identities: Mapping[str, Iterable[tuple[int, str, str, str]]],
) -> str:
    """Return promotion only for complete matched arm/seed/example/cell evidence.

    Each identity is ``(seed, example_id, cell_id, evaluation_mode)``.  Requiring
    the complete identity sets prevents seed/length metadata alone from being
    mistaken for promotion evidence.
    """

    if not isinstance(identities, Mapping):
        raise TypeError("RULER identities must be an arm mapping")
    if set(identities) != set(RULER_ARMS):
        return "feasibility"
    indexed: dict[str, set[tuple[int, str, str, str]]] = {}
    for arm in RULER_ARMS:
        values = tuple(identities[arm])
        normalized: set[tuple[int, str, str, str]] = set()
        for identity in values:
            if not isinstance(identity, tuple) or len(identity) != 4:
                raise TypeError("RULER identities must be four-field tuples")
            seed, example_id, cell_id, evaluation_mode = identity
            if type(seed) is not int:
                raise TypeError("RULER identity seeds must be ints")
            if any(
                type(value) is not str or not value
                for value in (example_id, cell_id, evaluation_mode)
            ):
                raise TypeError("RULER identity strings must be nonempty")
            normalized.add((seed, example_id, cell_id, evaluation_mode))
        if len(normalized) != len(values):
            raise ValueError(f"RULER arm {arm} contains duplicate identities")
        indexed[arm] = normalized
    reference = indexed["native"]
    if not reference or any(indexed[arm] != reference for arm in RULER_ARMS[1:]):
        return "feasibility"
    teacher_forced = tuple(identity for identity in reference if identity[3] == "teacher_forced")
    seeds = {identity[0] for identity in teacher_forced}
    if len(seeds) < RULER_HEAL_SEED_COUNT:
        return "feasibility"
    for seed in seeds:
        for cell in RULER_LONG_CELLS:
            count = sum(
                identity[0] == seed and identity[2] == cell
                for identity in teacher_forced
            )
            if count < RULER_MIN_EPISODES_PER_CELL:
                return "feasibility"
    return "promotion"


@dataclass(frozen=True)
class StateTrackingScore:
    task: str
    correct: tuple[bool, ...]
    lm_loss: float
    modulus: int | None = None


def score_state_tracking(
    task: str,
    predictions: Sequence[int],
    targets: Sequence[int],
    *,
    lm_loss: float,
    modulus: int | None = None,
) -> StateTrackingScore:
    """Score parity or modular-arithmetic state alongside finite LM loss."""
    if task not in {"parity", "modular"}:
        raise ValueError("state tracking task must be parity or modular")
    predicted = tuple(predictions)
    expected = tuple(targets)
    if not predicted or len(predicted) != len(expected) or any(type(v) is not int for v in predicted + expected):
        raise ValueError("state tracking predictions and targets must be matched ints")
    loss = float(lm_loss)
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("lm_loss must be finite and nonnegative")
    if task == "modular":
        if type(modulus) is not int or modulus < 2:
            raise ValueError("modular state tracking requires modulus >= 2")
        outcomes = tuple(a % modulus == b % modulus for a, b in zip(predicted, expected))
    else:
        if modulus is not None:
            raise ValueError("parity state tracking does not accept modulus")
        outcomes = tuple(a % 2 == b % 2 for a, b in zip(predicted, expected))
    return StateTrackingScore(task, outcomes, loss, modulus)


__all__ = [
    "RULER_ARMS",
    "RULER_CONTEXT_LENGTHS",
    "RULER_DEPTH_STRATA",
    "RULER_HEAL_SEED_COUNT",
    "RULER_LONG_CELLS",
    "RULER_MIN_EPISODES_PER_CELL",
    "RULER_QUERY_COUNTS",
    "RULER_SCHEMA_VERSION",
    "RulerCell",
    "RulerEpisode",
    "RulerScore",
    "StateTrackingScore",
    "build_ruler_episode",
    "ruler_evidence_scope",
    "score_free_generation",
    "score_teacher_forced",
    "score_state_tracking",
    "select_free_generation_subset",
]
