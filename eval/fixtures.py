"""Synthetic answer fixtures for LLM-mode eval, keyed by question index (0-7).

Reuses backend/mocks/data.py's MOCK_QUESTIONS (question text) and rubric
concepts/reference answers as ground truth, rather than hand-authoring a
second, possibly-divergent rubric — mock mode and eval fixtures should never
disagree about what a "correct" answer looks like.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.config import round_for_index
from backend.mocks.data import MOCK_QUESTIONS, get_rubric

FIXTURE_KINDS = ("blank", "weak", "strong")

# Hand-authored "weak" answers: plausible-length but vague/wrong, one per index.
# Deliberately avoid the rubric's own concept keywords so these read as
# genuinely weak, not just short.
_WEAK_ANSWERS: dict[int, str] = {
    0: "Um, I don't know, I just applied because I needed a job I guess.",
    1: "I'm not really sure. Maybe I'm okay at stuff. I don't have weaknesses I can think of.",
    2: "You could just loop through it and check somehow, I'm not 100% sure honestly.",
    3: "You'd make a table I think, with some columns for the data.",
    4: "I just used whatever the tutorial said, didn't really think about it much.",
    5: "It's like organizing a database somehow, not totally sure on specifics.",
    6: "Things didn't work out but it was fine, we moved on.",
    7: "I don't know, I guess I'm hardworking like everyone else.",
}

_FALLBACK_STRONG_ANSWER = (
    "I have strong hands-on experience here and can walk through the details "
    "with a specific example if that would help."
)


@dataclass(frozen=True)
class Fixture:
    index: int
    round_name: str
    question: str
    kind: str  # "blank" | "weak" | "strong"
    answer: str
    rubric: dict  # resolved rubric variant: {"concepts", "topic", "reference_answer"?}


def build_fixture(index: int, kind: str) -> Fixture:
    if kind not in FIXTURE_KINDS:
        raise ValueError(f"unknown fixture kind {kind!r}")
    # Fixtures always use the BASE question text. The "redirect" question text
    # is an *output* of the Strategy/Interview agents in llm mode (driven by
    # next_directive.focus_project), not something pre-selected as eval input —
    # redirect correctness is checked on the directive itself, see metrics.py.
    question = MOCK_QUESTIONS[index]["base"]
    rubric = get_rubric(index, question)
    if kind == "blank":
        answer = ""
    elif kind == "weak":
        answer = _WEAK_ANSWERS[index]
    else:  # "strong"
        answer = rubric.get("reference_answer") or _FALLBACK_STRONG_ANSWER
    return Fixture(
        index=index,
        round_name=round_for_index(index),
        question=question,
        kind=kind,
        answer=answer,
        rubric=rubric,
    )


def all_fixtures(indices: list[int] | None = None) -> list[Fixture]:
    """All (index, kind) fixtures for the given indices (default: all 0-7)."""
    idxs = indices if indices is not None else sorted(MOCK_QUESTIONS.keys())
    return [build_fixture(i, kind) for i in idxs for kind in FIXTURE_KINDS]
