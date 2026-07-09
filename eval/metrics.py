"""Pure, deterministic scoring/validity checks over agent outputs.

No LLM calls here — every function only inspects already-produced data, so
these are fast and independently pytest-testable with synthetic inputs (see
tests/test_eval_metrics.py). `severity` only matters when `passed=False`:
"error" fails a `--strict` run, "warning" is informational only (used for
checks with inherent LLM variance, like score monotonicity).
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.core.state import Directive, Scores

_DIM_KEYS = ["technical_accuracy", "communication", "confidence", "problem_solving", "relevance"]
_VALID_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # "error" | "warning" — only meaningful when passed=False


def to_dict(c: CheckResult) -> dict:
    return {"name": c.name, "passed": c.passed, "detail": c.detail, "severity": c.severity}


# --- schema / bounds ---------------------------------------------------------
def check_scores_bounds(scores: Scores) -> CheckResult:
    """All 5 dims in [0,10] and total <= 50. Pydantic's Field(ge=0,le=10)
    already enforces this at construction, so this can only fail if a
    Scores object was built bypassing validation — defense-in-depth, not the
    primary signal of scoring quality."""
    vals = [getattr(scores, k) for k in _DIM_KEYS]
    ok = all(0 <= v <= 10 for v in vals) and scores.total <= 50
    return CheckResult("scores_bounds", ok, f"total={scores.total}")


def check_raw_difficulty_valid(raw_difficulty) -> CheckResult:
    """Checks the LLM's RAW (pre-coercion) difficulty string against the
    requested schema. A real Directive object can never fail this (pydantic's
    Literal type rejects bad values at construction) — this checks the model's
    actual instruction-following, before our coercion papers over mistakes."""
    val = str(raw_difficulty).strip().lower() if raw_difficulty is not None else None
    ok = val in _VALID_DIFFICULTIES
    return CheckResult("raw_difficulty_valid", ok, f"raw={raw_difficulty!r}")


def check_question_validity(question_text: str) -> CheckResult:
    """Non-empty, a single question, and doesn't leak an answer."""
    q = (question_text or "").strip()
    if not q:
        return CheckResult("question_validity", False, "empty question")
    leak_markers = ("the answer is", "correct answer:", "for example, the answer")
    if any(m in q.lower() for m in leak_markers):
        return CheckResult("question_validity", False, "question text appears to leak an answer")
    question_marks = q.count("?")
    if question_marks > 2:
        return CheckResult("question_validity", False, f"{question_marks} '?' marks — likely multiple questions")
    if question_marks == 2:
        # Often a single question with a clarifying follow-up clause ("...
        # what happened, and how did you fix it?") — plausible, but flagged
        # for a human to glance at rather than silently passed.
        return CheckResult("question_validity", False, "2 '?' marks — borderline, review manually", severity="warning")
    return CheckResult("question_validity", True)


# --- monotonicity -------------------------------------------------------------
def check_monotonic_scores(blank_total: float, weak_total: float, strong_total: float, slack: float = 2.0) -> CheckResult:
    """blank <= weak <= strong (within `slack` points on the 50-point scale,
    to absorb LLM scoring noise). Warning-severity only — LLM variance means
    an occasional inversion isn't necessarily a real regression."""
    ok = (blank_total <= weak_total + slack) and (weak_total <= strong_total + slack)
    detail = f"blank={blank_total:g} weak={weak_total:g} strong={strong_total:g}"
    return CheckResult("monotonic_scores", ok, detail, severity="warning")


# --- strategy behavior ---------------------------------------------------------
def check_redirect_on_weak_score(directive: Directive, expected_project_title: str, score_total: int, threshold: int = 25) -> CheckResult:
    """Given a weak Technical score (< threshold/50) and a profile with one
    clearly-strongest project, the directive's focus_project should name that
    project (substring match, tolerating LLM paraphrase). This is the
    headline check: unlike mock mode's hardcoded redirect guardrails, llm
    mode's _llm_decision makes this call unprompted by any hardcoded rule."""
    if score_total >= threshold:
        return CheckResult("redirect_on_weak_score", True, "score not weak enough for this check — n/a")
    got = (directive.focus_project or "").strip().lower()
    expected = expected_project_title.strip().lower()
    ok = bool(got) and (expected in got or got in expected)
    return CheckResult("redirect_on_weak_score", ok, f"expected~={expected_project_title!r} got={directive.focus_project!r}")


def check_difficulty_escalation(directive: Directive, score_total: int, threshold: int = 40) -> CheckResult:
    """Given a strong score (>= threshold/50), difficulty should not stay
    'easy'. 'hard' is the ideal response; 'medium' is tolerated as a soft pass
    (noted in the detail) since a strategy agent holding steady for one strong
    answer isn't unreasonable — only staying at 'easy' is a real miss."""
    if score_total < threshold:
        return CheckResult("difficulty_escalation", True, "score not strong enough for this check — n/a")
    ok = directive.difficulty in ("medium", "hard")
    soft_note = "" if directive.difficulty == "hard" else " (soft pass — expected 'hard')"
    return CheckResult("difficulty_escalation", ok, f"difficulty={directive.difficulty}{soft_note if ok else ''}")


# --- latency -------------------------------------------------------------
@dataclass
class LatencyRecord:
    agent: str
    fixture_key: str  # e.g. "3:weak" or "weak_technical:0"
    seconds: float


def summarize_latency(records: list[LatencyRecord]) -> dict:
    if not records:
        return {}
    by_agent: dict[str, list[float]] = {}
    for r in records:
        by_agent.setdefault(r.agent, []).append(r.seconds)
    out = {}
    for agent, secs in by_agent.items():
        secs_sorted = sorted(secs)
        n = len(secs_sorted)
        out[agent] = {
            "count": n,
            "min": secs_sorted[0],
            "max": secs_sorted[-1],
            "mean": sum(secs_sorted) / n,
            "p50": secs_sorted[n // 2],
        }
    return out
