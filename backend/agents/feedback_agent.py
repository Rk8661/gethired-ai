"""Feedback Agent — scores each answer and builds the final report.

Two entry points:
- ``score_answer``  runs after every answer: scores it (5 dims), writes feedback,
  a suggested answer, and named missing concepts, and contributes weak/strong
  topics. It folds the now-complete question record into ``history``.
- ``build_report`` runs once at the end: aggregates patterns across the whole
  interview into a recruiter-style report and hiring recommendation.
"""
from __future__ import annotations

from backend.config import MAX_PER_QUESTION, MAX_TOTAL
from backend.core.llm_client import complete_json, safe_int, safe_str, safe_str_list
from backend.core.state import InterviewState, QuestionRecord, Scores, agent_mode
from backend.mocks.data import get_rubric

AGENT = "feedback_agent"

_DIM_KEYS = ["technical_accuracy", "communication", "confidence", "problem_solving", "relevance"]


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        seen.setdefault(it, None)
    return list(seen.keys())


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _to10(x: float) -> int:
    return int(round(_clip01(x) * 10))


# --- per-answer scoring (mock mode: deterministic heuristics, not LLM) ------
def _mock_eval(index: int, question: str, answer: str) -> dict:
    """Score the candidate's actual answer against a small rubric.

    Technical questions are scored on keyword/concept coverage; behavioral
    (HR / Hiring Manager) questions have no fixed concepts and are scored on
    depth/effort, since there's no single "correct" answer to match against.
    """
    rubric = get_rubric(index, question)
    concepts = rubric.get("concepts", [])
    topic = rubric.get("topic")
    reference = rubric.get("reference_answer", "")

    text = (answer or "").strip()
    word_count = len(text.split())

    if word_count == 0:
        return {
            "scores": {k: 0 for k in _DIM_KEYS},
            "feedback": "No answer was provided — this question scored zero across all dimensions.",
            "suggested_answer": reference or "Try answering with at least a few sentences covering the key idea.",
            "missing_concepts": _dedupe([label for _, label in concepts])[:4],
            "strong_topics": [],
            "weak_topics": [topic] if topic else [],
        }

    depth = _clip01(word_count / 40)  # detail signal, saturates around 40 words
    effort = _clip01(0.25 + word_count / 60)  # baseline credit for attempting, saturates ~45 words
    lower = text.lower()

    if concepts:
        all_labels = _dedupe([label for _, label in concepts])
        matched_labels = _dedupe([label for kw, label in concepts if kw in lower])
        missing_labels = [lbl for lbl in all_labels if lbl not in matched_labels]
        coverage = len(matched_labels) / len(all_labels)

        scores = {
            "technical_accuracy": _to10(coverage),
            "problem_solving": _to10(0.6 * coverage + 0.4 * depth),
            "relevance": _to10(0.7 * coverage + 0.3 * depth),
            "communication": _to10(0.5 * depth + 0.5 * effort),
            "confidence": _to10(0.4 * depth + 0.6 * effort),
        }
        if coverage >= 0.67:
            feedback = f"Strong answer — you clearly covered {', '.join(matched_labels)}."
        elif coverage >= 0.34:
            feedback = (
                f"Partial answer — you covered {', '.join(matched_labels) or 'the basics'}, "
                f"but missed {', '.join(missing_labels[:3])}."
            )
        else:
            feedback = f"Your answer didn't address the core ideas here ({', '.join(missing_labels[:3]) or 'see suggested answer'})."
        return {
            "scores": scores,
            "feedback": feedback,
            "suggested_answer": reference,
            "missing_concepts": missing_labels[:4] if coverage < 0.9 else [],
            "strong_topics": [topic] if topic and coverage >= 0.67 else [],
            "weak_topics": [topic] if topic and coverage < 0.34 else [],
        }

    # Behavioral question: no fixed concepts, judge on depth/effort.
    scores = {
        "technical_accuracy": _to10(0.5 + 0.3 * depth),
        "problem_solving": _to10(0.4 + 0.3 * depth),
        "relevance": _to10(0.5 + 0.3 * depth),
        "communication": _to10(0.5 * depth + 0.5 * effort),
        "confidence": _to10(0.4 * depth + 0.6 * effort),
    }
    if depth >= 0.6:
        feedback = "Clear, detailed answer with good structure."
        suggested = f"{text} — consider tightening this into a crisp 30-40 second version."
    elif depth >= 0.25:
        feedback = "Reasonable answer — a bit more detail or a concrete example would help."
        suggested = f"{text} — add a specific example with a measurable outcome."
    else:
        feedback = "Too brief to fully assess — try expanding with a specific example."
        suggested = f"{text} — expand significantly with a specific example and outcome."
    return {
        "scores": scores,
        "feedback": feedback,
        "suggested_answer": suggested,
        "missing_concepts": [],
        "strong_topics": [topic] if topic and depth >= 0.6 else [],
        "weak_topics": [topic] if topic and depth < 0.25 else [],
    }


def _llm_eval(state: InterviewState, question: str, answer: str, round_name: str) -> dict:
    technical = round_name == "Technical"
    suggestion_rule = (
        "suggested_answer = a strong model answer."
        if technical
        else "suggested_answer = an improved version of the candidate's OWN answer (do not invent a scripted ideal)."
    )
    return complete_json(
        system=(
            "You are an interview evaluator. Score the answer on five dimensions "
            "(technical_accuracy, communication, confidence, problem_solving, "
            "relevance), each an integer 0-10. " + suggestion_rule
        ),
        user=(
            f"Round: {round_name}\nQuestion: {question}\nAnswer: {answer}\n\n"
            "Return JSON: {\"scores\": {five keys}, \"feedback\": \"...\", "
            "\"suggested_answer\": \"...\", \"missing_concepts\": [...], "
            "\"strong_topics\": [...], \"weak_topics\": [...]}"
        ),
    )


def score_answer(state: InterviewState, answer: str) -> dict:
    """Score the answer to ``state.current_question`` and append to history."""
    cq = state.current_question
    if cq is None:
        raise RuntimeError("score_answer called with no current_question")

    if agent_mode(state, AGENT) == "llm":
        ev = _llm_eval(state, cq.question, answer, cq.round)
    else:
        ev = _mock_eval(cq.index, cq.question, answer)

    raw_scores = ev.get("scores") or {}
    record = QuestionRecord(
        round=cq.round,
        index=cq.index,
        question=cq.question,
        answer=answer,
        scores=Scores(
            technical_accuracy=safe_int(raw_scores.get("technical_accuracy")),
            communication=safe_int(raw_scores.get("communication")),
            confidence=safe_int(raw_scores.get("confidence")),
            problem_solving=safe_int(raw_scores.get("problem_solving")),
            relevance=safe_int(raw_scores.get("relevance")),
        ),
        feedback=safe_str(ev.get("feedback"), "No feedback generated."),
        suggested_answer=safe_str(ev.get("suggested_answer"), ""),
        missing_concepts=safe_str_list(ev.get("missing_concepts")),
    )
    return {
        "history": [record],
        "weak_topics": safe_str_list(ev.get("weak_topics")),
        "strong_topics": safe_str_list(ev.get("strong_topics")),
    }


# --- final report -----------------------------------------------------------
def _recommendation(total: int) -> str:
    pct = total / MAX_TOTAL
    if pct >= 0.8:
        return "Strong Hire"
    if pct >= 0.65:
        return "Hire"
    if pct >= 0.5:
        return "Lean Hire / Borderline"
    return "No Hire — needs more preparation"


def build_report(state: InterviewState) -> dict:
    """Aggregate the whole interview into a final report + recommendation."""
    total = sum(r.scores.total for r in state.history if r.scores)
    by_round: dict[str, int] = {}
    for r in state.history:
        if r.scores:
            by_round[r.round] = by_round.get(r.round, 0) + r.scores.total

    missing = _dedupe([c for r in state.history for c in r.missing_concepts])
    weak = _dedupe(state.weak_topics)
    strong = _dedupe(state.strong_topics)
    recommendation = _recommendation(total)

    report = {
        "total_score": total,
        "max_score": MAX_TOTAL,
        "per_question_max": MAX_PER_QUESTION,
        "scores_by_round": by_round,
        "strong_topics": strong,
        "weak_topics": weak,
        "concepts_to_study": missing,
        "recommendation": recommendation,
        "summary": (
            f"Scored {total}/{MAX_TOTAL}. Strengths: {', '.join(strong) or 'n/a'}. "
            f"Focus areas: {', '.join(weak) or 'n/a'}."
        ),
        "per_question": [
            {
                "index": r.index,
                "round": r.round,
                "question": r.question,
                "score": r.scores.total if r.scores else 0,
                "feedback": r.feedback,
                "suggested_answer": r.suggested_answer,
                "missing_concepts": r.missing_concepts,
            }
            for r in state.history
        ],
    }
    return {
        "final_report": report,
        "final_recommendation": recommendation,
        "current_round": "Complete",
    }
