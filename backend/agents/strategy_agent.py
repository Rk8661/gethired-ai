"""Strategy Agent — the USP. Never talks to the candidate.

After every scored answer it reads the latest performance plus the candidate
profile and decides how to adapt the *next* question: raise/lower difficulty,
ask a follow-up, or redirect to a stronger project / different topic. It records
each decision (``StrategyDecision``) to power the Interview Strategy Timeline and
writes the directive the Interview Agent consumes next.

It also advances ``current_question_index``  it is the last processing node
before routing, so the new index drives the loop-vs-finalize decision.
"""
from __future__ import annotations

from backend.config import round_for_index
from backend.core.llm_client import complete_json, safe_str
from backend.core.state import (
    Directive,
    InterviewState,
    StrategyDecision,
    agent_mode,
)
from backend.mocks.data import MOCK_QUESTIONS

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}

AGENT = "strategy_agent"


def _mock_decision(state: InterviewState) -> dict:
    """Derive a decision from the LAST scored answer, not a fixed script.

    This keeps the Strategy Timeline truthful in mock mode: it reacts to
    whatever the candidate actually typed (via feedback_agent's heuristic
    scores), the same shape of adaptation the llm path makes from a real
    model judgement.
    """
    if not state.history:
        return {
            "observation": "Interview is just starting.",
            "decision": "Begin at medium difficulty.",
            "directive": {"difficulty": "medium"},
        }

    last = state.history[-1]
    score_pct = (last.scores.total / 50) if last.scores else 0.5
    profile = state.candidate_profile
    best_project = profile.projects[0] if profile and profile.projects else None

    # Was this question already about the candidate's top project? (avoid
    # redirecting to a project we're literally already discussing.)
    redirect_text = MOCK_QUESTIONS.get(last.index, {}).get("redirect")
    already_on_project = bool(redirect_text) and last.question == redirect_text

    # A redirect only has effect if the NEXT question is still Technical and has
    # a project-redirect variant available — otherwise we'd log an adaptation
    # that can't actually happen (e.g. right as we move to the HM round).
    next_index = last.index + 1
    redirect_possible = (
        round_for_index(next_index) == "Technical"
        and MOCK_QUESTIONS.get(next_index, {}).get("redirect") is not None
    )

    if (
        last.round == "Technical"
        and score_pct < 0.5
        and best_project
        and redirect_possible
        and not already_on_project
    ):
        observation = (
            f"Scored {last.scores.total}/50 on {last.round} fundamentals — noticeable gaps. "
            f"Resume shows a strong, relevant project: '{best_project.title}'."
        )
        decision = (
            f"Stop drilling the weak area; redirect the next question to "
            f"'{best_project.title}' to assess real strength."
        )
        directive = {"difficulty": "medium", "focus_project": best_project.title}
    elif score_pct >= 0.8:
        observation = f"Scored {last.scores.total}/50 — excellent depth and clarity."
        decision = "Raise the difficulty and ask a probing follow-up."
        directive = {"difficulty": "hard", "follow_up": True}
    elif score_pct < 0.5:
        observation = f"Scored {last.scores.total}/50 — gaps in this area."
        decision = "Note this as a weak area and move on to a different topic."
        directive = {"difficulty": "easy"}
    else:
        observation = f"Scored {last.scores.total}/50 — solid, on-target answer."
        decision = "Hold steady and continue."
        directive = {"difficulty": "medium"}

    return {"observation": observation, "decision": decision, "directive": directive}


def _llm_decision(state: InterviewState) -> dict:
    last = state.history[-1]
    profile = state.candidate_profile.model_dump() if state.candidate_profile else {}
    scores = last.scores.model_dump() if last.scores else {}
    return complete_json(
        system=(
            "You are the strategy brain of an adaptive interviewer. Based on the "
            "candidate's latest answer and resume, decide how to adapt the NEXT "
            "question, like a human interviewer reading the room. You never speak "
            "to the candidate."
        ),
        user=(
            f"Just answered question #{last.index} ({last.round}).\n"
            f"Scores: {scores}\nMissing concepts: {last.missing_concepts}\n"
            f"Candidate profile: {profile}\n\n"
            "Return JSON: {\"observation\": \"...\", \"decision\": \"...\", "
            "\"directive\": {\"difficulty\": \"easy|medium|hard\", \"follow_up\": bool, "
            "\"focus_project\": str|null, \"focus_topic\": str|null}}"
        ),
    )


def run(state: InterviewState) -> dict:
    """LangGraph node: log a decision, set the next directive, advance the index."""
    answered_index = state.current_question_index
    if agent_mode(state, AGENT) == "llm" and state.history:
        d = _llm_decision(state)
    else:
        d = _mock_decision(state)

    decision = StrategyDecision(
        after_question=answered_index,
        observation=safe_str(d.get("observation"), "No observation."),
        decision=safe_str(d.get("decision"), "Continue at the same difficulty."),
    )
    raw_directive = d.get("directive") or {}
    difficulty = safe_str(raw_directive.get("difficulty"), "medium").strip().lower()
    if difficulty not in _VALID_DIFFICULTIES:
        difficulty = "medium"
    directive = Directive(
        difficulty=difficulty,
        follow_up=bool(raw_directive.get("follow_up", False)),
        focus_project=safe_str(raw_directive.get("focus_project")) or None,
        focus_topic=safe_str(raw_directive.get("focus_topic")) or None,
    )
    return {
        "strategy_log": [decision],
        "next_directive": directive,
        "current_question_index": answered_index + 1,
    }
