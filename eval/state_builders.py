"""Minimal InterviewState builders for direct (non-graph) agent invocation.

Mirrors the `_apply` reducer pattern in tests/test_pipeline.py: build the
smallest state that satisfies each agent's read requirements, call the agent
function directly (bypassing LangGraph entirely — none of these agents read
current_round, the full history, or any checkpoint/thread machinery), and
apply the returned partial-update dict by hand.
"""
from __future__ import annotations

from backend.core.state import (
    CandidateProfile,
    InterviewState,
    QuestionRecord,
    Scores,
)
from backend.mocks.data import MOCK_PROFILE

ACCUMULATING_CHANNELS = {"history", "strategy_log", "weak_topics", "strong_topics"}


def apply_update(state: InterviewState, update: dict) -> None:
    """Apply a node's partial-update dict, respecting operator.add channels."""
    for k, v in update.items():
        if k in ACCUMULATING_CHANNELS:
            setattr(state, k, getattr(state, k) + v)
        else:
            setattr(state, k, v)


def state_for_feedback_agent(job_role: str = "ML Engineer") -> InterviewState:
    """State with candidate_profile set, ready for feedback_agent.score_answer.
    Caller sets `current_question` per-fixture before scoring."""
    return InterviewState(
        job_role=job_role,
        candidate_profile=CandidateProfile(**MOCK_PROFILE),
        mode="llm",
    )


def state_with_weak_technical_history(round_name: str = "Technical", index: int = 3) -> InterviewState:
    """Simulates: candidate just answered a Technical question poorly, and the
    profile has one clearly-strongest project. Used to test whether the
    Strategy Agent's llm-mode decision redirects to that project — the
    headline USP behavior, exercised here without the mock-mode guardrails."""
    s = InterviewState(
        job_role="ML Engineer",
        candidate_profile=CandidateProfile(**MOCK_PROFILE),
        current_question_index=index + 1,
        mode="llm",
    )
    weak_scores = Scores(technical_accuracy=2, communication=4, confidence=3, problem_solving=2, relevance=2)  # total 13/50
    record = QuestionRecord(
        round=round_name,
        index=index,
        question="Let's talk databases — how would you design the schema for a ride-tracking app?",
        answer="Not totally sure, maybe a table with some fields.",
        scores=weak_scores,
        feedback="Weak.",
        suggested_answer="...",
        missing_concepts=["Schema design"],
    )
    apply_update(s, {"history": [record]})
    return s


def state_with_strong_history(round_name: str = "Technical", index: int = 2) -> InterviewState:
    """Simulates a strong answer, to test whether difficulty escalates."""
    s = InterviewState(
        job_role="ML Engineer",
        candidate_profile=CandidateProfile(**MOCK_PROFILE),
        current_question_index=index + 1,
        mode="llm",
    )
    strong_scores = Scores(technical_accuracy=9, communication=9, confidence=8, problem_solving=9, relevance=9)  # total 44/50
    record = QuestionRecord(
        round=round_name,
        index=index,
        question="Given a directed graph, how would you detect whether it contains a cycle?",
        answer="Run DFS tracking the recursion stack; a back-edge to an in-stack node means a cycle.",
        scores=strong_scores,
        feedback="Strong.",
        suggested_answer="...",
        missing_concepts=[],
    )
    apply_update(s, {"history": [record]})
    return s
