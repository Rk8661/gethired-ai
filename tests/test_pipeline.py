"""Unit + end-to-end tests for the GetHired AI interview pipeline (mock mode)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command

from backend.config import TOTAL_QUESTIONS, round_for_index
from backend.core.state import InterviewState, Scores
from backend.agents import resume_analyzer, interview_agent, strategy_agent, feedback_agent


# --- schema -----------------------------------------------------------------
def test_scores_total_and_bounds():
    s = Scores(technical_accuracy=10, communication=8, confidence=7, problem_solving=6, relevance=9)
    assert s.total == 40
    assert "total" in s.model_dump()
    with pytest.raises(Exception):
        Scores(technical_accuracy=11)


def test_round_mapping():
    rounds = [round_for_index(i) for i in range(TOTAL_QUESTIONS + 1)]
    assert rounds == ["HR", "HR", "Technical", "Technical", "Technical", "Technical",
                      "Hiring Manager", "Hiring Manager", "Complete"]


# --- agents (mock) ----------------------------------------------------------
def _apply(state: InterviewState, update: dict) -> None:
    lists = {"history", "strategy_log", "weak_topics", "strong_topics"}
    for k, v in update.items():
        setattr(state, k, getattr(state, k) + v if k in lists else v)


def test_resume_projects_ranked_by_relevance():
    s = InterviewState(job_role="ML Engineer", mode="mock")
    _apply(s, resume_analyzer.run(s))
    scores = [p.relevance_score for p in s.candidate_profile.projects]
    assert scores == sorted(scores, reverse=True)


def test_full_mock_interview_via_agents():
    s = InterviewState(job_role="ML Engineer", mode="mock")
    _apply(s, resume_analyzer.run(s))
    for i in range(TOTAL_QUESTIONS):
        _apply(s, interview_agent.run(s))
        _apply(s, feedback_agent.score_answer(s, f"answer {i}"))
        _apply(s, strategy_agent.run(s))
    _apply(s, feedback_agent.build_report(s))

    assert len(s.history) == TOTAL_QUESTIONS
    assert len(s.strategy_log) == TOTAL_QUESTIONS
    assert s.final_report["total_score"] <= s.final_report["max_score"] == 400
    # The signature adaptive pivot: after weak DSA (Q2), redirect to a project.
    assert "redirect" in s.strategy_log[2].decision.lower() or "DocuMind" in s.strategy_log[2].decision
    assert "DocuMind" in s.history[3].question


# --- graph (interrupt/resume) ----------------------------------------------
def test_graph_end_to_end():
    from backend.core.graph import GRAPH, pending_question

    cfg = {"configurable": {"thread_id": "test-graph"}}
    res = GRAPH.invoke({"job_role": "ML Engineer", "mode": "mock"}, cfg)
    assert pending_question(res)["index"] == 0
    for i in range(TOTAL_QUESTIONS):
        res = GRAPH.invoke(Command(resume=f"answer {i}"), cfg)
    state = GRAPH.get_state(cfg).values
    assert state["current_round"] == "Complete"
    assert state["final_recommendation"]
    assert len(state["strategy_log"]) == TOTAL_QUESTIONS


# --- API --------------------------------------------------------------------
def test_api_full_flow():
    from backend.main import app

    c = TestClient(app)
    start = c.post("/sessions", data={"job_role": "ML Engineer", "mode": "mock"})
    assert start.status_code == 200
    sid = start.json()["session_id"]

    last = None
    for i in range(TOTAL_QUESTIONS):
        r = c.post(f"/sessions/{sid}/answer", json={"answer": f"answer {i}"})
        assert r.status_code == 200
        last = r.json()
    assert last["complete"] is True
    assert last["final_report"]["total_score"] <= 400
    assert len(last["strategy_timeline"]) == TOTAL_QUESTIONS

    report = c.get(f"/sessions/{sid}/report")
    assert report.status_code == 200
    assert c.get("/sessions/does-not-exist").status_code == 404
