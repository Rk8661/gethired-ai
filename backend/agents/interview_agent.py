"""Interview Agent — asks the next question in the current round's persona.

Reads the Strategy Agent's ``next_directive`` (difficulty, follow-up, project /
topic to focus on) and produces a single ``current_question`` for the candidate
to answer. It never scores — that's the Feedback Agent's job.
"""
from __future__ import annotations

from backend.config import round_for_index
from backend.core.llm_client import complete_json, safe_str
from backend.core.state import InterviewState, QuestionRecord, agent_mode
from backend.mocks.data import MOCK_QUESTIONS

AGENT = "interview_agent"

PERSONA = {
    "HR": "a warm HR interviewer assessing motivation, communication and culture fit",
    "Technical": "a technical interviewer assessing CS fundamentals and project depth",
    "Hiring Manager": "a hiring manager assessing ownership, impact and role fit",
}


def _mock_question(state: InterviewState, index: int) -> str:
    spec = MOCK_QUESTIONS.get(index, {"base": "Tell me more about your experience."})
    # If the Strategy Agent redirected us to a specific project, prefer the
    # project-focused variant when one exists.
    if state.next_directive.focus_project and "redirect" in spec:
        return spec["redirect"]
    return spec["base"]


def _llm_question(state: InterviewState, index: int, round_name: str) -> str:
    d = state.next_directive
    profile = state.candidate_profile.model_dump() if state.candidate_profile else {}
    data = complete_json(
        system=(
            f"You are {PERSONA.get(round_name, 'an interviewer')}. Ask ONE concise "
            "interview question. Do not include the answer."
        ),
        user=(
            f"Job role: {state.job_role}\nRound: {round_name}\n"
            f"Difficulty: {d.difficulty}\nFollow-up: {d.follow_up}\n"
            f"Focus project: {d.focus_project}\nFocus topic: {d.focus_topic}\n"
            f"Candidate profile: {profile}\n"
            "Return JSON: {\"question\": \"...\"}"
        ),
    )
    return safe_str(data.get("question"), "Tell me about a project you're proud of.")


def run(state: InterviewState) -> dict:
    """LangGraph node: set ``current_question`` (and the round) for this index."""
    index = state.current_question_index
    round_name = round_for_index(index)
    if agent_mode(state, AGENT) == "llm":
        question = _llm_question(state, index, round_name)
    else:
        question = _mock_question(state, index)
    record = QuestionRecord(round=round_name, index=index, question=question)
    return {"current_question": record, "current_round": round_name}
