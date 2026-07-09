"""Shared state for the GetHired AI interview pipeline.

Every agent reads from and writes to one central ``InterviewState`` object
instead of passing ad-hoc JSON between LangGraph nodes. The accumulating list
channels (``history``, ``strategy_log``, ``weak_topics``, ``strong_topics``)
use ``operator.add`` reducers so a node can return *just its new items* and
LangGraph appends them rather than overwriting the whole list.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from backend.config import MAX_PER_QUESTION


# --- Candidate profile (produced by the Resume Analyzer) --------------------
class Project(BaseModel):
    title: str
    description: str = ""
    tech: list[str] = Field(default_factory=list)
    # Higher = more relevant to the selected job role. Drives which project the
    # Strategy Agent redirects to.
    relevance_score: float = 0.0


class CandidateProfile(BaseModel):
    name: str = "Candidate"
    education: str = ""
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


# --- Per-answer scoring -----------------------------------------------------
class Scores(BaseModel):
    technical_accuracy: int = Field(0, ge=0, le=10)
    communication: int = Field(0, ge=0, le=10)
    confidence: int = Field(0, ge=0, le=10)
    problem_solving: int = Field(0, ge=0, le=10)
    relevance: int = Field(0, ge=0, le=10)

    @computed_field  # serialized into API/JSON output
    @property
    def total(self) -> int:
        return (
            self.technical_accuracy
            + self.communication
            + self.confidence
            + self.problem_solving
            + self.relevance
        )


# --- Transcript records -----------------------------------------------------
class QuestionRecord(BaseModel):
    round: str
    index: int
    question: str
    answer: Optional[str] = None
    scores: Optional[Scores] = None
    feedback: Optional[str] = None
    # HR/HM: an improved version of the candidate's own answer. Technical: a
    # strong model answer.
    suggested_answer: Optional[str] = None
    missing_concepts: list[str] = Field(default_factory=list)


class StrategyDecision(BaseModel):
    """A single Strategy Agent decision — powers the Interview Strategy Timeline."""
    after_question: int
    observation: str
    decision: str


class Directive(BaseModel):
    """Strategy Agent -> Interview Agent instruction for the *next* question."""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    follow_up: bool = False
    focus_project: Optional[str] = None
    focus_topic: Optional[str] = None


# --- The one central object -------------------------------------------------
class InterviewState(BaseModel):
    job_role: str = ""
    # Raw resume text, set at session creation; consumed by the Resume Analyzer
    # in llm mode (ignored in mock mode).
    resume_text: str = ""
    candidate_profile: Optional[CandidateProfile] = None
    current_round: Literal["HR", "Technical", "Hiring Manager", "Complete"] = "HR"
    current_question_index: int = 0
    next_directive: Directive = Field(default_factory=Directive)
    # The question currently awaiting an answer (overwrite channel). Held here
    # until the answer arrives, then folded into a complete `history` record.
    current_question: Optional[QuestionRecord] = None

    # Accumulating channels (operator.add reducers -> nodes return new items).
    history: Annotated[list[QuestionRecord], operator.add] = Field(default_factory=list)
    strategy_log: Annotated[list[StrategyDecision], operator.add] = Field(default_factory=list)
    weak_topics: Annotated[list[str], operator.add] = Field(default_factory=list)
    strong_topics: Annotated[list[str], operator.add] = Field(default_factory=list)

    final_recommendation: Optional[str] = None
    final_report: Optional[dict] = None

    # Global mock/llm toggle, with an optional per-agent override.
    mode: Literal["mock", "llm"] = "mock"
    agent_modes: dict[str, str] = Field(default_factory=dict)


def agent_mode(state: InterviewState, agent_name: str) -> str:
    """Resolve the effective mode for an agent: per-agent override else global."""
    return state.agent_modes.get(agent_name, state.mode)


__all__ = [
    "Project",
    "CandidateProfile",
    "Scores",
    "QuestionRecord",
    "StrategyDecision",
    "Directive",
    "InterviewState",
    "agent_mode",
    "MAX_PER_QUESTION",
]
