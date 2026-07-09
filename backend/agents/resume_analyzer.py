"""Resume Analyzer — turns raw resume text into a structured CandidateProfile.

Projects are ranked by role-relevance so downstream agents (especially the
Strategy Agent) can redirect toward a candidate's strongest, most relevant work.
"""
from __future__ import annotations

from backend.core.llm_client import complete_json, safe_float, safe_str, safe_str_list
from backend.core.state import CandidateProfile, InterviewState, Project, agent_mode
from backend.mocks.data import MOCK_PROFILE

AGENT = "resume_analyzer"


def _rank_projects(profile: CandidateProfile) -> CandidateProfile:
    profile.projects.sort(key=lambda p: p.relevance_score, reverse=True)
    return profile


def _mock(state: InterviewState) -> CandidateProfile:
    return CandidateProfile(**MOCK_PROFILE)


def _llm(state: InterviewState, resume_text: str) -> CandidateProfile:
    data = complete_json(
        system=(
            "You are a resume analyzer for an interview-prep system. Extract a "
            "structured candidate profile. For each project, set relevance_score "
            "in [0,1] for how relevant it is to the target job role."
        ),
        user=(
            f"Target job role: {state.job_role}\n\nResume text:\n{resume_text}\n\n"
            "Return JSON with keys: name, education, skills (list), "
            "projects (list of {title, description, tech (list), relevance_score}), "
            "certifications (list)."
        ),
    )
    projects = []
    for p in data.get("projects") or []:
        if not isinstance(p, dict):
            continue
        projects.append(
            Project(
                title=safe_str(p.get("title"), "Untitled project"),
                description=safe_str(p.get("description"), ""),
                tech=safe_str_list(p.get("tech")),
                relevance_score=safe_float(p.get("relevance_score"), default=0.5),
            )
        )
    return CandidateProfile(
        name=safe_str(data.get("name"), "Candidate"),
        education=safe_str(data.get("education"), ""),
        skills=safe_str_list(data.get("skills")),
        projects=projects,
        certifications=safe_str_list(data.get("certifications")),
    )


def run(state: InterviewState, resume_text: str = "") -> dict:
    """LangGraph node: produce the candidate profile.

    Returns a partial state update (``candidate_profile``).
    """
    if agent_mode(state, AGENT) == "llm":
        profile = _llm(state, resume_text)
    else:
        profile = _mock(state)
    return {"candidate_profile": _rank_projects(profile)}
