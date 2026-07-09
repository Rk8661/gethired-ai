"""FastAPI app driving the interview turn-by-turn.

A "session" is just a LangGraph ``thread_id``; the compiled graph's checkpointer
holds that session's state. The API drives one question per request:

    POST /sessions               -> resume analysis + first question
    POST /sessions/{id}/answer   -> score prev answer + next question | final report
    GET  /sessions/{id}/report   -> final report + strategy timeline + history
    GET  /sessions/{id}          -> full InterviewState (debug/UI)
"""
from __future__ import annotations

import asyncio
import io
import uuid
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel
from pypdf import PdfReader

from backend.config import TOTAL_QUESTIONS
from backend.core.graph import GRAPH, STAGE_LABELS, pending_question, run_with_progress

app = FastAPI(title="GetHired AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon dev; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Known session ids (thread_ids live in the graph checkpointer).
_SESSIONS: set[str] = set()
# session_id -> current stage label, while a /answer call is in flight. Polled
# by the frontend so a slow llm-mode turn shows what's happening instead of a
# single static "loading" state.
_PROGRESS: dict[str, str] = {}


class AnswerIn(BaseModel):
    answer: str


# --- helpers ----------------------------------------------------------------


def _cfg(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}

def _dump(obj: Any) -> Any:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _require(session_id: str) -> None:
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")


def _state(session_id: str) -> dict:
    return GRAPH.get_state(_cfg(session_id)).values


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _last_evaluation(state: dict) -> Optional[dict]:
    history = state.get("history") or []
    if not history:
        return None
    r = _dump(history[-1])
    return {
        "index": r["index"],
        "round": r["round"],
        "question": r["question"],
        "scores": r["scores"],
        "feedback": r["feedback"],
        "suggested_answer": r["suggested_answer"],
        "missing_concepts": r["missing_concepts"],
    }


def _last_strategy(state: dict) -> Optional[dict]:
    log = state.get("strategy_log") or []
    return _dump(log[-1]) if log else None


# --- endpoints --------------------------------------------------------------
@app.post("/sessions")
async def create_session(
    job_role: str = Form(...),
    mode: str = Form("mock"),
    resume: Optional[UploadFile] = File(None),
):
    session_id = uuid.uuid4().hex
    resume_text = _extract_pdf(await resume.read()) if resume is not None else ""

    # GRAPH.invoke is synchronous and, in llm mode, makes blocking HTTP calls to
    # Fireworks that can take several seconds — run it off the event loop so it
    # doesn't freeze the whole server for other in-flight requests.
    result = await asyncio.to_thread(
        GRAPH.invoke,
        {"job_role": job_role, "mode": mode, "resume_text": resume_text},
        _cfg(session_id),
    )
    _SESSIONS.add(session_id)
    state = _state(session_id)
    return {
        "session_id": session_id,
        "candidate_profile": _dump(state.get("candidate_profile")),
        "question": pending_question(result),
        "total_questions": TOTAL_QUESTIONS,
    }


@app.post("/sessions/{session_id}/answer")
async def submit_answer(session_id: str, body: AnswerIn):
    _require(session_id)
    try:
        result = await asyncio.to_thread(
            run_with_progress,
            Command(resume=body.answer),
            _cfg(session_id),
            lambda label: _PROGRESS.__setitem__(session_id, label),
            STAGE_LABELS["evaluate"],
        )
    finally:
        _PROGRESS.pop(session_id, None)
    state = _state(session_id)

    question = pending_question(result)
    payload: dict = {
        "evaluation": _last_evaluation(state),
        "strategy_decision": _last_strategy(state),
        "complete": question is None,
    }
    if question is not None:
        payload["question"] = question
    else:
        payload["final_report"] = state.get("final_report")
        payload["strategy_timeline"] = [_dump(d) for d in state.get("strategy_log", [])]
    return payload


@app.get("/sessions/{session_id}/progress")
async def get_progress(session_id: str):
    _require(session_id)
    return {"stage": _PROGRESS.get(session_id)}


@app.get("/sessions/{session_id}/report")
async def get_report(session_id: str):
    _require(session_id)
    state = _state(session_id)
    if not state.get("final_report"):
        raise HTTPException(status_code=409, detail="interview not complete")
    return {
        "final_report": state["final_report"],
        "final_recommendation": state.get("final_recommendation"),
        "strategy_timeline": [_dump(d) for d in state.get("strategy_log", [])],
        "history": [_dump(r) for r in state.get("history", [])],
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    _require(session_id)
    state = _state(session_id)
    return {
        "job_role": state.get("job_role"),
        "current_round": state.get("current_round"),
        "current_question_index": state.get("current_question_index"),
        "candidate_profile": _dump(state.get("candidate_profile")),
        "history": [_dump(r) for r in state.get("history", [])],
        "strategy_log": [_dump(d) for d in state.get("strategy_log", [])],
        "weak_topics": state.get("weak_topics", []),
        "strong_topics": state.get("strong_topics", []),
        "final_recommendation": state.get("final_recommendation"),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
