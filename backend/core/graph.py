"""LangGraph orchestration of the interview.

The interview is literally a state machine, so it's built as a compiled
``StateGraph`` with a checkpointer. Flow:

    START -> resume_analysis -> ask_question -> [evaluate: interrupt for the
    candidate's answer, then score it] -> strategize -> route

``route`` loops back to ``ask_question`` until all questions are answered, then
goes to ``finalize`` -> END. The ``evaluate`` node calls ``interrupt()`` so the
graph pauses for the answer; the API resumes it with ``Command(resume=answer)``.

Each node delegates to a Phase-3 agent function (pure, independently testable).
"""
from __future__ import annotations

from typing import Callable, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from backend.agents import (
    feedback_agent,
    interview_agent,
    resume_analyzer,
    strategy_agent,
)
from backend.config import TOTAL_QUESTIONS
from backend.core.state import InterviewState


def _resume_analysis(state: InterviewState) -> dict:
    return resume_analyzer.run(state, state.resume_text)


def _ask_question(state: InterviewState) -> dict:
    return interview_agent.run(state)


def _evaluate(state: InterviewState) -> dict:
    # Pause and surface the pending question; resumes with the answer string.
    cq = state.current_question
    answer = interrupt(
        {"question": cq.question, "round": cq.round, "index": cq.index}
    )
    return feedback_agent.score_answer(state, answer)


def _strategize(state: InterviewState) -> dict:
    return strategy_agent.run(state)


def _finalize(state: InterviewState) -> dict:
    return feedback_agent.build_report(state)


def _route(state: InterviewState) -> str:
    return "ask_question" if state.current_question_index < TOTAL_QUESTIONS else "finalize"


def build_graph():
    g = StateGraph(InterviewState)
    g.add_node("resume_analysis", _resume_analysis)
    g.add_node("ask_question", _ask_question)
    g.add_node("evaluate", _evaluate)
    g.add_node("strategize", _strategize)
    g.add_node("finalize", _finalize)

    g.add_edge(START, "resume_analysis")
    g.add_edge("resume_analysis", "ask_question")
    g.add_edge("ask_question", "evaluate")
    g.add_edge("evaluate", "strategize")
    g.add_conditional_edges(
        "strategize", _route, {"ask_question": "ask_question", "finalize": "finalize"}
    )
    g.add_edge("finalize", END)
    # Allow our pydantic state models through the msgpack checkpoint serializer.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("backend.core.state", "InterviewState"),
            ("backend.core.state", "CandidateProfile"),
            ("backend.core.state", "Project"),
            ("backend.core.state", "Scores"),
            ("backend.core.state", "QuestionRecord"),
            ("backend.core.state", "StrategyDecision"),
            ("backend.core.state", "Directive"),
        ]
    )
    return g.compile(checkpointer=MemorySaver(serde=serde))


# Single compiled graph shared across sessions (each session = its own thread_id).
GRAPH = build_graph()


def pending_question(result: dict) -> dict | None:
    """Extract the interrupt payload (the question awaiting an answer), if any."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


# User-facing labels for each node, surfaced via run_with_progress's callback so
# the UI can show what's actually happening during a slow llm-mode turn instead
# of a single static "loading" state.
STAGE_LABELS = {
    "resume_analysis": "Analyzing your resume…",
    "ask_question": "Preparing your next question…",
    "evaluate": "Scoring your answer…",
    "strategize": "Deciding what to ask next…",
    "finalize": "Building your final report…",
}


# stream_mode="updates" only signals when a node FINISHES, not when the next
# one starts. To make on_stage describe what's CURRENTLY running (not what
# just completed), each node's completion announces its known successor's
# label, and the very first node's label is announced eagerly by the caller.
_SUCCESSOR_STAGE = {
    "resume_analysis": STAGE_LABELS["ask_question"],
    "evaluate": STAGE_LABELS["strategize"],
    "strategize": STAGE_LABELS["ask_question"],  # also covers the "finalize" tail
}


def run_with_progress(
    payload,
    config: dict,
    on_stage: Optional[Callable[[str], None]] = None,
    eager_first: Optional[str] = None,
) -> dict:
    """Drive the graph to its next interrupt (or END), reporting stage labels.

    Equivalent to ``GRAPH.invoke(payload, config)`` but streams node-by-node so
    ``on_stage`` can be called with the label of the node currently running.
    ``eager_first`` is announced immediately (before the first node completes),
    since the caller knows which node a given payload starts with. Returns a
    dict shaped like ``invoke()``'s result (containing ``__interrupt__`` if the
    graph paused for an answer, or empty if it reached END).
    """
    if on_stage and eager_first:
        on_stage(eager_first)
    result: dict = {}
    for chunk in GRAPH.stream(payload, config, stream_mode="updates"):
        for node_name in chunk:
            successor = _SUCCESSOR_STAGE.get(node_name)
            if on_stage and successor:
                on_stage(successor)
        if "__interrupt__" in chunk:
            result = chunk
    return result


__all__ = ["GRAPH", "build_graph", "pending_question", "run_with_progress", "STAGE_LABELS"]
