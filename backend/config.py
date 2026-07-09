"""Central configuration: model, interview layout, round mapping.

Everything that defines the *shape* of an interview (how many questions, which
round each question belongs to) lives here so the agents and graph stay free of
magic numbers.
"""
from __future__ import annotations

# --- LLM provider (Fireworks AI, OpenAI-compatible endpoint) ----------------
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/gpt-oss-20b"

# --- Interview layout -------------------------------------------------------
# 8 questions total: HR(2) -> Technical(4) -> Hiring Manager(2). The round
# ORDER is fixed; only difficulty / follow-ups / topic selection adapt.
ROUND_LAYOUT: list[str] = (
    ["HR"] * 2 + ["Technical"] * 4 + ["Hiring Manager"] * 2
)
TOTAL_QUESTIONS = len(ROUND_LAYOUT)  # 8

# Scoring: 5 dimensions x 10 marks = 50 per question, x 8 = 400 max.
MAX_PER_QUESTION = 50
MAX_TOTAL = MAX_PER_QUESTION * TOTAL_QUESTIONS  # 400


def round_for_index(index: int) -> str:
    """Return the interviewer round for a 0-based question index.

    Indices beyond the last question map to "Complete" (the interview is over).
    """
    if index < 0:
        raise ValueError(f"question index must be >= 0, got {index}")
    if index >= TOTAL_QUESTIONS:
        return "Complete"
    return ROUND_LAYOUT[index]
