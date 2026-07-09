"""Rubrics and scripted resume/questions for ``mock`` mode.

Mock mode runs entirely offline (no API calls) but still *evaluates the
candidate's actual answer* via simple, deterministic heuristics (concept
keyword coverage + answer length) rather than returning a fixed score per
question. This keeps the demo honest: a blank or off-topic answer scores low,
a strong on-topic answer scores high, and the Strategy Agent's adaptive pivot
(e.g. redirecting a weak technical round to the candidate's strongest, most
relevant project) is driven by the real computed score, not a hardcoded script.
"""
from __future__ import annotations

AI_PROJECT_TITLE = "DocuMind — RAG Document Assistant"

# --- Resume Analyzer output -------------------------------------------------
MOCK_PROFILE: dict = {
    "name": "Aarav Sharma",
    "education": "B.Tech, Computer Science (2025)",
    "skills": ["Python", "PyTorch", "LangChain", "FastAPI", "SQL", "Git"],
    "projects": [
        {
            "title": AI_PROJECT_TITLE,
            "description": "A retrieval-augmented Q&A app over PDFs using LangChain + FAISS embeddings.",
            "tech": ["LangChain", "FAISS", "OpenAI", "Streamlit"],
            "relevance_score": 0.94,
        },
        {
            "title": "CampusBus Tracker",
            "description": "Real-time bus tracking app with a REST backend.",
            "tech": ["FastAPI", "PostgreSQL", "React"],
            "relevance_score": 0.61,
        },
    ],
    "certifications": ["DeepLearning.AI — LangChain for LLM Apps"],
}

# --- Interview Agent questions ----------------------------------------------
# Base question per index. A few indices have a "redirect" variant used when the
# Strategy Agent set ``focus_project`` on the directive (because the candidate
# scored poorly on the default question for that slot).
MOCK_QUESTIONS: dict[int, dict] = {
    0: {"base": "Tell me about yourself and what drew you to this role."},
    1: {"base": "What would you say is your biggest strength, and one area you're actively improving?"},
    2: {"base": "Given a directed graph, how would you detect whether it contains a cycle?"},
    3: {
        "base": "Let's talk databases — how would you design the schema for a ride-tracking app?",
        "redirect": f"Your resume highlights '{AI_PROJECT_TITLE}'. Walk me through how retrieval works in it — from a user question to the final answer.",
    },
    4: {"base": "In that RAG system, how did you decide on chunking and which embeddings to use?"},
    5: {"base": "Explain database normalization and when you would denormalize on purpose."},
    6: {"base": "Tell me about a time a project didn't go as planned and how you handled it."},
    7: {"base": "Why should we hire you over other freshers, and where do you see yourself in two years?"},
}

# --- Feedback Agent rubrics --------------------------------------------------
# Each entry: "concepts" (search_term -> display_label) the candidate's answer
# is checked against, "topic" (for weak/strong topic tagging), and a
# "reference_answer" used as the suggested answer for Technical questions.
# Behavioral (HR / Hiring Manager) questions have no concepts — they're judged
# on depth/effort only, and their suggested_answer improves the candidate's OWN
# wording rather than substituting a scripted ideal.
_BASE = "default"
_REDIRECT = "redirect"

MOCK_RUBRIC: dict[int, dict] = {
    0: {_BASE: {"concepts": [], "topic": "Communication"}},
    1: {_BASE: {"concepts": [], "topic": "Self-awareness"}},
    2: {
        _BASE: {
            "concepts": [
                ("dfs", "DFS"),
                ("depth-first", "DFS"),
                ("recursion stack", "Recursion stack / back-edge"),
                ("back edge", "Recursion stack / back-edge"),
                ("topological", "Topological Sort (Kahn's)"),
                ("kahn", "Topological Sort (Kahn's)"),
                ("visited", "Visited-state tracking"),
                ("cycle", "Cycle detection"),
            ],
            "topic": "Graphs",
            "reference_answer": (
                "Run DFS tracking nodes in the current recursion stack; a back-edge to "
                "an in-stack node means a cycle. Alternatively, Kahn's topological sort "
                "detects a cycle when not all nodes are processed."
            ),
        }
    },
    3: {
        _BASE: {
            "concepts": [
                ("primary key", "Primary Keys"),
                ("foreign key", "Foreign Keys"),
                ("normali", "Normalization"),
                ("index", "Indexing"),
                ("schema", "Schema design"),
                ("one-to-many", "Relationships"),
                ("many-to-many", "Relationships"),
            ],
            "topic": "DBMS",
            "reference_answer": (
                "Separate tables for riders, drivers, rides and locations; foreign keys "
                "link rides to riders/drivers, with indexes on frequently queried "
                "columns like ride status and timestamps."
            ),
        },
        _REDIRECT: {
            "concepts": [
                ("embed", "Embeddings"),
                ("vector", "Vector Search"),
                ("faiss", "FAISS"),
                ("retriev", "Retrieval"),
                ("chunk", "Chunking"),
                ("rag", "RAG"),
                ("langchain", "LangChain"),
                ("similarity", "Similarity search"),
            ],
            "topic": "RAG",
            "reference_answer": (
                "The query is embedded, FAISS retrieves the top-k similar chunks via "
                "vector similarity search, and those chunks are passed to the LLM as "
                "context to generate the final answer."
            ),
        },
    },
    4: {
        _BASE: {
            "concepts": [
                ("chunk", "Chunk overlap"),
                ("overlap", "Chunk overlap"),
                ("embed", "Embedding choice"),
                ("cost", "Cost/quality trade-off"),
                ("semantic", "Semantic chunking"),
            ],
            "topic": "RAG",
            "reference_answer": (
                "Use semantic or fixed-size chunking with overlap so context isn't cut "
                "mid-thought, and pick an embedding model based on the cost/quality "
                "trade-off for the domain."
            ),
        }
    },
    5: {
        _BASE: {
            "concepts": [
                ("normal form", "Normal forms (1NF–3NF)"),
                ("1nf", "Normal forms (1NF–3NF)"),
                ("2nf", "Normal forms (1NF–3NF)"),
                ("3nf", "Normal forms (1NF–3NF)"),
                ("index", "Indexing"),
                ("transaction", "Transactions"),
                ("denormal", "Denormalization"),
            ],
            "topic": "DBMS",
            "reference_answer": (
                "Normalize through 1NF–3NF to remove redundancy, then denormalize "
                "read-heavy paths deliberately and add indexes on frequent "
                "filter/join columns."
            ),
        }
    },
    6: {_BASE: {"concepts": [], "topic": "Ownership"}},
    7: {_BASE: {"concepts": [], "topic": "Confidence"}},
}


def get_rubric(index: int, question_text: str) -> dict:
    """Pick the rubric variant matching the question actually asked."""
    spec = MOCK_RUBRIC.get(index, {_BASE: {"concepts": [], "topic": None}})
    redirect = spec.get(_REDIRECT)
    if redirect and MOCK_QUESTIONS.get(index, {}).get(_REDIRECT) == question_text:
        return redirect
    return spec[_BASE]
