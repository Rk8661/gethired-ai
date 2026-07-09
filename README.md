# GetHired AI

An adaptive AI interview coach for freshers. It reads a candidate's resume, runs a structured 8 question interview across three rounds (HR → Technical → Hiring Manager), and continuously adapts its questioning  difficulty, follow ups, which project to dig into  based on how the candidate is actually performing. At the end it produces a recruiter style report that explains not just the score, but *why* the interview unfolded the way it did.

Built for the AMD Developer Hackathon (ACT II), "Unicorn" track.

## Why

Most interview prep tools ask a fixed bank of questions in a fixed order, regardless of who's answering. They don't read the resume, don't adjust to how the candidate is doing, and don't explain *why* a question was asked or a score given. GetHired AI's differentiator isn't "uses an LLM"  it's that a dedicated **Strategy Agent** reads the room like a human interviewer would: if you stumble on data structures but your resume shows a strong ML project, it redirects the next question there instead of continuing to drill a weak spot. Every one of those decisions is logged and shown back to you as an **Interview Strategy Timeline**.

## Architecture

Four cooperating agents, orchestrated as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine (the interview genuinely *is* a state machine — the Strategy Agent's output drives a conditional edge, not just a prompt chain):

```
Resume Analyzer → Interview Agent ⇄ Strategy Agent → Feedback Agent
                         │                                  │
                    (asks Q, awaits            (scores the answer, drives
                     candidate's answer)        the final report)
```

- **Resume Analyzer** - parses the resume into a structured `CandidateProfile` (skills, projects ranked by role-relevance, education, certifications). Every other agent reads from this.
- **Interview Agent** - asks the next question in the current round's persona (HR / Technical / Hiring Manager), honoring whatever the Strategy Agent just directed (difficulty, follow-up, a specific project or topic to focus on).
- **Strategy Agent** - the USP. Never talks to the candidate. After every scored answer it decides how to adapt the *next* question and logs why, which powers the Strategy Timeline.
- **Feedback Agent** - scores each answer on 5 dimensions (technical accuracy, communication, confidence, problem solving, relevance - 50 points/question, 400 total), and at the end aggregates the whole interview into a final report with weak/strong topic clusters and a hiring recommendation.

Every agent runs in one of two modes, selectable per-session (and per-agent, via `agent_modes`):

- **`mock`** — fully offline, no API calls, instant. Not scripted/canned — it evaluates whatever the candidate actually typed via deterministic heuristics (keyword/concept coverage + answer depth), so a blank or off-topic answer genuinely scores low.
- **`llm`** — real calls to [Fireworks AI](https://fireworks.ai)'s OpenAI - compatible endpoint, model `accounts/fireworks/models/gpt-oss-20b`. Slower (each call is 3–50s; a full turn chains 3 sequential calls) and costs API credits, but is the real thing.

Shared state (`InterviewState`, a Pydantic model in `backend/core/state.py`) flows through every node - agents never call each other directly.

## Project structure

```
backend/
  agents/            Resume Analyzer, Interview Agent, Strategy Agent, Feedback Agent
  core/
    state.py          InterviewState + all Pydantic models  the shared contract every agent reads/writes
    llm_client.py      Single entry point for Fireworks calls (chat, complete_json, mode helpers)
    graph.py           LangGraph StateGraph wiring the agents together, with interrupt/resume for Q&A turns
  mocks/data.py        Rubrics, questions, and the scripted candidate profile used in mock mode
  config.py            Round layout, scoring constants
  main.py              FastAPI app (session lifecycle, turn-by-turn API)
frontend/              React (Vite) UI  upload, interview loop, strategy timeline, final report
eval/                  'LLM mode' evaluation harness (see below)  separate from the fast test suite
tests/                 Fast pytest suite (mock mode + pure logic only, no network calls, <1s)
```

## Setup

**Backend**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in FIREWORKS_API_KEY
```

Get a key at [fireworks.ai](https://fireworks.ai) - only required for `llm` mode; `mock` mode needs no key at all.

**Frontend**

```bash
cd frontend
npm install
```

## Running

Two terminals, both left running:

```bash
# Terminal 1 — backend (http://localhost:8000)
source venv/bin/activate
uvicorn backend.main:app --reload

# Terminal 2 — frontend (http://localhost:5173)
cd frontend
npm run dev
```

Open `http://localhost:5173`, pick a role, choose `mock` (instant) or `llm` (real model), optionally attach a resume PDF, and start the interview.

### API

| Endpoint | What it does |
|---|---|
| `POST /sessions` | Start a session (resume + job role + mode) → resume analysis + first question |
| `POST /sessions/{id}/answer` | Submit an answer → score + next question, or the final report on Q8 |
| `GET /sessions/{id}/progress` | Poll the current pipeline stage during a slow `llm`-mode turn |
| `GET /sessions/{id}/report` | Final report + strategy timeline + full history |
| `GET /sessions/{id}` | Full session state (debugging) |
| `GET /health` | Liveness check |

## Testing

```bash
pytest tests/          # fast: mock-mode pipeline + pure eval-metrics logic, no network calls, <1s
```

### LLM evaluation harness

`tests/` never calls the real model. `eval/` does — it's a separate, opt-in CLI that scores `llm`-mode output quality: does the Feedback Agent's scoring stay monotonic (blank < weak < strong answers), does the Strategy Agent actually redirect to the candidate's strongest project after a weak technical answer, does it escalate difficulty after a strong one, are generated questions well-formed. This is the harness that verifies the project's core adaptive-behavior claim holds against the *real* model, not just the scripted mock demo.

```bash
# Start small — question 3 has both a redirect-eligible rubric and a strong-project profile
python -m eval.run_eval --questions 3 --repeats 1

# Full sweep (all 8 questions) — ~8-17 minutes, well under $0.20 in API credits
python -m eval.run_eval --questions 0,1,2,3,4,5,6,7 --repeats 3 --strict
```

Reports are written to `eval/reports/` (gitignored) as both JSON and a skimmable Markdown summary.

## Tech stack

- **Backend**: Python, FastAPI, LangGraph, Pydantic
- **Frontend**: React (Vite)
- **LLM**: Fireworks AI, `gpt-oss-20b` (open-weight, Apache 2.0)

## Security note

`.env` (your real credentials) is gitignored. `.env.example` is a committed template — never put a real key in it.
