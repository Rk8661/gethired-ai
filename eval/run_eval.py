"""CLI driver: run llm-mode agents against synthetic fixtures, score them with
eval/metrics.py, and write a report.

Costs real Fireworks API credits and takes minutes (calls run 3-50s each with
high variance). NEVER run as part of the fast test suite — invoke explicitly:

    python -m eval.run_eval --questions 3 --repeats 1

Start with a single question (--questions 3, which has both a "redirect"
rubric variant and feeds the strategy redirect check) before running the
full 8-question sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.agents import feedback_agent, interview_agent, strategy_agent
from backend.config import round_for_index
from backend.core.state import CandidateProfile, Directive, InterviewState, QuestionRecord
from backend.mocks.data import MOCK_PROFILE
from eval import metrics
from eval.fixtures import Fixture, all_fixtures
from eval.state_builders import state_for_feedback_agent, state_with_strong_history, state_with_weak_technical_history

REPORTS_DIR = Path(__file__).parent / "reports"
REDIRECT_PROJECT_HINT = "DocuMind"  # substring of MOCK_PROFILE's top-ranked project title


def _timed_call(fn, *args, **kwargs):
    start = time.monotonic()
    result = fn(*args, **kwargs)
    return result, time.monotonic() - start


def _question_record(fx: Fixture) -> QuestionRecord:
    return QuestionRecord(round=fx.round_name, index=fx.index, question=fx.question)


def _coerce_directive(raw: dict) -> Directive:
    """Same coercion strategy_agent.run() itself applies, reproduced here since
    we call _llm_decision directly (see eval_strategy_agent's docstring)."""
    diff = str(raw.get("difficulty", "medium")).strip().lower()
    if diff not in ("easy", "medium", "hard"):
        diff = "medium"
    return Directive(
        difficulty=diff,
        follow_up=bool(raw.get("follow_up", False)),
        focus_project=raw.get("focus_project") or None,
        focus_topic=raw.get("focus_topic") or None,
    )


# --- per-agent eval runners ---------------------------------------------------
def eval_feedback_agent(fixtures: list[Fixture], repeats: int) -> dict:
    """Runs feedback_agent.score_answer in llm mode for each fixture."""
    results = []
    latency: list[metrics.LatencyRecord] = []
    for fx in fixtures:
        for rep in range(repeats):
            state = state_for_feedback_agent()
            state.current_question_index = fx.index
            state.current_question = _question_record(fx)
            update, secs = _timed_call(feedback_agent.score_answer, state, fx.answer)
            latency.append(metrics.LatencyRecord("feedback_agent", f"{fx.index}:{fx.kind}", secs))
            record = update["history"][0]
            checks = [metrics.check_scores_bounds(record.scores)]
            results.append(
                {
                    "index": fx.index,
                    "kind": fx.kind,
                    "round": fx.round_name,
                    "scores": record.scores.model_dump(),
                    "feedback": record.feedback,
                    "missing_concepts": record.missing_concepts,
                    "latency_s": round(secs, 2),
                    "repeat": rep,
                    "checks": [metrics.to_dict(c) for c in checks],
                }
            )

    # Monotonicity per index, averaged across repeats.
    monotonicity = []
    by_index: dict[int, dict[str, list[int]]] = {}
    for r in results:
        by_index.setdefault(r["index"], {}).setdefault(r["kind"], []).append(r["scores"]["total"])
    for idx, kinds in sorted(by_index.items()):
        if all(k in kinds for k in ("blank", "weak", "strong")):
            avg = {k: sum(v) / len(v) for k, v in kinds.items()}
            c = metrics.check_monotonic_scores(avg["blank"], avg["weak"], avg["strong"])
            monotonicity.append({"index": idx, **metrics.to_dict(c)})
    return {"per_fixture": results, "monotonicity": monotonicity, "latency": latency}


def eval_strategy_agent(repeats: int) -> dict:
    """Runs strategy_agent._llm_decision against hand-built weak/strong states.

    Calls _llm_decision directly rather than the public run(): run() silently
    falls back to mock behavior if state.history is empty (`if ... and
    state.history:`), and also does index-advancement bookkeeping unrelated to
    decision quality — already covered by tests/test_pipeline.py in mock mode.
    Calling _llm_decision isolates "did the model decide well" from that
    wrapper logic.
    """
    results = []
    latency: list[metrics.LatencyRecord] = []
    for rep in range(repeats):
        weak_state = state_with_weak_technical_history()
        raw, secs = _timed_call(strategy_agent._llm_decision, weak_state)
        latency.append(metrics.LatencyRecord("strategy_agent", f"weak_technical:{rep}", secs))
        raw_directive = raw.get("directive") or {}
        directive = _coerce_directive(raw_directive)
        checks = [
            metrics.check_redirect_on_weak_score(
                directive, expected_project_title=REDIRECT_PROJECT_HINT, score_total=weak_state.history[-1].scores.total
            ),
            metrics.check_raw_difficulty_valid(raw_directive.get("difficulty")),
        ]
        results.append(
            {"scenario": "weak_technical", "repeat": rep, "raw": raw, "latency_s": round(secs, 2), "checks": [metrics.to_dict(c) for c in checks]}
        )

        strong_state = state_with_strong_history()
        raw2, secs2 = _timed_call(strategy_agent._llm_decision, strong_state)
        latency.append(metrics.LatencyRecord("strategy_agent", f"strong_technical:{rep}", secs2))
        raw_directive2 = raw2.get("directive") or {}
        directive2 = _coerce_directive(raw_directive2)
        checks2 = [
            metrics.check_difficulty_escalation(directive2, score_total=strong_state.history[-1].scores.total),
            metrics.check_raw_difficulty_valid(raw_directive2.get("difficulty")),
        ]
        results.append(
            {"scenario": "strong_technical", "repeat": rep, "raw": raw2, "latency_s": round(secs2, 2), "checks": [metrics.to_dict(c) for c in checks2]}
        )
    return {"scenarios": results, "latency": latency}


def eval_interview_agent(indices: list[int], repeats: int) -> dict:
    """Runs interview_agent._llm_question for each index, checking question validity."""
    results = []
    latency: list[metrics.LatencyRecord] = []
    profile = CandidateProfile(**MOCK_PROFILE)
    for idx in indices:
        round_name = round_for_index(idx)
        for rep in range(repeats):
            state = InterviewState(job_role="ML Engineer", mode="llm", candidate_profile=profile)
            question, secs = _timed_call(interview_agent._llm_question, state, idx, round_name)
            latency.append(metrics.LatencyRecord("interview_agent", f"{idx}:{rep}", secs))
            check = metrics.check_question_validity(question)
            results.append(
                {
                    "index": idx,
                    "round": round_name,
                    "question": question,
                    "latency_s": round(secs, 2),
                    "repeat": rep,
                    "checks": [metrics.to_dict(check)],
                }
            )
    return {"per_question": results, "latency": latency}


# --- report building -----------------------------------------------------
def _all_checks(feedback_res: dict, strategy_res: dict, interview_res: dict) -> list[dict]:
    checks = []
    for r in feedback_res["per_fixture"]:
        checks.extend(r["checks"])
    checks.extend(feedback_res["monotonicity"])
    for s in strategy_res["scenarios"]:
        checks.extend(s["checks"])
    for r in interview_res["per_question"]:
        checks.extend(r["checks"])
    return checks


def build_report(feedback_res: dict, strategy_res: dict, interview_res: dict, questions: list[int], repeats: int) -> dict:
    all_checks = _all_checks(feedback_res, strategy_res, interview_res)
    n_errors = sum(1 for c in all_checks if not c["passed"] and c["severity"] == "error")
    n_warnings = sum(1 for c in all_checks if not c["passed"] and c["severity"] == "warning")
    all_latency = feedback_res["latency"] + strategy_res["latency"] + interview_res["latency"]
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {"questions": questions, "repeats": repeats},
        "summary": {
            "total_checks": len(all_checks),
            "errors": n_errors,
            "warnings": n_warnings,
            "passed": len(all_checks) - n_errors - n_warnings,
        },
        "latency": metrics.summarize_latency(all_latency),
        "feedback_agent": feedback_res,
        "strategy_agent": strategy_res,
        "interview_agent": interview_res,
    }


def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"# LLM Eval Report — {report['run_at']}",
        f"Questions: {report['config']['questions']} | Repeats: {report['config']['repeats']}",
        "",
        f"**{s['passed']} passed, {s['errors']} errors, {s['warnings']} warnings** (of {s['total_checks']} checks)",
        "",
    ]

    if report["latency"]:
        lines += ["## Latency (seconds)", "| Agent | n | min | p50 | mean | max |", "|---|---|---|---|---|---|"]
        for agent, stats in report["latency"].items():
            lines.append(f"| {agent} | {stats['count']} | {stats['min']:.1f} | {stats['p50']:.1f} | {stats['mean']:.1f} | {stats['max']:.1f} |")
        lines.append("")

    strategy_scenarios = report["strategy_agent"]["scenarios"]
    if strategy_scenarios:
        lines.append("## Headline check: Strategy Agent adaptive behavior (the USP)")
        for scenario_name, check_name in (("weak_technical", "redirect_on_weak_score"), ("strong_technical", "difficulty_escalation")):
            matches = [s for s in strategy_scenarios if s["scenario"] == scenario_name]
            for check_label in {check_name}:
                relevant = [c for m in matches for c in m["checks"] if c["name"] == check_label]
                if not relevant:
                    continue
                n_pass = sum(1 for c in relevant if c["passed"])
                verdict = "PASS" if n_pass == len(relevant) else ("PARTIAL" if n_pass > 0 else "FAIL")
                lines.append(f"  {scenario_name} -> {check_label}: {verdict}  ({n_pass}/{len(relevant)} repeats matched)")
        lines.append("")

    failures = [c for c in _all_checks(report["feedback_agent"], report["strategy_agent"], report["interview_agent"]) if not c["passed"]]
    failures.sort(key=lambda c: 0 if c["severity"] == "error" else 1)
    if failures:
        lines.append(f"## Failures ({s['errors']} error, {s['warnings']} warning)")
        for c in failures:
            tag = "ERROR" if c["severity"] == "error" else "WARN "
            lines.append(f"[{tag}] {c['name']}: {c['detail']}")
        lines.append("")
    else:
        lines.append("## Failures\nNone.\n")

    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the LLM-mode eval harness against real Fireworks calls.")
    p.add_argument("--questions", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated question indices (default: all 8)")
    p.add_argument("--repeats", type=int, default=1, help="Repeats per fixture/scenario (default 1)")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any ERROR-severity check fails")
    p.add_argument("--skip", type=str, default="", help="Comma-separated agents to skip: feedback,strategy,interview")
    args = p.parse_args(argv)

    questions = [int(x) for x in args.questions.split(",") if x.strip() != ""]
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    fixtures = all_fixtures(questions)

    print(
        f"Running LLM eval: {len(questions)} question(s) x {len(fixtures) // max(len(questions), 1)} fixture kinds, "
        f"{args.repeats} repeat(s). This calls the real Fireworks API and can take minutes.\n"
    )

    feedback_res = eval_feedback_agent(fixtures, args.repeats) if "feedback" not in skip else {"per_fixture": [], "monotonicity": [], "latency": []}
    strategy_res = eval_strategy_agent(args.repeats) if "strategy" not in skip else {"scenarios": [], "latency": []}
    interview_res = eval_interview_agent(questions, args.repeats) if "interview" not in skip else {"per_question": [], "latency": []}

    report = build_report(feedback_res, strategy_res, interview_res, questions, args.repeats)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORTS_DIR / f"eval_{ts}.json"
    md_path = REPORTS_DIR / f"eval_{ts}.md"
    md_text = render_markdown(report)
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path.write_text(md_text)

    print(md_text)
    print(f"Full report: {json_path}")

    if args.strict and report["summary"]["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
