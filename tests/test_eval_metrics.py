"""Unit tests for eval/metrics.py — pure functions, synthetic data only, no LLM calls."""
from backend.core.state import Directive, Scores
from eval import metrics
from eval.state_builders import state_with_strong_history, state_with_weak_technical_history


def test_scores_bounds_pass():
    s = Scores(technical_accuracy=8, communication=7, confidence=6, problem_solving=9, relevance=8)
    assert metrics.check_scores_bounds(s).passed


def test_question_validity_empty_fails():
    assert not metrics.check_question_validity("").passed
    assert not metrics.check_question_validity("   ").passed


def test_question_validity_leak_fails():
    r = metrics.check_question_validity("What is a hash map? The answer is a key-value store.")
    assert not r.passed
    assert r.severity == "error"


def test_question_validity_two_marks_is_warning_not_error():
    r = metrics.check_question_validity("What happened? And how did you fix it?")
    assert not r.passed
    assert r.severity == "warning"


def test_question_validity_three_marks_is_error():
    r = metrics.check_question_validity("What? Why? How?")
    assert not r.passed
    assert r.severity == "error"


def test_question_validity_single_question_passes():
    assert metrics.check_question_validity("Tell me about a time you fixed a bug?").passed


def test_monotonic_scores_pass():
    assert metrics.check_monotonic_scores(0, 20, 45).passed


def test_monotonic_scores_fail_but_warning_severity():
    r = metrics.check_monotonic_scores(30, 10, 5)  # clearly inverted, beyond slack
    assert not r.passed
    assert r.severity == "warning"


def test_monotonic_scores_within_slack_passes():
    r = metrics.check_monotonic_scores(10, 12, 11)  # weak > strong by 1, within default slack=2
    assert r.passed


def test_redirect_on_weak_score_matches_substring():
    d = Directive(focus_project="DocuMind — RAG Document Assistant")
    r = metrics.check_redirect_on_weak_score(d, expected_project_title="DocuMind", score_total=13)
    assert r.passed


def test_redirect_on_weak_score_no_match_fails():
    d = Directive(focus_project="CampusBus Tracker")
    r = metrics.check_redirect_on_weak_score(d, expected_project_title="DocuMind", score_total=13)
    assert not r.passed
    assert r.severity == "error"


def test_redirect_on_weak_score_none_fails():
    d = Directive(focus_project=None)
    r = metrics.check_redirect_on_weak_score(d, expected_project_title="DocuMind", score_total=13)
    assert not r.passed


def test_redirect_check_na_when_score_not_weak():
    d = Directive(focus_project=None)
    r = metrics.check_redirect_on_weak_score(d, expected_project_title="DocuMind", score_total=40)
    assert r.passed  # not applicable, so treated as pass


def test_difficulty_escalation_hard_passes():
    d = Directive(difficulty="hard")
    assert metrics.check_difficulty_escalation(d, score_total=45).passed


def test_difficulty_escalation_medium_soft_passes():
    d = Directive(difficulty="medium")
    r = metrics.check_difficulty_escalation(d, score_total=45)
    assert r.passed
    assert "soft pass" in r.detail


def test_difficulty_escalation_easy_fails():
    d = Directive(difficulty="easy")
    r = metrics.check_difficulty_escalation(d, score_total=45)
    assert not r.passed
    assert r.severity == "error"


def test_difficulty_escalation_na_when_score_not_strong():
    d = Directive(difficulty="easy")
    assert metrics.check_difficulty_escalation(d, score_total=20).passed


def test_raw_difficulty_valid():
    assert metrics.check_raw_difficulty_valid("hard").passed
    assert metrics.check_raw_difficulty_valid("MEDIUM").passed  # case-insensitive
    assert not metrics.check_raw_difficulty_valid("very hard").passed
    assert not metrics.check_raw_difficulty_valid(None).passed


def test_summarize_latency_computes_stats():
    records = [metrics.LatencyRecord("feedback_agent", f"0:{i}", float(i + 1)) for i in range(5)]
    out = metrics.summarize_latency(records)
    assert out["feedback_agent"]["count"] == 5
    assert out["feedback_agent"]["min"] == 1.0
    assert out["feedback_agent"]["max"] == 5.0


def test_summarize_latency_empty():
    assert metrics.summarize_latency([]) == {}


# --- state_builders sanity (no LLM calls; mode="llm" is just a state flag here) ---
def test_state_with_weak_technical_history_shape():
    s = state_with_weak_technical_history()
    assert s.history[-1].scores.total == 13
    assert s.candidate_profile.projects[0].relevance_score >= s.candidate_profile.projects[1].relevance_score


def test_state_with_strong_history_shape():
    s = state_with_strong_history()
    assert s.history[-1].scores.total == 44
