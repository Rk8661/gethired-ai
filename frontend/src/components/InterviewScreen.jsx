import { useState } from "react";
import LoadingDots from "./LoadingDots";
import ProgressStepper from "./ProgressStepper";

const ROUND_ICON = { HR: "🧑‍💼", Technical: "🖥️", "Hiring Manager": "🎯" };
const MIN_WORDS_HINT = 8;

// One question at a time: shows the persona/round, collects an answer, and
// renders the feedback for the answer just submitted. `stage` is the live
// backend stage label (e.g. "Scoring your answer…") polled while waiting —
// in llm mode a turn can take ~20-30s across 3 sequential model calls, so this
// keeps the wait from looking frozen.
export default function InterviewScreen({ question, total, lastEvaluation, onSubmit, loading, stage }) {
  const [answer, setAnswer] = useState("");
  const wordCount = answer.trim() ? answer.trim().split(/\s+/).length : 0;

  const submit = () => {
    if (!answer.trim()) return;
    onSubmit(answer);
    setAnswer("");
  };

  return (
    <div className="card" key={question.index}>
      <ProgressStepper currentIndex={question.index} />

      <div className="row spread">
        <span className={`badge round-${question.round.replace(/\s/g, "")}`}>
          {ROUND_ICON[question.round]} {question.round}
        </span>
        <span className="muted">Question {question.index + 1} / {total}</span>
      </div>

      <h2 className="question">{question.question}</h2>

      <textarea
        rows={5}
        placeholder="Type your answer…"
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        disabled={loading}
      />
      <div className={`word-count ${wordCount > 0 && wordCount < MIN_WORDS_HINT ? "low" : ""}`}>
        {wordCount === 0 ? "" : `${wordCount} word${wordCount === 1 ? "" : "s"}`}
      </div>

      <button disabled={loading} onClick={submit}>
        {loading ? <>{stage || "Evaluating"} <LoadingDots /></> : "Submit answer"}
      </button>

      {lastEvaluation && <FeedbackCard ev={lastEvaluation} />}
    </div>
  );
}

function FeedbackCard({ ev }) {
  const s = ev.scores;
  return (
    <div className="feedback">
      <div className="row spread">
        <strong>Feedback on Q{ev.index + 1}</strong>
        <strong>{s.total} / 50</strong>
      </div>
      <p>{ev.feedback}</p>
      <ScoreBars scores={s} />
      {ev.suggested_answer && (
        <p className="muted"><em>Suggested:</em> {ev.suggested_answer}</p>
      )}
      {ev.missing_concepts?.length > 0 && (
        <p className="muted">
          <em>Missing concepts:</em> {ev.missing_concepts.join(", ")}
        </p>
      )}
    </div>
  );
}

function ScoreBars({ scores }) {
  const dims = [
    ["Technical", scores.technical_accuracy],
    ["Communication", scores.communication],
    ["Confidence", scores.confidence],
    ["Problem Solving", scores.problem_solving],
    ["Relevance", scores.relevance],
  ];
  return (
    <div className="bars">
      {dims.map(([label, v]) => (
        <div className="bar-row" key={label}>
          <span className="bar-label">{label}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${v * 10}%` }} />
          </span>
          <span className="bar-val">{v}</span>
        </div>
      ))}
    </div>
  );
}
