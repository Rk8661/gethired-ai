import { useRef, useState } from "react";
import { createSession, getProgress, submitAnswer } from "./api";
import Header from "./components/Header";
import ResumeUpload from "./components/ResumeUpload";
import InterviewScreen from "./components/InterviewScreen";
import FinalReport from "./components/FinalReport";
import "./App.css";

// Drives the whole flow: upload -> ask/answer loop (8 Qs) -> report.
export default function App() {
  const [phase, setPhase] = useState("start"); // start | interview | report
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [sessionId, setSessionId] = useState(null);
  const [total, setTotal] = useState(8);
  const [question, setQuestion] = useState(null);
  const [lastEvaluation, setLastEvaluation] = useState(null);
  const [report, setReport] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [stage, setStage] = useState("");
  const pollRef = useRef(null);

  function startProgressPoll(sid) {
    stopProgressPoll();
    pollRef.current = setInterval(async () => {
      try {
        const { stage: s } = await getProgress(sid);
        if (s) setStage(s);
      } catch {
        // polling is best-effort; ignore transient failures
      }
    }, 600);
  }

  function stopProgressPoll() {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
    setStage("");
  }

  async function handleStart({ jobRole, mode, resume }) {
    setError("");
    setLoading(true);
    try {
      const data = await createSession({ jobRole, mode, resume });
      setSessionId(data.session_id);
      setTotal(data.total_questions);
      setQuestion(data.question);
      setLastEvaluation(null);
      setPhase("interview");
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function handleAnswer(answer) {
    setError("");
    setLoading(true);
    startProgressPoll(sessionId);
    try {
      const data = await submitAnswer(sessionId, answer);
      setLastEvaluation(data.evaluation);
      if (data.complete) {
        setReport(data.final_report);
        setTimeline(data.strategy_timeline);
        setPhase("report");
      } else {
        setQuestion(data.question);
      }
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      stopProgressPoll();
      setLoading(false);
    }
  }

  function handleRestart() {
    setPhase("start");
    setSessionId(null);
    setQuestion(null);
    setLastEvaluation(null);
    setReport(null);
    setTimeline([]);
  }

  return (
    <div className="app">
      <Header />
      {error && <div className="error">{error}</div>}

      {phase === "start" && <ResumeUpload onStart={handleStart} loading={loading} />}

      {phase === "interview" && question && (
        <InterviewScreen
          question={question}
          total={total}
          lastEvaluation={lastEvaluation}
          onSubmit={handleAnswer}
          loading={loading}
          stage={stage}
        />
      )}

      {phase === "report" && report && (
        <FinalReport report={report} timeline={timeline} onRestart={handleRestart} />
      )}
    </div>
  );
}
