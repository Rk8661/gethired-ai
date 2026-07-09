import { useState } from "react";
import LoadingDots from "./LoadingDots";

const ROLES = [
  "ML Engineer",
  "Software Engineer (Backend)",
  "Frontend Engineer",
  "Data Analyst",
];

// Start screen: pick a role, optionally upload a resume PDF, choose mode.
export default function ResumeUpload({ onStart, loading }) {
  const [jobRole, setJobRole] = useState(ROLES[0]);
  const [mode, setMode] = useState("mock");
  const [resume, setResume] = useState(null);

  return (
    <div className="card">
      <h1>Start your interview</h1>
      <p className="muted">
        Upload your resume and pick a role — the interviewer adapts to you as you go.
      </p>

      <label>Job role</label>
      <select value={jobRole} onChange={(e) => setJobRole(e.target.value)}>
        {ROLES.map((r) => (
          <option key={r}>{r}</option>
        ))}
      </select>

      <label>Resume (PDF, optional in mock mode)</label>
      <div className={`dropzone ${resume ? "has-file" : ""}`}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setResume(e.target.files?.[0] || null)}
        />
        <div className="dropzone-text">
          {resume ? (
            <span className="dropzone-filename">✓ {resume.name}</span>
          ) : (
            "Click to choose a PDF, or drag it here"
          )}
        </div>
      </div>

      <label>Mode</label>
      <div className="segmented">
        <button
          type="button"
          className={mode === "mock" ? "active" : ""}
          onClick={() => setMode("mock")}
        >
          Mock — instant, offline
        </button>
        <button
          type="button"
          className={mode === "llm" ? "active" : ""}
          onClick={() => setMode("llm")}
        >
          LLM — real Fireworks model
        </button>
      </div>

      <button disabled={loading} onClick={() => onStart({ jobRole, mode, resume })}>
        {loading ? <>Starting <LoadingDots /></> : "Start interview"}
      </button>
    </div>
  );
}
