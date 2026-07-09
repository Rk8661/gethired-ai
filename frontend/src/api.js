// Thin client for the GetHired AI backend.
const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function createSession({ jobRole, mode, resume }) {
  const form = new FormData();
  form.append("job_role", jobRole);
  form.append("mode", mode);
  if (resume) form.append("resume", resume);
  const res = await fetch(`${BASE}/sessions`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`createSession failed: ${res.status}`);
  return res.json();
}

export async function submitAnswer(sessionId, answer) {
  const res = await fetch(`${BASE}/sessions/${sessionId}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer }),
  });
  if (!res.ok) throw new Error(`submitAnswer failed: ${res.status}`);
  return res.json();
}

// Polled while an /answer call is in flight (llm mode can take a while) so the
// UI can show which stage is running instead of a single static spinner.
export async function getProgress(sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}/progress`);
  if (!res.ok) return { stage: null };
  return res.json();
}
