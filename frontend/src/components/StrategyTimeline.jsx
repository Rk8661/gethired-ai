// The headline USP visual: every Strategy Agent decision, in order, so the user
// can see WHY the interview adapted the way it did.
export default function StrategyTimeline({ entries }) {
  if (!entries?.length) return null;
  return (
    <div className="card">
      <h2>Interview Strategy Timeline</h2>
      <p className="muted">Why the interviewer adapted, after each answer.</p>
      <ol className="timeline">
        {entries.map((e, i) => (
          <li key={i}>
            <div className="timeline-dot" />
            <div>
              <strong>After Q{e.after_question + 1}</strong>
              <p className="observation">{e.observation}</p>
              <p className="decision">→ {e.decision}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
