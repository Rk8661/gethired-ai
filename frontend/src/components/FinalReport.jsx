import StrategyTimeline from "./StrategyTimeline";

// End screen: overall score, recommendation, strength/weakness clusters,
// concepts to study, and the strategy timeline.
export default function FinalReport({ report, timeline, onRestart }) {
  const pct = Math.round((report.total_score / report.max_score) * 100);
  return (
    <>
      <div className="card">
        <h1>Final Report</h1>
        <div className="score-hero">
          <ScoreRing pct={pct} />
          <div className="score-hero-text">
            <span className="score-big">{report.total_score}</span>
            <span className="muted">/ {report.max_score} points</span>
          </div>
        </div>
        <p className={`rec rec-${report.recommendation.split(" ")[0].toLowerCase()}`}>
          {report.recommendation}
        </p>
        <p>{report.summary}</p>

        <div className="grid2">
          <TopicBlock title="Strengths" items={report.strong_topics} kind="strong" />
          <TopicBlock title="Focus areas" items={report.weak_topics} kind="weak" />
        </div>

        {report.concepts_to_study?.length > 0 && (
          <>
            <h3>Concepts to study</h3>
            <div className="chips">
              {report.concepts_to_study.map((c) => (
                <span className="chip" key={c}>{c}</span>
              ))}
            </div>
          </>
        )}

        <h3>Scores by round</h3>
        <ul>
          {Object.entries(report.scores_by_round).map(([round, score]) => (
            <li key={round}>{round}: {score}</li>
          ))}
        </ul>

        <button onClick={onRestart}>Start a new interview</button>
      </div>

      <StrategyTimeline entries={timeline} />
    </>
  );
}

function ScoreRing({ pct, size = 84, stroke = 8 }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (Math.min(100, Math.max(0, pct)) / 100) * c;
  const color = pct >= 65 ? "var(--good)" : pct >= 50 ? "#ffcf6d" : "var(--bad)";
  return (
    <svg className="score-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line)" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={offset}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: "stroke-dashoffset 0.6s ease" }}
      />
      <text x="50%" y="52%" textAnchor="middle" dominantBaseline="middle" className="score-ring-pct">
        {pct}%
      </text>
    </svg>
  );
}

function TopicBlock({ title, items, kind }) {
  return (
    <div>
      <h3>{title}</h3>
      <div className="chips">
        {items?.length ? (
          items.map((t) => <span className={`chip chip-${kind}`} key={t}>{t}</span>)
        ) : (
          <span className="muted">None</span>
        )}
      </div>
    </div>
  );
}
