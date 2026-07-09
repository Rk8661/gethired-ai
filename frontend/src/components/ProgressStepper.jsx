// Visual position within the 8-question interview, grouped by round — mirrors
// backend/config.py's ROUND_LAYOUT (HR x2, Technical x4, Hiring Manager x2).
const ROUND_LAYOUT = ["HR", "HR", "Technical", "Technical", "Technical", "Technical", "Hiring Manager", "Hiring Manager"];

export default function ProgressStepper({ currentIndex }) {
  const groups = [];
  for (const round of ["HR", "Technical", "Hiring Manager"]) {
    const indices = ROUND_LAYOUT.map((r, i) => (r === round ? i : null)).filter((i) => i !== null);
    groups.push({ round, indices });
  }

  return (
    <div className="stepper">
      <div className="stepper-groups">
        {groups.map(({ round, indices }) => (
          <div className="stepper-group" key={round}>
            <span className="stepper-group-label">{round}</span>
            <div className="stepper-dots">
              {indices.map((i) => (
                <span
                  key={i}
                  className={`stepper-dot ${i < currentIndex ? "done" : ""} ${i === currentIndex ? "current" : ""}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
