// Persistent top bar shown across all phases so branding doesn't disappear
// once the user leaves the start screen.
export default function Header() {
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-mark">G</span>
        <div>
          <div className="brand-name">GetHired AI</div>
          <div className="brand-tag">Adaptive interview coach</div>
        </div>
      </div>
    </header>
  );
}
