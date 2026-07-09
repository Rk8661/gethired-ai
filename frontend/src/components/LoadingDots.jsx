// Small animated "..." indicator used anywhere we're waiting on a slow call
// (llm mode can take 5-40s per stage) so the wait doesn't look frozen.
export default function LoadingDots() {
  return (
    <span className="dots" aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  );
}
