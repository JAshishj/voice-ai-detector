import type { DetectResponse } from "../lib/api";
import { ConfidenceBar } from "./ConfidenceBar";
import { SegmentStrip } from "./SegmentStrip";

interface Props {
  result: DetectResponse;
  windows: number;
  onReset: () => void;
}

export function VerdictCard({ result, windows, onReset }: Props) {
  const human = result.classification === "HUMAN";
  return (
    <section
      aria-live="polite"
      className="cascade rounded-[20px] border border-hairline bg-panel p-6 md:p-8"
    >
      <p className="font-mono text-xs tracking-[0.08em] text-dim">
        VERDICT // {result.language.toUpperCase()}
      </p>
      <h2
        className={`mt-2 font-display text-4xl font-bold tracking-tight md:text-5xl ${
          human ? "text-emerald" : "text-rose"
        }`}
      >
        {human ? "HUMAN" : "AI_GENERATED"}
      </h2>
      <div className="mt-6">
        <ConfidenceBar
          classification={result.classification}
          confidence={result.confidenceScore}
        />
      </div>
      <p className="mt-4 max-w-[65ch] text-[15px] leading-relaxed text-ink">
        {result.explanation}
        {windows > 1 ? ` Averaged over ${windows} segments.` : ""}
      </p>
      <div className="mt-6">
        <SegmentStrip windows={windows} confidence={result.confidenceScore} />
      </div>
      <button
        type="button"
        onClick={onReset}
        className="mt-6 min-h-[44px] rounded-xl border border-hairline bg-transparent px-5 py-2.5 font-medium text-ink transition-colors hover:border-teal/60"
      >
        Analyze another clip
      </button>
    </section>
  );
}
