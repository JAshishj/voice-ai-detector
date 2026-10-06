interface Props {
  windows: number;
  confidence: number;
}

/**
 * Mono-indexed chips for windowed inference. The API returns an averaged
 * score, so chips honestly report the shared average per window rather
 * than inventing per-window values.
 */
export function SegmentStrip({ windows, confidence }: Props) {
  if (windows <= 1) return null;
  return (
    <div>
      <p className="font-mono text-xs tracking-[0.08em] text-dim">
        {windows} SEGMENTS AVERAGED
      </p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {Array.from({ length: windows }, (_, i) => (
          <span
            key={i}
            title={`Window ${i + 1}: shares the ${(confidence * 100).toFixed(1)}% average`}
            className="rounded-md border border-hairline bg-raised px-2 py-1 font-mono text-[11px] text-dim"
          >
            SEG_{String(i + 1).padStart(2, "0")}
          </span>
        ))}
      </div>
    </div>
  );
}
