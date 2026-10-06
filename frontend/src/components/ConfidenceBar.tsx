import type { Classification } from "../lib/api";

interface Props {
  classification: Classification;
  confidence: number;
}

/** 8px verdict-colored track with right-aligned mono readout (DESIGN.md §4). */
export function ConfidenceBar({ classification, confidence }: Props) {
  const fill = classification === "HUMAN" ? "bg-emerald" : "bg-rose";
  return (
    <div>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-raised"
        role="progressbar"
        aria-valuenow={Math.round(confidence * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Confidence"
      >
        <div
          className={`h-full rounded-full ${fill} transition-[width] duration-700`}
          style={{ width: `${Math.round(confidence * 100)}%` }}
        />
      </div>
      <p className="mt-2 text-right font-mono text-sm text-ink">
        {(confidence * 100).toFixed(2)}%
      </p>
    </div>
  );
}
