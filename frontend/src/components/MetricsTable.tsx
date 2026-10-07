/** Held-out figures — must match training/evaluate.py, never rounded. */
const ROWS: Array<[string, string]> = [
  ["Held-out accuracy", "96.3%"],
  ["F1", "0.964"],
  ["AUC", "0.992"],
  ["EER", "0.020 @ 0.89"],
  ["Operating threshold", "0.85"],
  ["Best val accuracy", "98.2%"],
];

export function MetricsTable() {
  return (
    <section className="rounded-[20px] border border-hairline bg-panel p-6">
      <h2 className="font-display text-xl font-bold tracking-tight text-ink">
        Model intel
      </h2>
      <dl className="mt-4 divide-y divide-hairline">
        {ROWS.map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between py-2.5">
            <dt className="text-sm text-dim">{k}</dt>
            <dd className="font-mono text-sm text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-sm leading-relaxed text-dim">
        wav2vec2 backbone with a frozen core plus an artifact-CNN head.
        Tones and white noise can still score confidently — silence is gated,
        tone rejection is planned.
      </p>
    </section>
  );
}
