import { MetricsTable } from "../components/MetricsTable";
import { LINKS } from "../lib/links";

/** The dossier page: what the model is, how it scored, where it lives. */
export function Model() {
  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8 md:px-6">
      <section className="max-w-[65ch]">
        <p className="font-mono text-xs tracking-[0.08em] text-dim">
          MODEL INTEL // DOSSIER
        </p>
        <h2 className="mt-2 font-display text-3xl font-bold tracking-tight md:text-4xl">
          What decides the verdict
        </h2>
        <p className="mt-3 text-[15px] leading-relaxed text-dim">
          A fine-tuned wav2vec2 backbone with a frozen core and an
          artifact-CNN head, trained on 2,544 clips and evaluated on a
          disjoint held-out split. Long clips are windowed into 6-second
          segments — the same framing used in training.
        </p>
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <MetricsTable />
        </div>

        <section
          aria-label="How it works"
          className="rounded-[20px] border border-hairline bg-panel p-6 lg:col-span-7"
        >
          <h3 className="font-display text-xl font-bold tracking-tight text-ink">
            How it works
          </h3>
          <ol className="mt-4 space-y-5">
            {[
              ["01", "Listen", "wav2vec2-base encodes 16 kHz mono into contextual speech representations. The core stays frozen; the final two transformer blocks adapt to synthetic artifacts."],
              ["02", "Inspect", "A 3-layer 1D artifact CNN (768 → 256 → 128 → 64 channels) scans the representations for vocoder fingerprints human speech doesn't carry."],
              ["03", "Decide", "Window probabilities average into one score. Scores at or above the 0.85 operating point read AI-generated — a recall-heavy cutoff tuned by Youden's J, so humans are rarely flagged."],
            ].map(([n, title, body]) => (
              <li key={n} className="flex gap-4">
                <span className="font-mono text-sm text-teal">{n}</span>
                <div>
                  <p className="font-display text-base font-bold text-ink">{title}</p>
                  <p className="mt-1 max-w-[65ch] text-sm leading-relaxed text-dim">{body}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section
          aria-label="API contract"
          className="rounded-[20px] border border-hairline bg-panel p-6 lg:col-span-7"
        >
          <h3 className="font-display text-xl font-bold tracking-tight text-ink">
            API contract
          </h3>
          <pre className="mt-4 overflow-x-auto rounded-xl bg-void p-4 font-mono text-[13px] leading-relaxed text-ink">
{`POST /api/voice-detection
{ "language": "English",
  "audioFormat": "wav",
  "audioBase64": "<clip>" }

→ { "classification": "HUMAN",
    "confidenceScore": 0.9464,
    "explanation": "…" }`}
          </pre>
          <p className="mt-3 text-sm text-dim">
            Rate-limited (30/min global, 5/min scans) with Retry-After. Silence,
            sub-0.5s, and oversize clips are rejected with 400 guidance.
          </p>
        </section>

        <section
          aria-label="Project links"
          className="rounded-[20px] border border-hairline bg-panel p-6 lg:col-span-5"
        >
          <h3 className="font-display text-xl font-bold tracking-tight text-ink">
            Where it lives
          </h3>
          <ul className="mt-4 space-y-3">
            {[
              ["Source code", "Training, API, and this console.", LINKS.github],
              ["Inference API", "Free Gradio Space serving /predict.", LINKS.spaceApi],
              ["Model weights", "detector.pt + operating config (363 MB).", LINKS.modelWeights],
            ].map(([title, body, href]) => (
              <li key={title}>
                <a
                  href={href}
                  target="_blank"
                  rel="noreferrer"
                  className="block rounded-xl border border-hairline bg-raised p-4 transition-colors hover:border-teal/60"
                >
                  <p className="font-medium text-ink">
                    {title} <span aria-hidden className="text-teal">↗</span>
                  </p>
                  <p className="mt-1 text-sm text-dim">{body}</p>
                </a>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
