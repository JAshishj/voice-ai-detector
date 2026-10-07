import { useEffect, useRef, useState } from "react";
import { useAnalyzeAudio } from "../hooks/useAnalyzeAudio";
import { prepareAudio, type PreparedAudio } from "../lib/audio";
import { ApiError, type Language } from "../lib/api";
import { UploadDropzone } from "../components/UploadDropzone";
import { RecorderPanel } from "../components/RecorderPanel";
import { LanguageSelect } from "../components/LanguageSelect";
import { Waveform } from "../components/Waveform";
import { VerdictCard } from "../components/VerdictCard";

type Notice = { kind: "guidance" | "error"; text: string } | null;

/** The prediction page: ingest → analyze → verdict. */
export function Console() {
  const [prepared, setPrepared] = useState<PreparedAudio | null>(null);
  const [clipName, setClipName] = useState("");
  const [language, setLanguage] = useState<Language>("English");
  const [notice, setNotice] = useState<Notice>(null);
  const [retryIn, setRetryIn] = useState(0);
  const lastInput = useRef<{ audioBase64: string; language: Language } | null>(null);

  const mutation = useAnalyzeAudio();

  const busy = mutation.isPending;
  const result = mutation.data ?? null;
  const apiError =
    mutation.error instanceof ApiError ? mutation.error : null;
  const rateLimited = apiError?.status === 429;

  // 429 countdown — the limiter becomes a visible, recoverable state.
  useEffect(() => {
    if (!rateLimited || !apiError?.retryAfter) {
      setRetryIn(0);
      return;
    }
    setRetryIn(apiError.retryAfter);
    const id = setInterval(
      () => setRetryIn((s) => (s > 0 ? s - 1 : 0)),
      1000,
    );
    return () => clearInterval(id);
  }, [rateLimited, apiError?.retryAfter, mutation.failureCount]);

  const ingest = async (blob: Blob, name: string) => {
    setNotice(null);
    mutation.reset();
    try {
      const audio = await prepareAudio(blob);
      setPrepared(audio);
      setClipName(name);
      if (audio.durationSec < 0.5) {
        setNotice({
          kind: "guidance",
          text: `Clip is ${audio.durationSec.toFixed(2)}s — the lab needs at least 0.5s of voiced audio.`,
        });
      }
    } catch {
      setPrepared(null);
      setNotice({
        kind: "error",
        text: "Could not decode that file. Send mp3, wav, or flac with actual speech in it.",
      });
    }
  };

  const analyze = () => {
    if (!prepared || busy) return;
    setNotice(null);
    const input = { audioBase64: prepared.wavBase64, language };
    lastInput.current = input;
    mutation.mutate({ ...input, audioFormat: "wav" });
  };

  const reset = () => {
    mutation.reset();
    setPrepared(null);
    setClipName("");
    setNotice(null);
  };

  const windows = prepared
    ? Math.min(5, Math.max(1, Math.ceil(prepared.durationSec / 6)))
    : 1;
  const tone = result
    ? result.classification === "HUMAN"
      ? "emerald"
      : "rose"
    : "teal";

  return (
    <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-8 md:px-6 lg:grid-cols-12">
      {/* Analyzer rail */}
      <div className="space-y-5 lg:col-span-7">
        <section className="rounded-[20px] border border-hairline bg-panel p-6">
          <h2 className="font-display text-2xl font-bold tracking-tight">
            Signal ingestion
          </h2>
          <p className="mt-1 max-w-[65ch] text-[15px] text-dim">
            Drop a voice clip or capture from the mic, pick the acoustic
            profile, run the forensic scan.
          </p>
          <div className="mt-5 space-y-5">
            <UploadDropzone
              disabled={busy}
              onFile={(f) => void ingest(f, f.name)}
            />
            <RecorderPanel
              disabled={busy}
              onReady={(audio, name) => {
                setNotice(null);
                mutation.reset();
                setPrepared(audio);
                setClipName(name);
              }}
            />
            <LanguageSelect
              value={language}
              onChange={setLanguage}
              disabled={busy}
            />
            {clipName && (
              <p className="font-mono text-xs text-dim">
                LOADED: {clipName} —{" "}
                {prepared ? prepared.durationSec.toFixed(1) : "?"}S // ~16KHZ MONO
              </p>
            )}
            {notice && (
              <p
                role={notice.kind === "error" ? "alert" : "status"}
                className={`text-sm ${notice.kind === "error" ? "text-rose" : "text-dim"}`}
              >
                {notice.text}
              </p>
            )}
            <button
              type="button"
              onClick={analyze}
              disabled={!prepared || busy}
              className={`min-h-12 w-full rounded-xl px-5 py-3 font-display text-base font-bold tracking-tight transition-all ${
                !prepared || busy
                  ? "cursor-not-allowed bg-raised text-dim"
                  : "bg-teal text-void active:-translate-y-px"
              }`}
            >
              {busy ? "Scanning…" : "Initiate forensic scan"}
            </button>
            {apiError && !rateLimited && (
              <p role="alert" className="text-sm text-rose">
                {apiError.message}
              </p>
            )}
            {mutation.error && !apiError && (
              <p role="alert" className="text-sm text-rose">
                Scan failed unexpectedly:{" "}
                {mutation.error instanceof Error
                  ? mutation.error.message
                  : "unknown error"}{" "}
                Open the browser console (F12) for details.
              </p>
            )}
            {rateLimited && (
              <div
                role="alert"
                className="rounded-xl border border-hairline bg-raised p-4"
              >
                <p className="text-sm text-ink">{apiError?.message}</p>
                <button
                  type="button"
                  disabled={retryIn > 0 || busy}
                  onClick={() => {
                    if (lastInput.current)
                      mutation.mutate({
                        ...lastInput.current,
                        audioFormat: "wav",
                      });
                  }}
                  className="mt-3 min-h-11 rounded-xl border border-teal/60 px-4 py-2 text-sm font-medium text-teal disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {retryIn > 0
                    ? `Retry available in ${retryIn}s`
                    : "Retry scan"}
                </button>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Verdict stage */}
      <div className="lg:col-span-5">
        <div className="lg:sticky lg:top-6">
          <section className="rounded-[20px] border border-hairline bg-panel p-6">
            <p className="font-mono text-xs tracking-[0.08em] text-dim">
              VERDICT STAGE // OSCILLOSCOPE
            </p>
            <div className="mt-3">
              <Waveform
                peaks={prepared?.peaks ?? null}
                state={busy ? "analyzing" : result ? "verdict" : "idle"}
                tone={tone}
              />
            </div>
            {!result && !busy && (
              <p className="mt-3 text-center font-display text-lg text-dim">
                Awaiting input stream
              </p>
            )}
          </section>
          <div className="mt-6">
            {busy && (
              <div
                className="rounded-[20px] border border-hairline bg-panel p-6 md:p-8"
                aria-label="Analyzing"
              >
                <div className="shimmer h-9 w-2/3 rounded-lg" />
                <div className="shimmer mt-4 h-2 w-full rounded-full" />
                <div className="shimmer mt-6 h-20 w-full rounded-xl" />
              </div>
            )}
            {result && !busy && (
              <VerdictCard
                result={result}
                windows={windows}
                onReset={reset}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
