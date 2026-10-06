import { useRecorder } from "../hooks/useRecorder";
import type { PreparedAudio } from "../lib/audio";

interface Props {
  onReady: (audio: PreparedAudio, name: string) => void;
  disabled?: boolean;
}

/** Mic capture panel with live level meter. Output matches upload path. */
export function RecorderPanel({ onReady, disabled }: Props) {
  const { state, level, start, stop } = useRecorder();

  const busy = disabled || state === "encoding";

  const handleToggle = async () => {
    if (state === "recording") {
      const prepared = await stop();
      if (prepared) onReady(prepared, "mic-capture.wav");
    } else if (state === "idle") {
      await start();
    }
  };

  return (
    <div className="rounded-[20px] border border-hairline bg-panel p-5">
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={handleToggle}
          disabled={busy && state !== "recording"}
          aria-label={state === "recording" ? "Stop recording" : "Record from mic"}
          className={`flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full transition-colors ${
            state === "recording"
              ? "bg-rose text-void"
              : "bg-teal font-semibold text-void"
          } ${busy && state !== "recording" ? "cursor-not-allowed opacity-50" : ""}`}
        >
          <span
            aria-hidden
            className={state === "recording" ? "h-3.5 w-3.5 rounded-[3px] bg-void" : "h-3.5 w-3.5 rounded-full bg-void"}
          />
        </button>
        <div className="flex-1">
          <p className="font-mono text-xs tracking-[0.08em] text-dim">
            {state === "recording"
              ? "CAPTURING — TAP TO STOP"
              : state === "encoding"
                ? "ENCODING 16KHZ MONO…"
                : state === "error"
                  ? "MIC UNAVAILABLE — CHECK PERMISSION"
                  : "MIC CAPTURE"}
          </p>
          <div
            className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-raised"
            aria-hidden
          >
            <div
              className="h-full rounded-full bg-teal transition-[width] duration-100"
              style={{ width: `${Math.round(Math.min(1, level) * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
