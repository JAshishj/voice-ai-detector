import { useEffect, useRef } from "react";

export type WaveState = "idle" | "analyzing" | "verdict";

interface Props {
  peaks: Float32Array | null;
  state: WaveState;
  /** verdict color key: null = teal (idle/analyzing) */
  tone: "teal" | "emerald" | "rose";
}

const TONES: Record<Props["tone"], string> = {
  teal: "#2DD4BF",
  emerald: "#34D399",
  rose: "#FB7185",
};

/** Oscilloscope canvas: ghost trace idle, drawing sweep while analyzing. */
export function Waveform({ peaks, state, tone }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const progress = useRef(0);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    let raf = 0;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      // Reticle grid
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      for (let x = 0; x <= w; x += 48) {
        ctx.beginPath();
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, h);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(0, h / 2 + 0.5);
      ctx.lineTo(w, h / 2 + 0.5);
      ctx.stroke();

      const bins = peaks?.length ?? 120;
      const color = TONES[tone];
      ctx.strokeStyle = peaks ? color : "rgba(142,142,147,0.45)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();

      const visible =
        state === "analyzing" && !reduced
          ? Math.floor(bins * progress.current)
          : bins;
      for (let b = 0; b < Math.max(1, visible); b++) {
        const amp = peaks ? peaks[b] : 0.06 + 0.04 * Math.sin(b * 0.4);
        const x = (b / (bins - 1)) * w;
        const y = h / 2 - amp * (h * 0.44);
        if (b === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      if (state === "analyzing" && !reduced) {
        progress.current = (progress.current + 0.012) % 1.15;
        raf = requestAnimationFrame(draw);
      }
    };

    progress.current = 0;
    draw();
    return () => cancelAnimationFrame(raf);
  }, [peaks, state, tone]);

  return (
    <canvas
      ref={ref}
      className="h-44 w-full"
      role="img"
      aria-label={
        state === "verdict"
          ? "Waveform of the analyzed clip"
          : "Oscilloscope display awaiting input"
      }
    />
  );
}
