import { useCallback, useEffect, useRef, useState } from "react";
import { prepareAudio, type PreparedAudio } from "../lib/audio";

export type RecorderState = "idle" | "recording" | "encoding" | "error";

/**
 * In-browser mic capture. Records to the browser's native container, then
 * re-encodes to 16 kHz mono WAV via lib/audio so the payload matches the
 * upload path exactly.
 */
export function useRecorder() {
  const [state, setState] = useState<RecorderState>("idle");
  const [level, setLevel] = useState(0);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const rafRef = useRef(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopTracks = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    cancelAnimationFrame(rafRef.current);
    setLevel(0);
  };

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const Ctx = window.AudioContext!;
      const ctx = new Ctx();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      analyserRef.current = analyser;
      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        for (const v of data) peak = Math.max(peak, Math.abs(v - 128) / 128);
        setLevel(peak);
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();

      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRef.current = rec;
      rec.start();
      setState("recording");
    } catch {
      setState("error");
    }
  }, []);

  const stop = useCallback(async (): Promise<PreparedAudio | null> => {
    const rec = mediaRef.current;
    if (!rec || state !== "recording") return null;
    setState("encoding");
    const done = new Promise<Blob>((resolve) => {
      rec.onstop = () =>
        resolve(new Blob(chunksRef.current, { type: rec.mimeType }));
      rec.stop();
    });
    const blob = await done;
    stopTracks();
    try {
      const prepared = await prepareAudio(blob);
      setState("idle");
      return prepared;
    } catch {
      setState("error");
      return null;
    }
  }, [state]);

  useEffect(() => () => stopTracks(), []);

  return { state, level, start, stop };
}
