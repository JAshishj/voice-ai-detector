import { useMutation } from "@tanstack/react-query";
import { ApiError, detectVoice, type AnalyzeInput } from "../lib/api";

/** POST mutation for voice analysis. No caching (every scan is fresh). */
export function useAnalyzeAudio() {
  return useMutation({
    mutationFn: (input: AnalyzeInput) => detectVoice(input),
    // Always log: the UI renders every error type, the console keeps detail.
    onError: (err) => console.error("[signal-lab] analyze failed:", err),
    // 4xx (validation / rate limit) must never retry; network errors get one.
    retry: (count, err) =>
      count < 1 && err instanceof ApiError ? err.status >= 500 : count < 1,
  });
}
