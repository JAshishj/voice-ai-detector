/** Typed client for the Voice AI Detector API (POST /api/voice-detection). */

export type Language = "Tamil" | "English" | "Hindi" | "Malayalam" | "Telugu";

export const LANGUAGES: Language[] = [
  "Tamil",
  "English",
  "Hindi",
  "Malayalam",
  "Telugu",
];

export type Classification = "AI_GENERATED" | "HUMAN";

export interface DetectResponse {
  status: "success";
  language: string;
  classification: Classification;
  confidenceScore: number;
  explanation: string;
}

export class ApiError extends Error {
  status: number;
  retryAfter?: number;

  constructor(status: number, message: string, retryAfter?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

/** Same-origin in production (FastAPI serves the build); override for dev. */
const API_BASE = import.meta.env.VITE_API_URL ?? "";

export interface AnalyzeInput {
  audioBase64: string;
  language: Language;
  audioFormat: "mp3" | "wav" | "flac";
}

export async function detectVoice(
  input: AnalyzeInput,
  signal?: AbortSignal,
): Promise<DetectResponse> {
  const res = await fetch(`${API_BASE}/api/voice-detection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal,
  });

  if (res.status === 429) {
    const retryAfter = Number(res.headers.get("Retry-After") ?? "30");
    throw new ApiError(
      429,
      "Scan budget exhausted — the shared console allows a few scans per minute.",
      Number.isFinite(retryAfter) ? retryAfter : 30,
    );
  }

  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    throw new ApiError(res.status, "Unreadable response from the lab.");
  }

  if (!res.ok) {
    const message =
      typeof body === "object" && body !== null && "message" in body
        ? String((body as { message: unknown }).message)
        : "The lab rejected this clip.";
    throw new ApiError(res.status, message);
  }

  return body as DetectResponse;
}

export interface HealthInfo {
  status: string;
  threshold?: number;
  best_val_acc?: number;
}

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new ApiError(res.status, "Lab offline.");
  return (await res.json()) as HealthInfo;
}
