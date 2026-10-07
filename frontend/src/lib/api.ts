/** Typed client for the Voice AI Detector API (POST /api/voice-detection). */

import { Client } from "@gradio/client";

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

/**
 * Backend selector. "fastapi" (default) talks to our REST envelope;
 * "gradio" talks to the free-tier Gradio Space via @gradio/client.
 * Set VITE_BACKEND=gradio + VITE_GRADIO_URL for split hosting.
 */
const BACKEND = import.meta.env.VITE_BACKEND ?? "fastapi";
const GRADIO_URL = import.meta.env.VITE_GRADIO_URL ?? "";
const GRADIO_THRESHOLD = Number(import.meta.env.VITE_THRESHOLD ?? "0.85");
/**
 * Optional: your own calls then bill to this account's ZeroGPU quota
 * instead of the small shared anonymous pool (which is currently empty).
 * WARNING: Vite bakes this into public JS — anyone can read it. Use a
 * dedicated throwaway HF account, or leave it unset and wait for the
 * monthly free-grant reset.
 */
const HF_TOKEN: string = import.meta.env.VITE_HF_TOKEN ?? "";

export interface AnalyzeInput {
  audioBase64: string;
  language: Language;
  audioFormat: "mp3" | "wav" | "flac";
}

function base64ToWavFile(b64: string): File {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // Must be a File (not a Blob): the Gradio client only uploads named files
  // as file inputs — a bare Blob arrives server-side as a bare string and
  // fails FileData validation before predict ever runs.
  return new File([bytes], "clip.wav", { type: "audio/wav" });
}

let gradioClient: Client | null = null;

async function detectViaGradio(
  input: AnalyzeInput,
  signal?: AbortSignal,
): Promise<DetectResponse> {
  if (!GRADIO_URL) {
    throw new ApiError(0, "VITE_GRADIO_URL is not configured.");
  }
  if (signal?.aborted) throw new ApiError(0, "Scan cancelled.");
  let client: Client;
  try {
    // @gradio/client types the token as `hf_${string}`; the env value is a
    // plain string, so narrow it (empty stays undefined = anonymous quota).
    const token = (HF_TOKEN || undefined) as `hf_${string}` | undefined;
    gradioClient ??= await Client.connect(GRADIO_URL, { hf_token: token });
    client = gradioClient;
  } catch (e) {
    console.error("[signal-lab] gradio connect failed:", GRADIO_URL, e);
    throw new ApiError(
      0,
      `Cannot reach the Space at ${GRADIO_URL} — check VITE_GRADIO_URL (exact https URL, no trailing slash) and that the Space is Running, not Paused or Building.`,
    );
  }
  let job: Awaited<ReturnType<Client["predict"]>>;
  try {
    job = await client.predict("/predict", [
      base64ToWavFile(input.audioBase64),
      input.language,
    ]);
  } catch (e) {
    console.error("[signal-lab] gradio scan failed:", e);
    const detail = e instanceof Error ? `: ${e.message}` : "";
    throw new ApiError(
      0,
      `Space reached but the scan failed${detail} — the ZeroGPU worker may be cold (first scans take up to a minute). Wait, then retry.`,
    );
  }
  const data: unknown = (job as { data?: unknown })?.data;
  if (!Array.isArray(data) || data.length < 3) {
    console.error("[signal-lab] unexpected gradio payload:", job);
    throw new ApiError(0, "Unexpected response from the analysis Space.");
  }
  const [label, confidence, explanation] = data as [unknown, unknown, unknown];
  if (label === "REJECTED") throw new ApiError(400, String(explanation));
  return {
    status: "success",
    language: input.language,
    classification: label as Classification,
    confidenceScore: Number(confidence),
    explanation: String(explanation),
  };
}

export async function detectVoice(
  input: AnalyzeInput,
  signal?: AbortSignal,
): Promise<DetectResponse> {
  if (BACKEND === "gradio") return detectViaGradio(input, signal);
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
  if (BACKEND === "gradio") {
    // Gradio exposes no threshold endpoint; the Space README pins it.
    return { status: "healthy", threshold: GRADIO_THRESHOLD };
  }
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new ApiError(res.status, "Lab offline.");
  return (await res.json()) as HealthInfo;
}
