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

export interface AnalyzeInput {
  audioBase64: string;
  language: Language;
  audioFormat: "mp3" | "wav" | "flac";
}

function base64ToWavBlob(b64: string): Blob {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: "audio/wav" });
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
  gradioClient ??= await Client.connect(GRADIO_URL);
  const client: Client = gradioClient;
  const job = await client.predict("/predict", [
    base64ToWavBlob(input.audioBase64),
    input.language,
  ]);
  const [label, confidence, explanation] = job.data as [string, number, string];
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
