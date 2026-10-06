---
title: Voice detector
emoji: 🎤
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Signal Lab — Voice AI Detector

Forensic console for detecting AI-generated vs human voices. React console +
FastAPI inference over a fine-tuned wav2vec2 artifact detector.

![Stitch design](https://lh3.googleusercontent.com/aida/AEtjO1WYxRqvhSJfyQRDzQ9pIja0vG0_3aWtb0xHkAU9V-hJReYImqmSKQeqhToFqnUfwDy0FbNfRUgBjwT4tAplv5nt1PzJEhfWlGCjgLg6-dkuXkVwYFAde9O7tssORC0FahHAN0unRIROn8UqNbX3BGWsjz7h05szDZaeUqXI4uJ5cCN8-d_081swLyawBEuONiNjniHqnKvM7uvN7-zNeqQHSXUk4sfc276QcWRE-w9tdjTaSYlhBvpKNs0)

## Architecture

```
browser ──► FastAPI (serves frontend/dist + /api/*) ──► wav2vec2 + artifact CNN
                 │── POST /api/voice-detection (5/min, no auth in demo mode)
                 │── GET  /health (threshold, val accuracy)
                 └── GET  / → React console (Signal Lab After Hours theme)
```

## Quickstart

```bash
docker compose up          # console + API on :7860
# dev:
uvicorn app.main:app --port 7860        # backend
cd frontend && npm install && npm run dev  # console (proxies /api → :7860)
```

## Model (v1.1.0 weights, served as-is)

| metric (held-out, 300 clips) | value |
|---|---|
| accuracy | 96.3% |
| F1 / AUC / EER | 0.964 / 0.992 / 0.020 @ 0.89 |
| operating threshold | 0.85 (auto-loaded from `model/detector_config.json`) |
| best val accuracy | 98.2% |

Inference windows long clips into 6 s segments (train/inference parity),
rejects silence / <0.5 s / oversize payloads with 400 guidance. Tones and
white noise can still score confidently — silence is gated, tone rejection
is planned.

## API

`POST /api/voice-detection` `{language, audioFormat: mp3|wav|flac, audioBase64}`
→ `{status, language, classification: AI_GENERATED|HUMAN, confidenceScore, explanation}`.
Rate-limited (30/min global, 5/min scans) with `Retry-After`; the console
shows a countdown + retry. Set `THRESHOLD_<LANG>` env vars to override the
0.85 operating point per language.

## Layout

- `frontend/` — React + TS + Tailwind + TanStack Query console (`DESIGN.md` is the frozen Stitch design contract)
- `app/` — FastAPI: audio validation, windowed inference, thresholds, limits
- `training/` — train / held-out evaluate / shared metrics
- `tests/` — no-model unit tests (`python tests/test_*.py`)
