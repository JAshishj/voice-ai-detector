---
title: Signal Lab Voice Detector API
emoji: 🎤
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
---

# Signal Lab — Voice AI Detector API

Free-tier Gradio backend for the Voice AI Detector console.

- Upload or mic-record a clip, pick a language profile, get a calibrated
  AI/HUMAN verdict (held-out accuracy 96.3%, EER 0.020, threshold 0.85).
- REST API enabled: the React console on Vercel calls `/predict` via
  `@gradio/client`.
- Weights load from the `Ashish-04007/voice-ai-detector-model` model repo
  (`MODEL_REPO_ID` env overrides); `HF_TOKEN` only needed for private repos.

Full-stack source: `Ashish-04007/voice-ai-detector` (GitHub).
