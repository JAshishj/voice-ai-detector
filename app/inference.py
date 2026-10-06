"""Efficient inference for the voice AI detector.

Fixes vs the previous version:
  * 6 s window parity with training (was: full-length fed to the model, so a
    30 s clip took ~2 s on CPU and risked OOM). Long clips are now split into
    non-overlapping 6 s windows (max 5 = 30 s) and window probabilities are
    averaged — bounded latency, consistent with how the model was trained.
  * Real attention masks (was: all-ones, so padding counted as speech).
  * Model source resolves via app.model.resolve_backbone_source instead of a
    hardcoded ./model/base_model path that crashes outside Docker.
  * Quantization is opt-out (QUANTIZE=0) and CPU-only; it barely helped on
    transformer blocks, so it no longer runs unconditionally at import.
  * Torch thread count is capped (TORCH_THREADS, default 4) to avoid
    oversubscription on shared CPU hosts.
"""
import gc
import os

import numpy as np
import torch
from transformers import Wav2Vec2Processor

from app.model import VoiceDetector, resolve_backbone_source

DEVICE = os.getenv("DEVICE", "cpu")  # keep CPU default for HF Spaces parity
SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 6   # train MAX_LEN parity
MAX_WINDOWS = 5                    # 5 x 6 s = 30 s = audio.MAX_SAMPLES
MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "model/detector.pt")

torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))

print("Initializing Voice Detector Service...")


def _resolve_processor_source() -> str:
    local_dir = "./model/base_model"
    if os.path.isdir(local_dir):
        return local_dir
    return resolve_backbone_source()


processor = Wav2Vec2Processor.from_pretrained(_resolve_processor_source())

model = VoiceDetector()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
del state_dict  # free temp RAM

model.to(DEVICE)
model.eval()

if os.getenv("QUANTIZE", "1") == "1" and DEVICE == "cpu":
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Dynamic quantization applied.")
    except Exception as e:  # quantized transformer can fail on some builds
        print(f"Quantization skipped: {e}")

gc.collect()
print("Service ready: Model loaded.")


def _window_prob(window: np.ndarray) -> float:
    encoding = processor(
        window,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,  # required: transformers>=5 omits it by default
    )
    input_values = encoding.input_values.to(DEVICE)
    try:
        attention_mask = encoding["attention_mask"].to(DEVICE)
    except KeyError:  # defensive: single window, mask = its true length
        attention_mask = torch.zeros_like(input_values)
        attention_mask[:, : min(len(window), attention_mask.shape[-1])] = 1
        attention_mask = attention_mask.to(DEVICE)

    with torch.inference_mode():
        logits = model(input_values, attention_mask)
        return float(torch.sigmoid(logits).item())


def predict(audio: np.ndarray) -> float:
    """Return P(AI-generated) in [0, 1] for 16 kHz mono float32 audio."""
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0:
        raise ValueError("empty audio")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak

    # Split into 6 s windows (last partial window is processor-padded, and the
    # real attention mask tells the backbone to ignore that padding).
    windows = [
        audio[i:i + WINDOW_SAMPLES]
        for i in range(0, len(audio), WINDOW_SAMPLES)
    ][:MAX_WINDOWS]

    probs = [_window_prob(w) for w in windows]
    return float(sum(probs) / len(probs))


def predict_with_meta(audio: np.ndarray) -> dict:
    """predict() plus diagnostics for logging/explanations."""
    prob = predict(audio)
    n_windows = min(
        MAX_WINDOWS,
        (int(np.asarray(audio).size) + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES,
    )
    return {"fake_prob": prob, "num_windows": max(n_windows, 1)}
