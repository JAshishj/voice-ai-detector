"""Audio decoding + input validation for the voice-detection API.

All limits are centralized here so train/inference/serving agree:
  TARGET_SR      16 kHz mono, peak-normalized float32
  MIN_SAMPLES    0.5 s  -> shorter clips are rejected (unreliable)
  MAX_SAMPLES   30   s  -> longer clips are truncated (latency/OOM guard)
  Silence / near-silence is rejected instead of confidently misclassified
  (previously silent input scored ~0.80 AI).
"""
import base64
import binascii
import io

import librosa
import numpy as np

TARGET_SR = 16000
MIN_SAMPLES = int(16000 * 0.5)   # 0.5 s
MAX_SAMPLES = int(16000 * 30)    # 30 s hard cap (inference windows this further)
MAX_BASE64_CHARS = 15_000_000    # ~11 MB decoded; 413 before we even decode
SILENCE_PEAK = 0.005             # below this peak amplitude -> silent
SILENCE_RMS = 0.002              # below this RMS -> near-silent


class AudioValidationError(ValueError):
    """Raised for invalid/unusable audio; mapped to HTTP 400 by the API."""


def decode_audio(base64_audio: str) -> np.ndarray:
    if not base64_audio:
        raise AudioValidationError("audioBase64 is empty")
    if len(base64_audio) > MAX_BASE64_CHARS:
        raise AudioValidationError(
            f"audio payload too large ({len(base64_audio)} chars, "
            f"max {MAX_BASE64_CHARS}); send at most ~30 s of audio"
        )

    try:
        audio_bytes = base64.b64decode(base64_audio, validate=True)
    except (binascii.Error, ValueError) as e:
        raise AudioValidationError(f"audioBase64 is not valid base64: {e}")
    if not audio_bytes:
        raise AudioValidationError("decoded audio is empty")

    # librosa.load handles WAV/MP3/FLAC (MP3 needs ffmpeg in the container).
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    except Exception as e:
        raise AudioValidationError(f"could not decode audio (need wav/mp3/flac): {e}")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise AudioValidationError("decoded audio has no samples")
    if audio.size < MIN_SAMPLES:
        raise AudioValidationError(
            f"audio too short ({audio.size / TARGET_SR:.2f}s, minimum "
            f"{MIN_SAMPLES / TARGET_SR:.1f}s)"
        )
    if audio.size > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]  # truncate; inference windows it further

    peak = float(np.max(np.abs(audio)))
    if peak < SILENCE_PEAK:
        raise AudioValidationError("audio is silent (peak amplitude near zero)")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < SILENCE_RMS:
        raise AudioValidationError("audio is near-silent (RMS too low to classify)")

    audio = audio / peak  # peak-normalize, matching training
    return audio.astype(np.float32)
