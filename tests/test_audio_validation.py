"""Unit tests that run WITHOUT loading the 363 MB model.

Run:  python -m pytest tests/ -q   (or: python tests/test_audio_validation.py)
"""
import base64
import io
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio import (
    MAX_BASE64_CHARS,
    AudioValidationError,
    decode_audio,
)


def _wav_b64(audio: np.ndarray, sr: int = 16000) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def test_valid_sine_decodes():
    sr = 16000
    sine = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)
    out = decode_audio(_wav_b64(sine))
    assert out.dtype == np.float32
    assert len(out) == sr * 2
    assert abs(float(np.max(np.abs(out))) - 1.0) < 1e-5  # peak-normalized


def test_silence_rejected():
    try:
        decode_audio(_wav_b64(np.zeros(16000 * 2, dtype=np.float32)))
    except AudioValidationError:
        return
    raise AssertionError("silent audio should be rejected")


def test_too_short_rejected():
    sine = (0.5 * np.sin(2 * np.pi * 440 * np.arange(1000) / 16000)).astype(np.float32)
    try:
        decode_audio(_wav_b64(sine))
    except AudioValidationError:
        return
    raise AssertionError("sub-0.5s audio should be rejected")


def test_invalid_base64_rejected():
    try:
        decode_audio("!!!not-base64!!!")
    except AudioValidationError:
        return
    raise AssertionError("invalid base64 should be rejected")


def test_oversize_rejected_without_decoding():
    try:
        decode_audio("A" * (MAX_BASE64_CHARS + 1))
    except AudioValidationError:
        return
    raise AssertionError("oversize payload should be rejected")


def test_long_audio_truncated_to_30s():
    sr = 16000
    sine = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 35) / sr)).astype(np.float32)
    out = decode_audio(_wav_b64(sine))
    assert len(out) == sr * 30


if __name__ == "__main__":
    test_valid_sine_decodes()
    test_silence_rejected()
    test_too_short_rejected()
    test_invalid_base64_rejected()
    test_oversize_rejected_without_decoding()
    test_long_audio_truncated_to_30s()
    print("audio validation tests: 6/6 passed")
