"""Tests for schemas + shared metrics (no model load required).

Run:  python -m pytest tests/ -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import DetectRequest
from training.metrics import (
    auc_score,
    best_threshold,
    eer,
    expected_calibration_error,
    f1_score,
)


def test_accepts_wav_and_flac_formats():
    r = DetectRequest(language="English", audioFormat="wav", audioBase64="AAAA")
    assert r.audioFormat == "wav"
    r = DetectRequest(language="Hindi", audioFormat="FLAC", audioBase64="AAAA")
    assert r.audioFormat == "flac"


def test_rejects_bad_language_and_format():
    for kwargs in (
        {"language": "French", "audioFormat": "mp3", "audioBase64": "AAAA"},
        {"language": "English", "audioFormat": "ogg", "audioBase64": "AAAA"},
        {"language": "English", "audioFormat": "mp3", "audioBase64": ""},
    ):
        try:
            DetectRequest(**kwargs)
        except Exception:
            continue
        raise AssertionError(f"should have rejected {kwargs}")


def test_metrics_perfect_predictions():
    targets = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert abs(auc_score(targets, scores) - 1.0) < 1e-9
    assert abs(f1_score(targets, [0, 0, 1, 1]) - 1.0) < 1e-9
    assert expected_calibration_error(targets, scores) < 0.25
    t = best_threshold(targets, scores)
    assert 0.2 <= t <= 0.8
    e, _ = eer(targets, scores)
    assert e < 0.05


def test_metrics_random_predictions():
    targets = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert abs(auc_score(targets, scores) - 0.5) < 1e-9


if __name__ == "__main__":
    test_accepts_wav_and_flac_formats()
    test_rejects_bad_language_and_format()
    test_metrics_perfect_predictions()
    test_metrics_random_predictions()
    print("schema+metrics tests: 4/4 passed")
