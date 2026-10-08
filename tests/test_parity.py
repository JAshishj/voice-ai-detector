"""Parity guard between the two serving backends.

app/main.py (FastAPI/Docker path) and spaces/gradio/app.py (free Gradio
Space) intentionally duplicate inference logic — the Space repo must be
self-contained (3 files, no monorepo imports). This test parses both files
as TEXT (no torch import, runs in seconds) and fails if the decision-critical
constants or policies drift apart.

Run:  python tests/test_parity.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def src(relative: str) -> str:
    with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
        return f.read()


MAIN = src(os.path.join("app", "main.py"))
INFER = src(os.path.join("app", "inference.py"))
AUDIO = src(os.path.join("app", "audio.py"))
SPACE = src(os.path.join("spaces", "gradio", "app.py"))


def check(name: str, condition: bool) -> None:
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(f"parity broken: {name}")


def test_window_policy() -> None:
    # 6 s windows, 30 s cap (5), stride down to 3 on long clips — both sides.
    check("6s window (main)", "SAMPLE_RATE * 6" in INFER)
    check("6s window (space)", "SAMPLE_RATE * 6" in SPACE)
    check(
        "max-5 stride-to-3 (main)",
        "MAX_WINDOWS" in INFER and "len(windows) // 2" in INFER,
    )
    check(
        "max-5 stride-to-3 (space)",
        "MAX_WINDOWS" in SPACE and "len(windows) // 2" in SPACE,
    )


def test_gates() -> None:
    # 0.5 s minimum, silence gates — both sides reject the same inputs.
    # (main-path gates live in app/audio.py; space gates are inline.)
    for label, code in (("main", MAIN + INFER + AUDIO), ("space", SPACE)):
        check(f"0.5s minimum ({label})", "0.5" in code and "MIN_SAMPLES" in code)
        check(f"silence gate ({label})", "SILENCE_PEAK" in code and "SILENCE_RMS" in code)


def test_thresholds() -> None:
    # Per-language THRESHOLD_* overrides with a trained-config default.
    for label, code in (("main", MAIN), ("space", SPACE)):
        check(f"per-language thresholds ({label})", "LANGUAGE_THRESHOLDS" in code)
        check(f"THRESHOLD_<LANG> env ({label})", "THRESHOLD_" in code)
    check("config default (space)", "suggested_threshold" in SPACE)


def test_verdict_copy() -> None:
    # Identical user-facing verdict strings on both backends.
    for phrase in (
        "Detected synthetic spectral patterns",
        "physiological micro-tremors",
        "averaged over",
    ):
        check(f"copy: {phrase[:30]}…", phrase in MAIN and phrase in SPACE)


if __name__ == "__main__":
    test_window_policy()
    test_gates()
    test_thresholds()
    test_verdict_copy()
    print("parity tests: all passed")
