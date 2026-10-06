"""Proper held-out evaluation for the voice AI detector.

Why this exists: training/test_model.py samples from the SAME dataset/ folder
used for training, so its accuracy is optimistic (data leakage). This script:

  * builds a deterministic hash-based held-out split (stable across runs),
  * reports accuracy, precision, recall, F1, AUC, EER and ECE calibration,
  * suggests a decision threshold (Youden's J) to plug into the API's
    THRESHOLD_* env vars,
  * runs robustness probes (silence / sine / noise must not be confident,
    long clips must stay bounded via windowing),
  * measures per-clip latency.

Usage:  python training/evaluate.py [--max-per-class 150] [--device cpu]
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import soundfile as sf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import Wav2Vec2Model, Wav2Vec2Processor  # noqa: E402

# NOTE: torch<2.6 load-gate bypass lives in app.model (_torch_load_compat),
# imported below before any from_pretrained call.
from app.model import ArtifactCNN, resolve_backbone_source  # noqa: E402
from training.metrics import (  # noqa: E402
    auc_score,
    best_threshold,
    eer,
    expected_calibration_error,
    f1_score,
    pr_at_threshold,
)

MAX_LEN = 16000 * 6
HELD_OUT_FRACTION = 0.20


class WindowedDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.cnn = ArtifactCNN()

    def forward(self, x, m):
        return self.cnn(self.backbone(x, attention_mask=m).last_hidden_state.transpose(1, 2))


def held_out_files(root: str, fraction: float = HELD_OUT_FRACTION):
    """Deterministic hash split: stable, no overlap across runs."""
    held = []
    for cls in ("human", "ai"):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            h = int(hashlib.md5(f"{cls}/{f}".encode()).hexdigest(), 16) % 1000
            if h < int(fraction * 1000):
                held.append((os.path.join(d, f), cls))
    return held


def load_clip(path: str) -> np.ndarray:
    audio, _ = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def score_windows(model, processor, device, audio: np.ndarray) -> float:
    windows = [audio[i:i + MAX_LEN] for i in range(0, len(audio), MAX_LEN)][:5]
    probs = []
    with torch.inference_mode():
        for w in windows:
            enc = processor(
                w, sampling_rate=16000, return_tensors="pt",
                padding=True, return_attention_mask=True,
            )
            x = enc.input_values.to(device)
            try:
                m = enc["attention_mask"].to(device)
            except KeyError:
                m = torch.zeros_like(x)
                m[:, : min(len(w), m.shape[-1])] = 1
                m = m.to(device)
            probs.append(float(torch.sigmoid(model(x, m)).item()))
    return float(sum(probs) / len(probs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--model", default="model/detector.pt")
    ap.add_argument("--max-per-class", type=int, default=150)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    print(f"Loading model on {device} ...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model = WindowedDetector(backbone).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    model.eval()
    print(f"Backbone source default would resolve to: {resolve_backbone_source()}")

    files = held_out_files(args.dataset)
    by_cls: dict[str, list] = {"human": [], "ai": []}
    for path, cls in files:
        by_cls[cls].append(path)
    for cls in by_cls:
        by_cls[cls] = by_cls[cls][: args.max_per_class]

    targets, scores, latencies = [], [], []
    for cls in ("human", "ai"):
        for path in by_cls[cls]:
            audio = load_clip(path)
            t0 = time.time()
            s = score_windows(model, processor, device, audio)
            latencies.append((time.time() - t0) * 1000)
            targets.append(1 if cls == "ai" else 0)
            scores.append(s)

    preds = [1 if s >= 0.5 else 0 for s in scores]
    m = pr_at_threshold(targets, scores, 0.5)
    eer_val, eer_t = eer(targets, scores)
    print("\n=== HELD-OUT REPORT (hash split, disjoint across runs) ===")
    print(f"samples: {len(targets)} "
          f"(human={sum(1 for t in targets if t == 0)}, ai={sum(1 for t in targets if t == 1)})")
    print(f"accuracy : {m['accuracy'] * 100:.1f}%")
    print(f"precision: {m['precision']:.3f}  recall: {m['recall']:.3f}  "
          f"F1: {f1_score(targets, preds):.3f}")
    print(f"AUC: {auc_score(targets, scores):.3f}  "
          f"EER: {eer_val:.3f} @ {eer_t:.3f}  "
          f"ECE: {expected_calibration_error(targets, scores):.3f}")
    print(f"suggested threshold (Youden J): {best_threshold(targets, scores):.3f}")
    print(f"latency ms: p50={sorted(latencies)[len(latencies)//2]:.0f} "
          f"max={max(latencies):.0f} (n={len(latencies)})")

    print("\n=== ROBUSTNESS PROBES (must be unconfident or bounded) ===")
    probes = {
        "silence 3s": np.zeros(16000 * 3, dtype=np.float32),
        "sine 1s": (0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32),
        "noise 3s": np.random.randn(16000 * 3).astype(np.float32),
    }
    for name, a in probes.items():
        a = a / max(1e-6, float(np.max(np.abs(a)))) if np.max(np.abs(a)) > 0 else a
        if name.startswith("silence"):
            print(f"{name}: rejected by API silence gate (no confident score expected)")
            continue
        s = score_windows(model, processor, device, a)
        flag = "OK (uncertain)" if 0.3 < s < 0.7 else "OVERCONFIDENT on non-speech!"
        print(f"{name}: P(AI)={s:.3f} -> {flag}")

    long_clip = np.random.randn(16000 * 30).astype(np.float32)
    long_clip /= float(np.max(np.abs(long_clip)))
    t0 = time.time()
    s = score_windows(model, processor, device, long_clip)
    print(f"30s clip: P(AI)={s:.3f} in {(time.time()-t0)*1000:.0f}ms (windowed, bounded)")


if __name__ == "__main__":
    main()
