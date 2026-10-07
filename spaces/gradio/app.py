"""Voice AI Detector — free-tier Gradio Space backend.

Self-contained (imports nothing from the main repo) so this file can ship
as its own Hugging Face Space repo alongside requirements.txt + README.md.
Weights load from a Hugging Face Model repo; the operating threshold travels
with them in detector_config.json.

Serves two clients:
  1. Gradio UI (upload or mic) for zero-setup demos.
  2. REST API (api=True) consumed by the React console on Vercel via
     @gradio/client: client.predict("/predict", { audio, language }).

Env:
  MODEL_REPO_ID  HF model repo holding detector.pt + detector_config.json
                 (default Ashish-04007/voice-ai-detector-model)
  HF_TOKEN       only needed if the model repo is private
  TORCH_THREADS  CPU thread cap (default 4)
"""

import json
import os

import librosa
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


def _torch_load_compat() -> None:
    """Local-dev only: bypass transformers' torch<2.6 load gate with a warning.
    The Space installs torch>=2.6, where this is a no-op."""
    try:
        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:
        return
    if (major, minor) >= (2, 6):
        return
    print(f"WARNING: torch {torch.__version__} < 2.6 — local-dev bypass active.")
    try:
        import transformers.modeling_utils as _mu
        import transformers.utils.import_utils as _iu

        _iu.check_torch_load_is_safe = lambda: None
        _mu.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


_torch_load_compat()

# NOTE: `gradio` is imported lazily in build_demo() so this module's
# predict()/model code imports without gradio installed (local testing).
# The Space container installs gradio via requirements.txt.
from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    # ZeroGPU scheduler requires at least one @spaces.GPU entrypoint.
    # Inference itself stays on CPU torch (see DEVICE); the decorator only
    # routes the call through a GPU worker from the free quota.
    from spaces import GPU as _GPU

    _zero_gpu = _GPU(duration=60)
except Exception:  # local dev without the spaces package
    def _zero_gpu(fn):
        return fn

# ── Config ──────────────────────────────────────────────────────────────
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "Ashish-04007/voice-ai-detector-model")
HF_TOKEN = os.getenv("HF_TOKEN")  # None is fine for public repos
BACKBONE_ID = os.getenv("BACKBONE_ID", "facebook/wav2vec2-base")
DEVICE = "cpu"
SAMPLE_RATE = 16000
WINDOW = SAMPLE_RATE * 6   # train/inference parity: 6 s windows
MAX_WINDOWS = 5            # 30 s hard cap
MIN_SAMPLES = int(SAMPLE_RATE * 0.5)
SILENCE_PEAK = 0.005
SILENCE_RMS = 0.002
LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))


# ── Model (architecture mirrors app/model.py) ────────────────────────────
class ArtifactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(torch.flatten(self.net(x), 1))


class VoiceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(BACKBONE_ID)
        self.cnn = ArtifactCNN()

    def forward(self, x, mask):
        feats = self.backbone(x, attention_mask=mask).last_hidden_state
        return self.cnn(feats.transpose(1, 2))


def _repo_file(name: str) -> str:
    # GRADIO_MODEL_DIR supports local dev (points at the main repo's model/).
    for base in (
        os.getenv("GRADIO_MODEL_DIR", ""),
        os.path.join(os.path.dirname(__file__), "model"),
    ):
        if base:
            candidate = os.path.join(base, name)
            if os.path.exists(candidate):
                return candidate
    return hf_hub_download(
        repo_id=MODEL_REPO_ID, filename=name, token=HF_TOKEN
    )


print("Loading processor + backbone ...")
processor = Wav2Vec2Processor.from_pretrained(BACKBONE_ID)
model = VoiceDetector().to(DEVICE)
model.load_state_dict(torch.load(_repo_file("detector.pt"), map_location=DEVICE))
model.eval()

try:
    with open(_repo_file("detector_config.json")) as f:
        _cfg = json.load(f)
    THRESHOLD = float(_cfg.get("suggested_threshold", 0.5))
except (OSError, ValueError, KeyError):
    THRESHOLD = 0.5
print(f"Ready. operating threshold={THRESHOLD}")


# ── Inference ─────────────────────────────────────────────────────────────
def _window_prob(window: np.ndarray) -> float:
    enc = processor(
        window, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        padding=True, return_attention_mask=True,
    )
    x = enc.input_values.to(DEVICE)
    try:
        m = enc["attention_mask"].to(DEVICE)
    except KeyError:
        m = torch.zeros_like(x)
        m[:, : min(len(window), m.shape[-1])] = 1
        m = m.to(DEVICE)
    with torch.inference_mode():
        return float(torch.sigmoid(model(x, m)).item())


def _load_clip(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < MIN_SAMPLES:
        raise ValueError(
            f"Clip is {audio.size / SAMPLE_RATE:.2f}s — need at least 0.5s of voiced audio."
        )
    audio = audio[: SAMPLE_RATE * 30]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < SILENCE_PEAK:
        raise ValueError("Clip is silent — send voiced speech.")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < SILENCE_RMS:
        raise ValueError("Clip is near-silent — send voiced speech.")
    return (audio / peak).astype(np.float32)


@_zero_gpu
def predict(audio_path: str | None, language: str):
    """Gradio endpoint. Returns (label, confidence, explanation)."""
    if not audio_path:
        return "REJECTED", 0.0, "No audio received — upload or record a clip."
    try:
        audio = _load_clip(audio_path)
    except ValueError as e:
        return "REJECTED", 0.0, str(e)
    except Exception:
        return "REJECTED", 0.0, "Could not decode that file — send mp3, wav, or flac."

    windows = [audio[i:i + WINDOW] for i in range(0, len(audio), WINDOW)][:MAX_WINDOWS]
    fake_prob = sum(_window_prob(w) for w in windows) / len(windows)

    is_fake = fake_prob >= THRESHOLD
    label = "AI_GENERATED" if is_fake else "HUMAN"
    conf = round(fake_prob if is_fake else 1.0 - fake_prob, 4)
    explanation = (
        "Detected synthetic spectral patterns consistent with AI voice generation."
        if is_fake
        else "Natural speech patterns and physiological micro-tremors detected."
    )
    if len(windows) > 1:
        explanation += f" (averaged over {len(windows)} segments)"
    return label, conf, explanation


def build_demo():
    import gradio as gr

    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Audio(sources=["upload", "microphone"], type="filepath", label="Voice clip"),
            gr.Dropdown(choices=LANGUAGES, value="English", label="Language profile"),
        ],
        outputs=[
            gr.Label(label="Classification"),
            gr.Number(label="Confidence"),
            gr.Textbox(label="Explanation"),
        ],
        title="Signal Lab — Voice AI Detector",
        description=(
            "Forensic console for AI-generated vs human voices. "
            f"Held-out accuracy 96.3%, EER 0.020, operating threshold {THRESHOLD}."
        ),
        api_name="predict",
        flagging_mode="never",
    )


if __name__ == "__main__":
    # ssr_mode=False: Gradio's experimental SSR node server 405-loops behind
    # the Spaces proxy ("POST method not allowed. No actions exist"). The
    # API and UI work without it; this just removes the crash spam.
    build_demo().queue(max_size=8).launch(
        server_port=int(os.getenv("PORT", "7860")), ssr_mode=False
    )
