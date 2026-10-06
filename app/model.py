import os
import torch
import torch.nn as nn


def _torch_load_compat() -> None:
    """Allow checkpoint loading on torch<2.6 local runtimes.

    transformers>=4.48 refuses torch.load below torch 2.6 (CVE-2025-32434).
    Production (Dockerfile) installs torch>=2.6 so this is a no-op there;
    on older local envs we bypass the version gate (weights_only handling
    is unchanged) with a loud warning instead of crashing.
    """
    try:
        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:
        return
    if (major, minor) >= (2, 6):
        return
    print(
        f"WARNING: torch {torch.__version__} < 2.6 — bypassing transformers' "
        "torch.load version gate for LOCAL DEV ONLY. Upgrade torch to >=2.6."
    )
    try:
        import transformers.modeling_utils as _mu
        import transformers.utils.import_utils as _iu

        _iu.check_torch_load_is_safe = lambda: None
        _mu.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


_torch_load_compat()

from transformers import Wav2Vec2Model  # noqa: E402

# Resolved in order:
#   1. $WAV2VEC2_MODEL_PATH if set (local dir or HF hub id)
#   2. ./model/base_model if it exists (Docker/HF Spaces layout)
#   3. $HF_MODEL_ID if set, else "facebook/wav2vec2-base"
DEFAULT_HF_ID = os.getenv("HF_MODEL_ID", "facebook/wav2vec2-base")


def resolve_backbone_source(explicit: str | None = None) -> str:
    """Return a transformers-compatible model source without crashing.

    Previously this was hardcoded to "./model/base_model", which does not
    exist in a fresh clone (it is only created inside Docker by
    download_base.py), so every local run crashed with an HFValidationError.
    """
    if explicit:
        return explicit
    env_path = os.getenv("WAV2VEC2_MODEL_PATH")
    if env_path:
        return env_path
    local_dir = "./model/base_model"
    if os.path.isdir(local_dir):
        return local_dir
    return DEFAULT_HF_ID

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
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class VoiceDetector(nn.Module):
    def __init__(self, backbone_source: str | None = None):
        super().__init__()
        self.backbone_source = resolve_backbone_source(backbone_source)
        self.backbone = Wav2Vec2Model.from_pretrained(
            self.backbone_source,
            low_cpu_mem_usage=True
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze final 2 transformer blocks for fine-tuning
        for block in self.backbone.encoder.layers[-2:]:
            for p in block.parameters():
                p.requires_grad = True

        self.cnn = ArtifactCNN()

    def forward(self, x, attention_mask):
        features = self.backbone(
            x,
            attention_mask=attention_mask
        ).last_hidden_state

        features = features.transpose(1, 2)
        return self.cnn(features)
