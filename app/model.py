import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

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
        return torch.sigmoid(self.fc(x))


class VoiceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(
            "./model/base_model",
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
