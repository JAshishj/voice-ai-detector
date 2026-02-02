import os
import random
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from app.model import VoiceDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 10
MAX_LEN = 16000 * 6

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

class VoiceDataset(Dataset):
    def __init__(self, root):
        self.data = []
        for label, cls in enumerate(["human", "ai"]):
            for lang in os.listdir(f"{root}/{cls}"):
                for f in os.listdir(f"{root}/{cls}/{lang}"):
                    if f.endswith(".wav"):
                        self.data.append(
                            (f"{root}/{cls}/{lang}/{f}", label)
                        )
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        audio, _ = torchaudio.load(path)
        audio = audio.squeeze()
        audio = audio[:MAX_LEN]
        return audio, label

def collate(batch):
    audios, labels = zip(*batch)
    inputs = processor(
        audios,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    return inputs.input_values, torch.tensor(labels).float()

dataset = VoiceDataset("dataset")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

model = VoiceDetector().to(DEVICE)
optimizer = torch.optim.Adam(model.cnn.parameters(), lr=1e-4)
criterion = nn.BCELoss()

model.train()

for epoch in range(EPOCHS):
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze()
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    print(f"Epoch {epoch+1} Loss: {total:.4f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/detector.pt")
print("Model saved.")
