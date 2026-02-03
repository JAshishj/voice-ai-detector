import os
import sys
import random
import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from tqdm import tqdm

os.environ["HF_HUB_OFFLINE"] = "1"

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.model import VoiceDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20  # Increased from 10
MAX_LEN = 16000 * 6  # 6 seconds
LEARNING_RATE = 3e-4
TRAIN_SPLIT = 0.85  # 85% train, 15% validation

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

class VoiceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        audio, sr = sf.read(path)
        audio = torch.from_numpy(audio).float()
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(dim=1)
        
        # Trim or pad to MAX_LEN
        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        elif len(audio) < MAX_LEN:
            # Pad with zeros if audio is shorter
            padding = torch.zeros(MAX_LEN - len(audio))
            audio = torch.cat([audio, padding])
        
        # Normalize
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        return audio, label


def collate_fn(batch):
    audios, labels = zip(*batch)

    inputs = processor(
        [a.numpy() for a in audios],
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    input_values = inputs.input_values
    attention_mask = torch.ones_like(input_values)

    return (
        input_values,
        attention_mask,
        torch.tensor(labels, dtype=torch.float32)
    )


def load_samples(root):
    """Load and split data into train and validation"""
    human_samples = []
    ai_samples = []

    for label, cls in enumerate(["human", "ai"]):
        cls_path = os.path.join(root, cls)
        if not os.path.exists(cls_path):
            print(f"⚠️ Warning: {cls_path} does not exist")
            continue

        for f in os.listdir(cls_path):
            if f.endswith(".wav"):
                sample = (os.path.join(cls_path, f), label)
                if label == 0:
                    human_samples.append(sample)
                else:
                    ai_samples.append(sample)

    # Shuffle each class separately
    random.shuffle(human_samples)
    random.shuffle(ai_samples)

    # Split each class into train and val (keeps class balance)
    human_split = int(len(human_samples) * TRAIN_SPLIT)
    ai_split = int(len(ai_samples) * TRAIN_SPLIT)

    train_samples = human_samples[:human_split] + ai_samples[:ai_split]
    val_samples = human_samples[human_split:] + ai_samples[ai_split:]

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def evaluate(model, loader):
    """Evaluate model on validation set"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    criterion = nn.BCELoss()

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(DEVICE)
            mask = mask.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x, mask).squeeze(-1)
            loss = criterion(preds, y)

            total_loss += loss.item()
            num_batches += 1

            # Convert predictions to labels
            predicted = (preds > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    model.train()
    accuracy = correct / total * 100 if total > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return accuracy, avg_loss


# ─── Main ─────────────────────────────────────────────
print(f"Using device: {DEVICE}")

# Load and split data
train_samples, val_samples = load_samples("dataset")
print(f"Train samples: {len(train_samples)}")
print(f"Val samples:   {len(val_samples)}")

train_dataset = VoiceDataset(train_samples)
val_dataset = VoiceDataset(val_samples)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

print("Initializing model...")
model = VoiceDetector().to(DEVICE)

# Class weights to handle imbalance (boost "human" class)
# pos_weight > 1 means penalize missing AI more
# pos_weight < 1 means penalize missing Human more
pos_weight = torch.tensor([0.8]).to(DEVICE)  # slightly boost human detection
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    model.cnn.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4  # L2 regularization
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.5
)

print("Starting training...")
model.train()

best_val_acc = 0.0
best_model_path = "model/detector_best.pt"

for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, mask, y in pbar:
        x = x.to(DEVICE)
        mask = mask.to(DEVICE)
        y = y.to(DEVICE)

        preds = model(x, mask).squeeze(-1)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_avg_loss = total_loss / num_batches

    # Validation
    val_acc, val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_avg_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  💾 New best model saved! Val Acc: {val_acc:.1f}%")

# Save final model too
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/detector.pt")

# Copy best model as the main one
import shutil
shutil.copy(best_model_path, "model/detector.pt")

print(f"\n✅ Training complete!")
print(f"   Best Val Accuracy: {best_val_acc:.1f}%")
print(f"   Model saved to model/detector.pt")