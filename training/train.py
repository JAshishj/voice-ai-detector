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
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import torch.cuda.amp as amp

os.environ["HF_HUB_OFFLINE"] = "1"

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.model import VoiceDetector
from training.metrics import auc_score, best_threshold, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
ACCUM_STEPS = 2  # Effective Batch Size = 16 * 2 = 32
EPOCHS = 10  # Reduced from 20 for faster iterations
MAX_LEN = 16000 * 6  # 6 seconds (must match app.inference.WINDOW_SAMPLES)
MIN_LEN = int(16000 * 0.5)  # skip clips shorter than 0.5 s (unreliable)
LEARNING_RATE = 2e-5
TRAIN_SPLIT = 0.85  # 85% train, 15% validation
EARLY_STOP_PATIENCE = 3  # stop if val loss doesn't improve for N epochs
# Windows uses spawn (no fork) and this is a flat script with no __main__
# guard, so worker processes would re-execute training and crash. Keep
# single-process loading there; Linux/macOS containers get real workers.
NUM_WORKERS = 0 if os.name == "nt" else min(4, (os.cpu_count() or 2))

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

class VoiceDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        if augment:
            self.augmenter = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        audio, sr = sf.read(path)
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float()

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(dim=1)

        # Trim to MAX_LEN (no zero-padding here: the collate fn pads the
        # batch and builds a REAL attention mask, so the backbone ignores
        # padding instead of treating it as speech).
        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]

        # Normalize
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        # Apply augmentation if in training mode
        if self.augment:
            audio_np = audio.numpy()
            augmented = self.augmenter(samples=audio_np, sample_rate=16000)
            audio = torch.from_numpy(np.asarray(augmented, dtype=np.float32)).float()
            if len(audio) > MAX_LEN:
                audio = audio[:MAX_LEN]

        return audio, label


def collate_fn(batch):
    audios, labels = zip(*batch)

    inputs = processor(
        [a.numpy() for a in audios],
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,  # required: transformers>=5 omits it by default
        return_tensors="pt",
    )

    input_values = inputs.input_values
    # Real mask from the processor: 1 for speech, 0 for padding.
    try:
        attention_mask = inputs["attention_mask"]
    except KeyError:  # defensive: build from pre-pad lengths
        lengths = [len(a) for a in audios]
        attention_mask = torch.zeros_like(input_values)
        for i, n in enumerate(lengths):
            attention_mask[i, : min(n, attention_mask.shape[-1])] = 1

    return (
        input_values,
        attention_mask,
        torch.tensor(labels, dtype=torch.float32),
    )


def load_samples(root):
    """Load and split data into train and validation"""
    human_samples = []
    ai_samples = []

    for label, cls in enumerate(["human", "ai"]):
        cls_path = os.path.join(root, cls)
        if not os.path.exists(cls_path):
            print(f"[!] Warning: {cls_path} does not exist")
            continue

        for f in os.listdir(cls_path):
            if f.endswith(".wav"):
                sample = (os.path.join(cls_path, f), label)
                # Skip unreadable / too-short clips up front so tiny files
                # can't poison batches with all-padding inputs.
                try:
                    info = sf.info(os.path.join(cls_path, f))
                    n = int(info.frames)
                    if n < MIN_LEN:
                        continue
                except Exception:
                    continue
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
    """Evaluate model on validation set.

    Returns (accuracy, avg_loss, f1, auc, scores, labels) where scores are
    P(AI) per sample. F1/AUC are computed without sklearn to keep the
    training env lean.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_scores: list[float] = []
    all_labels: list[int] = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(DEVICE)
            mask = mask.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x, mask).squeeze(-1)
            loss = criterion(logits, y)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits).detach().cpu().tolist()
            all_scores.extend(probs)
            all_labels.extend(y.detach().cpu().tolist())

    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    preds = [1 if s > 0.5 else 0 for s in all_scores]
    correct = sum(1 for p, t in zip(preds, all_labels) if p == int(t))
    total = len(all_labels)
    accuracy = correct / total * 100 if total > 0 else 0
    f1 = f1_score(all_labels, preds)
    auc = auc_score(all_labels, all_scores)

    return accuracy, avg_loss, f1, auc, all_scores, all_labels


# ─── Main ─────────────────────────────────────────────
print(f"Using device: {DEVICE}")

# Load and split data
train_samples, val_samples = load_samples("dataset")
print(f"Train samples: {len(train_samples)}")
print(f"Val samples:   {len(val_samples)}")

train_dataset = VoiceDataset(train_samples, augment=True)
val_dataset = VoiceDataset(val_samples, augment=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

print("Initializing model...")
model = VoiceDetector().to(DEVICE)

# Class weights to handle imbalance (boost "human" class)
# pos_weight > 1 means penalize missing AI more
# pos_weight < 1 means penalize missing Human more
# Treated equally (pos_weight=1.0) since we have a balanced dataset (933 each)
pos_weight = torch.tensor([1.0]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.5
)

# AMP GradScaler
scaler = amp.GradScaler(enabled=(DEVICE == "cuda"))

print("Starting training...")
model.train()

best_val_acc = 0.0
best_val_loss = float("inf")
epochs_no_improve = 0
best_model_path = "model/detector_best.pt"
best_threshold_value = 0.5

for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for i, (x, mask, y) in enumerate(pbar):
        x = x.to(DEVICE)
        mask = mask.to(DEVICE)
        y = y.to(DEVICE)

        # AMP Forward Pass
        with amp.autocast(enabled=(DEVICE == "cuda")):
            preds = model(x, mask).squeeze(-1)
            loss = criterion(preds, y)
            
            # Normalize loss for accumulation
            loss = loss / ACCUM_STEPS

        # AMP Backward Pass
        scaler.scale(loss).backward()

        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS # Scale back for logging
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item() * ACCUM_STEPS:.4f}'})

    # Flush any leftover accumulated gradients
    if (i + 1) % ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    train_avg_loss = total_loss / num_batches

    # Validation
    val_acc, val_loss, val_f1, val_auc, val_scores, val_labels = evaluate(model, val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_avg_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.1f}% | "
          f"Val F1: {val_f1:.3f} | "
          f"Val AUC: {val_auc:.3f}")

    # Track best by accuracy; early-stop on val loss.
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_threshold_value = best_threshold(val_labels, val_scores)
        torch.save(model.state_dict(), best_model_path)
        print(f"  [save] New best model saved! Val Acc: {val_acc:.1f}% "
              f"(val threshold suggestion: {best_threshold_value:.3f})")

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"  [stop] Early stopping: val loss stagnant for {EARLY_STOP_PATIENCE} epochs")
        break

# Save final model too
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/detector.pt")

# Copy best model as the main one
import shutil
import json
shutil.copy(best_model_path, "model/detector.pt")

# Persist inference parity config (window length + suggested threshold) so
# serving stays in sync with how the model was trained.
with open("model/detector_config.json", "w") as f:
    json.dump(
        {
            "sample_rate": 16000,
            "max_len": MAX_LEN,
            "min_len": MIN_LEN,
            "suggested_threshold": round(best_threshold_value, 4),
            "best_val_acc": round(best_val_acc, 2),
        },
        f,
        indent=2,
    )

print(f"\n[ok] Training complete!")
print(f"   Best Val Accuracy: {best_val_acc:.1f}%")
print(f"   Suggested threshold (Youden's J on val): {best_threshold_value:.3f}")
print(f"   Model saved to model/detector.pt (+ detector_config.json)")
