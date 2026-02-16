import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from transformers import Wav2Vec2Processor

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.model import VoiceDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
MAX_LEN = 16000 * 6

def verify_regional(model_path="model/detector.pt"):
    print(f"🔍 Verifying model: {model_path} on {DEVICE}")
    model = VoiceDetector().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    results = {
        "original": {"total": 0, "correct": 0},
        "regional": {"total": 0, "correct": 0}
    }

    test_dirs = ["dataset/human", "dataset/ai"]
    
    with torch.no_grad():
        for label, cls in enumerate(["human", "ai"]):
            path = os.path.join("dataset", cls)
            if not os.path.exists(path): continue
            
            files = [f for f in os.listdir(path) if f.endswith(".wav")]
            # Sample a few to be fast
            import random
            random.shuffle(files)
            files = files[:100] 

            for f in files:
                audio_path = os.path.join(path, f)
                is_regional = f.startswith("indic_")
                category = "regional" if is_regional else "original"
                
                try:
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1: audio = audio.mean(axis=1)
                    if len(audio) > MAX_LEN: audio = audio[:MAX_LEN]
                    elif len(audio) < MAX_LEN: audio = np.pad(audio, (0, MAX_LEN - len(audio)))
                    
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                    x = inputs.input_values.to(DEVICE)
                    mask = torch.ones_like(x).to(DEVICE)
                    
                    logits = model(x, mask).squeeze(-1)
                    pred = (torch.sigmoid(logits) > 0.5).float().item()
                    
                    results[category]["total"] += 1
                    if pred == float(label):
                        results[category]["correct"] += 1
                except Exception as e:
                    logger.error(f"Error processing {cls} file '{f}' at {audio_path}", exc_info=True)
                    continue

    print("\n--- Final Verification Report ---")
    for cat, data in results.items():
        if data["total"] > 0:
            acc = (data["correct"] / data["total"]) * 100
            print(f"📍 {cat.upper()} Accuracy: {acc:.2f}% ({data['correct']}/{data['total']})")
        else:
            print(f"📍 {cat.upper()} Data: No samples found.")

if __name__ == "__main__":
    verify_regional()
