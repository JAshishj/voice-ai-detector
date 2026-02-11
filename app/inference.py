import torch
from transformers import Wav2Vec2Processor
from app.model import VoiceDetector
import os
import gc

DEVICE = "cpu"
MODEL_PATH = "model/detector.pt"
BASE_MODEL_PATH = "./model/base_model"

print("Initializing Voice Detector Service...")

# 1. Load Processor
processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_PATH)

# 2. Load Model Structure
model = VoiceDetector()

# 3. Load Weights
# mmap=True helps but with 16GB it's optional; keeping for efficiency
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False, mmap=True)
model.load_state_dict(state_dict)
del state_dict # Free temp RAM

model.to(DEVICE)
model.eval()

# 4. Apply dynamic quantization for faster inference latency
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

gc.collect()
print("Service ready: Model loaded and quantized.")

def predict(audio):
    encoding = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    input_values = encoding.input_values.to(DEVICE)
    
    if hasattr(encoding, "attention_mask") and encoding.attention_mask is not None:
        attention_mask = encoding.attention_mask.to(DEVICE)
    else:
        attention_mask = torch.ones_like(input_values)

    with torch.inference_mode():
        # model expects (batch, seq_len)
        prob = model(input_values, attention_mask).item()

    return prob
