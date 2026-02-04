import torch
from transformers import Wav2Vec2Processor
from app.model import VoiceDetector

DEVICE = "cpu"

print("DEBUG: Initializing Wav2Vec2Processor...")
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)
print("DEBUG: Processor initialized.")

print("DEBUG: Initializing VoiceDetector model structure...")
model = VoiceDetector().to(DEVICE)

print("DEBUG: Loading model weights...")
model.load_state_dict(
    torch.load("model/detector.pt", map_location=DEVICE, weights_only=False)
)
model.eval()
print("DEBUG: Model weights loaded.")

# Apply dynamic quantization to reduce memory usage (critical for Railway free tier)
print("DEBUG: Applying dynamic quantization...")
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("DEBUG: Model quantized successfully.")

def predict(audio):
    encoding = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    input_values = encoding.input_values.to(DEVICE)
    attention_mask = encoding.attention_mask.to(DEVICE)

    with torch.no_grad():
        prob = model(input_values, attention_mask).item()

    return prob
