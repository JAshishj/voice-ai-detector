import torch
from transformers import Wav2Vec2Processor
from app.model import VoiceDetector

DEVICE = "cpu"

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

model = VoiceDetector().to(DEVICE)
model.load_state_dict(
    torch.load("model/detector.pt", map_location=DEVICE)
)
model.eval()

# Apply dynamic quantization to reduce memory usage (critical for Railway free tier)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

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
