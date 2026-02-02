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

def predict(audio):
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values.to(DEVICE)

    with torch.no_grad():
        prob = model(inputs).item()

    return prob
