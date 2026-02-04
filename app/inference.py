import torch
from transformers import Wav2Vec2Processor
from app.model import VoiceDetector

import os
import gc

DEVICE = "cpu"
_processor = None
_model = None

print(f"DEBUG: Application starting on PORT: {os.getenv('PORT', '8000')}")
print(f"DEBUG: Model file exists: {os.path.exists('model/detector.pt')}")

def get_model():
    global _processor, _model
    
    if _model is not None:
        return _processor, _model

    print("DEBUG: Loading Processor...")
    _processor = Wav2Vec2Processor.from_pretrained("./model/base_model")
    
    print("DEBUG: Initializing Model structure...")
    # Using low_cpu_mem_usage=True to reduce peak RAM during load
    _model = VoiceDetector()
    
    print("DEBUG: Loading weights (weights_only=False, mmap=True)...")
    # mmap=True maps weights to disk/virtual memory instead of loading all into RAM at once
    state_dict = torch.load("model/detector.pt", map_location=DEVICE, weights_only=False, mmap=True)
    _model.load_state_dict(state_dict)
    del state_dict # Free RAM immediately
    
    _model.to(DEVICE)
    _model.eval()
    
    print("DEBUG: Applying dynamic quantization...")
    _model = torch.quantization.quantize_dynamic(
        _model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    gc.collect() # Force garbage collection
    print("DEBUG: Model ready.")
    return _processor, _model

def predict(audio):
    processor, model = get_model()
    
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

# DISABLED startup pre-load to allow container to start. 
# Loading will happen on the first request.
# try:
#     get_model()
# except Exception as e:
#     print(f"DEBUG: Startup pre-load failed: {e}")

