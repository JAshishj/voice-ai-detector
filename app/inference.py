import torch
from transformers import Wav2Vec2Processor
from app.model import VoiceDetector

import os
import gc

DEVICE = "cpu"
_processor = None
_model = None
import psutil

def log_mem(step):
    mem = psutil.virtual_memory()
    print(f"DEBUG MEM [{step}]: Available: {mem.available / (1024**2):.2f}MB, Used: {mem.percent}%")

print(f"DEBUG: Application starting on PORT: {os.getenv('PORT', '8000')}")
log_mem("Startup")

def get_model():
    global _processor, _model
    
    if _model is not None:
        return _processor, _model

    log_mem("Before Processor")
    print("DEBUG: Loading Processor...")
    _processor = Wav2Vec2Processor.from_pretrained("./model/base_model")
    
    log_mem("Before Model Init")
    print("DEBUG: Initializing Model structure...")
    _model = VoiceDetector()
    
    log_mem("Before Weights Load")
    print("DEBUG: Loading weights (weights_only=False, mmap=True)...")
    state_dict = torch.load("model/detector.pt", map_location=DEVICE, weights_only=False, mmap=True)
    _model.load_state_dict(state_dict)
    del state_dict 
    
    _model.to(DEVICE)
    _model.eval()
    log_mem("After Weights Loaded")
    
    # Temporarily disabling quantization to see if we can at least reach "ready" state
    # print("DEBUG: Applying dynamic quantization...")
    # _model = torch.quantization.quantize_dynamic(
    #     _model, {torch.nn.Linear}, dtype=torch.qint8
    # )
    
    gc.collect() 
    log_mem("After GC")
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

