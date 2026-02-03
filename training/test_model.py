import os
import sys
import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.model import VoiceDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("Loading model...")
model = VoiceDetector().to(DEVICE)
model.load_state_dict(torch.load("model/detector.pt", map_location=DEVICE))
model.eval()

# Load processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def predict(audio_path):
    """Predict if audio is AI or Human"""
    audio, sr = sf.read(audio_path)
    audio = np.array(audio, dtype=np.float32)
    
    # Mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    
    # Process
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    input_values = inputs.input_values.to(DEVICE)
    attention_mask = torch.ones_like(input_values).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_values, attention_mask)
    
    score = prediction.item()
    label = "AI" if score > 0.5 else "Human"
    confidence = score if score > 0.5 else 1 - score
    
    return label, confidence, score


def test_from_dataset():
    """Test using samples from your own dataset"""
    print("\n📂 Testing from your dataset samples...")
    print("-" * 50)
    
    results = {"human": {"correct": 0, "total": 0},
               "ai": {"correct": 0, "total": 0}}
    
    # Pick 20 random samples from each category
    for actual_label in ["human", "ai"]:
        folder = os.path.join("dataset", actual_label)
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        
        # Pick min(20, available) samples
        sample_files = files[:20]
        
        for f in sample_files:
            path = os.path.join(folder, f)
            predicted_label, confidence, score = predict(path)
            
            is_correct = predicted_label.lower() == actual_label
            results[actual_label]["total"] += 1
            if is_correct:
                results[actual_label]["correct"] += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} File: {actual_label}/{f:20s} | "
                  f"Actual: {actual_label:5s} | "
                  f"Predicted: {predicted_label:5s} | "
                  f"Confidence: {confidence:.2%}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 ACCURACY REPORT")
    print("=" * 50)
    
    total_correct = 0
    total_samples = 0
    
    for label in ["human", "ai"]:
        correct = results[label]["correct"]
        total = results[label]["total"]
        accuracy = correct / total * 100 if total > 0 else 0
        total_correct += correct
        total_samples += total
        print(f"  {label.upper():5s} | Correct: {correct:2d}/{total:2d} | Accuracy: {accuracy:.1f}%")
    
    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    print("-" * 50)
    print(f"  OVERALL   | Correct: {total_correct:2d}/{total_samples:2d} | Accuracy: {overall:.1f}%")
    print("=" * 50)


def test_from_base64():
    """Test using a base64 encoded audio (simulates hackathon input)"""
    import base64
    import io
    
    print("\n🔧 Testing Base64 input (like hackathon)...")
    print("-" * 50)
    
    # Pick one sample from each category
    for label in ["human", "ai"]:
        folder = os.path.join("dataset", label)
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        sample_path = os.path.join(folder, files[0])
        
        # Read file and convert to base64
        with open(sample_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Decode and predict (simulating API input)
        audio_bytes = base64.b64decode(audio_b64)
        
        # Save temporarily
        temp_path = "temp_test.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        predicted_label, confidence, score = predict(temp_path)
        os.remove(temp_path)
        
        status = "✅" if predicted_label.lower() == label else "❌"
        print(f"{status} Actual: {label:5s} | Predicted: {predicted_label:5s} | Confidence: {confidence:.2%}")
    
    print("-" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("🎤 Voice AI Detector - Model Testing")
    print("=" * 50)
    
    # Test 1: Test accuracy from dataset
    test_from_dataset()
    
    # Test 2: Test base64 input (like hackathon)
    test_from_base64()