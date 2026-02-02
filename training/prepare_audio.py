import os
import librosa
import soundfile as sf

TARGET_SR = 16000

for root, _, files in os.walk("dataset"):
    for f in files:
        if f.endswith((".wav", ".mp3")):
            path = os.path.join(root, f)
            audio, sr = librosa.load(path, sr=None, mono=True)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, sr, TARGET_SR)
            audio = librosa.util.normalize(audio)
            sf.write(path.replace(".mp3", ".wav"), audio, TARGET_SR)
            print(f"Processed and saved: {path}")