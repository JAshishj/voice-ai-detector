import base64
import io
import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000

def decode_audio(base64_audio: str) -> np.ndarray:
    audio_bytes = base64.b64decode(base64_audio)
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, sr, TARGET_SR)

    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=25)

    return audio
