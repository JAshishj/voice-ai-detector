import base64
import io
import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000

def decode_audio(base64_audio: str) -> np.ndarray:
    audio_bytes = base64.b64decode(base64_audio)
    
    # Use librosa.load with io.BytesIO. 
    # librosa.load supports MP3 if ffmpeg is installed in the system.
    # We force sr=TARGET_SR and mono=True here to streamline the process.
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)

    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=25)

    return audio
