import os 
import librosa 
import soundfile as sf 
import numpy as np 
from tqdm import tqdm

TARGET_SR = 16000 
SUPPORTED = (".wav", ".mp3", ".flac") 
 
def process_file(path): 
    try:
        audio, sr = librosa.load(path, sr=None, mono=True) 
     
        # Resample if needed 
        if sr != TARGET_SR: 
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR) 
     
        # Normalize safely 
        peak = np.max(np.abs(audio)) 
        if peak > 0: 
            audio = audio / peak 
     
        # Always save as WAV 
        new_path = os.path.splitext(path)[0] + ".wav" 
        sf.write(new_path, audio, TARGET_SR) 
     
        # Delete original only if it's not already a WAV 
        if path != new_path and os.path.exists(path): 
            os.remove(path) 
     
        return True
        
    except Exception as e:
        print(f"\n⚠️ Error processing {path}: {e}")
        return False
 
def main(): 
    root = "dataset" 
 
    # First, collect all files to process
    files_to_process = []
    for dirpath, _, filenames in os.walk(root): 
        for f in filenames: 
            if f.lower().endswith(SUPPORTED): 
                files_to_process.append(os.path.join(dirpath, f))
    
    print(f"Found {len(files_to_process)} audio files to process")
    
    # Process with progress bar
    success_count = 0
    for filepath in tqdm(files_to_process, desc="Processing audio"):
        if process_file(filepath):
            success_count += 1
    
    print(f"✅ Audio preparation complete: {success_count}/{len(files_to_process)} files processed successfully")
 
if __name__ == "__main__": 
    main()