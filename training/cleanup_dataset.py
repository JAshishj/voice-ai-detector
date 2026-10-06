
import os
import soundfile as sf
import tqdm

def clean_dataset(root_dir="dataset"):
    print(f"🧹 Scanning {root_dir} for corrupted files...")
    deleted = 0
    checked = 0
    
    for r, d, f_list in os.walk(root_dir):
        for f in f_list:
            if f.endswith(".wav") or f.endswith(".mp3"):
                path = os.path.join(r, f)
                checked += 1
                try:
                    # Try to open the file to see if it's valid
                    data, samplerate = sf.read(path)
                    if len(data) < 100: # Remove tiny files
                         print(f"❌ Deleting too small file: {path}")
                         os.remove(path)
                         deleted += 1
                except Exception as e:
                    print(f"❌ Deleting CORRUPTED file: {path} ({e})")
                    try:
                        os.remove(path)
                        deleted += 1
                    except:
                        pass

    print(f"✅ Finished! Checked {checked} files. Deleted {deleted} bad files.")

if __name__ == "__main__":
    clean_dataset()
