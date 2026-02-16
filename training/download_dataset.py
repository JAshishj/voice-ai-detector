import os
from datasets import load_dataset, Audio
from tqdm import tqdm

DATASET_NAME = "kartikeykushwaha14/deepfake-audio-detection"
SAVE_ROOT = "dataset"

# Load with decode=False to prevent torchcodec from being triggered
dataset = load_dataset(
    DATASET_NAME, 
    split="train", 
    streaming=True
)

# Cast the audio column to prevent automatic decoding
dataset = dataset.cast_column("audio", Audio(decode=False))

os.makedirs(SAVE_ROOT, exist_ok=True)

# Create human and ai folders
os.makedirs(os.path.join(SAVE_ROOT, "human"), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, "ai"), exist_ok=True)

# Counters for each category
human_count = 0
ai_count = 0

for idx, item in enumerate(tqdm(dataset, desc="Downloading files")):
    try:
        label = item.get('label', None)
        
        # Map label to folder name
        if label == 0:
            folder_name = "human"
            file_idx = human_count
            human_count += 1
        elif label == 1:
            folder_name = "ai"
            file_idx = ai_count
            ai_count += 1
        else:
            # Unknown label, skip or put in a separate folder
            print(f"\n⚠️ Unknown label {label} for file {idx}, skipping")
            continue
        
        folder = os.path.join(SAVE_ROOT, folder_name)
        
        # Use the original filename if available, otherwise use counter
        original_filename = item['audio'].get('path', f"{file_idx}.flac")
        if '.' in original_filename:
            ext = original_filename.split('.')[-1]
        else:
            ext = 'flac'
        
        save_path = os.path.join(folder, f"{file_idx}.{ext}")
        
        if not os.path.exists(save_path):
            # Get the raw bytes directly from the dataset
            audio_bytes = item['audio']['bytes']
            
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
                    
    except Exception as e:
        print(f"\n⚠️ Error saving file {idx}: {e}")
        continue

print(f"✅ Dataset downloaded to '{SAVE_ROOT}'")
print(f"   Human samples: {human_count}")
print(f"   AI samples: {ai_count}")