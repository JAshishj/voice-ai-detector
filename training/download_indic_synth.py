import os
import argparse
from datasets import load_dataset, Audio
from tqdm import tqdm

# Mapping IndicSynth names to fleurs codes
LANG_MAP = {
    "Hindi": "hi_in",
    "Tamil": "ta_in",
    "Telugu": "te_in",
    "Malayalam": "ml_in"
}

def download_regional_data(languages=["Hindi", "Tamil", "Telugu", "Malayalam"], limit=1000, save_root="dataset"):
    print(f"🚀 Starting MASTER balanced download for: {languages}")
    
    if not languages:
        raise ValueError("languages must contain at least one language")
    
    os.makedirs(os.path.join(save_root, "human"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "ai"), exist_ok=True)

    counts = {"human": 0, "ai": 0, "ai_skipped": 0, "human_skipped": 0}
    target_per_lang = limit // len(languages)
    target_per_class = target_per_lang // 2

    for lang in languages:
        print(f"\n🌍 Processing Language: {lang}")
        
        # 1. AI samples from IndicSynth
        print(f"   🤖 Fetching AI samples (IndicSynth)...")
        try:
            ai_dataset = load_dataset("vdivyasharma/IndicSynth", name=lang, split="train", streaming=True)
            ai_dataset = ai_dataset.cast_column("audio", Audio(decode=False))
            ai_count = 0
            for item in ai_dataset:
                if ai_count >= target_per_class: break
                
                folder = os.path.join(save_root, "ai")
                filename = f"indic_ai_{lang}_{ai_count}.wav"
                save_path = os.path.join(folder, filename)
                
                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(item["audio"]["bytes"])
                    ai_count += 1
                    counts["ai"] += 1
                else:
                    counts["ai_skipped"] += 1
            print(f"   ✅ Collected {ai_count} AI samples.")
        except Exception as e:
            print(f"   ⚠️ Error fetching AI for {lang}: {e}")

        # 2. Human samples from fleurs
        f_code = LANG_MAP.get(lang)
        if f_code:
            print(f"   👤 Fetching Human samples (google/fleurs - {f_code})...")
            try:
                human_dataset = load_dataset("google/fleurs", f_code, split="train", streaming=True, trust_remote_code=True)
                human_dataset = human_dataset.cast_column("audio", Audio(decode=False))
                human_count = 0
                for item in human_dataset:
                    if human_count >= target_per_class: break
                    
                    folder = os.path.join(save_root, "human")
                    filename = f"indic_human_{lang}_{human_count}.wav"
                    save_path = os.path.join(folder, filename)
                    
                    if not os.path.exists(save_path):
                        with open(save_path, "wb") as f:
                            f.write(item["audio"]["bytes"])
                        human_count += 1
                        counts["human"] += 1
                    else:
                        counts["human_skipped"] += 1
                print(f"   ✅ Collected {human_count} Human samples.")
            except Exception as e:
                print(f"   ⚠️ Error fetching Human for {lang}: {e}")

    print(f"\n🏁 Finished!")
    print(f"📊 Total Indian Regional Samples: Human={counts['human']}, AI={counts['ai']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--langs", nargs="+", default=["Hindi", "Tamil", "Telugu", "Malayalam"])
    parser.add_argument("--root", default="dataset")
    args = parser.parse_args()
    
    download_regional_data(languages=args.langs, limit=args.limit, save_root=args.root)
