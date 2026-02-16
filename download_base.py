from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os

model_name = "facebook/wav2vec2-base"
save_path = "./model/base_model"

print(f"Downloading {model_name}...")
model = Wav2Vec2Model.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

print(f"Saving to {save_path}...")
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print("Done.")
