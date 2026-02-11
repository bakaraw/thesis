from transformers import BlipProcessor, BlipForConditionalGeneration
import os

MODEL_DIR = ".././models/blip-large"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading BLIP models...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

processor.save_pretrained(MODEL_DIR)
blip_model.save_pretrained(MODEL_DIR)
print(f"Models saved to {MODEL_DIR}")