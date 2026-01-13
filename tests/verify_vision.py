import torch
from PIL import Image
import requests
from torchoptim.models.registry import ModelRegistry
import torchoptim.models.vision # Triggers registration

def verify_vision():
    print("🚀 Day 1: Starting CLIP Vision Verification...")
    
    # 1. Create CLIP via Registry
    model_id = "openai/clip-vit-base-patch32"
    model = ModelRegistry.create("vision", model_id)
    model.load()
    
    # 2. Get a sample image (Two cats)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    # 3. Run Inference (Image Encoding)
    print("📸 Encoding Image...")
    img_result = model.infer({"image": image})
    
    # 4. Run Inference (Text Encoding)
    print("🔤 Encoding Text...")
    txt_result = model.infer({"text": "a photo of two cats"})

    # 5. Metadata Check
    meta = model.get_metadata()
    print(f"\n✅ SUCCESS!")
    print(f"📊 Model: {meta.name}")
    print(f"🧩 Params: {meta.num_parameters / 1e6:.2f}M")
    print(f"📏 Embedding Dim: {img_result['embedding_dim']}")

if __name__ == "__main__":
    verify_vision()