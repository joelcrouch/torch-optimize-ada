import torch
from PIL import Image
import requests
from torchoptim.models.registry import ModelRegistry
import torchoptim.models.vision  # Triggers registration

def test_vision():
    print("\n--- Starting CLIP Verification ---")
    
    # 1. Create CLIP via Registry
    model = ModelRegistry.create("vision", "openai/clip-vit-base-patch32")
    model.load()
    
    # 2. Grab a test image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Two cats image
    image = Image.open(requests.get(url, stream=True).raw)
    
    # 3. Run Inference (Image Encoding)
    print("📸 Encoding Image...")
    img_result = model.infer({"image": image})
    print(f"✅ Image Embedding Shape: {img_result['embedding'].shape}")
    
    # 4. Run Inference (Text Encoding)
    print("🔤 Encoding Text...")
    txt_result = model.infer({"text": "a photo of two cats"})
    print(f"✅ Text Embedding Shape: {txt_result['embedding'].shape}")

    # 5. Check Metadata
    meta = model.get_metadata()
    print(f"\n📊 Model: {meta.name}")
    print(f"🧩 Params: {meta.num_parameters / 1e6:.2f}M")

if __name__ == "__main__":
    test_vision()