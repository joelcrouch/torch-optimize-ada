import pytest
import torch
from PIL import Image
from torchoptim.models.text_generation import TextGenerationModel
from torchoptim.models.vision import VisionModel



@pytest.fixture
def tiny_text_model():
    # A tiny GPT-2 model (only a few MB) for fast testing
    return "hf-internal-testing/tiny-random-gpt2"

def test_text_model_metadata(tiny_text_model):
    """Verify metadata extraction works without crashing"""
    model = TextGenerationModel(tiny_text_model, device="cpu")
    model.load()
    meta = model.get_metadata()
    
    assert meta.name == tiny_text_model
    assert meta.model_type == "text-generation"
    assert meta.num_parameters > 0

def test_text_model_inference_shape(tiny_text_model):
    """Verify that infer() returns the correct dictionary keys"""
    model = TextGenerationModel(tiny_text_model, device="cpu")
    model.load()
    
    result = model.infer({"prompt": "Hello", "max_tokens": 5})
    assert "generated_text" in result
    assert "num_tokens" in result
    assert isinstance(result["generated_text"], str)

# def test_vision_model_metadata():
#     """Verify CLIP metadata extraction with a tiny model"""
#     # Using a tiny CLIP-like model to keep tests fast
#     tiny_vision = "hf-internal-testing/tiny-random-clip" 
#     model = VisionModel(tiny_vision, device="cpu")
#     model.load()
#     meta = model.get_metadata()
    
#     assert meta.model_type == "vision"
#     assert "CLIP" in meta.architecture or "clip" in meta.architecture.lower()

# def test_vision_inference_output():
#     """Verify vision adapter produces correct embedding dimensions"""
#     tiny_vision = "hf-internal-testing/tiny-random-clip"
#     model = VisionModel(tiny_vision, device="cpu")
#     model.load()
    
#     # Create a fake 224x224 image
#     dummy_image = Image.new('RGB', (224, 224), color='red')
#     result = model.infer({"image": dummy_image})
    
#     assert "embedding" in result
#     assert isinstance(result["embedding"], torch.Tensor)