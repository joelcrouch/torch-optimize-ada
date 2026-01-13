import pytest
import torch
from torchoptim.core.model_interface import ModelInterface, ModelSignature, ModelMetadata
from torchoptim.models.registry import ModelRegistry

# 1. Create a Dummy class because we can't instantiate an ABC directly
class DummyModel(ModelInterface):
    def load(self, **kwargs): pass
    def infer(self, inputs): return {"output": "test"}
    def get_signature(self): return ModelSignature({}, {})
    def get_metadata(self): return ModelMetadata("test", "test", 0, "test")

def test_interface_initialization():
    """Verify that the base constructor stores variables correctly."""
    model = DummyModel(model_id="test-id", device="cpu")
    assert model.model_id == "test-id"
    assert model.device == "cpu"

def test_memory_info_logic():
    """Verify the GB conversion math in get_memory_info."""
    model = DummyModel(model_id="test-id")
    mem_info = model.get_memory_info()
    
    # On a machine with GPU, it should return GB numbers
    if torch.cuda.is_available():
        assert "allocated_gb" in mem_info
        assert isinstance(mem_info["allocated_gb"], float)
    else:
        assert mem_info["error"] == "CUDA not available"

def test_warmup_execution():
    """Ensure warmup can be called without crashing."""
    model = DummyModel(model_id="test-id")
    # This just ensures the method exists and runs
    model.warmup(num_iterations=1)

from torchoptim.models.registry import ModelRegistry

def test_registry_registration():
    """Verify that we can register and create a dummy model."""
    
    @ModelRegistry.register("test-type")
    class MockModel(DummyModel): # Using the DummyModel from before
        pass
    
    # Test creation
    instance = ModelRegistry.create("test-type", "some-name")
    assert isinstance(instance, MockModel)
    assert instance.model_id == "some-name"

def test_registry_error():
    """Verify registry raises error for unknown types."""
    with pytest.raises(ValueError, match="Unknown model type"):
        ModelRegistry.create("non-existent", "name")