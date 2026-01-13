import pytest
from torchoptim.models.registry import ModelRegistry
from torchoptim.core.model_interface import ModelInterface

def test_registry_stores_models():
    """Check if the @register decorator actually stores classes"""
    @ModelRegistry.register("test-type")
    class MockModel(ModelInterface):
        def load(self, name): pass
        def infer(self, inputs): return {}
        def get_signature(self): pass
        def get_metadata(self): pass
        def warmup(self): pass

    assert "test-type" in ModelRegistry._models
    instance = ModelRegistry.create("test-type", "dummy-name")
    assert isinstance(instance, MockModel)

def test_registry_error_on_unknown():
    """Ensure it raises ValueError for unregistered types"""
    with pytest.raises(ValueError, match="Unknown model type"):
        ModelRegistry.create("non-existent", "model-name")