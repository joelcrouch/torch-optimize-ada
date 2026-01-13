import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelSignature:
    """Input/Output schema to prevent data mismatches."""
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

@dataclass
class ModelMetadata:
    """Hardware-relevant info for optimization decisions."""
    name: str
    model_type: str  # e.g., "text-generation", "vision"
    num_parameters: int
    architecture: str

class ModelInterface(ABC):
    """Generic interface for any ML model in the TorchOptim ecosystem."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load model into memory (VRAM)."""
        pass

    @abstractmethod
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Unified inference method."""
        pass

    @abstractmethod
    def get_signature(self) -> ModelSignature:
        """Enforces schema-checking."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Provides stats for benchmarking/optimization."""
        pass

    def warmup(self, num_iterations: int = 3) -> None:
        """Warms up GPU kernels for accurate benchmarking."""
        if not torch.cuda.is_available(): return
        
        # dummy_input logic would be implemented in children
        print(f"🔥 Warming up {self.model_id} for {num_iterations} iterations...")
        # (Child classes will fill this with a real dummy pass)

    def get_memory_info(self) -> Dict[str, float]:
        """Helper to monitor your RTX 500's tight 3.65GB VRAM."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3
            }
        return {"error": "CUDA not available"}