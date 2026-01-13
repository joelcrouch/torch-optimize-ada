# SPRINT 1: Framework Foundation & Baseline
## User Stories & Tasks (Days 1-4)

---

## **Sprint 1 Goal**
Build the core abstraction layer that works with any PyTorch model, demonstrate with 2 different model types (LLM + Vision), and establish baseline performance metrics.

**Definition of Done:**
- ✅ Core framework interfaces implemented (ModelInterface, Profiler, Benchmarker)
- ✅ Model registry can load any HuggingFace/PyTorch model
- ✅ Generic profiling and benchmarking working
- ✅ Llama 3.2-1B + CLIP both registered and running through framework
- ✅ Baseline metrics documented
- ✅ Framework generality proven

---

## **DAY 1: Core Abstractions & Project Setup DONE!**

### 📋 User Story 1.1: Model Interface Abstraction
**As a** framework user  
**I want** a generic interface to load any PyTorch model  
**So that** I can optimize different model types without rewriting code

**Acceptance Criteria:**
- [✅] `ModelInterface` abstract class created
- [ ] Can load models from HuggingFace
- [ ] Can load models from PyTorch
- [ ] Automatic input/output schema detection
- [ ] Model metadata extraction (params, architecture, etc.)
- [ ] Works with at least 2 different model types

**Tasks:**
- [✅] Create project structure:
  ```
  torchoptim/
  ├── README.md
  ├── requirements.txt
  ├── setup.py
  ├── torchoptim/
  │   ├── __init__.py
  │   ├── core/
  │   │   ├── __init__.py
  │   │   ├── model_interface.py
  │   │   ├── profiler.py
  │   │   └── benchmarker.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── registry.py
  │   │   ├── text_generation.py
  │   │   └── vision.py
  │   ├── optimizations/
  │   │   ├── __init__.py
  │   │   └── base.py
  │   └── utils/
  │       ├── __init__.py
  │       └── helpers.py
  ├── examples/
  │   └── basic_usage.py
  ├── tests/
  │   └── test_interface.py
  └── notebooks/
      └── 01_getting_started.ipynb
  ```

- [✅] Implement `torchoptim/core/model_interface.py`:
  ```python
  from abc import ABC, abstractmethod
  from typing import Dict, Any, List
  from dataclasses import dataclass
  
  @dataclass
  class ModelSignature:
      """Model input/output schema"""
      input_schema: Dict[str, Any]
      output_schema: Dict[str, Any]
      
  @dataclass
  class ModelMetadata:
      """Model metadata"""
      name: str
      model_type: str  # "text-generation", "vision", "multimodal"
      num_parameters: int
      architecture: str
      
  class ModelInterface(ABC):
      """Generic interface for any ML model"""
      
      @abstractmethod
      def load(self, model_name: str, **kwargs) -> None:
          """Load model from registry"""
          pass
      
      @abstractmethod
      def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
          """Run inference"""
          pass
      
      @abstractmethod
      def get_signature(self) -> ModelSignature:
          """Get input/output schema"""
          pass
      
      @abstractmethod
      def get_metadata(self) -> ModelMetadata:
          """Get model metadata"""
          pass
      
      @abstractmethod
      def warmup(self, num_iterations: int = 3) -> None:
          """Warmup model for benchmarking"""
          pass
  ```

- [✅] Create `torchoptim/models/registry.py`:
  ```python
  from typing import Dict, Type
  from torchoptim.core.model_interface import ModelInterface
  
  class ModelRegistry:
      """Registry for model implementations"""
      
      _models: Dict[str, Type[ModelInterface]] = {}
      
      @classmethod
      def register(cls, model_type: str):
          """Decorator to register model implementations"""
          def decorator(model_class: Type[ModelInterface]):
              cls._models[model_type] = model_class
              return model_class
          return decorator
      
      @classmethod
      def create(cls, model_type: str, model_name: str, **kwargs) -> ModelInterface:
          """Create model instance"""
          if model_type not in cls._models:
              raise ValueError(f"Unknown model type: {model_type}")
          return cls._models[model_type](model_name, **kwargs)
  ```

- [✅] Write basic tests in `tests/test_interface.py`
- [✅] Set up development environment (Docker or virtualenv)
- [✅] Install dependencies:
  ```
  torch>=2.1.0
  transformers>=4.35.0
  accelerate>=0.25.0
  sentencepiece>=0.1.99
  pillow>=10.0.0
  numpy>=1.24.0
  ```

**Time Box:** 4-5 hours

**Verification:**
```python
# Should be able to do this at end of day
from torchoptim.models.registry import ModelRegistry

# This won't work yet, but interface should be ready
# model = ModelRegistry.create("text-generation", "meta-llama/Llama-3.2-1B")
```

---

### 📋 User Story 1.2: Text Generation Model Adapter
**As a** framework user  
**I want** to load LLMs through the generic interface  
**So that** I can optimize text generation models

**Acceptance Criteria:**
- [ ] `TextGenerationModel` class implements `ModelInterface`
- [ ] Can load any HuggingFace causal LM
- [ ] Handles tokenization automatically
- [ ] Returns structured output
- [ ] Registered in ModelRegistry

**Tasks:**
- [ ] Implement `torchoptim/models/text_generation.py`:
  ```python
  from typing import Dict, Any
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from torchoptim.core.model_interface import ModelInterface, ModelSignature, ModelMetadata
  from torchoptim.models.registry import ModelRegistry
  
  @ModelRegistry.register("text-generation")
  class TextGenerationModel(ModelInterface):
      def __init__(self, model_name: str, device: str = "cuda", **kwargs):
          self.model_name = model_name
          self.device = device
          self.model = None
          self.tokenizer = None
          
      def load(self, model_name: str = None, **kwargs) -> None:
          """Load HuggingFace causal LM"""
          name = model_name or self.model_name
          self.tokenizer = AutoTokenizer.from_pretrained(name)
          self.model = AutoModelForCausalLM.from_pretrained(
              name,
              torch_dtype=torch.float16,
              device_map=self.device
          )
          self.model.eval()
          
      def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
          """Run text generation"""
          prompt = inputs.get("prompt", "")
          max_tokens = inputs.get("max_tokens", 50)
          
          input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
          
          with torch.no_grad():
              outputs = self.model.generate(
                  input_ids,
                  max_new_tokens=max_tokens,
                  do_sample=False
              )
          
          generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
          
          return {
              "generated_text": generated_text,
              "num_tokens": len(outputs[0])
          }
      
      def get_signature(self) -> ModelSignature:
          return ModelSignature(
              input_schema={"prompt": "str", "max_tokens": "int"},
              output_schema={"generated_text": "str", "num_tokens": "int"}
          )
      
      def get_metadata(self) -> ModelMetadata:
          num_params = sum(p.numel() for p in self.model.parameters())
          return ModelMetadata(
              name=self.model_name,
              model_type="text-generation",
              num_parameters=num_params,
              architecture=self.model.config.architectures[0]
          )
      
      def warmup(self, num_iterations: int = 3) -> None:
          """Warmup for stable benchmarking"""
          dummy_input = {"prompt": "Hello", "max_tokens": 10}
          for _ in range(num_iterations):
              self.infer(dummy_input)
  ```

- [ ] Test loading Llama 3.2-1B: `meta-llama/Llama-3.2-1B-Instruct`
- [ ] Verify inference works
- [ ] Add error handling

**Time Box:** 2-3 hours

**Verification:**
```python
from torchoptim.models.registry import ModelRegistry

model = ModelRegistry.create("text-generation", "meta-llama/Llama-3.2-1B-Instruct")
model.load()
result = model.infer({"prompt": "What is AI?", "max_tokens": 50})
print(result["generated_text"])
```

---

### 📋 User Story 1.3: Vision Model Adapter
**As a** framework user  
**I want** to load vision models through the generic interface  
**So that** I can optimize image models

**Acceptance Criteria:**
- [ ] `VisionModel` class implements `ModelInterface`
- [ ] Can load CLIP and similar vision models
- [ ] Handles image preprocessing automatically
- [ ] Returns embeddings
- [ ] Registered in ModelRegistry

**Tasks:**
- [ ] Implement `torchoptim/models/vision.py`:
  ```python
  from typing import Dict, Any
  import torch
  from PIL import Image
  from transformers import CLIPModel, CLIPProcessor
  from torchoptim.core.model_interface import ModelInterface, ModelSignature, ModelMetadata
  from torchoptim.models.registry import ModelRegistry
  
  @ModelRegistry.register("vision")
  class VisionModel(ModelInterface):
      def __init__(self, model_name: str, device: str = "cuda", **kwargs):
          self.model_name = model_name
          self.device = device
          self.model = None
          self.processor = None
          
      def load(self, model_name: str = None, **kwargs) -> None:
          """Load vision model (CLIP)"""
          name = model_name or self.model_name
          self.processor = CLIPProcessor.from_pretrained(name)
          self.model = CLIPModel.from_pretrained(name).to(self.device)
          self.model.eval()
          
      def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
          """Encode image or text"""
          if "image" in inputs:
              return self._encode_image(inputs["image"])
          elif "text" in inputs:
              return self._encode_text(inputs["text"])
          else:
              raise ValueError("Must provide 'image' or 'text'")
      
      def _encode_image(self, image: Image.Image) -> Dict[str, Any]:
          inputs = self.processor(images=image, return_tensors="pt").to(self.device)
          with torch.no_grad():
              image_features = self.model.get_image_features(**inputs)
          return {
              "embedding": image_features.cpu().numpy(),
              "embedding_dim": image_features.shape[-1]
          }
      
      def _encode_text(self, text: str) -> Dict[str, Any]:
          inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
          with torch.no_grad():
              text_features = self.model.get_text_features(**inputs)
          return {
              "embedding": text_features.cpu().numpy(),
              "embedding_dim": text_features.shape[-1]
          }
      
      def get_signature(self) -> ModelSignature:
          return ModelSignature(
              input_schema={"image": "PIL.Image or text": "str"},
              output_schema={"embedding": "ndarray", "embedding_dim": "int"}
          )
      
      def get_metadata(self) -> ModelMetadata:
          num_params = sum(p.numel() for p in self.model.parameters())
          return ModelMetadata(
              name=self.model_name,
              model_type="vision",
              num_parameters=num_params,
              architecture="CLIP"
          )
      
      def warmup(self, num_iterations: int = 3) -> None:
          """Warmup for stable benchmarking"""
          dummy_image = Image.new('RGB', (224, 224))
          for _ in range(num_iterations):
              self.infer({"image": dummy_image})
  ```

- [ ] Test loading CLIP: `openai/clip-vit-base-patch32`
- [ ] Verify image and text encoding works
- [ ] Add error handling

**Time Box:** 2-3 hours

**Verification:**
```python
from torchoptim.models.registry import ModelRegistry
from PIL import Image

model = ModelRegistry.create("vision", "openai/clip-vit-base-patch32")
model.load()

# Test image encoding
image = Image.open("test.jpg")
result = model.infer({"image": image})
print(f"Embedding dimension: {result['embedding_dim']}")

# Test text encoding
result = model.infer({"text": "a photo of a cat"})
print(f"Embedding dimension: {result['embedding_dim']}")
```
# **DAY 1 done!** 
---

## **DAY 2: Generic Profiling Engine**

### 📋 User Story 2.1: Model Profiler Interface
**As a** framework user  
**I want** to profile any model's performance  
**So that** I can identify bottlenecks automatically

**Acceptance Criteria:**
- [ ] `Profiler` class works with any `ModelInterface`
- [ ] Collects GPU utilization, memory, latency
- [ ] Works with both model types
- [ ] Returns structured profiling data
- [ ] No model-specific code needed

**Tasks:**
- [ ] Implement `torchoptim/core/profiler.py`:
  ```python
  from typing import Dict, Any, List
  import time
  import torch
  import numpy as np
  from dataclasses import dataclass, asdict
  from torchoptim.core.model_interface import ModelInterface
  
  try:
      import pynvml
      pynvml.nvmlInit()
      NVML_AVAILABLE = True
  except:
      NVML_AVAILABLE = False
  
  @dataclass
  class ProfileResult:
      """Profiling results"""
      model_name: str
      model_type: str
      latency_mean_ms: float
      latency_std_ms: float
      latency_p50_ms: float
      latency_p95_ms: float
      latency_p99_ms: float
      gpu_memory_used_mb: float
      gpu_memory_reserved_mb: float
      gpu_utilization_percent: float
      num_parameters: int
      
      def to_dict(self) -> Dict[str, Any]:
          return asdict(self)
  
  class Profiler:
      """Generic model profiler"""
      
      def __init__(self, model: ModelInterface):
          self.model = model
          
      def profile(self, 
                  test_inputs: List[Dict[str, Any]], 
                  num_warmup: int = 5,
                  num_iterations: int = 30) -> ProfileResult:
          """Profile model performance"""
          
          # Warmup
          print(f"Warming up for {num_warmup} iterations...")
          for i in range(num_warmup):
              self.model.infer(test_inputs[i % len(test_inputs)])
          
          # Profile
          print(f"Profiling for {num_iterations} iterations...")
          latencies = []
          
          for i in range(num_iterations):
              test_input = test_inputs[i % len(test_inputs)]
              
              torch.cuda.synchronize()
              start = time.perf_counter()
              
              self.model.infer(test_input)
              
              torch.cuda.synchronize()
              end = time.perf_counter()
              
              latencies.append((end - start) * 1000)  # Convert to ms
          
          # Collect GPU metrics
          gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
          gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
          
          # Get GPU utilization if available
          gpu_util = 0.0
          if NVML_AVAILABLE:
              try:
                  handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                  utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                  gpu_util = utilization.gpu
              except:
                  pass
          
          # Calculate statistics
          latencies_arr = np.array(latencies)
          metadata = self.model.get_metadata()
          
          return ProfileResult(
              model_name=metadata.name,
              model_type=metadata.model_type,
              latency_mean_ms=float(np.mean(latencies_arr)),
              latency_std_ms=float(np.std(latencies_arr)),
              latency_p50_ms=float(np.percentile(latencies_arr, 50)),
              latency_p95_ms=float(np.percentile(latencies_arr, 95)),
              latency_p99_ms=float(np.percentile(latencies_arr, 99)),
              gpu_memory_used_mb=gpu_memory_used,
              gpu_memory_reserved_mb=gpu_memory_reserved,
              gpu_utilization_percent=gpu_util,
              num_parameters=metadata.num_parameters
          )
      
      def detect_bottlenecks(self, profile: ProfileResult) -> List[str]:
          """Identify optimization opportunities"""
          bottlenecks = []
          
          if profile.gpu_utilization_percent < 50:
              bottlenecks.append("Low GPU utilization - consider batching")
          
          if profile.latency_std_ms > profile.latency_mean_ms * 0.2:
              bottlenecks.append("High latency variance - inconsistent performance")
          
          if profile.gpu_memory_used_mb > 10000:  # >10GB
              bottlenecks.append("High memory usage - consider quantization")
          
          return bottlenecks
  ```

- [ ] Add GPU monitoring utilities
- [ ] Test with both model types
- [ ] Add error handling for CPU-only systems

**Time Box:** 3-4 hours

**Verification:**
```python
from torchoptim.models.registry import ModelRegistry
from torchoptim.core.profiler import Profiler

model = ModelRegistry.create("text-generation", "meta-llama/Llama-3.2-1B-Instruct")
model.load()

profiler = Profiler(model)
test_inputs = [{"prompt": f"Test prompt {i}", "max_tokens": 20} for i in range(10)]
results = profiler.profile(test_inputs)

print(f"Mean latency: {results.latency_mean_ms:.2f}ms")
print(f"P95 latency: {results.latency_p95_ms:.2f}ms")
print(f"GPU memory: {results.gpu_memory_used_mb:.2f}MB")
```

---

### 📋 User Story 2.2: Baseline Profiling for Both Models
**As a** framework user  
**I want** documented baseline performance for both models  
**So that** I can measure optimization improvements

**Acceptance Criteria:**
- [ ] Llama 3.2-1B profiled and documented
- [ ] CLIP profiled and documented
- [ ] Results saved to JSON
- [ ] Bottlenecks identified for both

**Tasks:**
- [ ] Create `examples/profile_models.py`:
  ```python
  import json
  from pathlib import Path
  from torchoptim.models.registry import ModelRegistry
  from torchoptim.core.profiler import Profiler
  from PIL import Image
  
  def profile_llm():
      print("Profiling LLM...")
      model = ModelRegistry.create("text-generation", "meta-llama/Llama-3.2-1B-Instruct")
      model.load()
      
      profiler = Profiler(model)
      test_inputs = [
          {"prompt": "What is machine learning?", "max_tokens": 50},
          {"prompt": "Explain neural networks.", "max_tokens": 50},
          {"prompt": "What is deep learning?", "max_tokens": 50},
      ]
      
      results = profiler.profile(test_inputs, num_iterations=30)
      bottlenecks = profiler.detect_bottlenecks(results)
      
      return results, bottlenecks
  
  def profile_vision():
      print("Profiling Vision model...")
      model = ModelRegistry.create("vision", "openai/clip-vit-base-patch32")
      model.load()
      
      profiler = Profiler(model)
      # Create dummy images
      test_inputs = [
          {"image": Image.new('RGB', (224, 224))} for _ in range(10)
      ]
      
      results = profiler.profile(test_inputs, num_iterations=30)
      bottlenecks = profiler.detect_bottlenecks(results)
      
      return results, bottlenecks
  
  if __name__ == "__main__":
      Path("results/baseline").mkdir(parents=True, exist_ok=True)
      
      llm_results, llm_bottlenecks = profile_llm()
      vision_results, vision_bottlenecks = profile_vision()
      
      # Save results
      with open("results/baseline/llm_profile.json", "w") as f:
          json.dump({
              "profile": llm_results.to_dict(),
              "bottlenecks": llm_bottlenecks
          }, f, indent=2)
      
      with open("results/baseline/vision_profile.json", "w") as f:
          json.dump({
              "profile": vision_results.to_dict(),
              "bottlenecks": vision_bottlenecks
          }, f, indent=2)
      
      print("\n=== LLM Results ===")
      print(f"Latency (p50): {llm_results.latency_p50_ms:.2f}ms")
      print(f"GPU Memory: {llm_results.gpu_memory_used_mb:.2f}MB")
      print(f"Bottlenecks: {llm_bottlenecks}")
      
      print("\n=== Vision Results ===")
      print(f"Latency (p50): {vision_results.latency_p50_ms:.2f}ms")
      print(f"GPU Memory: {vision_results.gpu_memory_used_mb:.2f}MB")
      print(f"Bottlenecks: {vision_bottlenecks}")
  ```

- [ ] Run profiling on both models
- [ ] Save results to `results/baseline/`
- [ ] Document findings

**Time Box:** 2 hours

---

### 📋 User Story 2.3: Profiling Report Generation
**As a** framework user  
**I want** a readable profiling report  
**So that** I can understand performance characteristics

**Acceptance Criteria:**
- [ ] Markdown report generated from profiling data
- [ ] Includes model comparison
- [ ] Lists bottlenecks for each model
- [ ] Suggests optimization priorities

**Tasks:**
- [ ] Create report generation script
- [ ] Generate `results/baseline/BASELINE_REPORT.md`:
  ```markdown
  # Baseline Profiling Report
  
  Date: [auto-generated]
  Framework Version: 0.1.0
  GPU: [detect from system]
  
  ## Llama 3.2-1B (Text Generation)
  
  ### Performance Metrics
  - Mean Latency: XXms
  - P50 Latency: XXms
  - P95 Latency: XXms
  - P99 Latency: XXms
  - GPU Memory: XXMB
  - GPU Utilization: XX%
  - Parameters: XXM
  
  ### Bottlenecks Identified
  - [List from profiler]
  
  ### Optimization Priorities
  1. Quantization (FP32 → FP16/INT8)
  2. Batch inference
  3. KV cache optimization
  
  ## CLIP ViT-B/32 (Vision)
  
  [Similar structure]
  
  ## Comparison
  
  | Metric | LLM | Vision |
  |--------|-----|--------|
  | Latency (p50) | XX | XX |
  | Memory | XX | XX |
  | Parameters | XX | XX |
  
  ## Next Steps for Sprint 2
  
  Based on profiling, we will focus on:
  1. [Priority 1]
  2. [Priority 2]
  3. [Priority 3]
  ```

**Time Box:** 1-2 hours

---

## **DAY 3: Generic Benchmarking Framework**

### 📋 User Story 3.1: Benchmarker Interface
**As a** framework user  
**I want** to benchmark any model systematically  
**So that** I can compare different optimization variants

**Acceptance Criteria:**
- [ ] `Benchmarker` class works with any `ModelInterface`
- [ ] Supports different benchmark types (latency, throughput, etc.)
- [ ] Can compare multiple model variants
- [ ] Saves results in structured format

**Tasks:**
- [ ] Implement `torchoptim/core/benchmarker.py`:
  ```python
  from typing import Dict, Any, List
  import time
  import json
  from pathlib import Path
  from dataclasses import dataclass, asdict
  import numpy as np
  from torchoptim.core.model_interface import ModelInterface
  
  @dataclass
  class BenchmarkResult:
      """Benchmark results"""
      model_name: str
      model_type: str
      variant: str  # e.g., "baseline", "fp16", "int8"
      test_type: str  # e.g., "latency", "throughput"
      
      # Latency metrics
      latency_mean_ms: float = 0.0
      latency_p50_ms: float = 0.0
      latency_p95_ms: float = 0.0
      latency_p99_ms: float = 0.0
      
      # Throughput metrics
      throughput_items_per_sec: float = 0.0
      
      # Resource metrics
      gpu_memory_mb: float = 0.0
      
      num_samples: int = 0
      timestamp: str = ""
      
      def to_dict(self) -> Dict[str, Any]:
          return asdict(self)
  
  class Benchmarker:
      """Generic model benchmarker"""
      
      def __init__(self, model: ModelInterface, variant_name: str = "baseline"):
          self.model = model
          self.variant_name = variant_name
          
      def benchmark_latency(self,
                           test_inputs: List[Dict[str, Any]],
                           num_warmup: int = 5,
                           num_iterations: int = 50) -> BenchmarkResult:
          """Benchmark inference latency"""
          
          metadata = self.model.get_metadata()
          
          # Warmup
          for i in range(num_warmup):
              self.model.infer(test_inputs[i % len(test_inputs)])
          
          # Benchmark
          latencies = []
          for i in range(num_iterations):
              test_input = test_inputs[i % len(test_inputs)]
              
              import torch
              torch.cuda.synchronize()
              start = time.perf_counter()
              
              self.model.infer(test_input)
              
              torch.cuda.synchronize()
              end = time.perf_counter()
              
              latencies.append((end - start) * 1000)
          
          latencies_arr = np.array(latencies)
          
          import torch
          gpu_memory = torch.cuda.memory_allocated() / 1024**2
          
          from datetime import datetime
          
          return BenchmarkResult(
              model_name=metadata.name,
              model_type=metadata.model_type,
              variant=self.variant_name,
              test_type="latency",
              latency_mean_ms=float(np.mean(latencies_arr)),
              latency_p50_ms=float(np.percentile(latencies_arr, 50)),
              latency_p95_ms=float(np.percentile(latencies_arr, 95)),
              latency_p99_ms=float(np.percentile(latencies_arr, 99)),
              gpu_memory_mb=gpu_memory,
              num_samples=num_iterations,
              timestamp=datetime.now().isoformat()
          )
      
      def benchmark_throughput(self,
                               test_inputs: List[Dict[str, Any]],
                               duration_seconds: int = 10) -> BenchmarkResult:
          """Benchmark throughput (items/second)"""
          
          metadata = self.model.get_metadata()
          
          # Warmup
          for _ in range(5):
              self.model.infer(test_inputs[0])
          
          # Benchmark
          start_time = time.time()
          num_processed = 0
          
          while time.time() - start_time < duration_seconds:
              self.model.infer(test_inputs[num_processed % len(test_inputs)])
              num_processed += 1
          
          elapsed = time.time() - start_time
          throughput = num_processed / elapsed
          
          import torch
          gpu_memory = torch.cuda.memory_allocated() / 1024**2
          
          from datetime import datetime
          
          return BenchmarkResult(
              model_name=metadata.name,
              model_type=metadata.model_type,
              variant=self.variant_name,
              test_type="throughput",
              throughput_items_per_sec=throughput,
              gpu_memory_mb=gpu_memory,
              num_samples=num_processed,
              timestamp=datetime.now().isoformat()
          )
      
      def save_results(self, result: BenchmarkResult, output_dir: str = "results"):
          """Save benchmark results"""
          output_path = Path(output_dir) / self.variant_name
          output_path.mkdir(parents=True, exist_ok=True)
          
          filename = f"{result.model_type}_{result.test_type}.json"
          with open(output_path / filename, "w") as f:
              json.dump(result.to_dict(), f, indent=2)
  
  class BenchmarkComparison:
      """Compare multiple benchmark results"""
      
      @staticmethod
      def compare(results: List[Bench