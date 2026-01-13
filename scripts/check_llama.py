# from torchoptim.models.registry import ModelRegistry
# import torch

# # This imports our newly registered class
# import torchoptim.models.text_generation 

# print("Loading Llama-3.2-1B...")
# model = ModelRegistry.create("text-generation", "meta-llama/Llama-3.2-1B-Instruct")
# model.load()

# print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# result = model.infer({"prompt": "Explain GPU optimization in one sentence.", "max_tokens": 30})
# print(f"\nResponse: {result['generated_text']}")


import torch
from torchoptim.models.registry import ModelRegistry
import torchoptim.models.text_generation 

print("Loading Open Llama-3.2-1B (Unsloth)...")
# We use the Unsloth version because it bypasses the "Gated" requirement
model = ModelRegistry.create("text-generation", "unsloth/Llama-3.2-1B-Instruct")
model.load()

print(f"✅ Model Loaded!")
meta = model.get_metadata()
print(f"📊 Params: {meta.num_parameters / 1e6:.2f}M")

result = model.infer({"prompt": "The best way to optimize a neural network is", "max_tokens": 20})
print(f"\nResponse: {result['generated_text']}")