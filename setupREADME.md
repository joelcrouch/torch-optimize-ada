# TorchOptim Environment Setup

Quick setup guide for the TorchOptim inference optimization framework.

## Prerequisites

- **Conda/Miniconda** - [Download here](https://docs.conda.io/en/latest/miniconda.html)
- **NVIDIA GPU** (optional but recommended)
- **NVIDIA Drivers** (if using GPU) - [Download here](https://www.nvidia.com/Download/index.aspx)
- **CUDA Toolkit 12.1+** (installed via conda)

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make the setup script executable
chmod +x setup_environment.sh

# Run the setup (takes 5-10 minutes)
./setup_environment.sh

# Optional: Run GPU verification tests
./setup_environment.sh --gpu-check
```

### Option 2: Manual Setup

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate torchoptim

# Install package in development mode (if setup.py exists)
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Verify Your Installation

### Basic Verification

```bash
conda activate torchoptim
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Test Core Dependencies

```bash
conda activate torchoptim
python << EOF
# Test imports
import torch
import transformers
import numpy as np
import PIL
from vllm import LLM  # Might fail without GPU

print("✓ All core dependencies imported successfully")
EOF
```

### GPU Verification (if available)

```bash
# Check NVIDIA driver
nvidia-smi

# Test GPU computation
python << EOF
import torch

# Create tensors on GPU
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()

# Perform computation
z = torch.matmul(x, y)
print(f"✓ GPU computation successful")
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
EOF
```

## What Gets Installed

### Core Libraries
- **PyTorch 2.1.0** - Deep learning framework
- **CUDA 12.1** - GPU acceleration
- **Transformers 4.35.0** - HuggingFace models
- **vLLM 0.2.6** - Fast LLM inference
- **NumPy, Pillow** - Data processing

### Development Tools
- **Jupyter** - Interactive notebooks
- **pytest** - Testing framework
- **black, isort** - Code formatting
- **FastAPI** - API serving

### Optional Components
- **TensorRT** - NVIDIA inference optimization (commented out - install separately)
- **Triton** - Model serving (commented out - install separately)
- **Weights & Biases** - Experiment tracking (commented out)

## Project Structure

After setup, your directory should look like:

```
torchoptim/
├── environment.yml           # Conda environment definition
├── setup_environment.sh      # Automated setup script
├── setup.py                  # Package installation config
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── torchoptim/              # Main package
│   ├── __init__.py
│   ├── core/                # Core abstractions
│   ├── models/              # Model adapters
│   ├── optimizations/       # Optimization plugins
│   ├── utils/               # Utilities
│   └── deployment/          # Deployment tools
├── examples/                # Example scripts
├── tests/                   # Test suite
├── notebooks/              # Jupyter notebooks
├── results/                # Benchmark results
│   ├── baseline/
│   ├── optimized/
│   └── comparisons/
├── docs/                   # Documentation
└── configs/                # Configuration files
```

## Common Issues and Solutions

### Issue: `conda: command not found`
**Solution:** Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

### Issue: `CUDA not available`
**Solution:** 
1. Check NVIDIA drivers: `nvidia-smi`
2. If not installed, download from NVIDIA website
3. Verify GPU compatibility: CUDA 12.1 requires compute capability 5.0+

### Issue: `vLLM import fails`
**Solution:** vLLM requires GPU. For CPU-only testing:
- Comment out vLLM in environment.yml
- Use standard transformers for inference
- Or install CPU-compatible version (limited features)

### Issue: `Out of memory` errors
**Solution:**
- Reduce batch size
- Use smaller models for testing
- Clear GPU cache: `torch.cuda.empty_cache()`
- Check memory: `nvidia-smi`

### Issue: Environment creation is slow
**Solution:**
- This is normal - conda resolves dependencies (5-10 mins)
- Use `mamba` for faster installation: `conda install mamba -c conda-forge`
- Then: `mamba env create -f environment.yml`

## Updating the Environment

### Add New Dependencies

```bash
# Edit environment.yml
# Then update the environment
conda env update -n torchoptim -f environment.yml --prune
```

### Update All Packages

```bash
conda activate torchoptim
conda update --all
```

## Removing the Environment

```bash
# Deactivate if active
conda deactivate

# Remove environment
conda env remove -n torchoptim

# Verify removal
conda env list
```

## GPU Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with compute capability 5.0+ (Maxwell architecture or newer)
- **VRAM:** 8GB minimum (16GB+ recommended for larger models)
- **Driver:** 525.60.13+ for CUDA 12.1
- **CUDA:** 12.1 (installed via conda)

### Recommended Specifications
- **GPU:** NVIDIA RTX 3090, A100, or newer
- **VRAM:** 24GB+
- **Driver:** Latest NVIDIA drivers
- **System RAM:** 32GB+

### Supported GPUs
- ✓ RTX 4090, 4080, 4070, 3090, 3080, 3070
- ✓ A100, A40, A30, A10
- ✓ V100, P100
- ✓ GTX 1080 Ti and newer (with limitations)

## CPU-Only Mode

If you don't have a GPU, you can still use the framework with limitations:

```yaml
# In environment.yml, replace:
- pytorch-cuda=12.1

# With:
- cpuonly
```

Then:
```bash
conda env update -n torchoptim -f environment.yml
```

**Limitations in CPU mode:**
- Slower inference (10-100x slower)
- vLLM won't work (use standard transformers)
- TensorRT optimization unavailable
- Limited batching capabilities

## Development Workflow

```bash
# Activate environment
conda activate torchoptim

# Make changes to code
# ...

# Run tests
pytest tests/

# Format code
black torchoptim/
isort torchoptim/

# Run examples
python examples/basic_usage.py

# Start Jupyter
jupyter notebook
```

## Getting Help

### Check Installation
```bash
./setup_environment.sh --gpu-check
```

### Test Specific Component
```bash
python -c "import vllm; print('vLLM works')"
python -c "import transformers; print('Transformers works')"
```

### Environment Information
```bash
conda activate torchoptim
conda list
conda info
```

### System Information
```bash
nvidia-smi  # GPU info
nvcc --version  # CUDA compiler version
python --version
```

## Next Steps

After setup:

1. **Review the Sprint Plan:** See `master_sprint_plan.md`
2. **Start Day 1:** Follow `sprint1_user_stories.md`
3. **Run Examples:** Check `examples/` directory
4. **Read Docs:** Review `docs/` for detailed guides

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

---

**Questions?** Open an issue or consult the documentation in `docs/`