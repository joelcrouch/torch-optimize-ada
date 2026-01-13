#!/bin/bash

################################################################################
# TorchOptim Environment Setup Script
# 
# This script sets up the complete development environment for the
# TorchOptim inference optimization framework.
#
# Usage:
#   ./setup_environment.sh [--gpu-check] [--skip-verify]
#
# Options:
#   --gpu-check    Run GPU verification tests
#   --skip-verify  Skip environment verification
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="torchoptim"
PYTHON_VERSION="3.10"
CONDA_ENV_FILE="environment.yml"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Check Prerequisites
################################################################################

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed!"
        echo "Please install Miniconda or Anaconda from:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda is installed: $(conda --version)"
    
    # Check if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        print_warning "NVIDIA drivers not detected - GPU acceleration will not be available"
        print_info "For GPU support, install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx"
    fi
    
    # Check if environment file exists
    if [ ! -f "$CONDA_ENV_FILE" ]; then
        print_error "Environment file '$CONDA_ENV_FILE' not found!"
        echo "Please make sure environment.yml is in the current directory"
        exit 1
    fi
    print_success "Environment file found: $CONDA_ENV_FILE"
    
    echo ""
}

################################################################################
# Create Conda Environment
################################################################################

create_environment() {
    print_header "Creating Conda Environment"
    
    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Environment '$ENV_NAME' already exists"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n $ENV_NAME -y
            print_success "Existing environment removed"
        else
            print_info "Updating existing environment..."
            conda env update -n $ENV_NAME -f $CONDA_ENV_FILE --prune
            print_success "Environment updated"
            return
        fi
    fi
    
    # Create new environment
    print_info "Creating new environment (this may take 5-10 minutes)..."
    conda env create -f $CONDA_ENV_FILE
    print_success "Environment created successfully"
    
    echo ""
}

################################################################################
# Install Development Package
################################################################################

install_dev_package() {
    print_header "Installing TorchOptim Package"
    
    # Check if setup.py exists
    if [ -f "setup.py" ]; then
        print_info "Installing package in development mode..."
        conda run -n $ENV_NAME pip install -e .
        print_success "Package installed in development mode"
    else
        print_warning "setup.py not found - skipping package installation"
        print_info "You can install later with: pip install -e ."
    fi
    
    echo ""
}

################################################################################
# Verify Installation
################################################################################

verify_installation() {
    print_header "Verifying Installation"
    
    # Test Python
    print_info "Testing Python..."
    if conda run -n $ENV_NAME python --version; then
        print_success "Python is working"
    else
        print_error "Python test failed"
        return 1
    fi
    
    # Test PyTorch
    print_info "Testing PyTorch..."
    if conda run -n $ENV_NAME python -c "import torch; print(f'PyTorch {torch.__version__}')"; then
        print_success "PyTorch is working"
    else
        print_error "PyTorch import failed"
        return 1
    fi
    
    # Test CUDA availability
    print_info "Testing CUDA availability..."
    CUDA_AVAILABLE=$(conda run -n $ENV_NAME python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        print_success "CUDA is available"
        CUDA_VERSION=$(conda run -n $ENV_NAME python -c "import torch; print(torch.version.cuda)")
        GPU_COUNT=$(conda run -n $ENV_NAME python -c "import torch; print(torch.cuda.device_count())")
        GPU_NAME=$(conda run -n $ENV_NAME python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")
        echo "  CUDA Version: $CUDA_VERSION"
        echo "  GPU Count: $GPU_COUNT"
        echo "  GPU: $GPU_NAME"
    else
        print_warning "CUDA is not available - CPU-only mode"
    fi
    
    # Test Transformers
    print_info "Testing Transformers..."
    if conda run -n $ENV_NAME python -c "import transformers; print(f'Transformers {transformers.__version__}')"; then
        print_success "Transformers is working"
    else
        print_error "Transformers import failed"
        return 1
    fi
    
    # Test vLLM (optional - might fail without GPU)
    print_info "Testing vLLM..."
    if conda run -n $ENV_NAME python -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null; then
        print_success "vLLM is working"
    else
        print_warning "vLLM test failed (this is OK if no GPU)"
    fi
    
    echo ""
}

################################################################################
# GPU Verification Tests
################################################################################

run_gpu_tests() {
    print_header "Running GPU Verification Tests"
    
    print_info "Creating temporary test script..."
    
    cat > /tmp/gpu_test.py << 'EOF'
import torch
import sys

print("=" * 60)
print("GPU Verification Test")
print("=" * 60)

# CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n⚠️  CUDA not available - tests will be limited")
    sys.exit(0)

# CUDA details
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")

# GPU information
num_gpus = torch.cuda.device_count()
print(f"\nNumber of GPUs: {num_gpus}")

for i in range(num_gpus):
    print(f"\nGPU {i}:")
    print(f"  Name: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Multi-Processor Count: {props.multi_processor_count}")

# Memory test
print("\n" + "=" * 60)
print("Testing GPU Memory Allocation")
print("=" * 60)

try:
    # Allocate 1GB tensor
    x = torch.randn(1024, 1024, 256).cuda()
    print(f"✓ Successfully allocated ~1GB tensor on GPU")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    del x
    torch.cuda.empty_cache()
    print(f"✓ Memory released successfully")
except Exception as e:
    print(f"✗ GPU memory test failed: {e}")
    sys.exit(1)

# Computation test
print("\n" + "=" * 60)
print("Testing GPU Computation")
print("=" * 60)

try:
    # Matrix multiplication test
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    
    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✓ Matrix multiplication (1000x1000) x 100 iterations:")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Avg per iteration: {elapsed*10:.2f} ms")
    
except Exception as e:
    print(f"✗ GPU computation test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All GPU tests passed!")
print("=" * 60)
EOF

    conda run -n $ENV_NAME python /tmp/gpu_test.py
    
    if [ $? -eq 0 ]; then
        print_success "GPU verification tests passed"
    else
        print_warning "Some GPU tests failed (this might be OK if no GPU)"
    fi
    
    rm /tmp/gpu_test.py
    echo ""
}

################################################################################
# Create Project Structure
################################################################################

create_project_structure() {
    print_header "Creating Project Structure"
    
    if [ -d "torchoptim" ]; then
        print_warning "Project structure already exists - skipping"
        return
    fi
    
    print_info "Creating directory structure..."
    
    mkdir -p torchoptim/{core,models,optimizations,utils,deployment}
    mkdir -p examples
    mkdir -p tests
    mkdir -p notebooks
    mkdir -p results/{baseline,optimized,comparisons}
    mkdir -p docs
    mkdir -p configs
    
    # Create __init__.py files
    touch torchoptim/__init__.py
    touch torchoptim/core/__init__.py
    touch torchoptim/models/__init__.py
    touch torchoptim/optimizations/__init__.py
    touch torchoptim/utils/__init__.py
    touch torchoptim/deployment/__init__.py
    
    print_success "Project structure created"
    
    echo ""
}

################################################################################
# Create Setup Files
################################################################################

create_setup_files() {
    print_header "Creating Setup Files"
    
    # Create setup.py if it doesn't exist
    if [ ! -f "setup.py" ]; then
        print_info "Creating setup.py..."
        cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="torchoptim",
    version="0.1.0",
    description="Model-agnostic ML inference optimization framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "serving": [
            "vllm>=0.2.6",
            "fastapi>=0.104.1",
            "pydantic==1.10.13",
            "uvicorn>=0.24.0",
        ],
    },
)
EOF
        print_success "setup.py created"
    fi
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        print_info "Creating .gitignore..."
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/create_setup_files() {
    print_header "Creating Setup Files"
    
    # Create setup.py if it doesn't exist
    if [ ! -f "setup.py" ]; then
        print_info "Creating setup.py..."
        cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="torchoptim",
    version="0.1.0",
    description="Model-agnostic ML inference optimization framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "serving": [
            "vllm>=0.2.6",
            "fastapi>=0.104.1",
            "pydantic==1.10.13",
            "uvicorn>=0.24.0",
        ],
    },
)
EOF
        print_success "setup.py created"
    fi
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        print_info "Creating .gitignore..."
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# PyTorch
*.pth
*.pt
*.ckpt

# Models (don't commit large model files)
models/
*.bin
*.safetensors

# Results
results/
runs/
logs/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.envrc

# Temporary
tmp/
temp/
*.tmp
EOF
        print_success ".gitignore created"
    fi
    
    echo ""
}

var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# PyTorch
*.pth
*.pt
*.ckpt

# Models (don't commit large model files)
models/
*.bin
*.safetensors

# Results
results/
runs/
logs/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.envrc

# Temporary
tmp/
temp/
*.tmp
EOF
        print_success ".gitignore created"
    fi
    
    echo ""
}

################################################################################
# Print Next Steps
################################################################################

print_next_steps() {
    print_header "Setup Complete! 🎉"
    
    echo -e "${GREEN}Your TorchOptim environment is ready!${NC}\n"
    
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo -e "     ${YELLOW}conda activate $ENV_NAME${NC}\n"
    
    echo "  2. Verify the installation:"
    echo -e "     ${YELLOW}python -c 'import torch; print(torch.cuda.is_available())'${NC}\n"
    
    echo "  3. Start with the Quick Start guide:"
    echo -e "     ${YELLOW}jupyter notebook notebooks/01_getting_started.ipynb${NC}\n"
    
    echo "  4. Or run the example scripts:"
    echo -e "     ${YELLOW}python examples/basic_usage.py${NC}\n"
    
    echo "To deactivate the environment:"
    echo -e "  ${YELLOW}conda deactivate${NC}\n"
    
    echo "To remove the environment:"
    echo -e "  ${YELLOW}conda env remove -n $ENV_NAME${NC}\n"
    
    if [ "$CUDA_AVAILABLE" != "True" ]; then
        print_warning "GPU not detected - some features will be limited"
        echo "For GPU support, ensure NVIDIA drivers are installed"
        echo ""
    fi
}

################################################################################
# Main Script
################################################################################

main() {
    # Parse arguments
    RUN_GPU_CHECK=false
    SKIP_VERIFY=false
    
    for arg in "$@"; do
        case $arg in
            --gpu-check)
                RUN_GPU_CHECK=true
                shift
                ;;
            --skip-verify)
                SKIP_VERIFY=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --gpu-check    Run comprehensive GPU verification tests"
                echo "  --skip-verify  Skip environment verification"
                echo "  --help, -h     Show this help message"
                exit 0
                ;;
        esac
    done
    
    # Run setup steps
    check_prerequisites
    create_environment
    create_project_structure
    create_setup_files
    install_dev_package
    
    if [ "$SKIP_VERIFY" != true ]; then
        verify_installation
    fi
    
    if [ "$RUN_GPU_CHECK" = true ]; then
        run_gpu_tests
    fi
    
    print_next_steps
}

# Run main function
main "$@"