#!/bin/bash

# Configuration
ENV_NAME="torchoptim"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

################################################################################
# Verification Logic
################################################################################

verify_installation() {
    print_header "Verifying Installation"
    
    print_info "Testing Python..."
    if conda run -n $ENV_NAME python --version; then
        print_success "Python is working"
    else
        print_error "Python test failed. Is the environment '$ENV_NAME' created?"
        return 1
    fi
    
    print_info "Testing PyTorch..."
    if conda run -n $ENV_NAME python -c "import torch; print(f'PyTorch {torch.__version__}')"; then
        print_success "PyTorch is working"
    else
        print_error "PyTorch import failed"
        return 1
    fi
    
    print_info "Testing CUDA availability..."
    CUDA_AVAILABLE=$(conda run -n $ENV_NAME python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        print_success "CUDA is available"
        conda run -n $ENV_NAME python -c "import torch; print(f'  CUDA Version: {torch.version.cuda}\n  GPU: {torch.cuda.get_device_name(0)}')"
    else
        print_warning "CUDA is not available - CPU-only mode detected"
    fi
    
    print_info "Testing Transformers..."
    if conda run -n $ENV_NAME python -c "import transformers; print(f'Transformers {transformers.__version__}')"; then
        print_success "Transformers is working"
    else
        print_error "Transformers import failed"
    fi
    
    print_info "Testing vLLM..."
    if conda run -n $ENV_NAME python -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null; then
        print_success "vLLM is working"
    else
        print_warning "vLLM import failed (check CUDA/Driver compatibility)"
    fi
}

run_gpu_tests() {
    print_header "Running GPU Verification Tests"
    print_info "Creating temporary test script..."
    
    cat > /tmp/gpu_test.py << 'EOF'
import torch
import sys
import time

print("=" * 60)
print("GPU Stress & Computation Test")
print("=" * 60)

if not torch.cuda.is_available():
    print("\n❌ CUDA not available")
    sys.exit(1)

# GPU information
props = torch.cuda.get_device_properties(0)
print(f"Device: {props.name}")
print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")

# Memory test
try:
    x = torch.randn(1024, 1024, 256).cuda() # ~1GB
    print(f"✓ Memory Allocation Success: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used")
    del x
    torch.cuda.empty_cache()
except Exception as e:
    print(f"✗ Memory Test Failed: {e}")
    sys.exit(1)

# Computation test
try:
    a, b = torch.randn(2000, 2000).cuda(), torch.randn(2000, 2000).cuda()
    for _ in range(10): torch.matmul(a, b) # Warmup
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100): torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"✓ MatMul Test Success: {((time.time() - start) * 10):.2f} ms avg per iteration")
except Exception as e:
    print(f"✗ Computation Test Failed: {e}")
    sys.exit(1)

print("\n✓ All Hardware Tests Passed!")
EOF

    conda run -n $ENV_NAME python /tmp/gpu_test.py
    TEST_RESULT=$?
    rm /tmp/gpu_test.py
    
    if [ $TEST_RESULT -eq 0 ]; then
        print_success "Hardware handshake complete."
    else
        print_error "Hardware tests failed."
    fi
}

# --- Execution ---
verify_installation
run_gpu_tests