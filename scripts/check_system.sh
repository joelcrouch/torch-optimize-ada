#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

# Configuration
CONDA_ENV_FILE="environment.yml"

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# --- THE PREREQUISITE CHECK ---
print_header "Checking Prerequisites"

# 1. Conda Check
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed!"
    exit 1
fi
print_success "Conda is installed: $(conda --version)"

# 2. NVIDIA GPU Check
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA drivers detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    print_warning "NVIDIA drivers not detected - GPU acceleration will not be available"
fi

# 3. Environment File Check
if [ ! -f "$CONDA_ENV_FILE" ]; then
    print_error "Environment file '$CONDA_ENV_FILE' not found!"
    exit 1
fi
print_success "Environment file found: $CONDA_ENV_FILE"

echo -e "\n${GREEN}Foundation is solid. Ready for the next module.${NC}"