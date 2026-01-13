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
print_header "makeing env..."
ENV_NAME="torchoptim"
CONDA_ENV_FILE="environment.yml"

# Logic for testing the Conda creation specifically
print_header "Testing Environment Creation"

if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '$ENV_NAME' exists."
    # Forcing a recreate is safer for debugging dependency hell
    print_info "Removing to ensure a clean slate..."
    conda env remove -n $ENV_NAME -y
fi

print_info "Attempting build from $CONDA_ENV_FILE..."
if conda env create -f $CONDA_ENV_FILE; then
    print_success "Conda solved the environment!"
else
    print_error "Conda failed to solve. Check the logs above for version conflicts."
    exit 1
fi