#!/bin/bash
set -e

echo "=== PDF Inference Pipeline Setup (2026 Stack) ==="

# Update package lists
sudo apt-get update

# Check for Python 3.12 and install with dev headers
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 not found. Installing..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
fi

# Ensure Python 3.12 with dev headers and build tools are installed
echo "Installing Python 3.12 and build dependencies..."
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential gcc ninja-build

# Install CUDA toolkit if not present
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found. Installing minimal CUDA components..."
    
    # Install only the essential CUDA components (avoids nsight-systems dependency issues on Ubuntu 24.04)
    sudo apt-get install -y --no-install-recommends \
        cuda-nvcc-12-4 \
        cuda-cudart-dev-12-4 \
        cuda-nvrtc-dev-12-4 \
        libcublas-dev-12-4 \
        libcusparse-dev-12-4 \
        libcurand-dev-12-4
    
    echo "CUDA toolkit installed."
fi

# Set CUDA environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "CUDA_HOME set to: $CUDA_HOME"
else
    # Try to find CUDA
    for cuda_dir in /usr/local/cuda-12* /usr/local/cuda; do
        if [ -d "$cuda_dir" ]; then
            export CUDA_HOME="$cuda_dir"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
            echo "CUDA_HOME set to: $CUDA_HOME"
            break
        fi
    done
fi

# Create virtual environment
echo "Creating Python 3.12 virtual environment..."
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip and install uv for fast package management
echo "Installing uv package manager..."
pip install --upgrade pip
pip install uv

# Install SGLang with CUDA support (for L4 GPU)
echo "Installing SGLang with CUDA support..."
uv pip install "sglang[all]"

# Install application dependencies
echo "Installing application dependencies..."
uv pip install -r requirements.txt

# Configure iptables to redirect port 80 to 8000
echo "Configuring iptables (requires sudo)..."
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8000

# Make iptables rule persistent (optional, requires iptables-persistent)
if command -v netfilter-persistent &> /dev/null; then
    echo "Saving iptables rules..."
    sudo netfilter-persistent save
else
    echo "Note: Install iptables-persistent to make the port redirect permanent:"
    echo "  sudo apt-get install -y iptables-persistent"
fi

echo ""
echo "=== Setup Complete ==="
echo "To start the server, run: ./start.sh"
echo ""
