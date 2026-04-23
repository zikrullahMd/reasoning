#!/bin/bash
set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL_HERE}"  # Override with: MODEL_PATH="meta-llama/..." ./start.sh
SGLANG_PORT=30000
FASTAPI_PORT=8000
SGLANG_LOG="sglang.log"

echo "=== PDF Inference Pipeline Startup ==="

# Set CUDA_HOME if not already set
if [ -z "$CUDA_HOME" ]; then
    # Search common CUDA locations (including versioned paths)
    for cuda_dir in /usr/local/cuda /usr/local/cuda-12* /usr/local/cuda-11* /usr/lib/cuda; do
        if [ -d "$cuda_dir" ]; then
            export CUDA_HOME="$cuda_dir"
            break
        fi
    done
    
    # Fallback: try to find nvcc and derive CUDA_HOME from it
    if [ -z "$CUDA_HOME" ]; then
        NVCC_PATH=$(which nvcc 2>/dev/null || true)
        if [ -n "$NVCC_PATH" ]; then
            export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        fi
    fi
fi

if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    echo "CUDA_HOME: $CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "ERROR: CUDA installation not found!"
    echo "Please install CUDA toolkit: sudo apt-get install cuda-toolkit-12-4"
    echo "Or set CUDA_HOME manually: export CUDA_HOME=/path/to/cuda"
    exit 1
fi

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

source venv/bin/activate

# Check if MODEL_PATH is set
if [ "$MODEL_PATH" == "YOUR_MODEL_HERE" ]; then
    echo ""
    echo "ERROR: MODEL_PATH not configured!"
    echo ""
    echo "Set the model path using one of these methods:"
    echo "  1. Environment variable: MODEL_PATH='openai/gpt-oss-20b' ./start.sh"
    echo "  2. Edit this script and replace YOUR_MODEL_HERE"
    echo ""
    echo "Recommended model for L4 GPU (24GB VRAM):"
    echo "  - openai/gpt-oss-20b (22B params, optimized for reasoning)"
    echo ""
    exit 1
fi

# Check if SGLang is already running
if curl -s "http://localhost:${SGLANG_PORT}/v1/models" > /dev/null 2>&1; then
    echo "SGLang server already running on port ${SGLANG_PORT}"
else
    echo "Starting SGLang server with model: ${MODEL_PATH}"
    echo "This may take a few minutes to load the model into GPU memory..."
    
    # Start SGLang in background
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port $SGLANG_PORT \
        --context-length 8192 \
        --mem-fraction-static 0.9 \
        > "$SGLANG_LOG" 2>&1 &
    
    SGLANG_PID=$!
    echo "SGLang PID: $SGLANG_PID (logs: $SGLANG_LOG)"
    
    # Wait for SGLang to be ready
    echo "Waiting for SGLang to initialize (this can take 5-10 minutes for large models)..."
    MAX_WAIT=600  # 10 minutes max
    WAITED=0
    
    while ! curl -s "http://localhost:${SGLANG_PORT}/v1/models" > /dev/null 2>&1; do
        if ! kill -0 $SGLANG_PID 2>/dev/null; then
            echo "Error: SGLang process died. Check $SGLANG_LOG for details."
            tail -50 "$SGLANG_LOG"
            exit 1
        fi
        
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "Error: SGLang failed to start within ${MAX_WAIT} seconds."
            echo "The model may still be loading. Check logs with: tail -f $SGLANG_LOG"
            echo "Last 50 lines of log:"
            tail -50 "$SGLANG_LOG"
            exit 1
        fi
        
        sleep 10
        WAITED=$((WAITED + 10))
        echo "  Still initializing... (${WAITED}s / ${MAX_WAIT}s max)"
    done
    
    echo "SGLang server ready!"
fi

# Start FastAPI server
echo ""
echo "Starting FastAPI server on port ${FASTAPI_PORT}..."
echo "API endpoint: http://0.0.0.0:${FASTAPI_PORT}/analyze"
echo "Health check: http://0.0.0.0:${FASTAPI_PORT}/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

export SGLANG_URL="http://localhost:${SGLANG_PORT}"
exec uvicorn server:app --host 0.0.0.0 --port $FASTAPI_PORT
