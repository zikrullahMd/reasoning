# PDF Inference Pipeline

A FastAPI-based service for analyzing PDF documents using LLMs. Upload a PDF, ask a question, and get AI-powered answers.

## Requirements

- **EC2 Instance**: `g6.2xlarge` (NVIDIA L4 GPU with 24GB VRAM)
- **OS**: Ubuntu 24.04 LTS
- **Storage**: Minimum 100GB EBS volume (for model weights)
- **Python**: 3.12

## Quick Start

```bash
# 1. Clone/copy files to your EC2 instance
# 2. Run setup
chmod +x setup.sh start.sh
./setup.sh

# 3. Login to HuggingFace (required for gated models like Gemma)
source venv/bin/activate
pip install huggingface_hub
huggingface-cli login

# 4. Start the server
MODEL_PATH="google/gemma-3-4b-it" ./start.sh
```

## Detailed Setup

### Step 1: Launch EC2 Instance

1. Launch a `g6.2xlarge` instance with Ubuntu 24.04 LTS AMI
2. Configure security group to allow inbound traffic on port 80 (HTTP)
3. Attach at least 100GB EBS storage

### Step 2: Connect and Prepare

```bash
ssh -i your-key.pem ubuntu@<your-ec2-ip>

# Clone your project or upload files
cd ~
mkdir reasoning && cd reasoning
# ... upload setup.sh, start.sh, server.py, requirements.txt
```

### Step 3: Run Setup Script

```bash
chmod +x setup.sh start.sh
./setup.sh
```

This will:
- Install Python 3.12 with development headers
- Install CUDA toolkit components
- Create a virtual environment
- Install SGLang and dependencies
- Configure iptables to redirect port 80 → 8000

### Step 4: Configure HuggingFace Access

For gated models (like Google Gemma), you need to:

1. **Accept the model license** at https://huggingface.co/google/gemma-3-4b-it
2. **Create an access token** at https://huggingface.co/settings/tokens
3. **Login on your server**:

```bash
source venv/bin/activate
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

### Step 5: Start the Server

```bash
# For Google Gemma 3 4B (recommended for L4 GPU)
MODEL_PATH="google/gemma-3-4b-it" ./start.sh

# Or for OpenAI gpt-oss-20b (larger, uses quantization)
MODEL_PATH="openai/gpt-oss-20b" ./start.sh
```

The server will:
1. Load the model into GPU memory (2-5 minutes)
2. Start the FastAPI server on port 8000
3. Redirect port 80 → 8000 via iptables

## API Usage

### Health Check

```bash
curl http://<your-ec2-ip>/health
```

### Analyze a PDF

```bash
curl -X POST \
  -F "file=@document.pdf" \
  -F "question=What are the main points in this document?" \
  http://<your-ec2-ip>/analyze
```

### Using Postman

1. **Method**: POST
2. **URL**: `http://<your-ec2-ip>/analyze`
3. **Body**: form-data
   - `file` (File): Select your PDF
   - `question` (Text): Your question about the document

## Supported Models

| Model | Size | Fits L4? | Notes |
|-------|------|----------|-------|
| `google/gemma-3-4b-it` | 4B | Yes | Recommended, requires HF login |
| `google/gemma-3n-E4B-it` | ~4B | Yes | Efficient variant |
| `openai/gpt-oss-20b` | 22B | Yes | Uses mxfp4 quantization |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Yes | Good quality |

## Troubleshooting

### "CUDA out of memory"

The model is too large. Use a smaller model:
```bash
MODEL_PATH="google/gemma-3-4b-it" ./start.sh
```

### "No space left on device"

Clear HuggingFace cache:
```bash
rm -rf ~/.cache/huggingface/hub
```

### "Could not find CUDA installation"

Install CUDA toolkit:
```bash
sudo apt-get install -y cuda-nvcc-12-4 cuda-cudart-dev-12-4
```

Or set CUDA_HOME manually:
```bash
export CUDA_HOME=/usr/local/cuda
MODEL_PATH="google/gemma-3-4b-it" ./start.sh
```

### "Transformers does not recognize this architecture"

Upgrade transformers:
```bash
source venv/bin/activate
pip install --upgrade transformers
```

### "Access denied" for gated models

1. Accept the license at the model's HuggingFace page
2. Login: `huggingface-cli login`

### Server won't start / process killed

Check the logs:
```bash
tail -100 sglang.log
```

Common causes:
- Out of memory (use smaller model)
- Missing CUDA (check `nvidia-smi`)
- Port already in use (kill existing process)

## File Structure

```
reasoning/
├── server.py          # FastAPI application
├── setup.sh           # Environment setup script
├── start.sh           # Server startup script
├── requirements.txt   # Python dependencies
├── sglang.log         # SGLang server logs
└── venv/              # Python virtual environment
```

## Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Client    │────▶│  FastAPI (:8000)    │────▶│ SGLang (:30000) │
│  (Postman)  │     │  - PDF extraction   │     │  - LLM inference│
└─────────────┘     │  - Prompt assembly  │     │  - Streaming    │
                    └─────────────────────┘     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼─────┐      ┌──────▼─────┐
              │  PyMuPDF  │      │  Surya OCR │
              │ (digital) │      │ (scanned)  │
              └───────────┘      └────────────┘
```

## Performance Notes

- **Cold start**: 2-5 minutes (model loading)
- **Inference**: 30-60 tokens/second on L4 GPU
- **PDF extraction**: Milliseconds for digital PDFs, 1-2 seconds for OCR

## Stopping the Server

```bash
# Find and kill the processes
pkill -f sglang
pkill -f uvicorn

# Or use Ctrl+C if running in foreground
```
