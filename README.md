# DeepSeek-OCR Windows Setup with Docker Backend

✅ **Fully Working** - Production-ready Docker setup for running DeepSeek-OCR on Windows with NVIDIA CUDA support, using **exact requirements** from the official repository.

## Features

- ✅ **Working OCR Extraction**: Successfully extracts text from images and documents
- 🚀 **Official Requirements**: Uses exact versions from DeepSeek-OCR GitHub (Python 3.12.9, CUDA 11.8, PyTorch 2.6.0)
- ⚡ **UV Package Manager**: Fast dependency installation using `uv` for optimized builds
- ⚙️ **Environment Configuration**: Full `.env` file support for model parameters (base_size, image_size, crop_mode)
- 🎮 **CUDA Support**: NVIDIA Docker for GPU acceleration with CUDA 11.8 + Flash Attention 2.0
- 🐳 **Docker Compose**: Easy container management with proper volume mounting
- 💾 **Persistent Models**: Models cached in `D:/data/models` (shared between host and container)
- 🌐 **Flask API**: RESTful interface with `/ocr` and `/ocr/stream` endpoints
- 📊 **Progress Tracking**: Real-time animated progress during OCR inference

## Exact Requirements

Based on [official DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR):

- **Python**: 3.12.9 (exact)
- **CUDA**: 11.8 (exact)
- **PyTorch**: 2.6.0 (exact)
- **Transformers**: 4.46.3 (exact)
- **Tokenizers**: 0.20.3 (exact)
- **Flash-attn**: 2.7.3 (exact)
- **Model Size**: 3B parameters (~6.68 GB download)

## Prerequisites

1. **Docker Desktop for Windows** with WSL2 backend
   - Download: https://www.docker.com/products/docker-desktop
   - Enable WSL2 during installation

2. **NVIDIA GPU** with recent drivers
   - Docker Desktop includes nvidia-docker support
   - Verify: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

3. **Python 3.8+** (for the client script)
   ```bash
   pip install Pillow requests
   ```

4. **Model Storage Path** (Customizable)
   - Default: `D:/data/models` (see [Customizing Model Storage Path](#customizing-model-storage-path) below)
   - Ensure the directory exists and has sufficient space (~10GB for model + cache)

## Quick Start

### Option 1: Using Python Client Script (Recommended)

1. **Run the automated script**:

   ```bash
   python ocr_inference.py
   ```

   The script will automatically:
   - Build the Docker image (first time: ~10-15 minutes)
   - Start the container with GPU support
   - Download the model (~6.68 GB, first time only)
   - Wait for the backend to be ready

2. **Perform OCR**:
   - Enter image paths: `D:/path/to/image.jpg`
   - Enter URLs: `https://example.com/image.png`
   - See real-time progress: 🔍 Loading image → Processing → Extracting text
   - Type `quit` to exit

### Option 2: Using Docker Compose

1. **Configure environment** (optional):

   ```bash
   cp .env.example .env
   # Edit .env to customize model parameters
   ```

2. **Start the container**:

   ```bash
   docker-compose up -d
   ```

3. **Check logs**:

   ```bash
   docker-compose logs -f
   ```

4. **Test the API**:

   ```bash
   curl http://localhost:5000/health
   ```

## Configuration (.env file)

Edit the [.env](.env) file to customize model behavior:

```env
# Model configuration presets:
# tiny: base_size=512, image_size=512, crop_mode=False (fastest)
# small: base_size=640, image_size=640, crop_mode=False
# base: base_size=1024, image_size=1024, crop_mode=False
# large: base_size=1280, image_size=1280, crop_mode=False
# gundam: base_size=1024, image_size=640, crop_mode=True (balanced)

BASE_SIZE=1024
IMAGE_SIZE=640
CROP_MODE=true
DEVICE=cuda
TORCH_DTYPE=bfloat16
```

## Customizing Model Storage Path

**IMPORTANT**: The default configuration uses `D:/data/models` for model storage. You can customize this to any location on your system.

### Step 1: Choose Your Model Storage Location

Pick any directory on your system where you want to store the model files (~10GB):

- Windows: `C:/Users/YourName/models`, `E:/ml-models`, etc.
- Linux/Mac: `/home/yourname/models`, `/data/models`, etc.

### Step 2: Update docker-compose.yml

Edit [docker-compose.yml](docker-compose.yml) and change the volume mount:

```yaml
volumes:
  - D:/data/models:/data/models  # Change D:/data/models to YOUR path
```

Example for different locations:

```yaml
# Windows C: drive
volumes:
  - C:/Users/YourName/models:/data/models

# Windows E: drive
volumes:
  - E:/ml-models:/data/models

# Linux/Mac
volumes:
  - /home/yourname/models:/data/models
```

### Step 3: Update ocr_inference.py (if using Python client)

Edit [ocr_inference.py](ocr_inference.py) line 15:

```python
# Change this line to match your model storage path
MODEL_VOLUME = "D:/data/models:/data/models"
```

Example:

```python
MODEL_VOLUME = "C:/Users/YourName/models:/data/models"
```

### Step 4: Create the Directory

Make sure the directory exists on your system:

```bash
# Windows
mkdir C:\Users\YourName\models

# Linux/Mac
mkdir -p /home/yourname/models
```

### Volume Mount Explained

The format is: `HOST_PATH:CONTAINER_PATH`

- **HOST_PATH**: Your actual folder on Windows/Linux/Mac (e.g., `D:/data/models`)
- **CONTAINER_PATH**: Where it appears inside Docker (always `/data/models`)
- The colon `:` separates them
- Files are synced between both locations
- Model downloads are saved to HOST_PATH and persist after container stops

**Example**:
```yaml
volumes:
  - D:/data/models:/data/models
    ↑              ↑
    Your Windows   Inside Docker
    folder         (always this)
```

## Architecture

```text
Windows (Host)
    ├─ ocr_inference.py          # Client script
    ├─ Dockerfile                # Container definition
    ├─ docker-compose.yml        # Docker Compose config
    ├─ .env                      # Configuration file
    └─ D:/data/models/           # Model storage (mounted in container)
        └─ DeepSeek-OCR/         # Model files

Docker Container (Ubuntu 22.04 + CUDA 11.8)
    ├─ Python 3.12.9             # Exact version from official repo
    ├─ PyTorch 2.6.0 + CUDA 11.8 # Exact versions
    ├─ flash-attn 2.7.3          # Required for model
    ├─ uv (package manager)      # Fast dependency installation
    ├─ ocr_backend.py            # Flask API server
    └─ /data/models/             # Mounted from host
```

## How It Works

1. **Client Script** (`ocr_inference.py`):
   - Detects Windows OS
   - Manages Docker container lifecycle
   - Sends OCR requests via HTTP API
   - Displays results

2. **Backend Server** (`ocr_backend.py`):
   - Runs in Linux Docker container
   - Loads DeepSeek-OCR model with CUDA
   - Exposes Flask API on port 5000
   - Handles model patching and inference

## Troubleshooting

### Docker not found
```
✗ Docker is not installed!
```
**Solution**: Install Docker Desktop for Windows

### Docker not running
```
✗ Docker is not running!
```
**Solution**: Start Docker Desktop

### GPU errors
```
✗ Failed to start container: ...
```
**Solution**: Try without GPU:
```bash
docker run -d --name deepseek-ocr-server -p 5000:5000 -v D:/data/models:/data/models deepseek-ocr-backend
```

### Check container logs
```bash
docker logs deepseek-ocr-server
```

### Restart container
```bash
docker restart deepseek-ocr-server
```

### Remove and rebuild
```bash
docker stop deepseek-ocr-server
docker rm deepseek-ocr-server
docker rmi deepseek-ocr-backend
python ocr_inference.py  # Will rebuild
```

## Manual Docker Commands

**Build image**:
```bash
docker build -t deepseek-ocr-backend .
```

**Run container with GPU**:
```bash
docker run -d --name deepseek-ocr-server --gpus all -p 5000:5000 -v D:/data/models:/data/models deepseek-ocr-backend
```

**Run container without GPU**:
```bash
docker run -d --name deepseek-ocr-server -p 5000:5000 -v D:/data/models:/data/models deepseek-ocr-backend
```

**Test backend directly**:
```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/ocr -H "Content-Type: application/json" -d "{\"image\": \"https://example.com/image.jpg\"}"
```

## Docker Compose Commands

```bash
# Start container (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Stop and remove volumes
docker-compose down -v
```

## API Endpoints

### Health Check

```bash
GET http://localhost:5000/health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

### OCR Inference

```bash
POST http://localhost:5000/ocr
Content-Type: application/json

{
  "image_path": "/data/models/test.jpg"
}
```

Response:

```json
{
  "text": "Extracted text from image...",
  "time_taken": 1.23
}
```

## Files

- [ocr_inference.py](ocr_inference.py) - Client script (Windows)
- [ocr_backend.py](ocr_backend.py) - Flask API server (Docker)
- [Dockerfile](Dockerfile) - Container definition with CUDA 11.8
- [docker-compose.yml](docker-compose.yml) - Docker Compose configuration
- [requirements.txt](requirements.txt) - Python dependencies (exact versions)
- [.env](.env) - Configuration file
- [.env.example](.env.example) - Configuration template

## Performance

- **First run**: ~10-15 minutes (Docker build + model download)
- **Subsequent runs**: ~30 seconds startup time
- **OCR Speed**: 1-30 seconds depending on image complexity
  - Simple images: ~1-2 seconds
  - Documents with tables: ~30 seconds
- **Model caching**: Models cached in `D:/data/models` and reused
- **Model size**: ~6.68 GB
- **GPU**: Requires NVIDIA GPU with CUDA support (tested on RTX 3090)

## Technical Details

### What's Inside

- **Base Image**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Python**: 3.12.9 (installed from deadsnakes PPA)
- **Package Manager**: `uv` (official astral-sh image) for fast installs
- **Flash Attention**: 2.7.3 compiled with CUDA support
- **Model Format**: Reads output from `result.mmd` file generated by model

### API Endpoints

1. **GET /health** - Health check

   ```json
   {
     "status": "ok",
     "model_loaded": true,
     "device": "cuda"
   }
   ```

2. **POST /ocr** - OCR inference

   ```json
   {
     "image_path": "/tmp/image.jpg"
   }
   ```

   Response:

   ```json
   {
     "text": "Extracted text here...",
     "time_taken": 2.5
   }
   ```

3. **POST /ocr/stream** - Streaming OCR with Server-Sent Events

   Returns real-time progress updates

## Troubleshooting

### Issue: "Text file not generated"

**Solution**: The model now correctly reads from `result.mmd` file. If you still see this, check container logs:

```bash
docker logs deepseek-ocr-server -f
```

### Issue: Docker build fails on flash-attn

**Solution**: Make sure you have:

- Docker Desktop with WSL2
- Enough disk space (~20GB for build)
- NVIDIA Docker runtime enabled

### Issue: Model returns None or empty text

**Solution**: This is fixed in the current version. The backend now properly reads the `result.mmd` file generated by the model.

## Credits

- **DeepSeek-OCR**: <https://github.com/deepseek-ai/DeepSeek-OCR>
- **Model**: <https://huggingface.co/deepseek-ai/DeepSeek-OCR>
- **UV Package Manager**: <https://github.com/astral-sh/uv>

## Notes

- Uses **exact versions** from official DeepSeek-OCR repo
- Properly configured with CUDA 11.8 + Flash Attention 2.0
- Tested and working on Windows 11 with Docker Desktop
- Model output format: Markdown (`.mmd` files) with extracted text
