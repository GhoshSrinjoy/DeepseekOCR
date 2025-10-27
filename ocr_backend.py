import os
import time
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration from .env
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/models/DeepSeek-OCR")
BACKEND = os.getenv("BACKEND", "transformers")
DEVICE = os.getenv("DEVICE", "cuda")
BASE_SIZE = int(os.getenv("BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("CROP_MODE", "true").lower() == "true"
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "bfloat16")
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# Global variables for model
model = None
tokenizer = None
device = None

def download_model():
    """Download model if not present locally"""
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"Model found at {MODEL_PATH}")
        return MODEL_PATH

    print(f"Downloading {MODEL_NAME} (~6.68 GB)...")
    print("This will be cached by transformers...")
    return MODEL_NAME

def load_model_startup():
    """Load model on startup"""
    global model, tokenizer, device

    print("=" * 60)
    print("DeepSeek-OCR Backend (Docker + CUDA 11.8)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Backend: {BACKEND}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Base Size: {BASE_SIZE}")
    print(f"  Image Size: {IMAGE_SIZE}")
    print(f"  Crop Mode: {CROP_MODE}")
    print(f"  Torch dtype: {TORCH_DTYPE}")

    # Use configured device, but verify CUDA availability
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = DEVICE

    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Clear transformers cache to avoid stale files
    import shutil
    cache_dir = "/root/.cache/huggingface/modules/transformers_modules/DeepSeek-OCR"
    if os.path.exists(cache_dir):
        print(f"Clearing stale cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    model_path = download_model()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        force_download=False,
        resume_download=True
    )

    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
        force_download=False,
        resume_download=True,
        _attn_implementation='flash_attention_2'
    )

    # Move model to device and set dtype
    model = model.eval()
    if device == "cuda":
        if TORCH_DTYPE == "bfloat16":
            model = model.cuda().to(torch.bfloat16)
        elif TORCH_DTYPE == "float16":
            model = model.cuda().to(torch.float16)
        else:
            model = model.cuda()

    print("Model loaded successfully!")
    print("=" * 60)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device)
    })

@app.route('/ocr', methods=['POST'])
def ocr():
    """OCR inference endpoint - saves to file and returns result"""
    try:
        data = request.json
        image_path = data.get('image_path')

        print(f"Received OCR request with image_path: '{image_path}'")

        if not image_path:
            return jsonify({"error": "No image_path provided"}), 400

        # Check if file exists
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image file not found: {image_path}"}), 404

        start_time = time.time()

        # Use the EXACT code from HuggingFace model card
        prompt = "<image>\nFree OCR."

        print(f"Running model.infer with image_file='{image_path}'")

        # Use configuration from .env - save to file
        import uuid
        output_dir = f"/tmp/ocr_output_{uuid.uuid4().hex}"
        os.makedirs(output_dir, exist_ok=True)

        # Run model inference (it will save text to file)
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            base_size=BASE_SIZE,
            image_size=IMAGE_SIZE,
            crop_mode=CROP_MODE,
            save_results=True,
            output_path=output_dir
        )

        # Find and read the generated text file
        # The model saves to result.mmd file
        text_content = None
        result_file = os.path.join(output_dir, 'result.mmd')

        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            print(f"Successfully read {len(text_content)} characters from result.mmd")
        else:
            # Fallback: check for any .txt or .mmd files
            for file in os.listdir(output_dir):
                if file.endswith(('.txt', '.mmd')):
                    with open(os.path.join(output_dir, file), 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    break

        if not text_content:
            text_content = "No text extracted"

        print(f"Final extracted text length: {len(text_content)} characters")

        # Clean up temp files
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

        time_taken = time.time() - start_time

        return jsonify({
            "text": text_content,
            "time_taken": time_taken
        })

    except Exception as e:
        print(f"Error in OCR: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/ocr/stream', methods=['POST'])
def ocr_stream():
    """OCR inference with streaming output using Server-Sent Events"""
    from flask import Response, stream_with_context

    def generate():
        try:
            data = request.json
            image_path = data.get('image_path')

            yield f"data: {{\"status\": \"started\", \"message\": \"Starting OCR...\"}}\n\n"

            if not image_path:
                yield f"data: {{\"error\": \"No image_path provided\"}}\n\n"
                return

            if not os.path.exists(image_path):
                yield f"data: {{\"error\": \"Image file not found: {image_path}\"}}\n\n"
                return

            yield f"data: {{\"status\": \"processing\", \"message\": \"Loading image...\"}}\n\n"

            prompt = "<image>\nFree OCR."

            import uuid
            output_dir = f"/tmp/ocr_output_{uuid.uuid4().hex}"
            os.makedirs(output_dir, exist_ok=True)

            yield f"data: {{\"status\": \"processing\", \"message\": \"Running model inference...\"}}\n\n"

            # Run inference
            start_time = time.time()
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                base_size=BASE_SIZE,
                image_size=IMAGE_SIZE,
                crop_mode=CROP_MODE,
                save_results=True,
                output_path=output_dir
            )

            # Read generated text file
            text_content = None
            for file in os.listdir(output_dir):
                if file.endswith('.txt'):
                    with open(os.path.join(output_dir, file), 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    break

            time_taken = time.time() - start_time

            if text_content:
                # Send text in chunks for streaming effect
                import json
                yield f"data: {{\"status\": \"complete\", \"text\": {json.dumps(text_content)}, \"time_taken\": {time_taken}}}\n\n"
            else:
                yield f"data: {{\"error\": \"No text extracted\"}}\n\n"

            # Cleanup
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

        except Exception as e:
            import json
            yield f"data: {{\"error\": {json.dumps(str(e))}}}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    load_model_startup()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
