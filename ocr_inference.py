"""https://huggingface.co/deepseek-ai/DeepSeek-OCR"""
import os
import time
import platform
import subprocess
import requests
from PIL import Image
from io import BytesIO

# Configuration
DOCKER_IMAGE_NAME = "deepseek-ocr-backend"
DOCKER_CONTAINER_NAME = "deepseek-ocr-server"
BACKEND_PORT = 5000
MODEL_PATH_HOST = "D:/data/models"

def is_docker_installed():
    """Check if Docker is installed"""
    try:
        subprocess.run(['docker', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def is_docker_running():
    """Check if Docker daemon is running"""
    try:
        subprocess.run(['docker', 'ps'], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_container_exists():
    """Check if container already exists"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', f'name={DOCKER_CONTAINER_NAME}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        return DOCKER_CONTAINER_NAME in result.stdout
    except subprocess.CalledProcessError:
        return False

def check_container_running():
    """Check if container is running"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={DOCKER_CONTAINER_NAME}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        return DOCKER_CONTAINER_NAME in result.stdout
    except subprocess.CalledProcessError:
        return False

def build_docker_image():
    """Build Docker image"""
    print("\n" + "=" * 60)
    print("Building Docker image (this may take a few minutes)...")
    print("=" * 60)

    try:
        subprocess.run(
            ['docker', 'build', '-t', DOCKER_IMAGE_NAME, '.'],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("✓ Docker image built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build Docker image: {e}")
        return False

def start_docker_container():
    """Start Docker container"""
    print("\n" + "=" * 60)
    print("Starting Docker container with CUDA support...")
    print("=" * 60)

    # Create models directory if it doesn't exist
    os.makedirs(MODEL_PATH_HOST, exist_ok=True)

    try:
        # Check if container exists
        if check_container_exists():
            if check_container_running():
                print("✓ Container already running!")
                return True
            else:
                print("Starting existing container...")
                subprocess.run(['docker', 'start', DOCKER_CONTAINER_NAME], check=True)
                print("✓ Container started!")
                return True

        # Run new container
        print("Creating and starting new container...")
        subprocess.run([
            'docker', 'run', '-d',
            '--name', DOCKER_CONTAINER_NAME,
            '--gpus', 'all',  # Enable GPU support
            '-p', f'{BACKEND_PORT}:5000',
            '-v', f'{MODEL_PATH_HOST}:/data/models',
            DOCKER_IMAGE_NAME
        ], check=True)

        print("✓ Container started successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to start container: {e}")
        print("\nIf you see GPU errors, make sure nvidia-docker is installed.")
        print("Try without GPU: docker run -d --name {} -p {}:5000 -v {}:/data/models {}".format(
            DOCKER_CONTAINER_NAME, BACKEND_PORT, MODEL_PATH_HOST, DOCKER_IMAGE_NAME
        ))
        return False

def wait_for_backend():
    """Wait for backend to be ready"""
    print("\nWaiting for backend to initialize...")
    max_retries = 60
    for i in range(max_retries):
        try:
            response = requests.get(f'http://localhost:{BACKEND_PORT}/health', timeout=2)
            if response.status_code == 200:
                data = response.json()
                print(f"\n✓ Backend ready! Device: {data.get('device', 'unknown')}")
                return True
        except requests.exceptions.RequestException:
            pass

        print(f"Waiting... ({i+1}/{max_retries})", end='\r')
        time.sleep(2)

    print("\n✗ Backend failed to start. Check Docker logs:")
    print(f"  docker logs {DOCKER_CONTAINER_NAME}")
    return False

def setup_linux_backend():
    """Setup Linux backend in Docker"""
    print("=" * 60)
    print("DeepSeek-OCR Inference Tool")
    print("Windows detected - Setting up Linux backend in Docker")
    print("=" * 60)

    # Check Docker installation
    if not is_docker_installed():
        print("\n✗ Docker is not installed!")
        print("\nPlease install Docker Desktop from:")
        print("https://www.docker.com/products/docker-desktop")
        return False

    if not is_docker_running():
        print("\n✗ Docker is not running!")
        print("Please start Docker Desktop and try again.")
        return False

    print("✓ Docker is installed and running")

    # Check if image exists
    try:
        result = subprocess.run(
            ['docker', 'images', '-q', DOCKER_IMAGE_NAME],
            capture_output=True,
            text=True,
            check=True
        )
        image_exists = bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        image_exists = False

    # Build image if needed
    if not image_exists:
        if not build_docker_image():
            return False
    else:
        print("✓ Docker image already exists")

    # Start container
    if not start_docker_container():
        return False

    # Wait for backend
    if not wait_for_backend():
        return False

    return True

def load_image(path_or_url):
    """Load and validate image"""
    if path_or_url.startswith(('http://', 'https://')):
        print(f"Downloading image from URL...")
        response = requests.get(path_or_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Image file not found: {path_or_url}")
        image = Image.open(path_or_url)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

def upload_image_to_container(local_path):
    """Upload image file to Docker container"""
    container_path = f"/tmp/image_{int(time.time())}.jpg"

    # Copy file into container
    try:
        subprocess.run(
            ['docker', 'cp', local_path, f'{DOCKER_CONTAINER_NAME}:{container_path}'],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Uploaded to container: {container_path}")
        return container_path
    except subprocess.CalledProcessError as e:
        print(f"Docker cp failed: {e.stderr}")
        raise Exception(f"Failed to copy image to container: {e.stderr}")

def perform_ocr(image_path):
    """Perform OCR by calling backend API with progress tracking"""
    import threading
    import sys

    print("\n🔍 Starting OCR inference...")

    # Progress animation
    progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    done = False

    def show_progress():
        idx = 0
        stages = [
            "Loading image into model",
            "Processing vision encoder",
            "Extracting text content",
            "Finalizing results"
        ]
        stage_idx = 0
        counter = 0

        while not done:
            if counter % 10 == 0 and counter > 0:
                stage_idx = min(stage_idx + 1, len(stages) - 1)

            sys.stdout.write(f'\r{progress_chars[idx]} {stages[stage_idx]}...')
            sys.stdout.flush()
            idx = (idx + 1) % len(progress_chars)
            counter += 1
            time.sleep(0.1)

        sys.stdout.write('\r✓ OCR extraction completed!        \n')
        sys.stdout.flush()

    # Start progress animation in background
    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()

    try:
        response = requests.post(
            f'http://localhost:{BACKEND_PORT}/ocr',
            json={'image_path': image_path},
            timeout=300  # 5 minutes timeout
        )

        done = True
        progress_thread.join(timeout=0.5)

        if response.status_code == 200:
            data = response.json()
            return data['text'], data['time_taken']
        else:
            error_data = response.json()
            raise Exception(f"Backend error: {error_data.get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        done = True
        progress_thread.join(timeout=0.5)
        raise Exception(f"Failed to connect to backend: {e}")

def main():
    # Check if running on Windows
    if platform.system() == "Windows":
        # Setup Linux backend
        if not setup_linux_backend():
            print("\nFailed to setup backend. Exiting...")
            return
    else:
        print("Running on non-Windows system. Please use the native version.")
        return

    print("\n" + "=" * 60)
    print("Model ready! You can now perform OCR inference.")
    print("=" * 60)

    # Interactive inference loop
    while True:
        print("\nEnter image path or URL (or 'quit' to exit):")
        user_input = input("> ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nStopping container...")
            subprocess.run(['docker', 'stop', DOCKER_CONTAINER_NAME], capture_output=True)
            print("Exiting...")
            break

        if not user_input:
            print("Please enter a valid path or URL.")
            continue

        try:
            # Validate image
            image = load_image(user_input)
            print(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")

            # Handle local files vs URLs
            if user_input.startswith(('http://', 'https://')):
                # For URLs, download and save locally, then upload to container
                temp_local = "temp_downloaded.jpg"
                image.save(temp_local)
                container_path = upload_image_to_container(temp_local)
                os.remove(temp_local)
            else:
                # For local files, upload to container
                container_path = upload_image_to_container(user_input)

            # Perform OCR
            text, time_taken = perform_ocr(container_path)

            # Display results
            print("\n" + "=" * 60)
            print("EXTRACTED TEXT:")
            print("=" * 60)
            print(text)
            print("=" * 60)
            print(f"Time taken: {time_taken:.2f} seconds")
            print("=" * 60)

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
