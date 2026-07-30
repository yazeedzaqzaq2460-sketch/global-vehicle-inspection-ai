from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories
CONFIG_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# AI Models
YOLO_MODEL_DIR = MODEL_DIR / "yolo"
QWEN_MODEL_DIR = MODEL_DIR / "qwen"
OCR_MODEL_DIR = MODEL_DIR / "ocr"

# Device
DEVICE = "cuda"

# Image Settings
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25

# Supported Formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov"]

# Create required directories automatically
for directory in [
    CONFIG_DIR,
    DATASET_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    YOLO_MODEL_DIR,
    QWEN_MODEL_DIR,
    OCR_MODEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)