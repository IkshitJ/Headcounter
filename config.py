"""
=============================================================================
CCTV HEAD COUNTER - MASTER CONFIGURATION
=============================================================================
This file is the single source of truth for all system configurations.
You can adjust models, detection sensitivity, RTSP stream URLs, tripwire
coordinates, and server settings directly from here without modifying any
core algorithms or frontend code.
=============================================================================
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "sample_data"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. SERVER & NETWORK SETTINGS
# =============================================================================
SERVER_HOST = "0.0.0.0"      # "0.0.0.0" listens on all network interfaces; "127.0.0.1" for localhost only
SERVER_PORT = 8000           # Change this port if port 8000 is occupied on your system

# =============================================================================
# 2. DEFAULT VIDEO INPUT SOURCE
# =============================================================================
# Can be:
# - An integer for a webcam (e.g., 0, 1)
# - A path to a local video file (e.g., SAMPLE_DIR / "cctv_crowd_demo.mp4")
# - An RTSP CCTV camera stream URL (e.g., "rtsp://admin:pass@192.168.1.100:554/live")
DEFAULT_VIDEO_SOURCE = str(SAMPLE_DIR / "cctv_crowd_demo.mp4")

# =============================================================================
# 3. AI DETECTION MODEL SETTINGS
# =============================================================================
# Default model to load on startup.
# Options:
#   - "yolo11n-head": Ultra-fast YOLO11 Nano fine-tuned on human heads (recommended for real-time 40+ FPS)
#   - "yolov8s-head": YOLOv8 Small model (higher recall in extremely dense crowds)
#   - "yolo11n-person": Standard full-body person detection model (fallback)
DEFAULT_MODEL = "yolo11n-head"

# Detection Sensitivity Thresholds
# Lower confidence = detects more heads (including small/distant heads), but might increase false positives.
# Higher confidence = only detects very clear heads. Range: 0.10 to 0.90
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# Non-Maximum Suppression (NMS) IoU Threshold
# Controls overlap suppression. 0.45 is standard for crowded spaces.
DEFAULT_IOU_THRESHOLD = 0.45

# Hardware Accelerator: "mps" (Apple Silicon GPU), "cuda" (NVIDIA GPU), "cpu", or None (auto-detect)
ACCELERATOR_DEVICE = None

# Model Registry with download sources
MODEL_REGISTRY = {
    "yolo11n-head": {
        "url": "https://huggingface.co/Xulee-0806/YOLO-head-detection/resolve/main/11n-head.pt",
        "filename": "11n-head.pt",
        "description": "YOLO11 Nano Head Detector - Ultra-fast, optimized for high FPS CCTV surveillance",
        "classes": [0],
    },
    "yolov8s-head": {
        "url": "https://huggingface.co/Xulee-0806/YOLO-head-detection/resolve/main/v8s-head.pt",
        "filename": "v8s-head.pt",
        "description": "YOLOv8 Small Head Detector - High accuracy for dense crowds",
        "classes": [0],
    },
    "yolo11n-person": {
        "url": None,  # Downloaded automatically by Ultralytics
        "filename": "yolo11n.pt",
        "description": "Standard YOLO11 Full-Body Person Detector",
        "classes": [0],
    }
}

# =============================================================================
# 4. TRACKING PARAMETERS
# =============================================================================
# Maximum pixel distance between consecutive frames to consider a head as the same person
MAX_TRACKING_DISTANCE = 65.0

# Number of consecutive frames a lost head can be occluded before deleting its track ID
MAX_MISSED_FRAMES = 15

# How many past coordinate points to keep per head to draw motion trails
TRAJECTORY_HISTORY_LENGTH = 30

# =============================================================================
# 5. TRIPWIRE & REGION OF INTEREST (ROI) DEFAULTS
# =============================================================================
# Virtual tripwire line endpoints: ((x1, y1), (x2, y2))
# Head trajectories crossing this line trigger the bidirectional IN/OUT counter.
DEFAULT_TRIPWIRE_LINE = ((50.0, 200.0), (590.0, 200.0))

# Default monitored polygonal zone (points in clockwise order: [(x, y), ...])
# Heads inside this perimeter are counted in the 'ROI Zone Count' metric.
DEFAULT_ROI_POLYGON = [
    (80.0, 60.0),
    (560.0, 60.0),
    (560.0, 320.0),
    (80.0, 320.0)
]

# =============================================================================
# 6. VISUAL RENDERING SETTINGS (OpenCV BGR Colors)
# =============================================================================
SHOW_BOUNDING_BOXES = True
SHOW_MOTION_TRAILS = True
SHOW_HEAD_CROSSHAIRS = True
SHOW_TOP_HUD = True

# Colors in BGR (Blue, Green, Red) format
COLOR_HEAD_BOX = (0, 255, 128)        # Neon Emerald Green
COLOR_HEAD_CENTER = (0, 255, 255)     # Bright Yellow
COLOR_CROSSHAIR_RING = (0, 180, 255)  # Orange-Yellow
COLOR_TRIPWIRE_LINE = (0, 0, 255)     # Red
COLOR_ROI_ZONE = (255, 165, 0)        # Amber Cyan/Orange
COLOR_HUD_BACKGROUND = (15, 18, 25)   # Charcoal Dark Slate

