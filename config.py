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

# Pre-packaged CCTV Surveillance Channels
DEMO_SOURCES = {
    "demo1": str(SAMPLE_DIR / "cctv_crowd_demo.mp4"),
    "demo2": str(SAMPLE_DIR / "cctv_howrah_dense_crowd.mp4"),
    "demo3": str(SAMPLE_DIR / "cctv_india_market_kolkata.mp4"),
    "demo4": str(SAMPLE_DIR / "cctv_shibuya_dense_crossing.mp4"),
    "demo5": str(SAMPLE_DIR / "cctv_paris_station_rush.mp4"),
}

# =============================================================================
# 3. AI DETECTION MODEL SETTINGS
# =============================================================================
# Default model to load on startup.
# Options:
#   - "yolo11n-head": Ultra-fast YOLO11 Nano (recommended for real-time 40+ FPS)
#   - "yolo11s-head": YOLO11 Small model (excellent speed-accuracy balance)
#   - "yolo11m-head": YOLO11 Medium model (superior recall in dense gatherings)
#   - "yolov8n-head": YOLOv8 Nano model (ultra-low latency ~90+ FPS)
#   - "yolov8s-head": YOLOv8 Small model (zero false-box precision baseline)
#   - "yolov8m-head": YOLOv8 Medium model (robust high-capacity crowd head detector)
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
    "yolo11s-head": {
        "url": "https://huggingface.co/Xulee-0806/YOLO-head-detection/resolve/main/11s-head.pt",
        "filename": "11s-head.pt",
        "description": "YOLO11 Small Head Detector - Balanced accuracy and speed",
        "classes": [0],
    },
    "yolo11m-head": {
        "url": "https://huggingface.co/Xulee-0806/YOLO-head-detection/resolve/main/11m-head.pt",
        "filename": "11m-head.pt",
        "description": "YOLO11 Medium Head Detector - Highest recall in dense, occluded crowds",
        "classes": [0],
    },
    "yolov8n-head": {
        "url": "https://huggingface.co/Xulee-0806/YOLO-head-detection/resolve/main/v8n-head.pt",
        "filename": "v8n-head.pt",
        "description": "YOLOv8 Nano Head Detector - Ultra-lightweight & high FPS",
        "classes": [0],
    },
    "yolov8s-head": {
        "url": "https://huggingface.co/Xulee-0806/YOLO-head-detection/resolve/main/v8s-head.pt",
        "filename": "v8s-head.pt",
        "description": "YOLOv8 Small Head Detector - High precision with zero false boxes on empty scenes",
        "classes": [0],
    },
    "yolov8m-head": {
        "url": "https://huggingface.co/Xulee-0806/YOLO-head-detection/resolve/main/v8m-head.pt",
        "filename": "v8m-head.pt",
        "description": "YOLOv8 Medium Head Detector - High capacity for complex surveillance scenes",
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
# Normalized (0.0 to 1.0) coordinates for resolution independence
DEFAULT_TRIPWIRE_LINE = ((0.1, 0.5), (0.9, 0.5))

# Default monitored polygonal zone (points in clockwise order: [(x, y), ...])
# Heads inside this perimeter are counted in the 'ROI Zone Count' metric.
# Normalized (0.0 to 1.0) coordinates for resolution independence
DEFAULT_ROI_POLYGON = [
    (0.15, 0.15),
    (0.85, 0.15),
    (0.85, 0.85),
    (0.15, 0.85)
]

# =============================================================================
# 6. VISUAL RENDERING SETTINGS (OpenCV BGR Colors)
# =============================================================================
SHOW_BOUNDING_BOXES = True
SHOW_MOTION_TRAILS = True
SHOW_HEAD_CROSSHAIRS = True
SHOW_TOP_HUD = False

# Monochrome Pale White & Black BGR Colors
COLOR_HEAD_BOX = (245, 245, 245)        # Crisp Pale White
COLOR_HEAD_CENTER = (255, 255, 255)     # Bright Pale White
COLOR_CROSSHAIR_RING = (160, 160, 160)  # Silver Gray
COLOR_TRIPWIRE_LINE = (220, 220, 220)   # Pale White Line
COLOR_ROI_ZONE = (140, 140, 140)        # Translucent Monochrome Gray
COLOR_HUD_BACKGROUND = (5, 5, 5)        # Deep Pure Black


# =============================================================================
# 7. CROWD DYNAMICS ANALYSIS (Claude + GTFS-RT)
# =============================================================================
# POST /api/analyze fuses live Headcounter metrics with upstream transit feeds
# and asks Claude for a 20-minute density forecast + countermeasure playbook.
# Credentials: set ANTHROPIC_API_KEY in the environment (never in this file).
ANALYSIS_MODEL = "claude-opus-5-5"
ANALYSIS_EFFORT = "high"          # low | medium | high | xhigh | max

# Physical description of the monitored zone. The ROI polygon is in pixels,
# so the real-world area must be supplied here (or per request).
ZONE_NAME = "Demo corridor (camera 1)"
ZONE_AREA_M2 = 120.0              # usable floor area covered by the ROI polygon
ZONE_TOPOLOGY = (
    "Single covered corridor, 4 m wide, narrowing to a 2.2 m turnstile bank "
    "with 3 gates (2 open). One stairway down to platform level at the far end."
)
# Density thresholds (pax/m^2). Fruin LOS F begins ~4.0; the forecast reports
# time-to-critical against DENSITY_CRITICAL.
DENSITY_CRITICAL = 4.0
DENSITY_FATAL = 5.5

# Flow-rate window: lambda/mu are derived from tripwire IN/OUT deltas over the
# most recent N seconds of counter history.
FLOW_WINDOW_SECONDS = 60

# GTFS-Realtime upstream feeds (protobuf). Leave as None to skip transit fusion.
# Example (Delhi OTD): "https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb?key=..."
GTFS_RT_TRIP_UPDATES_URL = os.environ.get("GTFS_RT_TRIP_UPDATES_URL")
GTFS_RT_VEHICLE_POSITIONS_URL = os.environ.get("GTFS_RT_VEHICLE_POSITIONS_URL")
# GTFS stop_ids that feed this zone (arrivals here become inbound pulses).
GTFS_STOP_IDS: list = []
GTFS_LOOKAHEAD_MINUTES = 25
# Assumed passengers alighting per arrival when no occupancy data is published.
GTFS_DEFAULT_ALIGHTING_PER_ARRIVAL = 150
