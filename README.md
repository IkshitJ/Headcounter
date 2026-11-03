# 📹 CCTV Live Head Counter made by Ikshit for Calance


```
Headcounter/
├── config.py               # ⭐ Central master configuration file (edit settings here)
├── app.py                  # FastAPI server with MJPEG streaming & REST API
├── cli.py                  # Standalone CLI tool for terminal & OpenCV GUI
├── download_models.py      # Downloader for pretrained YOLO head checkpoints
├── run_dashboard.sh        # One-click script to start the Web Dashboard
├── run_cli.sh              # One-click script to run the CLI tool
├── requirements.txt        # Python dependencies
├── engine/
│   ├── detector.py         # YOLO Head detector with MPS/CUDA/CPU device selection
│   ├── tracker.py          # Multi-object tracker with trajectory trail history
│   ├── counter.py          # Real-time in-frame, ROI zone, & tripwire counter
│   └── pipeline.py         # Unified video processing orchestrator & visualizer
├── models/
│   ├── 11n-head.pt         # YOLO11 Nano Head detection weights (5.2 MB)
│   └── v8s-head.pt         # YOLOv8 Small Head detection weights (21.5 MB)
├── sample_data/
│   └── cctv_crowd_demo.mp4 # Sample CCTV crowd footage
└── templates/
    └── index.html          # Surveillance web dashboard
```

---

## For Customization & Scaling (`config.py`)

All settings can be changed inside **`config.py`** 

* **Change Server Port / Host:** Modify `SERVER_PORT = 8000` or `SERVER_HOST = "0.0.0.0"`.
* **Set Default CCTV / RTSP Stream:** Modify `DEFAULT_VIDEO_SOURCE = "rtsp://admin:pass@192.168.1.100:554/live"`.
* **Adjust Detection Sensitivity:** Modify `DEFAULT_CONFIDENCE_THRESHOLD = 0.25`.
* **Customize Virtual Tripwire Position:** Modify `DEFAULT_TRIPWIRE_LINE = ((x1, y1), (x2, y2))`.
* **Customize Monitored Polygon Zone:** Modify `DEFAULT_ROI_POLYGON = [(x1, y1), (x2, y2), ...]`.
* **Change Visual HUD Colors:** Modify `COLOR_HEAD_BOX`, `COLOR_HEAD_CENTER`, `COLOR_TRIPWIRE_LINE`.

---


