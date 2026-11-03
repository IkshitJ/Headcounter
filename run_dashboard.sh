#!/usr/bin/env bash
# Quickstart script to launch the CCTV AI Head Counter Web Dashboard
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python3 download_models.py
else
    source venv/bin/activate
fi

echo "=============================================================="
echo "  📹 CCTV AI LIVE HEAD COUNTER DASHBOARD"
echo "  Open your browser at: http://localhost:8000"
echo "=============================================================="
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

