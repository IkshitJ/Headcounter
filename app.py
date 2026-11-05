"""
Web Application & REST API Server
Provides the surveillance web dashboard, real-time MJPEG video stream,
video upload handling, and metrics endpoints.
"""

import os
import io
import time
import shutil
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, Response, FileResponse
from pydantic import BaseModel

from config import (
    BASE_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
    SAMPLE_DIR,
    DEFAULT_VIDEO_SOURCE,
    DEMO_SOURCES,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TRIPWIRE_LINE,
    DEFAULT_ROI_POLYGON,
    SERVER_HOST,
    SERVER_PORT
)
from engine.pipeline import VideoPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HeadCounterApp")

# Initialize Pipeline with defaults from config.py
pipeline = VideoPipeline(
    source=DEFAULT_VIDEO_SOURCE if os.path.exists(DEFAULT_VIDEO_SOURCE) else 0,
    model_name=DEFAULT_MODEL,
    conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD
)
pipeline.open_source(DEFAULT_VIDEO_SOURCE if os.path.exists(DEFAULT_VIDEO_SOURCE) else 0)

app = FastAPI(title="CCTV Live Head Counter API")


class SourceRequest(BaseModel):
    source_type: str  # 'demo', 'webcam', 'rtsp', 'file'
    demo_id: Optional[str] = None
    url: Optional[str] = None
    index: Optional[int] = 0
    filepath: Optional[str] = None


class SettingsRequest(BaseModel):
    model_name: Optional[str] = None
    conf_threshold: Optional[float] = None
    is_paused: Optional[bool] = None
    overlay_type: Optional[str] = None
    enabled: Optional[bool] = None


class TripwireRequest(BaseModel):
    enabled: bool
    pt1: Optional[List[float]] = None
    pt2: Optional[List[float]] = None


class ROIRequest(BaseModel):
    enabled: bool
    points: Optional[List[List[float]]] = None


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the main CCTV Surveillance Web Dashboard."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    return FileResponse(index_file)


@app.get("/video_feed")
def video_feed():
    """
    MJPEG Video streaming endpoint for live head counting feed.
    Streams annotated frames directly to HTML <img> tags or video players.
    """
    return StreamingResponse(
        pipeline.stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.get("/api/frame")
def get_frame():
    """Returns the latest annotated JPEG frame (Cloudflare & Safari safe)."""
    frame_bytes = pipeline.get_current_frame_jpeg()
    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns current live head count, peak count, FPS, and detection metrics."""
    metrics = dict(pipeline.current_metrics)
    metrics["model_name"] = pipeline.detector.model_name
    metrics["device"] = pipeline.detector.device
    metrics["conf_threshold"] = pipeline.conf_threshold
    metrics["is_paused"] = pipeline.is_paused
    metrics["total_unique_heads"] = len(pipeline.counter.seen_track_ids)
    metrics["tripwire"] = {
        "enabled": pipeline.raw_counting_line is not None,
        "pt1": pipeline.raw_counting_line[0] if pipeline.raw_counting_line else None,
        "pt2": pipeline.raw_counting_line[1] if pipeline.raw_counting_line else None,
    }
    metrics["roi"] = {
        "enabled": pipeline.raw_roi_polygon is not None,
        "points": pipeline.raw_roi_polygon if pipeline.raw_roi_polygon else None,
    }
    return metrics


@app.get("/api/history")
def get_history() -> List[Dict[str, Any]]:
    """Returns rolling temporal history for time-series charts."""
    return pipeline.counter.history


@app.post("/api/set_source")
def set_source(req: SourceRequest):
    """Switches the active CCTV video input source."""
    try:
        if req.source_type.startswith("demo"):
            target_key = req.demo_id or req.source_type
            if target_key in DEMO_SOURCES and os.path.exists(DEMO_SOURCES[target_key]):
                target_video = DEMO_SOURCES[target_key]
            elif os.path.exists(DEFAULT_VIDEO_SOURCE):
                target_video = DEFAULT_VIDEO_SOURCE
            else:
                raise HTTPException(status_code=404, detail="Demo video not found")
            success = pipeline.open_source(target_video)
        elif req.source_type == "webcam":
            success = pipeline.open_source(int(req.index or 0))
        elif req.source_type == "rtsp":
            if not req.url:
                raise HTTPException(status_code=400, detail="RTSP URL required")
            success = pipeline.open_source(req.url)
        elif req.source_type == "file":
            if not req.filepath or not os.path.exists(req.filepath):
                raise HTTPException(status_code=404, detail="File not found")
            success = pipeline.open_source(req.filepath)
        else:
            raise HTTPException(status_code=400, detail="Unknown source_type")

        return {"success": success}
    except Exception as e:
        logger.error(f"Error changing source: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Handles surveillance video file uploads and starts head counting on the uploaded clip."""
    try:
        filename = f"{int(time.time())}_{file.filename}"
        dest_path = os.path.join(UPLOADS_DIR, filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        success = pipeline.open_source(dest_path)
        return {"success": success, "filename": file.filename, "saved_path": dest_path}
    except Exception as e:
        logger.error(f"Error handling video upload: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    """Updates AI model, confidence thresholds, and visual overlays in real-time."""
    if req.model_name:
        pipeline.set_model(req.model_name)
    if req.conf_threshold is not None:
        pipeline.set_confidence(req.conf_threshold)
    if req.is_paused is not None:
        pipeline.is_paused = req.is_paused
    if req.overlay_type and req.enabled is not None:
        if req.overlay_type == "boxes":
            pipeline.show_boxes = req.enabled
        elif req.overlay_type == "trails":
            pipeline.show_trails = req.enabled
        elif req.overlay_type == "crosshairs":
            pipeline.show_crosshairs = req.enabled
        elif req.overlay_type == "hud":
            pipeline.show_hud = req.enabled
    return {"success": True}


@app.post("/api/tripwire")
def configure_tripwire(req: TripwireRequest):
    """Configures virtual tripwire coordinates for bidirectional IN/OUT line crossing."""
    if req.enabled:
        pt1 = tuple(req.pt1) if req.pt1 is not None else DEFAULT_TRIPWIRE_LINE[0]
        pt2 = tuple(req.pt2) if req.pt2 is not None else DEFAULT_TRIPWIRE_LINE[1]
        pipeline.set_counting_line(pt1, pt2)
    else:
        pipeline.set_counting_line(None, None)
    return {
        "success": True,
        "tripwire_enabled": req.enabled,
        "pt1": pipeline.raw_counting_line[0] if pipeline.raw_counting_line else None,
        "pt2": pipeline.raw_counting_line[1] if pipeline.raw_counting_line else None,
    }


@app.post("/api/roi")
def configure_roi(req: ROIRequest):
    """Configures monitored region of interest polygon points."""
    if req.enabled:
        points = [tuple(p) for p in req.points] if req.points is not None else DEFAULT_ROI_POLYGON
        pipeline.set_roi_polygon(points)
    else:
        pipeline.set_roi_polygon(None)
    return {
        "success": True,
        "roi_enabled": req.enabled,
        "points": pipeline.raw_roi_polygon if pipeline.raw_roi_polygon else None,
    }


@app.post("/api/reset_counts")
def reset_counts():
    """Resets peak counts, unique track IDs, and crossing counters."""
    pipeline.counter.reset()
    pipeline.tracker.reset()
    return {"success": True}


@app.get("/api/export_csv")
def export_csv():
    """Exports a timestamped CSV of headcount records."""
    output = io.StringIO()
    output.write("timestamp,datetime,head_count,roi_count,in_count,out_count\n")
    for entry in pipeline.counter.history:
        ts = entry["time"]
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        output.write(f"{ts},{dt_str},{entry['count']},{entry['roi_count']},{entry['in']},{entry['out']}\n")

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cctv_headcount_{int(time.time())}.csv"}
    )


class AnalyzeRequest(BaseModel):
    zone_name: Optional[str] = None
    area_m2: Optional[float] = None
    topology: Optional[str] = None
    operator_notes: Optional[str] = None
    extra_telemetry: Optional[Dict[str, Any]] = None  # e.g. {"rf_probes": {...}, "environment": {...}}


_last_analysis: Optional[Dict[str, Any]] = None


@app.post("/api/analyze")
def run_crowd_analysis(req: AnalyzeRequest):
    """Fuses live counts with GTFS-RT and returns a 20-minute stampede-risk forecast from Claude."""
    global _last_analysis
    import anthropic
    from engine import crowd_analyst

    try:
        result = crowd_analyst.analyze(
            pipeline,
            zone_name=req.zone_name or crowd_analyst.ZONE_NAME,
            area_m2=req.area_m2 or crowd_analyst.ZONE_AREA_M2,
            topology=req.topology or crowd_analyst.ZONE_TOPOLOGY,
            operator_notes=req.operator_notes,
            extra_telemetry=req.extra_telemetry,
        )
    except (anthropic.AuthenticationError, TypeError):
        # TypeError: the SDK could not resolve any credential (no key, token or profile)
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is missing or invalid")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Claude API rate limit hit; retry shortly")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the Claude API")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    _last_analysis = result
    return result


@app.get("/api/analyze/last")
def get_last_analysis():
    """Returns the most recent crowd analysis without calling the model again."""
    if _last_analysis is None:
        raise HTTPException(status_code=404, detail="No analysis has been run yet")
    return _last_analysis


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
