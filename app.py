"""
Web Application & REST API Server
Provides the surveillance web dashboard, real-time multi-camera CCTV feeds (Quad 2x2 grid),
per-camera head counting & overcrowding alerts, interactive tripwire & ROI, and analytics.
"""

import os
# Prevent OpenMP and threading conflicts on cloud containers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import time
import shutil
import logging
import threading
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
    QUAD_CAMERAS,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TRIPWIRE_LINE,
    DEFAULT_ROI_POLYGON,
    ACCELERATOR_DEVICE,
    SERVER_HOST,
    SERVER_PORT
)
from engine.detector import HeadDetector
from engine.pipeline import VideoPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HeadCounterApp")


class CameraManager:
    """
    Manages all CCTV camera streams simultaneously.
    Shares a single HeadDetector instance across all pipelines to ensure efficient memory usage
    while maintaining independent tracking, counts, tripwires, and ROI zones per camera.
    Uses on-demand frame retrieval with high-frequency caching so that inference runs only when
    viewed, completely eliminating background thread locks, OpenMP segfaults, and CPU starvation.
    """
    def __init__(self):
        logger.info("Initializing Shared HeadDetector for Multi-Camera CCTV Grid...")
        self.detector = HeadDetector(
            model_name=DEFAULT_MODEL,
            conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            iou_threshold=DEFAULT_IOU_THRESHOLD,
            device=ACCELERATOR_DEVICE
        )
        self.pipelines: Dict[str, VideoPipeline] = {}
        self.cached_frames: Dict[str, bytes] = {}
        self.last_update_time: Dict[str, float] = {}
        self.locks: Dict[str, threading.Lock] = {}

        for cam_id, cfg in QUAD_CAMERAS.items():
            source_path = cfg["source"]
            # Fallback to default video or webcam if file not present
            if not (isinstance(source_path, str) and os.path.exists(source_path)):
                source_path = DEFAULT_VIDEO_SOURCE if os.path.exists(DEFAULT_VIDEO_SOURCE) else 0

            pipe = VideoPipeline(
                source=source_path,
                model_name=DEFAULT_MODEL,
                conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                detector=self.detector,
                camera_id=cam_id,
                camera_name=cfg["name"]
            )
            pipe.open_source(source_path)
            self.pipelines[cam_id] = pipe
            self.cached_frames[cam_id] = b""
            self.last_update_time[cam_id] = 0.0
            self.locks[cam_id] = threading.Lock()
            logger.info(f"Initialized Camera Channel {cam_id}: {cfg['name']}")

    def get_pipeline(self, cam_id: str = "cam1") -> VideoPipeline:
        return self.pipelines.get(cam_id, self.pipelines.get("cam1"))

    def get_frame(self, cam_id: str = "cam1") -> bytes:
        pipe = self.get_pipeline(cam_id)
        if not pipe:
            return b""

        lock = self.locks.get(cam_id)
        with lock:
            now = time.time()
            # Cache for 40ms (~25 FPS max) to prevent duplicate work if multiple clients poll
            if self.cached_frames[cam_id] and (now - self.last_update_time[cam_id] < 0.04):
                return self.cached_frames[cam_id]

            try:
                frame_bytes = pipe.get_current_frame_jpeg()
                if frame_bytes:
                    self.cached_frames[cam_id] = frame_bytes
                    self.last_update_time[cam_id] = now
                    return frame_bytes
            except Exception as e:
                logger.error(f"Error reading frame for {cam_id}: {e}")

            return self.cached_frames.get(cam_id, b"")

    def stream_generator(self, cam_id: str = "cam1"):
        while True:
            frame_bytes = self.get_frame(cam_id)
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.04)


# Initialize Multi-Camera Grid Manager
camera_manager = CameraManager()
# Primary pipeline reference for legacy endpoints
pipeline = camera_manager.get_pipeline("cam1")

app = FastAPI(title="CCTV Multi-Feed Live Head Counter API")


class SourceRequest(BaseModel):
    camera_id: Optional[str] = "cam1"
    source_type: str  # 'demo', 'webcam', 'rtsp', 'file'
    demo_id: Optional[str] = None
    url: Optional[str] = None
    index: Optional[int] = 0
    filepath: Optional[str] = None


class SettingsRequest(BaseModel):
    camera_id: Optional[str] = None
    model_name: Optional[str] = None
    conf_threshold: Optional[float] = None
    is_paused: Optional[bool] = None
    overlay_type: Optional[str] = None
    enabled: Optional[bool] = None


class TripwireRequest(BaseModel):
    camera_id: Optional[str] = "cam1"
    enabled: bool
    pt1: Optional[List[float]] = None
    pt2: Optional[List[float]] = None


class ROIRequest(BaseModel):
    camera_id: Optional[str] = "cam1"
    enabled: bool
    points: Optional[List[List[float]]] = None


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the main CCTV Surveillance Web Dashboard."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    return FileResponse(index_file)


@app.get("/video_feed")
def video_feed(cam_id: str = "cam1"):
    """
    MJPEG Video streaming endpoint for live head counting feed.
    Streams annotated frames directly to HTML <img> tags or video players.
    """
    return StreamingResponse(
        camera_manager.stream_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.get("/api/stream/{cam_id}")
def video_stream_camera(cam_id: str):
    """MJPEG stream for a specific camera channel."""
    return StreamingResponse(
        camera_manager.stream_generator(cam_id),
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
def get_frame(cam_id: str = "cam1"):
    """Returns the latest annotated JPEG frame for the primary or specified camera."""
    frame_bytes = camera_manager.get_frame(cam_id)
    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/api/frame/{cam_id}")
def get_camera_frame(cam_id: str):
    """Returns the latest annotated JPEG frame for a specific camera channel."""
    frame_bytes = camera_manager.get_frame(cam_id)
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
    """
    Returns current live head count, peak count, FPS, and detection metrics
    for all 4 simultaneous CCTV feeds as well as aggregated surveillance metrics.
    """
    cameras_status = {}
    total_heads_all = 0
    total_unique_all = 0
    total_in_all = 0
    total_out_all = 0

    for cid, pipe in camera_manager.pipelines.items():
        m = dict(pipe.current_metrics)
        m["camera_id"] = cid
        m["name"] = pipe.camera_name
        m["location"] = QUAD_CAMERAS.get(cid, {}).get("location", "")
        m["is_paused"] = pipe.is_paused
        m["total_unique_heads"] = len(pipe.counter.seen_track_ids)
        m["tripwire"] = {
            "enabled": pipe.raw_counting_line is not None,
            "pt1": pipe.raw_counting_line[0] if pipe.raw_counting_line else None,
            "pt2": pipe.raw_counting_line[1] if pipe.raw_counting_line else None,
        }
        m["roi"] = {
            "enabled": pipe.raw_roi_polygon is not None,
            "points": pipe.raw_roi_polygon if pipe.raw_roi_polygon else None,
        }
        total_heads_all += m.get("current_head_count", 0)
        total_unique_all += m["total_unique_heads"]
        total_in_all += m.get("in_count", 0)
        total_out_all += m.get("out_count", 0)
        cameras_status[cid] = m

    primary = camera_manager.get_pipeline("cam1")
    top = dict(primary.current_metrics if primary else {})
    top["cameras"] = cameras_status
    top["total_head_count_all"] = total_heads_all
    top["total_unique_heads_all"] = total_unique_all
    top["total_in_all"] = total_in_all
    top["total_out_all"] = total_out_all
    top["model_name"] = camera_manager.detector.model_name
    top["device"] = camera_manager.detector.device
    top["conf_threshold"] = primary.conf_threshold if primary else DEFAULT_CONFIDENCE_THRESHOLD
    top["is_paused"] = primary.is_paused if primary else False
    top["total_unique_heads"] = len(primary.counter.seen_track_ids) if primary else 0
    top["tripwire"] = cameras_status.get("cam1", {}).get("tripwire", {})
    top["roi"] = cameras_status.get("cam1", {}).get("roi", {})
    return top


@app.get("/api/history")
def get_history(cam_id: str = "cam1") -> List[Dict[str, Any]]:
    """Returns rolling temporal history for time-series charts for a camera."""
    pipe = camera_manager.get_pipeline(cam_id)
    return pipe.counter.history if pipe else []


@app.post("/api/set_source")
def set_source(req: SourceRequest):
    """Switches the active video source for the target camera."""
    cam_id = req.camera_id or "cam1"
    target_pipe = camera_manager.get_pipeline(cam_id)
    if not target_pipe:
        raise HTTPException(status_code=404, detail="Camera not found")

    try:
        if req.source_type.startswith("demo"):
            target_key = req.demo_id or req.source_type
            if target_key in DEMO_SOURCES and os.path.exists(DEMO_SOURCES[target_key]):
                target_video = DEMO_SOURCES[target_key]
            elif os.path.exists(DEFAULT_VIDEO_SOURCE):
                target_video = DEFAULT_VIDEO_SOURCE
            else:
                raise HTTPException(status_code=404, detail="Demo video not found")
            success = target_pipe.open_source(target_video)
        elif req.source_type == "webcam":
            success = target_pipe.open_source(int(req.index or 0))
        elif req.source_type == "rtsp":
            if not req.url:
                raise HTTPException(status_code=400, detail="RTSP URL required")
            success = target_pipe.open_source(req.url)
        elif req.source_type == "file":
            if not req.filepath or not os.path.exists(req.filepath):
                raise HTTPException(status_code=404, detail="File not found")
            success = target_pipe.open_source(req.filepath)
        else:
            raise HTTPException(status_code=400, detail="Unknown source_type")

        return {"success": success, "camera_id": cam_id}
    except Exception as e:
        logger.error(f"Error changing source for {cam_id}: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...), camera_id: str = "cam1"):
    """Handles surveillance video file uploads and assigns to target camera."""
    target_pipe = camera_manager.get_pipeline(camera_id)
    if not target_pipe:
        raise HTTPException(status_code=404, detail="Camera not found")

    try:
        filename = f"{int(time.time())}_{file.filename}"
        dest_path = os.path.join(UPLOADS_DIR, filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        success = target_pipe.open_source(dest_path)
        return {"success": success, "filename": file.filename, "saved_path": dest_path, "camera_id": camera_id}
    except Exception as e:
        logger.error(f"Error handling video upload: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    """Updates AI model, confidence thresholds, and visual overlays."""
    # Update AI model globally across all pipelines
    if req.model_name:
        camera_manager.detector.load_model(req.model_name)
        for pipe in camera_manager.pipelines.values():
            pipe.detector = camera_manager.detector

    target_pipes = [camera_manager.get_pipeline(req.camera_id)] if req.camera_id else list(camera_manager.pipelines.values())

    for pipe in target_pipes:
        if not pipe:
            continue
        if req.conf_threshold is not None:
            pipe.set_confidence(req.conf_threshold)
        if req.is_paused is not None:
            pipe.is_paused = req.is_paused
        if req.overlay_type and req.enabled is not None:
            if req.overlay_type == "boxes":
                pipe.show_boxes = req.enabled
            elif req.overlay_type == "trails":
                pipe.show_trails = req.enabled
            elif req.overlay_type == "crosshairs":
                pipe.show_crosshairs = req.enabled
            elif req.overlay_type == "hud":
                pipe.show_hud = req.enabled

    return {"success": True}


@app.post("/api/tripwire")
def configure_tripwire(req: TripwireRequest):
    """Configures virtual tripwire coordinates for bidirectional IN/OUT line crossing."""
    pipe = camera_manager.get_pipeline(req.camera_id or "cam1")
    if not pipe:
        raise HTTPException(status_code=404, detail="Camera not found")

    if req.enabled:
        pt1 = tuple(req.pt1) if req.pt1 is not None else DEFAULT_TRIPWIRE_LINE[0]
        pt2 = tuple(req.pt2) if req.pt2 is not None else DEFAULT_TRIPWIRE_LINE[1]
        pipe.set_counting_line(pt1, pt2)
    else:
        pipe.set_counting_line(None, None)

    return {
        "success": True,
        "camera_id": req.camera_id or "cam1",
        "tripwire_enabled": req.enabled,
        "pt1": pipe.raw_counting_line[0] if pipe.raw_counting_line else None,
        "pt2": pipe.raw_counting_line[1] if pipe.raw_counting_line else None,
    }


@app.post("/api/roi")
def configure_roi(req: ROIRequest):
    """Configures monitored region of interest polygon points."""
    pipe = camera_manager.get_pipeline(req.camera_id or "cam1")
    if not pipe:
        raise HTTPException(status_code=404, detail="Camera not found")

    if req.enabled:
        points = [tuple(p) for p in req.points] if req.points is not None else DEFAULT_ROI_POLYGON
        pipe.set_roi_polygon(points)
    else:
        pipe.set_roi_polygon(None)

    return {
        "success": True,
        "camera_id": req.camera_id or "cam1",
        "roi_enabled": req.enabled,
        "points": pipe.raw_roi_polygon if pipe.raw_roi_polygon else None,
    }


@app.post("/api/reset_counts")
def reset_counts(camera_id: Optional[str] = None):
    """Resets peak counts, unique track IDs, and crossing counters."""
    targets = [camera_manager.get_pipeline(camera_id)] if camera_id else list(camera_manager.pipelines.values())
    for pipe in targets:
        if pipe:
            pipe.counter.reset()
            pipe.tracker.reset()
    return {"success": True}


@app.get("/api/export_csv")
def export_csv(cam_id: str = "cam1"):
    """Exports a timestamped CSV of headcount records for a camera."""
    pipe = camera_manager.get_pipeline(cam_id)
    history = pipe.counter.history if pipe else []

    output = io.StringIO()
    output.write(f"# Camera: {pipe.camera_name if pipe else cam_id}\n")
    output.write("timestamp,datetime,head_count,roi_count,in_count,out_count\n")
    for entry in history:
        ts = entry["time"]
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        output.write(f"{ts},{dt_str},{entry['count']},{entry['roi_count']},{entry['in']},{entry['out']}\n")

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cctv_{cam_id}_headcount_{int(time.time())}.csv"}
    )


class AnalyzeRequest(BaseModel):
    camera_id: Optional[str] = "cam1"
    zone_name: Optional[str] = None
    area_m2: Optional[float] = None
    topology: Optional[str] = None
    operator_notes: Optional[str] = None
    extra_telemetry: Optional[Dict[str, Any]] = None


_last_analysis: Optional[Dict[str, Any]] = None


@app.post("/api/analyze")
def run_crowd_analysis(req: AnalyzeRequest):
    """Fuses live counts with GTFS-RT and returns a 20-minute stampede-risk forecast from Claude."""
    global _last_analysis
    import anthropic
    from engine import crowd_analyst

    target_pipe = camera_manager.get_pipeline(req.camera_id or "cam1")
    if not target_pipe:
        raise HTTPException(status_code=404, detail="Camera pipeline not found")

    try:
        result = crowd_analyst.analyze(
            target_pipe,
            zone_name=req.zone_name or crowd_analyst.ZONE_NAME,
            area_m2=req.area_m2 or crowd_analyst.ZONE_AREA_M2,
            topology=req.topology or crowd_analyst.ZONE_TOPOLOGY,
            operator_notes=req.operator_notes,
            extra_telemetry=req.extra_telemetry,
        )
    except (anthropic.AuthenticationError, TypeError):
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


@app.get("/dashboard", response_class=FileResponse)
async def serve_dashboard():
    """Serves the main CCTV Surveillance Web Dashboard for iframe mounting."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    return FileResponse(index_file)


# Gradio Wrapper for Hugging Face Spaces & ZeroGPU compatibility
try:
    import gradio as gr
    with gr.Blocks(
        title="CCTV Multi-Feed Live Head Counter",
        fill_width=True,
        css="""
        footer {display: none !important;}
        .gradio-container {padding: 0 !important; max-width: 100% !important; margin: 0 !important; background: #000 !important;}
        body {background-color: #000 !important; margin: 0 !important;}
        """
    ) as demo:
        gr.HTML("""
        <iframe src="/dashboard" style="width: 100vw; height: 96vh; border: none; overflow: hidden;" allow="camera; microphone; autoplay"></iframe>
        """)

    app = gr.mount_gradio_app(app, demo, path="/")
except Exception as e:
    logger.warning(f"Gradio mount skipped: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

