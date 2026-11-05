"""
Video Processing Pipeline Module
Orchestrates frame capture, head detection, tracking, counting, and HUD rendering.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Tuple, List, Generator
import cv2
import numpy as np

from config import (
    DEFAULT_VIDEO_SOURCE,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    ACCELERATOR_DEVICE,
    SHOW_BOUNDING_BOXES,
    SHOW_MOTION_TRAILS,
    SHOW_HEAD_CROSSHAIRS,
    SHOW_TOP_HUD,
    COLOR_HEAD_BOX,
    COLOR_HEAD_CENTER,
    COLOR_CROSSHAIR_RING,
    COLOR_TRIPWIRE_LINE,
    COLOR_ROI_ZONE,
    COLOR_HUD_BACKGROUND
)
from .detector import HeadDetector
from .tracker import HeadTracker
from .counter import HeadCounter

logger = logging.getLogger("HeadPipeline")


class VideoPipeline:
    """
    Unified CCTV Video Processing Pipeline for Head Detection, Tracking & Counting.
    Handles RTSP streams, local video files, and live webcams.
    """

    def __init__(
        self,
        source: Any = DEFAULT_VIDEO_SOURCE,
        model_name: str = DEFAULT_MODEL,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: Optional[str] = ACCELERATOR_DEVICE
    ):
        self.source = source
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.detector = HeadDetector(
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        self.tracker = HeadTracker()
        self.counter = HeadCounter()
        self.raw_counting_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.raw_roi_polygon: Optional[List[Tuple[float, float]]] = None

        # Video Capture handle
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False

        # Visual settings
        self.show_boxes = SHOW_BOUNDING_BOXES
        self.show_trails = SHOW_MOTION_TRAILS
        self.show_hud = SHOW_TOP_HUD
        self.show_crosshairs = SHOW_HEAD_CROSSHAIRS

        # Performance stats
        self.fps = 0.0
        self.latency_ms = 0.0
        self._last_time = time.time()
        self.current_metrics: Dict[str, Any] = {
            "current_head_count": 0,
            "roi_head_count": 0,
            "peak_head_count": 0,
            "in_count": 0,
            "out_count": 0,
            "fps": 0.0,
            "latency_ms": 0.0
        }

    def open_source(self, source: Any) -> bool:
        """Opens or switches the active video input (webcam index, RTSP URL, or video file)."""
        if self.cap is not None:
            self.cap.release()

        self.source = source
        self.tracker.reset()
        self.counter.reset()

        logger.info(f"Opening video source: {source}")
        if isinstance(source, str) and source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return False

        logger.info(f"Successfully opened video source: {source}")
        return True

    def set_model(self, model_name: str):
        """Switches the active detection model."""
        self.detector.load_model(model_name)
        self.tracker.reset()

    def set_confidence(self, conf: float):
        """Updates detection confidence threshold."""
        self.conf_threshold = conf
        self.detector.conf_threshold = conf

    def set_counting_line(self, pt1: Optional[Tuple[float, float]], pt2: Optional[Tuple[float, float]]):
        """Sets virtual tripwire coordinates for IN/OUT counting (normalized 0.0-1.0 or pixels)."""
        if pt1 is None or pt2 is None:
            self.raw_counting_line = None
            self.counter.counting_line = None
        else:
            self.raw_counting_line = (pt1, pt2)

    def set_roi_polygon(self, points: Optional[List[Tuple[float, float]]]):
        """Sets monitored region of interest polygon points (normalized 0.0-1.0 or pixels)."""
        if not points:
            self.raw_roi_polygon = None
            self.counter.roi_polygon = None
        else:
            self.raw_roi_polygon = points

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single image/video frame through detection, tracking, counting, and rendering.
        """
        start_t = time.time()
        h, w = frame.shape[:2]

        # Resolution-aware tripwire mapping (supports normalized 0.0-1.0 or pixel coordinates)
        if self.raw_counting_line:
            p1, p2 = self.raw_counting_line
            if p1[0] <= 1.05 and p1[1] <= 1.05 and p2[0] <= 1.05 and p2[1] <= 1.05:
                self.counter.counting_line = (
                    (float(p1[0]) * w, float(p1[1]) * h),
                    (float(p2[0]) * w, float(p2[1]) * h)
                )
            else:
                self.counter.counting_line = (p1, p2)
        else:
            self.counter.counting_line = None

        # Resolution-aware ROI mapping (supports normalized 0.0-1.0 or pixel coordinates)
        if self.raw_roi_polygon and len(self.raw_roi_polygon) >= 3:
            if all(p[0] <= 1.05 and p[1] <= 1.05 for p in self.raw_roi_polygon):
                self.counter.roi_polygon = [
                    (float(p[0]) * w, float(p[1]) * h) for p in self.raw_roi_polygon
                ]
            else:
                self.counter.roi_polygon = self.raw_roi_polygon
        else:
            self.counter.roi_polygon = None

        # 1. Detect Heads
        raw_detections = self.detector.detect(frame)

        # 2. Track Heads
        tracked_heads = self.tracker.update(raw_detections)

        # 3. Update Counts
        counts = self.counter.update(tracked_heads)

        # Compute FPS and latency
        end_t = time.time()
        curr_latency = (end_t - start_t) * 1000.0
        self.latency_ms = 0.8 * self.latency_ms + 0.2 * curr_latency if self.latency_ms > 0 else curr_latency

        dt = end_t - self._last_time
        self._last_time = end_t
        curr_fps = 1.0 / dt if dt > 0 else 0.0
        self.fps = 0.85 * self.fps + 0.15 * curr_fps if self.fps > 0 else curr_fps

        counts["fps"] = round(self.fps, 1)
        counts["latency_ms"] = round(self.latency_ms, 1)
        counts["frame_width"] = w
        counts["frame_height"] = h
        self.current_metrics = counts

        # 4. Draw Visual Overlays
        annotated_frame = self._render_annotations(frame.copy(), tracked_heads, counts)

        return annotated_frame, counts

    def _render_annotations(
        self,
        frame: np.ndarray,
        tracked_heads: List[Dict[str, Any]],
        counts: Dict[str, Any]
    ) -> np.ndarray:
        """Renders bounding boxes, tracks, lines, and HUD on the frame."""
        h, w = frame.shape[:2]

        # Draw ROI Polygon if active
        if self.counter.roi_polygon and len(self.counter.roi_polygon) >= 3:
            pts = np.array(self.counter.roi_polygon, np.int32).reshape((-1, 1, 2))
            roi_overlay = frame.copy()
            cv2.fillPoly(roi_overlay, [pts], COLOR_ROI_ZONE)
            cv2.addWeighted(roi_overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [pts], True, COLOR_ROI_ZONE, 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "MONITORED ROI ZONE",
                (pts[0][0][0], max(20, pts[0][0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA
            )

        # Draw Tripwire Line if active
        if self.counter.counting_line:
            p1, p2 = self.counter.counting_line
            p1_i = (int(p1[0]), int(p1[1]))
            p2_i = (int(p2[0]), int(p2[1]))
            cv2.line(frame, p1_i, p2_i, COLOR_TRIPWIRE_LINE, 2, cv2.LINE_AA)
            cv2.circle(frame, p1_i, 4, (250, 250, 250), -1)
            cv2.circle(frame, p2_i, 4, (250, 250, 250), -1)
            cv2.putText(
                frame,
                "TRIPWIRE COUNT LINE",
                (p1_i[0] + 10, p1_i[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA
            )

        # Draw Head Detections & Trajectories
        for head in tracked_heads:
            x1, y1, x2, y2 = map(int, head["bbox"])
            cx, cy = map(int, head["center"])
            track_id = head["track_id"]
            conf = head["confidence"]
            history = head.get("history", [])

            # Movement Trajectory Trail (Monochrome Fading White)
            if self.show_trails and len(history) > 1:
                for j in range(1, len(history)):
                    pt_a = (int(history[j - 1][0]), int(history[j - 1][1]))
                    pt_b = (int(history[j][0]), int(history[j][1]))
                    alpha = float(j) / len(history)
                    g_val = int(220 * alpha)
                    color = (g_val, g_val, g_val)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)

            # Head Bounding Box
            if self.show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HEAD_BOX, 2, cv2.LINE_AA)

                # Head Center Crosshair / Point
                if self.show_crosshairs:
                    cv2.circle(frame, (cx, cy), 3, COLOR_HEAD_CENTER, -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 6, COLOR_CROSSHAIR_RING, 1, cv2.LINE_AA)

                # Head Tag Pill
                label = f"Head #{track_id} {int(conf * 100)}%"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tag_y = max(y1 - 6, th + 4)
                cv2.rectangle(
                    frame,
                    (x1, tag_y - th - 4),
                    (x1 + tw + 6, tag_y + baseline - 2),
                    (10, 10, 10),
                    -1
                )
                cv2.rectangle(
                    frame,
                    (x1, tag_y - th - 4),
                    (x1 + tw + 6, tag_y + baseline - 2),
                    COLOR_HEAD_BOX,
                    1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, tag_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA
                )

        # Top CCTV HUD Overlay Banner
        if self.show_hud:
            hud_height = 55
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, hud_height), COLOR_HUD_BACKGROUND, -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.line(frame, (0, hud_height), (w, hud_height), (90, 90, 90), 1)

            # Live Head Count Badge (Monochrome Pale White)
            head_cnt_str = f"LIVE HEAD COUNT: {counts['current_head_count']}"
            cv2.putText(frame, head_cnt_str, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (250, 250, 250), 2, cv2.LINE_AA)

            # Peak Count
            peak_str = f"PEAK: {counts['peak_head_count']}"
            cv2.putText(frame, peak_str, (320, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # In / Out Crossings (if any)
            if counts.get("in_count", 0) > 0 or counts.get("out_count", 0) > 0:
                flow_str = f"IN: {counts['in_count']} | OUT: {counts['out_count']}"
                cv2.putText(frame, flow_str, (460, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            # CCTV Live Recording Status
            status_str = "REC [1080P HD]"
            (pw, _), _ = cv2.getTextSize(status_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.circle(frame, (w - pw - 28, 30), 4, (240, 240, 240), -1)
            cv2.putText(frame, status_str, (w - pw - 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        return frame

    def _create_error_frame(self, title: str, subtitle: str = "") -> np.ndarray:
        """Generates a fallback informational HUD frame if video source fails or is missing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)  # Pure dark monochrome
        cv2.rectangle(frame, (8, 8), (632, 472), (60, 60, 60), 1)
        cv2.putText(frame, "CCTV SURVEILLANCE FEED - STANDBY", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(frame, (8, 65), (632, 65), (50, 50, 50), 1)
        cv2.putText(frame, title, (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(frame, subtitle, (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(frame, "Select a channel on the right panel or upload video.", (30, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
        return frame

    def stream_generator(self) -> Generator[bytes, None, None]:
        """
        Yields MJPEG encoded video stream for live browser streaming.
        """
        self.is_running = True
        logger.info("Starting video stream generator loop...")

        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            if self.cap is None or not self.cap.isOpened():
                if not self.open_source(self.source):
                    error_frame = self._create_error_frame(
                        "VIDEO INPUT SOURCE NOT FOUND",
                        f"Target source '{os.path.basename(str(self.source))}' is unavailable"
                    )
                    ret, buffer = cv2.imencode(".jpg", error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                        )
                    time.sleep(1.0)
                    continue

            success, frame = self.cap.read()
            if not success:
                # If looped video file, seek back to beginning
                if isinstance(self.source, str) and os.path.exists(self.source):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                    continue
                else:
                    logger.warning("Frame read failed or stream closed.")
                    error_frame = self._create_error_frame(
                        "STREAM DISCONNECTED OR ENDED",
                        "Waiting for video input..."
                    )
                    ret, buffer = cv2.imencode(".jpg", error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                        )
                    time.sleep(1.0)
                    continue

            # Process frame
            annotated_frame, _ = self.process_frame(frame)

            # Encode as JPEG
            ret, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            # Gentle yield to prevent 100% CPU lockup on cloud free-tiers
            time.sleep(0.02)

    def get_current_frame_jpeg(self) -> bytes:
        """Fetches, processes, and encodes a single frame on demand (Cloudflare & Safari safe)."""
        if self.cap is None or not self.cap.isOpened():
            if not self.open_source(self.source):
                err = self._create_error_frame("VIDEO SOURCE STANDBY", "Select a CCTV channel on the right panel")
                ret, buf = cv2.imencode(".jpg", err, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                return buf.tobytes() if ret else b""

        success, frame = self.cap.read()
        if not success:
            if isinstance(self.source, str) and os.path.exists(self.source):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
            if not success:
                err = self._create_error_frame("STREAM ENDED", "Reconnecting video channel...")
                ret, buf = cv2.imencode(".jpg", err, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                return buf.tobytes() if ret else b""

        annotated_frame, _ = self.process_frame(frame)
        ret, buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ret else b""

    def process_video_file(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Processes a full CCTV video file offline, rendering annotated video to output_path.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

        self.tracker.reset()
        self.counter.reset()

        frame_idx = 0
        counts_summary = []

        logger.info(f"Processing video {input_path} ({total_frames} frames) -> {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, counts = self.process_frame(frame)
            out.write(annotated_frame)
            counts_summary.append(counts)

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

        cap.release()
        out.release()

        logger.info(f"Finished processing {frame_idx} frames. Output saved to {output_path}")
        return {
            "total_frames": frame_idx,
            "peak_head_count": self.counter.peak_count,
            "total_unique_heads": len(self.counter.seen_track_ids),
            "in_count": self.counter.in_count,
            "out_count": self.counter.out_count,
            "output_path": output_path
        }

    def close(self):
        """Releases resources."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
