"""
Real-time Hand Gesture Recognition Backend
Flask server with WebSocket support for live video stream processing.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from typing import Any

import cv2
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from core.gesture_recognizer import GestureRecognizer
from core.skin_detector import SkinColorDetector
from utils.helpers import decode_base64_frame, encode_frame_to_base64, resize_frame


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("gesture_backend")


def _parse_allowed_origins() -> list[str]:
    configured = os.environ.get("FRONTEND_URL", "").strip()
    if not configured:
        return ["*"]

    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if "*" not in origins:
        origins.append("*")
    return origins


def _resolve_socketio_async_mode() -> str:
    """
    Pick a Socket.IO async mode that is safe for the current runtime.

    Priority:
    1) Explicit env override via SOCKETIO_ASYNC_MODE
    2) Python 3.14+ defaults to threading (eventlet is currently incompatible)
    3) Try eventlet when importable
    4) Fallback to threading
    """
    forced_mode = os.environ.get("SOCKETIO_ASYNC_MODE", "").strip().lower()
    if forced_mode:
        return forced_mode

    if sys.version_info >= (3, 14):
        return "threading"

    try:
        import eventlet  # noqa: F401

        return "eventlet"
    except Exception:
        return "threading"


ALLOWED_ORIGINS = _parse_allowed_origins()
ASYNC_MODE = _resolve_socketio_async_mode()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "gesture-recognition-secret")
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=False)
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode=ASYNC_MODE,
    ping_timeout=30,
    ping_interval=20,
    max_http_buffer_size=8_000_000,
    logger=False,
    engineio_logger=False,
    always_connect=True,
)

skin_detector = SkinColorDetector()
gesture_recognizer = GestureRecognizer()

session_stats = {
    "frames_processed": 0,
    "gestures_detected": {},
    "start_time": time.time(),
    "last_processing_ms": 0.0,
    "last_error": None,
}


@app.route("/api/health", methods=["GET"])
def health_check():
    runtime_info = gesture_recognizer.get_runtime_info()
    return jsonify(
        {
            "status": "ok",
            "uptime": round(time.time() - session_stats["start_time"], 2),
            "frames_processed": session_stats["frames_processed"],
            "last_processing_ms": session_stats["last_processing_ms"],
            "detector": runtime_info["detector"],
            "landmark_detection_enabled": runtime_info["landmark_detection_enabled"],
            "runtime_warning": runtime_info["runtime_warning"],
            "socketio_async_mode": ASYNC_MODE,
            "python_version": platform.python_version(),
        }
    )


@app.errorhandler(404)
def not_found(_: Any):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_: Any):
    return jsonify({"error": "Internal server error"}), 500


@app.route("/api/gestures", methods=["GET"])
def get_gestures():
    return jsonify(
        {
            "gestures": gesture_recognizer.get_supported_gestures(),
            "description": "Supported hand gestures for recognition",
        }
    )


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(session_stats)


@app.route("/api/stats/reset", methods=["POST"])
def reset_stats():
    global session_stats
    session_stats = {
        "frames_processed": 0,
        "gestures_detected": {},
        "start_time": time.time(),
        "last_processing_ms": 0.0,
        "last_error": None,
    }
    return jsonify({"message": "Stats reset successfully"})


@app.route("/api/skin-model", methods=["GET"])
def get_skin_model():
    return jsonify(skin_detector.get_model_params())


@app.route("/api/skin-model", methods=["POST"])
def update_skin_model():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    skin_detector.update_params(data)
    return jsonify(
        {"message": "Skin model updated", "params": skin_detector.get_model_params()}
    )


@socketio.on("connect")
def handle_connect():
    runtime_info = gesture_recognizer.get_runtime_info()
    LOGGER.info("Client connected sid=%s origin=%s", request.sid, request.headers.get("Origin"))
    emit(
        "connected",
        {
            "message": "Connected to gesture recognition server",
            "detector": runtime_info["detector"],
            "landmark_detection_enabled": runtime_info["landmark_detection_enabled"],
            "runtime_warning": runtime_info["runtime_warning"],
        },
    )


@socketio.on("disconnect")
def handle_disconnect():
    LOGGER.info("Client disconnected sid=%s", request.sid)


@socketio.on("frame")
def handle_frame(data):
    """Receive a frame, process it, and return gesture + annotated frame."""
    started = time.perf_counter()

    try:
        frame_payload = data.get("frame") if isinstance(data, dict) else None
        if not frame_payload:
            emit("error", {"message": "Missing frame payload"})
            return

        frame = decode_base64_frame(frame_payload)
        if frame is None or frame.size == 0:
            emit("error", {"message": "Failed to decode frame"})
            return

        # Keep processing cost bounded on cloud free instances.
        frame = resize_frame(frame, max_width=960)

        skin_mask, _ = skin_detector.detect(frame)
        hand_region = skin_detector.extract_hand_region(skin_mask)
        gesture_result = gesture_recognizer.process_frame(frame, hand_region)
        gesture_result["stable_gesture"] = (
            gesture_result.get("stable_gesture") or gesture_result["gesture"]
        )

        session_stats["frames_processed"] += 1
        session_stats["last_error"] = None

        stable_name = gesture_result.get("stable_gesture", "none")
        if stable_name != "none":
            session_stats["gestures_detected"][stable_name] = (
                session_stats["gestures_detected"].get(stable_name, 0) + 1
            )

        annotated = gesture_recognizer.annotate_frame(frame, gesture_result, skin_mask)
        processing_ms = round((time.perf_counter() - started) * 1000, 2)
        session_stats["last_processing_ms"] = processing_ms

        emit(
            "result",
            {
                "gesture": gesture_result,
                "annotated_frame": encode_frame_to_base64(annotated, quality=78),
                "skin_mask": encode_frame_to_base64(
                    cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR),
                    quality=70,
                ),
                "frame_count": session_stats["frames_processed"],
                "processing_ms": processing_ms,
                "debug": {
                    "detector": gesture_result.get("detector"),
                    "reason": gesture_result.get("reason"),
                },
            },
        )
    except Exception as exc:  # pragma: no cover - defensive path
        session_stats["last_error"] = str(exc)
        LOGGER.exception("Frame processing failed sid=%s", request.sid)
        emit("error", {"message": "Frame processing failed"})


@socketio.on("calibrate")
def handle_calibrate(data):
    """Calibrate skin color model from a sample region."""
    try:
        frame_payload = data.get("frame") if isinstance(data, dict) else None
        region = data.get("region") if isinstance(data, dict) else None
        frame = decode_base64_frame(frame_payload) if frame_payload else None

        if frame is None or region is None:
            emit("error", {"message": "Invalid calibration data"})
            return

        skin_detector.calibrate(frame, region)
        emit(
            "calibrated",
            {
                "message": "Skin model calibrated successfully",
                "params": skin_detector.get_model_params(),
            },
        )
    except Exception as exc:
        session_stats["last_error"] = str(exc)
        LOGGER.exception("Calibration failed sid=%s", request.sid)
        emit("error", {"message": "Calibration failed"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    LOGGER.info(
        "Starting Hand Gesture Recognition Server host=0.0.0.0 port=%s origins=%s async_mode=%s",
        port,
        ALLOWED_ORIGINS,
        ASYNC_MODE,
    )
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
