"""
Utility helpers for frame encoding/decoding between browser and backend.
"""

import cv2
import numpy as np
import base64
import re


def decode_base64_frame(data_url: str) -> np.ndarray | None:
    """
    Decode a base64 data URL (from browser canvas/video) to a BGR numpy array.
    
    Accepts:
      - Full data URL: "data:image/jpeg;base64,/9j/..."
      - Raw base64 string
    """
    try:
        # Strip data URL prefix if present
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]

        # Decode base64
        img_bytes = base64.b64decode(data_url)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[decode_base64_frame] Error: {e}")
        return None


def encode_frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """
    Encode a BGR numpy array as a JPEG base64 data URL.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def resize_frame(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Resize frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter on frame."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (frame.shape[1] - 110, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (100, 255, 100), 2
    )
    return frame
