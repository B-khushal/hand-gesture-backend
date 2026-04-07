"""
Gesture Recognizer
Prefers MediaPipe hand landmarks when available and falls back to
OpenCV contour analysis when MediaPipe is unavailable or misses a frame.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import Counter, deque

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - depends on local Python version
    mp = None


LOGGER = logging.getLogger(__name__)

GESTURES = {
    "fist": {"name": "fist", "emoji": "✊", "action": "Stop / Select"},
    "one": {"name": "one", "emoji": "☝️", "action": "Point / Click"},
    "peace": {"name": "peace", "emoji": "✌️", "action": "Scroll / Navigate"},
    "three": {"name": "three", "emoji": "🤟", "action": "Volume +"},
    "four": {"name": "four", "emoji": "🖖", "action": "Volume -"},
    "open_palm": {"name": "open_palm", "emoji": "🖐️", "action": "Pause / Play"},
    "thumbs_up": {"name": "thumbs_up", "emoji": "👍", "action": "Confirm / Like"},
    "thumbs_down": {"name": "thumbs_down", "emoji": "👎", "action": "Reject / Dislike"},
    "ok": {"name": "ok", "emoji": "👌", "action": "OK / Confirm"},
    "none": {"name": "none", "emoji": "-", "action": "No hand detected"},
}

FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_PIPS = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}


class GestureRecognizer:
    """Robust gesture recognizer with optional MediaPipe landmark support."""

    def __init__(
        self,
        min_contour_area: int = 3500,
        defect_angle_threshold: int = 80,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.5,
        smoothing_window: int = 5,
    ):
        self.min_contour_area = min_contour_area
        self.defect_angle_threshold = defect_angle_threshold
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.gesture_history: deque[str] = deque(maxlen=smoothing_window)
        self.mp_hands = None
        self.mp_drawing = None
        self.hand_connections = []
        self.detector_name = "opencv"
        self.landmark_detection_enabled = False
        self.runtime_warning = ""

        if mp is not None:
            try:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.mp_drawing = mp.solutions.drawing_utils
                self.hand_connections = tuple(mp.solutions.hands.HAND_CONNECTIONS)
                self.detector_name = "mediapipe"
                self.landmark_detection_enabled = True
            except Exception as exc:  # pragma: no cover - runtime/platform specific
                self.runtime_warning = (
                    "MediaPipe is installed but could not initialize in this runtime. "
                    f"Falling back to contour detection. Details: {exc}"
                )
                LOGGER.warning("MediaPipe initialization failed: %s", exc)
        else:
            self.runtime_warning = (
                "MediaPipe is unavailable in this Python runtime. "
                f"Current Python is {sys.version_info.major}.{sys.version_info.minor}; "
                "MediaPipe support is typically available on Python 3.10-3.12."
            )
            LOGGER.warning(
                "MediaPipe is not installed. Falling back to contour-based recognition only."
            )

    def get_runtime_info(self) -> dict:
        return {
            "detector": self.detector_name,
            "landmark_detection_enabled": self.landmark_detection_enabled,
            "runtime_warning": self.runtime_warning,
        }

    def recognize(self, hand_mask: np.ndarray, original_frame: np.ndarray) -> dict:
        """Compatibility wrapper used by the existing tests/app."""
        return self.process_frame(original_frame, hand_mask)

    def process_frame(self, frame: np.ndarray, hand_mask: np.ndarray | None = None) -> dict:
        """Process a frame and return the best available gesture result."""
        if frame is None or frame.size == 0:
            return self._no_hand(reason="empty_frame")

        if self.mp_hands is not None:
            mp_result = self._recognize_with_mediapipe(frame)
            if mp_result["gesture"] != "none" or mp_result["confidence"] >= 0.35:
                mp_result["stable_gesture"] = self._stabilize(mp_result["gesture"])
                return mp_result

        fallback_mask = hand_mask
        if fallback_mask is None:
            fallback_mask = self._build_fallback_mask(frame)

        fallback_result = self._recognize_from_mask(fallback_mask, frame)
        fallback_result["stable_gesture"] = self._stabilize(fallback_result["gesture"])
        return fallback_result

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: dict,
        skin_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw landmarks, contours, bounding boxes, and debug information."""
        out = frame.copy()
        gesture_name = result.get("stable_gesture") or result.get("gesture", "none")
        action = result.get("action", "")
        fingers = result.get("finger_count", 0)
        conf = result.get("confidence", 0.0)
        bbox = result.get("bbox")
        detector = result.get("detector", self.detector_name)
        handedness = result.get("handedness", "unknown")
        reason = result.get("reason", "")

        if skin_mask is not None and skin_mask.size:
            mask_color = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
            mask_overlay = cv2.resize(mask_color, (180, 135), interpolation=cv2.INTER_AREA)
            out[12:147, out.shape[1] - 192:out.shape[1] - 12] = mask_overlay
            cv2.rectangle(
                out,
                (out.shape[1] - 192, 12),
                (out.shape[1] - 12, 147),
                (80, 160, 255),
                1,
            )
            cv2.putText(
                out,
                "Skin Mask",
                (out.shape[1] - 188, 162),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 220, 255),
                1,
            )

        if bbox and gesture_name != "none":
            x, y, w, h = bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 120), 2)

        landmarks = result.get("landmarks")
        if landmarks:
            for start, end in self.hand_connections:
                p1 = landmarks[start]
                p2 = landmarks[end]
                cv2.line(out, p1, p2, (255, 140, 0), 2)
            for point in landmarks:
                cv2.circle(out, point, 4, (255, 255, 255), -1)
                cv2.circle(out, point, 2, (0, 120, 255), -1)
        elif skin_mask is not None and skin_mask.size:
            contours, _ = cv2.findContours(
                skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(out, [largest], -1, (0, 200, 255), 2)
                hull = cv2.convexHull(largest)
                cv2.drawContours(out, [hull], -1, (255, 80, 0), 2)

        panel_h = 128
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (390, panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

        lines = [
            f"Gesture: {gesture_name.upper()}",
            f"Action: {action}",
            f"Fingers: {fingers}   Conf: {conf:.0%}",
            f"Detector: {detector}   Hand: {handedness}",
        ]
        if reason:
            lines.append(f"Debug: {reason}")

        y = 28
        for idx, text in enumerate(lines):
            size = 0.72 if idx == 0 else 0.55
            color = (0, 255, 120) if idx == 0 else (200, 220, 255)
            thickness = 2 if idx == 0 else 1
            cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
            y += 28

        return out

    def get_supported_gestures(self) -> list:
        return [
            {
                "key": key,
                "name": value["name"],
                "emoji": value["emoji"],
                "action": value["action"],
            }
            for key, value in GESTURES.items()
        ]

    def _recognize_with_mediapipe(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        if not results.multi_hand_landmarks:
            return self._no_hand(detector="mediapipe", reason="no_landmarks")

        landmarks_obj = results.multi_hand_landmarks[0]
        handedness = "unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label.lower()

        h, w = frame.shape[:2]
        landmarks = []
        normalized = []
        for lm in landmarks_obj.landmark:
            px = min(max(int(lm.x * w), 0), w - 1)
            py = min(max(int(lm.y * h), 0), h - 1)
            landmarks.append((px, py))
            normalized.append((lm.x, lm.y, lm.z))

        xs = [pt[0] for pt in landmarks]
        ys = [pt[1] for pt in landmarks]
        x_min = max(0, min(xs) - 20)
        y_min = max(0, min(ys) - 20)
        x_max = min(w - 1, max(xs) + 20)
        y_max = min(h - 1, max(ys) + 20)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        finger_states = self._finger_states_from_landmarks(landmarks, handedness)
        finger_count = sum(int(state) for state in finger_states.values())
        gesture_key, confidence, reason = self._classify_from_landmarks(
            landmarks, finger_states, handedness
        )
        gesture = GESTURES.get(gesture_key, GESTURES["none"])

        return {
            "gesture": gesture["name"],
            "emoji": gesture["emoji"],
            "action": gesture["action"],
            "finger_count": finger_count,
            "confidence": round(float(confidence), 2),
            "bbox": bbox,
            "area": int(bbox[2] * bbox[3]),
            "detector": "mediapipe",
            "handedness": handedness,
            "landmarks": landmarks,
            "landmarks_normalized": normalized,
            "finger_states": finger_states,
            "reason": reason,
        }

    def _recognize_from_mask(self, hand_mask: np.ndarray, frame: np.ndarray) -> dict:
        contours, _ = cv2.findContours(
            hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return self._no_hand(detector="opencv", reason="no_contours")

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_contour_area:
            return self._no_hand(detector="opencv", reason="contour_too_small")

        finger_count = self._count_fingers(contour)
        shape_info = self._shape_features(contour)
        gesture_key = self._classify_fallback(finger_count, shape_info, contour)
        gesture_name = self._gesture_name_from_fallback(gesture_key)
        gesture = GESTURES.get(gesture_name, GESTURES["none"])

        return {
            "gesture": gesture["name"],
            "emoji": gesture["emoji"],
            "action": gesture["action"],
            "finger_count": finger_count,
            "confidence": self._confidence(shape_info, area),
            "bbox": shape_info["bbox"],
            "area": int(area),
            "detector": "opencv",
            "handedness": "unknown",
            "landmarks": None,
            "reason": "fallback_contour_detection",
        }

    def _build_fallback_mask(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

        hsv_mask = cv2.inRange(
            hsv,
            np.array([0, 20, 55], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
        ycrcb_mask = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 180, 135], dtype=np.uint8),
        )
        combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined

    def _finger_states_from_landmarks(self, landmarks: list[tuple[int, int]], handedness: str) -> dict:
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        palm_width = max(abs(index_mcp[0] - pinky_mcp[0]), 1)

        thumb_tip = landmarks[FINGER_TIPS["thumb"]]
        thumb_ip = landmarks[FINGER_PIPS["thumb"]]
        if handedness == "right":
            thumb_extended = thumb_tip[0] < thumb_ip[0] - max(6, palm_width * 0.08)
        elif handedness == "left":
            thumb_extended = thumb_tip[0] > thumb_ip[0] + max(6, palm_width * 0.08)
        else:
            thumb_extended = abs(thumb_tip[0] - thumb_ip[0]) > max(10, palm_width * 0.12)

        states = {"thumb": bool(thumb_extended)}
        for finger in ("index", "middle", "ring", "pinky"):
            tip = landmarks[FINGER_TIPS[finger]]
            pip = landmarks[FINGER_PIPS[finger]]
            states[finger] = tip[1] < pip[1] - 10 and tip[1] < wrist[1]

        return states

    def _classify_from_landmarks(
        self,
        landmarks: list[tuple[int, int]],
        states: dict,
        handedness: str,
    ) -> tuple[str, float, str]:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]
        bbox = cv2.boundingRect(np.array(landmarks, dtype=np.int32))
        _, y, _, h = bbox
        palm_scale = max(h, 1)
        thumb_index_distance = math.dist(thumb_tip, index_tip) / palm_scale

        active = {name for name, enabled in states.items() if enabled}
        finger_count = len(active)

        if finger_count == 0:
            return "fist", 0.86, "all_fingers_folded"

        if thumb_index_distance < 0.12 and states["middle"] and states["ring"] and states["pinky"]:
            return "ok", 0.84, "thumb_index_pinched"

        if active == {"thumb"}:
            if thumb_tip[1] < wrist[1] - 20:
                return "thumbs_up", 0.9, "thumb_only_upward"
            if thumb_tip[1] > wrist[1] + 20:
                return "thumbs_down", 0.9, "thumb_only_downward"
            return "one", 0.55, f"thumb_only_{handedness}"

        if active == {"index"}:
            return "one", 0.88, "index_only"

        if active == {"index", "middle"}:
            return "peace", 0.9, "index_middle_extended"

        if active == {"index", "middle", "ring"} or active == {"thumb", "index", "middle"}:
            return "three", 0.8, "three_fingers_extended"

        if active == {"index", "middle", "ring", "pinky"}:
            return "four", 0.84, "four_fingers_extended"

        if finger_count >= 5 or active == {"thumb", "index", "middle", "ring", "pinky"}:
            return "open_palm", 0.92, "all_fingers_extended"

        fallback_name = self._gesture_name_from_fallback(finger_count)
        return fallback_name, 0.58, "partial_landmark_match"

    def _stabilize(self, gesture_name: str) -> str:
        if gesture_name != "none":
            self.gesture_history.append(gesture_name)
        elif self.gesture_history:
            self.gesture_history.append(self.gesture_history[-1])

        if not self.gesture_history:
            return gesture_name

        most_common, count = Counter(self.gesture_history).most_common(1)[0]
        if count >= max(2, len(self.gesture_history) // 2):
            return most_common
        return gesture_name

    def _classify_fallback(self, fingers: int, shape: dict, contour: np.ndarray) -> str | int:
        ar = shape["aspect_ratio"]
        circ = shape["circularity"]
        x, y, w, h = shape["bbox"]
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = shape["area"] / hull_area if hull_area > 0 else 0

        if fingers <= 1 and ar < 0.7 and solidity > 0.72:
            _, cy = shape["centroid"]
            return "thumbs_up" if cy < y + h // 2 else "thumbs_down"

        if circ > 0.62 and fingers <= 2:
            return "ok"

        return fingers

    def _gesture_name_from_fallback(self, gesture_key: str | int) -> str:
        if isinstance(gesture_key, str):
            return gesture_key if gesture_key in GESTURES else "none"
        if gesture_key == 0:
            return "fist"
        if gesture_key == 1:
            return "one"
        if gesture_key == 2:
            return "peace"
        if gesture_key == 3:
            return "three"
        if gesture_key == 4:
            return "four"
        if gesture_key >= 5:
            return "open_palm"
        return "none"

    def _count_fingers(self, contour: np.ndarray) -> int:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return 0

        try:
            defects = cv2.convexityDefects(contour, hull_indices)
        except cv2.error:
            return 0

        if defects is None:
            return 0

        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            angle = self._angle_between(start, far, end)
            depth = d / 256.0

            if angle < self.defect_angle_threshold and depth > 15:
                finger_count += 1

        return min(finger_count + 1, 5)

    def _shape_features(self, contour: np.ndarray) -> dict:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0
        aspect_ratio = w / h if h > 0 else 1
        extent = area / (w * h) if w * h > 0 else 0

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        return {
            "bbox": (x, y, w, h),
            "area": area,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "centroid": (cx, cy),
        }

    @staticmethod
    def _angle_between(p1, vertex, p2) -> float:
        v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
        v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag = math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag == 0:
            return 180.0
        cos_angle = max(-1.0, min(1.0, dot / mag))
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def _confidence(shape: dict, area: float) -> float:
        score = min(area / 30000, 1.0) * 0.5
        score += min(shape["circularity"] * 0.5, 0.3)
        score += min(shape["extent"] * 0.5, 0.2)
        return round(min(score, 0.99), 2)

    @staticmethod
    def _no_hand(detector: str = "none", reason: str = "no_hand") -> dict:
        gesture = GESTURES["none"]
        return {
            "gesture": gesture["name"],
            "stable_gesture": gesture["name"],
            "emoji": gesture["emoji"],
            "action": gesture["action"],
            "finger_count": 0,
            "confidence": 0.0,
            "bbox": None,
            "area": 0,
            "detector": detector,
            "handedness": "unknown",
            "landmarks": None,
            "reason": reason,
        }
