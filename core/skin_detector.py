"""
Skin Color Detector
Skin segmentation is kept as a debug view and fallback when landmark
tracking is unavailable.
"""

from __future__ import annotations

import cv2
import numpy as np


class SkinColorDetector:
    """Adaptive skin segmentation using HSV and YCrCb thresholds."""

    def __init__(self):
        self.hsv_lower = np.array([0, 20, 50], dtype=np.uint8)
        self.hsv_upper = np.array([25, 255, 255], dtype=np.uint8)
        self.ycrcb_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_upper = np.array([255, 180, 135], dtype=np.uint8)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.calibrated = False

    def detect(self, frame: np.ndarray):
        """
        Detect skin pixels in a BGR frame.

        Returns:
            skin_mask  (H, W) uint8 binary mask
            skin_frame (H, W, 3) masked BGR image
        """
        if frame is None or frame.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8)

        denoised = cv2.GaussianBlur(frame, (5, 5), 0)
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_channel = clahe.apply(y_channel)
        ycrcb_equalized = cv2.merge((y_channel, cr_channel, cb_channel))

        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask_ycrcb = cv2.inRange(ycrcb_equalized, self.ycrcb_lower, self.ycrcb_upper)

        combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        combined = self._suppress_frame_edges(combined)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel_open)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel_close)
        combined = cv2.erode(combined, self.kernel_erode, iterations=1)
        combined = cv2.medianBlur(combined, 5)
        combined = self._keep_prominent_regions(combined)

        skin_frame = cv2.bitwise_and(frame, frame, mask=combined)
        return combined, skin_frame

    def extract_hand_region(self, skin_mask: np.ndarray) -> np.ndarray:
        """Find the largest plausible hand contour and return it as a mask."""
        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return np.zeros_like(skin_mask)

        best = self._select_hand_contour(contours, skin_mask.shape)
        if best is None:
            return np.zeros_like(skin_mask)

        hand_mask = np.zeros_like(skin_mask)
        cv2.drawContours(hand_mask, [best], -1, 255, thickness=cv2.FILLED)
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, self.kernel_close)
        return hand_mask

    def calibrate(self, frame: np.ndarray, region: dict):
        """Adapt the skin model from a user-selected BGR region."""
        x = int(region["x"])
        y = int(region["y"])
        w = int(region["w"])
        h = int(region["h"])

        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))

        sample = frame[y:y + h, x:x + w]
        if sample.size == 0:
            raise ValueError("Empty sample region")

        hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        ycrcb_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2YCrCb)

        h_mean, s_mean, v_mean = np.mean(hsv_sample.reshape(-1, 3), axis=0)
        _, cr_mean, cb_mean = np.mean(ycrcb_sample.reshape(-1, 3), axis=0)

        self.hsv_lower = np.array(
            [max(0, h_mean - 18), max(20, s_mean - 70), max(30, v_mean - 90)],
            dtype=np.uint8,
        )
        self.hsv_upper = np.array(
            [min(179, h_mean + 18), min(255, s_mean + 90), 255],
            dtype=np.uint8,
        )
        self.ycrcb_lower = np.array(
            [0, max(120, cr_mean - 22), max(70, cb_mean - 22)],
            dtype=np.uint8,
        )
        self.ycrcb_upper = np.array(
            [255, min(180, cr_mean + 22), min(145, cb_mean + 22)],
            dtype=np.uint8,
        )

        self.calibrated = True

    def get_model_params(self) -> dict:
        return {
            "hsv_lower": self.hsv_lower.tolist(),
            "hsv_upper": self.hsv_upper.tolist(),
            "ycrcb_lower": self.ycrcb_lower.tolist(),
            "ycrcb_upper": self.ycrcb_upper.tolist(),
            "calibrated": self.calibrated,
        }

    def update_params(self, params: dict):
        if "hsv_lower" in params:
            self.hsv_lower = np.array(params["hsv_lower"], dtype=np.uint8)
        if "hsv_upper" in params:
            self.hsv_upper = np.array(params["hsv_upper"], dtype=np.uint8)
        if "ycrcb_lower" in params:
            self.ycrcb_lower = np.array(params["ycrcb_lower"], dtype=np.uint8)
        if "ycrcb_upper" in params:
            self.ycrcb_upper = np.array(params["ycrcb_upper"], dtype=np.uint8)

    @staticmethod
    def _suppress_frame_edges(mask: np.ndarray, margin_ratio: float = 0.04) -> np.ndarray:
        h, w = mask.shape[:2]
        margin_x = max(8, int(w * margin_ratio))
        margin_y = max(8, int(h * margin_ratio))
        cleaned = mask.copy()
        cleaned[:, :margin_x] = 0
        cleaned[:, w - margin_x:] = 0
        cleaned[:margin_y, :] = 0
        cleaned[h - margin_y:, :] = 0
        return cleaned

    @staticmethod
    def _keep_prominent_regions(mask: np.ndarray, max_regions: int = 3) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask)

        filtered = np.zeros_like(mask)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]:
            if cv2.contourArea(contour) < 1200:
                continue
            cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
        return filtered

    def _select_hand_contour(
        self, contours: list[np.ndarray], frame_shape: tuple[int, int]
    ) -> np.ndarray | None:
        h, w = frame_shape[:2]
        frame_area = float(h * w)
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        best_contour = None
        best_score = float("-inf")

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2500 or area > frame_area * 0.45:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            if cw < 40 or ch < 40:
                continue

            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            extent = area / (cw * ch) if cw * ch else 0.0
            solidity = area / hull_area if hull_area > 0 else 0.0
            aspect_ratio = cw / ch if ch else 0.0

            if extent < 0.2 or extent > 0.9:
                continue
            if solidity < 0.35 or solidity > 1.01:
                continue
            if aspect_ratio < 0.35 or aspect_ratio > 1.9:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                cx, cy = x + cw / 2.0, y + ch / 2.0
            else:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]

            centroid = np.array([cx, cy], dtype=np.float32)
            distance = np.linalg.norm(centroid - center)
            max_distance = max(np.linalg.norm(center), 1.0)
            center_score = 1.0 - min(distance / max_distance, 1.0)
            fill_score = 1.0 - abs(extent - 0.55)
            solidity_score = 1.0 - abs(solidity - 0.72)
            complexity_score = min(perimeter / max(np.sqrt(area), 1.0), 12.0) / 12.0
            size_score = min(area / (frame_area * 0.1), 1.0)

            score = (
                center_score * 0.35
                + fill_score * 0.2
                + solidity_score * 0.2
                + complexity_score * 0.15
                + size_score * 0.1
            )

            touches_edge = x <= 4 or y <= 4 or x + cw >= w - 4 or y + ch >= h - 4
            if touches_edge:
                score -= 0.35

            if score > best_score:
                best_score = score
                best_contour = contour

        return best_contour
