"""
Beacon detection and candidate extraction for FSOC-PAT-26169.

This module provides spatial candidate detection algorithms that extract region-of-interest
candidates from sensor images without ground-truth dependency.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration parameters for BeaconDetector."""

    threshold: float = 128.0
    min_area: float = 4.0
    max_area: float = 1000.0
    max_candidates: int = 10

    def __post_init__(self) -> None:
        """Validate detection configuration values."""
        if not math.isfinite(self.threshold) or self.threshold < 0.0 or self.threshold > 255.0:
            raise ValueError(f"threshold must be in [0, 255], got {self.threshold}")
        if not math.isfinite(self.min_area) or self.min_area <= 0.0:
            raise ValueError(f"min_area must be finite and > 0, got {self.min_area}")
        if not math.isfinite(self.max_area) or self.max_area < self.min_area:
            raise ValueError(
                f"max_area ({self.max_area}) must be >= min_area ({self.min_area})"
            )
        if self.max_candidates <= 0:
            raise ValueError(f"max_candidates must be > 0, got {self.max_candidates}")


@dataclass(frozen=True)
class DetectionCandidate:
    """Immutable representation of a single detected ROI candidate."""

    candidate_id: int
    centroid: Tuple[float, float]
    area: float
    bounding_box: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    peak_intensity: float
    mean_intensity: float

    def __post_init__(self) -> None:
        """Validate candidate fields."""
        if self.candidate_id <= 0:
            raise ValueError(f"candidate_id must be > 0, got {self.candidate_id}")
        if not math.isfinite(self.centroid[0]) or not math.isfinite(self.centroid[1]):
            raise ValueError(f"centroid coordinates must be finite, got {self.centroid}")
        if not math.isfinite(self.area) or self.area <= 0.0:
            raise ValueError(f"area must be finite and > 0, got {self.area}")
        xmin, ymin, xmax, ymax = self.bounding_box
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"invalid bounding box dimensions: {self.bounding_box}")


class BeaconDetector:
    """Spatial ROI detector that extracts candidate target regions from monochrome frames."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config if config is not None else DetectionConfig()

    def detect(self, image: np.ndarray) -> List[DetectionCandidate]:
        """Detect candidate regions of interest in the input image.

        Args:
            image: 2D monochrome numpy array (uint8 or float scaled to 0-255).

        Returns:
            List of DetectionCandidate dataclasses sorted deterministically by location.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Input image must be a 2D numpy array")

        if image.dtype != np.uint8:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image

        thresh_val = int(round(self.config.threshold))
        _, binary_img = cv2.threshold(img_uint8, thresh_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        raw_extracted = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (self.config.min_area <= area <= self.config.max_area):
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])

            x, y, w, h = cv2.boundingRect(contour)
            xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)

            mask = np.zeros_like(img_uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            mean_val = float(cv2.mean(img_uint8, mask=mask)[0])
            _, max_val, _, _ = cv2.minMaxLoc(img_uint8, mask=mask)
            peak_val = float(max_val)

            raw_extracted.append(
                {
                    "centroid_sort_key": (round(cy, 4), round(cx, 4)),
                    "centroid": (cx, cy),
                    "area": area,
                    "bounding_box": (xmin, ymin, xmax, ymax),
                    "peak_intensity": peak_val,
                    "mean_intensity": mean_val,
                }
            )

        # Sort candidates deterministically by spatial location (y then x)
        raw_extracted.sort(key=lambda c: c["centroid_sort_key"])
        selected = raw_extracted[: self.config.max_candidates]

        candidates = []
        for idx, item in enumerate(selected, start=1):
            candidates.append(
                DetectionCandidate(
                    candidate_id=idx,
                    centroid=item["centroid"],
                    area=item["area"],
                    bounding_box=item["bounding_box"],
                    peak_intensity=item["peak_intensity"],
                    mean_intensity=item["mean_intensity"],
                )
            )

        return candidates