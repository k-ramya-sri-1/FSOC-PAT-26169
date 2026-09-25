"""Authoritative Classical Computer Vision Beacon Detection for FSOC-PAT-26169.

This module processes a 2D monochrome image to locate bright beacon candidates using
classical image processing and intensity-weighted connected-component centroid estimation.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration parameters for classical computer vision beacon detection."""

    threshold: float = 100.0
    min_area: float = 2.0
    max_area: float = 500.0
    min_peak: float = 120.0

    def __post_init__(self) -> None:
        """Validate detection configuration parameters."""
        if self.threshold < 0.0 or self.threshold > 255.0:
            raise ValueError(f"threshold must be in [0, 255], got {self.threshold}")
        if self.min_area <= 0.0:
            raise ValueError(f"min_area must be positive, got {self.min_area}")
        if self.max_area <= self.min_area:
            raise ValueError(
                f"max_area ({self.max_area}) must be strictly greater than min_area ({self.min_area})"
            )
        if self.min_peak < 0.0 or self.min_peak > 255.0:
            raise ValueError(f"min_peak must be in [0, 255], got {self.min_peak}")


@dataclass(frozen=True)
class DetectionCandidate:
    """Detected candidate metadata extracted from image connected components."""

    centroid_x: float
    centroid_y: float
    area: float
    peak_intensity: float
    mean_intensity: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int


class BeaconDetector:
    """Classical Computer Vision detector locating optical beacon candidates in monochrome images."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config if config is not None else DetectionConfig()

    def detect(self, image: np.ndarray) -> List[DetectionCandidate]:
        """Detect beacon candidates from a 2D monochrome image.

        Args:
            image: 2D monochrome image array (e.g., float32 or uint8).

        Returns:
            List of DetectionCandidate objects ordered deterministically by:
            1. Descending peak intensity
            2. Descending area
            3. Ascending centroid_y
            4. Ascending centroid_x
        """
        self._validate_input_image(image)

        # Work on uint8 binary threshold image for connected components
        img_float = image.astype(np.float32)
        thresh_val = float(self.config.threshold)

        # Binarize image using threshold
        _, binary_img = cv2.threshold(
            img_float, thresh_val, 255, cv2.THRESH_BINARY
        )
        binary_uint8 = binary_img.astype(np.uint8)

        # Connected component analysis with 8-connectivity
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_uint8, connectivity=8
        )

        candidates: List[DetectionCandidate] = []

        # Iterate over all components (label 0 is background)
        for label_id in range(1, num_labels):
            area = float(stats[label_id, cv2.CC_STAT_AREA])

            # Area filtering
            if area < self.config.min_area or area > self.config.max_area:
                continue

            bx = int(stats[label_id, cv2.CC_STAT_LEFT])
            by = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            # Extract 2D component image bounding region and label mask
            component_image = img_float[by : by + bh, bx : bx + bw]
            comp_mask = (labels[by : by + bh, bx : bx + bw] == label_id)

            comp_pixels = component_image[comp_mask]

            if comp_pixels.size == 0:
                continue

            peak_intensity = float(np.max(comp_pixels))

            # Peak intensity filtering
            if peak_intensity < self.config.min_peak:
                continue

            mean_intensity = float(np.mean(comp_pixels))

            # Intensity-weighted 2D weights for subpixel centroid calculation
            weights = np.where(comp_mask, component_image, 0.0)
            total_weight = float(np.sum(weights))

            if total_weight > 0.0:
                grid_y, grid_x = np.ogrid[:bh, :bw]
                cx_local = float(np.sum(grid_x * weights) / total_weight)
                cy_local = float(np.sum(grid_y * weights) / total_weight)
                cx = float(bx) + cx_local
                cy = float(by) + cy_local
            else:
                # Fallback to geometric centroid from stats if weight is zero
                cx = float(centroids[label_id][0])
                cy = float(centroids[label_id][1])

            candidates.append(
                DetectionCandidate(
                    centroid_x=cx,
                    centroid_y=cy,
                    area=area,
                    peak_intensity=peak_intensity,
                    mean_intensity=mean_intensity,
                    bbox_x=bx,
                    bbox_y=by,
                    bbox_width=bw,
                    bbox_height=bh,
                )
            )

        # Deterministic ordering:
        # 1. Descending peak intensity
        # 2. Descending area
        # 3. Ascending centroid_y
        # 4. Ascending centroid_x
        candidates.sort(
            key=lambda c: (
                -c.peak_intensity,
                -c.area,
                c.centroid_y,
                c.centroid_x,
            )
        )

        return candidates

    @staticmethod
    def _validate_input_image(image: np.ndarray) -> None:
        """Validate input image array shape and format."""
        if image is None:
            raise ValueError("Input image cannot be None")
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Input image must be a NumPy array, got {type(image)}"
            )
        if image.ndim != 2:
            raise ValueError(
                f"Input image must be a 2D monochrome array, got shape {image.shape}"
            )
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Input image array cannot be empty")