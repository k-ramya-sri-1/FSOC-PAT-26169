"""Authoritative Sensor / Image Formation Foundation for FSOC-PAT-26169.

This module converts the physical Scene state into a synthetic monochrome
focal-plane array (FPA) image containing the target optical beacon.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/COORDINATE_CONVENTION.md,
and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import numpy as np

from core.geometry import (
    CameraConfig,
    is_inside_fpa,
    is_visible_in_fov,
    project_camera_to_pixel,
    rotate_world_to_camera,
    world_to_camera_relative,
)
from core.scene import Scene


@dataclass(frozen=True)
class SensorConfig:
    """Sensor hardware and rendering configuration parameters."""

    camera_config: CameraConfig = field(default_factory=CameraConfig)
    beacon_diameter: float = 10.0
    peak_intensity: float = 255.0

    def __post_init__(self) -> None:
        """Validate sensor configuration parameters."""
        if self.beacon_diameter <= 0.0:
            raise ValueError(
                f"Beacon diameter must be strictly positive, got {self.beacon_diameter}"
            )
        if self.peak_intensity <= 0.0 or self.peak_intensity > 255.0:
            raise ValueError(
                f"Peak intensity must be in range (0, 255], got {self.peak_intensity}"
            )


@dataclass(frozen=True)
class SensorFrame:
    """Sensor image capture output frame."""

    image: np.ndarray
    target_visible: bool
    projected_pixel: Optional[Tuple[float, float]]

    def __post_init__(self) -> None:
        """Validate sensor frame data shapes and types."""
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 2:
            raise ValueError("Sensor image must be a 2D NumPy array")


class MonochromeSensor:
    """Monochrome focal-plane array sensor producing synthetic beacon imagery."""

    def __init__(self, config: Optional[SensorConfig] = None) -> None:
        self.config = config if config is not None else SensorConfig()

    def capture(self, scene: Scene) -> SensorFrame:
        """Capture a synthetic monochrome frame from current scene state."""
        width = self.config.camera_config.width
        height = self.config.camera_config.height

        # Initialize clean dark monochrome background (2D float32 array)
        image = np.zeros((height, width), dtype=np.float32)

        # 1. World-to-camera relative transformation via core.geometry
        p_rel = world_to_camera_relative(
            scene.target_state.position, scene.camera_state.position
        )

        # 2. Camera rotation transformation via core.geometry
        p_cam = rotate_world_to_camera(
            p_rel, scene.camera_state.pan_rad, scene.camera_state.tilt_rad
        )

        # 3. Geometric FOV & Forward visibility check via core.geometry
        if not is_visible_in_fov(p_cam, self.config.camera_config):
            return SensorFrame(
                image=image, target_visible=False, projected_pixel=None
            )

        # 4. Pinhole projection via core.geometry
        u_proj, v_proj = project_camera_to_pixel(
            p_cam, self.config.camera_config
        )

        # 5. Check if projected pixel falls within physical FPA boundaries
        if not is_inside_fpa(u_proj, v_proj, self.config.camera_config):
            return SensorFrame(
                image=image, target_visible=False, projected_pixel=None
            )

        # 6. Render synthetic beacon with subpixel precision
        self._render_beacon(image, u_proj, v_proj)

        return SensorFrame(
            image=image,
            target_visible=True,
            projected_pixel=(u_proj, v_proj),
        )

    def _render_beacon(
        self, image: np.ndarray, u_center: float, v_center: float
    ) -> None:
        """Render a Gaussian beacon spot with subpixel accuracy into FPA image array."""
        width = self.config.camera_config.width
        height = self.config.camera_config.height

        # Gaussian sigma based on specified diameter (diameter ~ 4 * sigma)
        sigma = self.config.beacon_diameter / 4.0
        radius_bound = int(math.ceil(self.config.beacon_diameter * 1.5))

        # Determine bounding grid over FPA
        u_min = max(0, int(math.floor(u_center - radius_bound)))
        u_max = min(width, int(math.ceil(u_center + radius_bound)) + 1)
        v_min = max(0, int(math.floor(v_center - radius_bound)))
        v_max = min(height, int(math.ceil(v_center + radius_bound)) + 1)

        if u_min >= u_max or v_min >= v_max:
            return

        # Meshgrid centered at fractional subpixel coordinate
        u_coords = np.arange(u_min, u_max, dtype=np.float32)
        v_coords = np.arange(v_min, v_max, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        # Exact subpixel distance squared
        dist_sq = (u_grid - u_center) ** 2 + (v_grid - v_center) ** 2

        # 2D Gaussian profile
        beacon_spot = self.config.peak_intensity * np.exp(-dist_sq / (2.0 * sigma**2))

        # Additively render onto image array and clip to [0, 255] range
        image[v_min:v_max, u_min:u_max] = np.clip(
            image[v_min:v_max, u_min:u_max] + beacon_spot, 0.0, 255.0
        )