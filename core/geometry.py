"""Authoritative Camera Geometry Implementation for FSOC-PAT-26169.

This module provides coordinate transformations, pinhole projection equations,
pixel-to-angle conversions, line-of-sight calculations, and field-of-view checks.

All functions follow docs/COORDINATE_CONVENTION.md exactly.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Camera geometry configuration parameters."""

    width: int = 640
    height: int = 480
    fov_x_deg: float = 4.0
    fov_y_deg: float = 3.0

    def __post_init__(self) -> None:
        """Validate camera geometry configuration upon initialization."""
        if self.width <= 0:
            raise ValueError(f"Image width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Image height must be positive, got {self.height}")
        if self.fov_x_deg <= 0 or self.fov_x_deg >= 180:
            raise ValueError(
                f"Horizontal FOV must be in range (0, 180) degrees, got {self.fov_x_deg}"
            )
        if self.fov_y_deg <= 0 or self.fov_y_deg >= 180:
            raise ValueError(
                f"Vertical FOV must be in range (0, 180) degrees, got {self.fov_y_deg}"
            )

    @property
    def fov_x_rad(self) -> float:
        """Horizontal field of view in radians."""
        return math.radians(self.fov_x_deg)

    @property
    def fov_y_rad(self) -> float:
        """Vertical field of view in radians."""
        return math.radians(self.fov_y_deg)

    @property
    def fx(self) -> float:
        """Focal length in horizontal pixel units."""
        return self.width / (2.0 * math.tan(self.fov_x_rad / 2.0))

    @property
    def fy(self) -> float:
        """Focal length in vertical pixel units."""
        return self.height / (2.0 * math.tan(self.fov_y_rad / 2.0))

    @property
    def cx(self) -> float:
        """Horizontal principal point (image center x)."""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """Vertical principal point (image center y)."""
        return self.height / 2.0


def get_image_center(config: CameraConfig) -> Tuple[float, float]:
    """Calculate the principal/image center coordinates (cx, cy)."""
    return config.cx, config.cy


def world_to_camera_relative(
    target_world: np.ndarray, camera_world: np.ndarray
) -> np.ndarray:
    """Calculate target position relative to camera in world frame (Prelative = Pw - Pc)."""
    target = np.asarray(target_world, dtype=float)
    camera = np.asarray(camera_world, dtype=float)
    if target.shape != (3,) or camera.shape != (3,):
        raise ValueError("Positions must be 3D vectors of shape (3,)")
    return target - camera


def rotate_world_to_camera(
    p_relative: np.ndarray, pan_rad: float, tilt_rad: float
) -> np.ndarray:
    """Transform relative world position vector into camera-frame coordinates [Xc, Yc, Zc].

    Specification (matching docs/COORDINATE_CONVENTION.md):
    - Input Frame: Relative world vector Prelative = P_world - P_camera [X_rel, Y_rel, Z_rel]
                   where world frame is +X right, +Y up, +Z forward.
    - Output Frame: Camera vector P_camera = [Xc, Yc, Zc]
                    where camera frame is +X right, +Y up, +Z forward.
    - Transformation Type: Passive world-to-camera coordinate transformation.
    - Angle Units: Radians (pan_rad, tilt_rad).
    - Pan Axis & Physical Direction:
        * Pan axis = world vertical Y-axis.
        * Positive pan rotates the camera to the RIGHT.
        * Under world-to-camera transformation, a stationary forward target moves toward
          negative camera X (-Xc) and therefore appears LEFT in the projected image.
        * Matrix: R_pan(phi) = [[cos(phi), 0, -sin(phi)],
                                [0,        1,  0       ],
                                [sin(phi), 0,  cos(phi)]]
    - Tilt Axis & Physical Direction:
        * Tilt axis = intermediate horizontal X-axis after pan.
        * Positive tilt rotates the camera DOWN.
        * Under world-to-camera transformation, a stationary forward target moves toward
          positive camera Y (+Yc) and therefore appears UP in the projected image.
        * Matrix: R_tilt(theta) = [[1, 0,          0         ],
                                  [0, cos(theta), sin(theta)],
                                  [0, -sin(theta),cos(theta)]]
    - Composition Order: Pan is applied first, followed by tilt:
                         P_camera = R_tilt(theta) @ R_pan(phi) @ P_relative
    """
    rel = np.asarray(p_relative, dtype=float)
    if rel.shape != (3,):
        raise ValueError("Relative position must be a 3D vector of shape (3,)")

    # Rotate by pan around Y axis
    c_p, s_p = math.cos(pan_rad), math.sin(pan_rad)
    r_pan = np.array([[c_p, 0.0, -s_p], [0.0, 1.0, 0.0], [s_p, 0.0, c_p]])

    # Rotate by tilt around X axis
    c_t, s_t = math.cos(tilt_rad), math.sin(tilt_rad)
    r_tilt = np.array([[1.0, 0.0, 0.0], [0.0, c_t, s_t], [0.0, -s_t, c_t]])

    # Total composition order: R_tilt @ R_pan
    r_cam = r_tilt @ r_pan
    return r_cam @ rel


def project_camera_to_pixel(
    p_camera: np.ndarray, config: CameraConfig
) -> Tuple[float, float]:
    """Project a point in camera coordinates [Xc, Yc, Zc] onto the FPA image plane.

    Equations:
        u = fx * (Xc / Zc) + cx
        v = cy - fy * (Yc / Zc)
    """
    p_cam = np.asarray(p_camera, dtype=float)
    if p_cam.shape != (3,):
        raise ValueError("Camera position must be a 3D vector of shape (3,)")

    xc, yc, zc = p_cam[0], p_cam[1], p_cam[2]
    if zc <= 0:
        raise ValueError(
            f"Point must be strictly in front of camera (Zc > 0), got Zc={zc}"
        )

    u = config.fx * (xc / zc) + config.cx
    v = config.cy - config.fy * (yc / zc)
    return float(u), float(v)


def pixel_to_angular_offset(
    u: float, v: float, config: CameraConfig
) -> Tuple[float, float]:
    """Convert pixel position (u, v) to horizontal and vertical angular offsets in radians.

    Equations:
        dx = u - cx
        dy = cy - v
        angle_x = atan(dx / fx)
        angle_y = atan(dy / fy)
    """
    dx = u - config.cx
    dy = config.cy - v
    angle_x = math.atan(dx / config.fx)
    angle_y = math.atan(dy / config.fy)
    return angle_x, angle_y


def angular_offset_to_pixel(
    angle_x_rad: float, angle_y_rad: float, config: CameraConfig
) -> Tuple[float, float]:
    """Convert angular offsets in radians to pixel coordinates (u, v).

    Equations:
        dx = fx * tan(angle_x)
        dy = fy * tan(angle_y)
        u = cx + dx
        v = cy - dy
    """
    dx = config.fx * math.tan(angle_x_rad)
    dy = config.fy * math.tan(angle_y_rad)
    u = config.cx + dx
    v = config.cy - dy
    return float(u), float(v)


def is_visible_in_fov(p_camera: np.ndarray, config: CameraConfig) -> bool:
    """Check if target in camera coordinates [Xc, Yc, Zc] lies within camera FOV."""
    p_cam = np.asarray(p_camera, dtype=float)
    if p_cam.shape != (3,):
        raise ValueError("Camera position must be a 3D vector of shape (3,)")

    xc, yc, zc = p_cam[0], p_cam[1], p_cam[2]
    if zc <= 0:
        return False

    ang_x = abs(math.atan2(xc, zc))
    ang_y = abs(math.atan2(yc, zc))

    return (ang_x <= config.fov_x_rad / 2.0) and (ang_y <= config.fov_y_rad / 2.0)


def is_inside_fpa(u: float, v: float, config: CameraConfig) -> bool:
    """Check if pixel coordinates (u, v) lie within finite physical FPA bounds."""
    return (0.0 <= u < float(config.width)) and (0.0 <= v < float(config.height))


def get_line_of_sight(p_camera: np.ndarray) -> np.ndarray:
    """Calculate normalized camera-frame line-of-sight unit vector L = Pc / ||Pc||."""
    p_cam = np.asarray(p_camera, dtype=float)
    if p_cam.shape != (3,):
        raise ValueError("Camera position must be a 3D vector of shape (3,)")

    norm = np.linalg.norm(p_cam)
    if norm == 0:
        raise ValueError("Cannot calculate line-of-sight for zero vector")
    return p_cam / norm