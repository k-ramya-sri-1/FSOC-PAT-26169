"""Unit tests for core/geometry.py following docs/TEST_STRATEGY.md."""

import math
import numpy as np
import pytest

from core.geometry import (
    CameraConfig,
    angular_offset_to_pixel,
    get_image_center,
    get_line_of_sight,
    is_inside_fpa,
    is_visible_in_fov,
    pixel_to_angular_offset,
    project_camera_to_pixel,
    rotate_world_to_camera,
    world_to_camera_relative,
)


def test_image_center():
    cfg = CameraConfig(width=640, height=480, fov_x_deg=4.0, fov_y_deg=3.0)
    cx, cy = get_image_center(cfg)
    assert cx == 320.0
    assert cy == 240.0


def test_center_optical_axis_projection():
    cfg = CameraConfig()
    p_cam = np.array([0.0, 0.0, 100.0])
    u, v = project_camera_to_pixel(p_cam, cfg)
    assert u == pytest.approx(320.0)
    assert v == pytest.approx(240.0)

    ang_x, ang_y = pixel_to_angular_offset(u, v, cfg)
    assert ang_x == pytest.approx(0.0)
    assert ang_y == pytest.approx(0.0)


def test_directional_sanity_checks():
    cfg = CameraConfig()
    p_center = np.array([0.0, 0.0, 100.0])
    u_c, v_c = project_camera_to_pixel(p_center, cfg)

    # Right -> u increases
    u_r, _ = project_camera_to_pixel(np.array([1.0, 0.0, 100.0]), cfg)
    assert u_r > u_c

    # Left -> u decreases
    u_l, _ = project_camera_to_pixel(np.array([-1.0, 0.0, 100.0]), cfg)
    assert u_l < u_c

    # Upward -> v decreases
    _, v_up = project_camera_to_pixel(np.array([0.0, 1.0, 100.0]), cfg)
    assert v_up < v_c

    # Downward -> v increases
    _, v_dn = project_camera_to_pixel(np.array([0.0, -1.0, 100.0]), cfg)
    assert v_dn > v_c


def test_pixel_angle_round_trip():
    cfg = CameraConfig()
    u_orig, v_orig = 450.0, 120.0

    ang_x, ang_y = pixel_to_angular_offset(u_orig, v_orig, cfg)
    u_reconstructed, v_reconstructed = angular_offset_to_pixel(ang_x, ang_y, cfg)

    assert u_reconstructed == pytest.approx(u_orig, abs=1e-7)
    assert v_reconstructed == pytest.approx(v_orig, abs=1e-7)


def test_angle_pixel_round_trip():
    cfg = CameraConfig()
    ang_x_orig, ang_y_orig = math.radians(0.8), math.radians(-0.5)

    u, v = angular_offset_to_pixel(ang_x_orig, ang_y_orig, cfg)
    ang_x_rec, ang_y_rec = pixel_to_angular_offset(u, v, cfg)

    assert ang_x_rec == pytest.approx(ang_x_orig, abs=1e-7)
    assert ang_y_rec == pytest.approx(ang_y_orig, abs=1e-7)


def test_corner_behavior_and_fpa_bounds():
    cfg = CameraConfig(width=640, height=480, fov_x_deg=4.0, fov_y_deg=3.0)

    # Top-left corner
    assert is_inside_fpa(0.0, 0.0, cfg) is True
    # Bottom-right corner
    assert is_inside_fpa(639.9, 479.9, cfg) is True
    # Outside FPA bounds
    assert is_inside_fpa(640.0, 480.0, cfg) is False
    assert is_inside_fpa(-0.1, 100.0, cfg) is False


def test_visibility_and_fov():
    cfg = CameraConfig(width=640, height=480, fov_x_deg=4.0, fov_y_deg=3.0)

    # Directly in center (visible)
    p_in = np.array([0.0, 0.0, 100.0])
    assert is_visible_in_fov(p_in, cfg) is True

    # Behind camera (Zc <= 0)
    p_behind = np.array([0.0, 0.0, -10.0])
    assert is_visible_in_fov(p_behind, cfg) is False

    # Out of horizontal FOV (> 2 deg off axis)
    x_out = 100.0 * math.tan(math.radians(2.5))
    p_out_h = np.array([x_out, 0.0, 100.0])
    assert is_visible_in_fov(p_out_h, cfg) is False


def test_non_default_resolution_and_fov():
    cfg = CameraConfig(width=1920, height=1080, fov_x_deg=10.0, fov_y_deg=5.0)
    cx, cy = get_image_center(cfg)
    assert cx == 960.0
    assert cy == 540.0

    p_cam = np.array([0.0, 0.0, 500.0])
    u, v = project_camera_to_pixel(p_cam, cfg)
    assert u == pytest.approx(960.0)
    assert v == pytest.approx(540.0)


def test_invalid_camera_config_rejection():
    with pytest.raises(ValueError):
        CameraConfig(width=0)

    with pytest.raises(ValueError):
        CameraConfig(height=-10)

    with pytest.raises(ValueError):
        CameraConfig(fov_x_deg=0.0)

    with pytest.raises(ValueError):
        CameraConfig(fov_y_deg=180.0)


def test_invalid_projection_behind_camera():
    cfg = CameraConfig()
    p_cam_behind = np.array([0.0, 0.0, -1.0])
    with pytest.raises(ValueError):
        project_camera_to_pixel(p_cam_behind, cfg)


def test_line_of_sight():
    p_cam = np.array([3.0, 4.0, 0.0])
    los = get_line_of_sight(p_cam)
    np.testing.assert_allclose(los, [0.6, 0.8, 0.0])


# ==================================================
# ROTATION & WORLD-TO-CAMERA PIPELINE UNIT TESTS
# ==================================================

def test_world_to_camera_relative_subtraction():
    target_w = np.array([10.0, 20.0, 100.0])
    camera_w = np.array([2.0, 5.0, 10.0])
    p_rel = world_to_camera_relative(target_w, camera_w)
    np.testing.assert_allclose(p_rel, [8.0, 15.0, 90.0])


def test_rotation_zero():
    p_rel = np.array([0.0, 0.0, 100.0])
    p_cam = rotate_world_to_camera(p_rel, pan_rad=0.0, tilt_rad=0.0)
    # Zero rotation must leave vector unchanged
    np.testing.assert_allclose(p_cam, [0.0, 0.0, 100.0], atol=1e-12)


def test_rotation_positive_pan():
    # +pan rotates camera RIGHT.
    # A forward target [0, 0, 100] with pan=+90 deg must move toward -camera-X [-100, 0, 0].
    p_rel = np.array([0.0, 0.0, 100.0])
    pan = math.radians(90.0)
    p_cam = rotate_world_to_camera(p_rel, pan_rad=pan, tilt_rad=0.0)

    expected = np.array([-100.0, 0.0, 0.0])
    np.testing.assert_allclose(p_cam, expected, atol=1e-12)


def test_rotation_positive_tilt():
    # +tilt rotates camera DOWN.
    # A forward target [0, 0, 100] with tilt=+90 deg must move toward +camera-Y [0, +100, 0].
    p_rel = np.array([0.0, 0.0, 100.0])
    tilt = math.radians(90.0)
    p_cam = rotate_world_to_camera(p_rel, pan_rad=0.0, tilt_rad=tilt)

    expected = np.array([0.0, 100.0, 0.0])
    np.testing.assert_allclose(p_cam, expected, atol=1e-12)


def test_rotation_combined_pan_tilt():
    p_rel = np.array([10.0, 20.0, 100.0])
    pan = math.radians(30.0)
    tilt = math.radians(45.0)

    c_p, s_p = math.cos(pan), math.sin(pan)
    c_t, s_t = math.cos(tilt), math.sin(tilt)

    # Analytical verification of R_tilt @ R_pan @ p_rel:
    # R_pan @ [10, 20, 100] = [10*c_p - 100*s_p, 20, 10*s_p + 100*c_p]
    # R_tilt @ [X1, Y1, Z1] = [X1, Y1*c_t + Z1*s_t, -Y1*s_t + Z1*c_t]
    x1 = 10.0 * c_p - 100.0 * s_p
    y1 = 20.0
    z1 = 10.0 * s_p + 100.0 * c_p

    expected = np.array([
        x1,
        y1 * c_t + z1 * s_t,
        -y1 * s_t + z1 * c_t
    ])

    p_cam = rotate_world_to_camera(p_rel, pan_rad=pan, tilt_rad=tilt)
    np.testing.assert_allclose(p_cam, expected, atol=1e-12)


def test_world_to_camera_full_pipeline_projection():
    cfg = CameraConfig()
    target_w = np.array([0.0, 0.0, 100.0])
    camera_w = np.array([0.0, 0.0, 0.0])

    p_rel = world_to_camera_relative(target_w, camera_w)
    
    # Positive pan (+1 deg) moves forward target to -camera-X -> image u < 320 (left)
    p_cam_pan = rotate_world_to_camera(p_rel, pan_rad=math.radians(1.0), tilt_rad=0.0)
    u_pan, _ = project_camera_to_pixel(p_cam_pan, cfg)
    assert u_pan < 320.0

    # Positive tilt (+1 deg) moves forward target to +camera-Y -> image v < 240 (up)
    p_cam_tilt = rotate_world_to_camera(p_rel, pan_rad=0.0, tilt_rad=math.radians(1.0))
    _, v_tilt = project_camera_to_pixel(p_cam_tilt, cfg)
    assert v_tilt < 240.0