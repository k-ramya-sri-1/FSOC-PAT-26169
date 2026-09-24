"""Unit tests for core/sensor.py following docs/TEST_STRATEGY.md."""

import math
import numpy as np
import pytest

from core.geometry import CameraConfig, project_camera_to_pixel, rotate_world_to_camera, world_to_camera_relative
from core.scene import CameraState, Scene, TargetState
from core.sensor import MonochromeSensor, SensorConfig, SensorFrame


def test_sensor_config_validation():
    with pytest.raises(ValueError):
        SensorConfig(beacon_diameter=0.0)

    with pytest.raises(ValueError):
        SensorConfig(peak_intensity=-10.0)

    with pytest.raises(ValueError):
        SensorConfig(peak_intensity=300.0)


def test_sensor_default_capture_center():
    scene = Scene(
        target_state=TargetState(position=[0.0, 0.0, 100.0]),
        camera_state=CameraState(position=[0.0, 0.0, 0.0], pan_rad=0.0, tilt_rad=0.0),
    )
    sensor = MonochromeSensor()
    frame = sensor.capture(scene)

    assert isinstance(frame, SensorFrame)
    assert frame.target_visible is True
    assert frame.projected_pixel is not None
    assert frame.projected_pixel[0] == pytest.approx(320.0)
    assert frame.projected_pixel[1] == pytest.approx(240.0)

    # Peak intensity should be at image center (240, 320)
    assert frame.image[240, 320] == pytest.approx(255.0, abs=1e-2)


def test_sensor_target_movement_right_and_up():
    sensor = MonochromeSensor()

    # Move target right (+X = 1.0) -> projected u > 320
    scene_right = Scene(target_state=TargetState(position=[1.0, 0.0, 100.0]))
    frame_right = sensor.capture(scene_right)
    assert frame_right.projected_pixel[0] > 320.0

    # Move target up (+Y = 1.0) -> projected v < 240
    scene_up = Scene(target_state=TargetState(position=[0.0, 1.0, 100.0]))
    frame_up = sensor.capture(scene_up)
    assert frame_up.projected_pixel[1] < 240.0


def test_sensor_positive_pan_and_tilt():
    sensor = MonochromeSensor()
    p_target = np.array([0.0, 0.0, 100.0])

    # +pan rotates camera right -> target moves -X in camera frame -> u < 320
    pan_rad = math.radians(1.0)
    scene_pan = Scene(
        target_state=TargetState(position=p_target),
        camera_state=CameraState(pan_rad=pan_rad, tilt_rad=0.0),
    )
    frame_pan = sensor.capture(scene_pan)
    assert frame_pan.projected_pixel[0] < 320.0

    # Verify frame projection matches public core.geometry calculation
    p_rel = world_to_camera_relative(p_target, np.zeros(3))
    p_cam_expected = rotate_world_to_camera(p_rel, pan_rad, 0.0)
    u_exp, v_exp = project_camera_to_pixel(p_cam_expected, sensor.config.camera_config)
    assert frame_pan.projected_pixel[0] == pytest.approx(u_exp)
    assert frame_pan.projected_pixel[1] == pytest.approx(v_exp)

    # +tilt rotates camera down -> target moves +Y in camera frame -> v < 240
    tilt_rad = math.radians(1.0)
    scene_tilt = Scene(
        target_state=TargetState(position=p_target),
        camera_state=CameraState(pan_rad=0.0, tilt_rad=tilt_rad),
    )
    frame_tilt = sensor.capture(scene_tilt)
    assert frame_tilt.projected_pixel[1] < 240.0


def test_sensor_visibility_behind_and_outside_fov():
    sensor = MonochromeSensor()

    # Target behind camera (Z = -10.0)
    scene_behind = Scene(target_state=TargetState(position=[0.0, 0.0, -10.0]))
    frame_behind = sensor.capture(scene_behind)
    assert frame_behind.target_visible is False
    assert frame_behind.projected_pixel is None
    assert np.all(frame_behind.image == 0.0)
    assert frame_behind.image.shape == (480, 640)

    # Target outside horizontal FOV
    x_out = 100.0 * math.tan(math.radians(3.0))  # > 2.0 deg FOV half-angle
    scene_out = Scene(target_state=TargetState(position=[x_out, 0.0, 100.0]))
    frame_out = sensor.capture(scene_out)
    assert frame_out.target_visible is False
    assert frame_out.projected_pixel is None
    assert np.all(frame_out.image == 0.0)


def test_beacon_subpixel_precision_and_rendering():
    sensor = MonochromeSensor()

    # Create two scenes with target positions differing by fractional pixel amounts
    scene1 = Scene(target_state=TargetState(position=[0.123, -0.456, 100.0]))
    scene2 = Scene(target_state=TargetState(position=[0.145, -0.456, 100.0]))

    frame1 = sensor.capture(scene1)
    frame2 = sensor.capture(scene2)

    p_rel1 = world_to_camera_relative([0.123, -0.456, 100.0], np.zeros(3))
    p_cam1 = rotate_world_to_camera(p_rel1, 0.0, 0.0)
    u_exp1, v_exp1 = project_camera_to_pixel(p_cam1, sensor.config.camera_config)

    p_rel2 = world_to_camera_relative([0.145, -0.456, 100.0], np.zeros(3))
    p_cam2 = rotate_world_to_camera(p_rel2, 0.0, 0.0)
    u_exp2, v_exp2 = project_camera_to_pixel(p_cam2, sensor.config.camera_config)

    # 1. Verify fractional coordinate values are stored precisely
    assert frame1.projected_pixel[0] == pytest.approx(u_exp1)
    assert frame1.projected_pixel[1] == pytest.approx(v_exp1)
    assert frame2.projected_pixel[0] == pytest.approx(u_exp2)
    assert frame2.projected_pixel[1] == pytest.approx(v_exp2)

    # 2. Verify image arrays reflect fractional shift and are not identical
    assert not np.array_equal(frame1.image, frame2.image)


def test_beacon_diameter_and_intensity_scaling():
    cfg_small = SensorConfig(beacon_diameter=6.0, peak_intensity=100.0)
    cfg_large = SensorConfig(beacon_diameter=18.0, peak_intensity=200.0)

    sensor_small = MonochromeSensor(cfg_small)
    sensor_large = MonochromeSensor(cfg_large)

    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))

    frame_small = sensor_small.capture(scene)
    frame_large = sensor_large.capture(scene)

    # Check center pixel intensity matches peak_intensity
    assert frame_small.image[240, 320] == pytest.approx(100.0, abs=1e-2)
    assert frame_large.image[240, 320] == pytest.approx(200.0, abs=1e-2)

    # Count pixels exceeding threshold > 10.0 intensity
    count_small = np.sum(frame_small.image > 10.0)
    count_large = np.sum(frame_large.image > 10.0)

    assert count_large > count_small


test_edge_clipping_visible_positions = [
    ([3.4, 0.0, 100.0]),   # Near right boundary (approx u = 635 px, inside FOV)
    ([-3.4, 0.0, 100.0]),  # Near left boundary (approx u = 5 px, inside FOV)
]


@pytest.mark.parametrize("pos", test_edge_clipping_visible_positions)
def test_edge_clipping_no_error(pos):
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=pos))

    frame = sensor.capture(scene)

    # 1. Target is geometrically visible
    assert frame.target_visible is True
    assert frame.projected_pixel is not None

    # 2. Projected pixel is near an image boundary (u close to 0 or 640)
    u_proj, v_proj = frame.projected_pixel
    assert (u_proj < 10.0) or (u_proj > 630.0)

    # 3. Capture completes without indexing error and image shape remains correct
    assert frame.image.shape == (480, 640)

    # 4. Beacon is clipped safely inside array bounds
    assert np.max(frame.image) > 0.0


def test_sensor_determinism():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.5, -0.2, 80.0]))

    frame1 = sensor.capture(scene)
    frame2 = sensor.capture(scene)

    np.testing.assert_array_equal(frame1.image, frame2.image)
    assert frame1.projected_pixel == frame2.projected_pixel
    assert frame1.target_visible == frame2.target_visible