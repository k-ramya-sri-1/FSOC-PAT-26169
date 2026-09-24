"""Unit tests for core/scene.py following docs/TEST_STRATEGY.md."""

import math
import numpy as np
import pytest

from core.scene import (
    BoundedRandomMotionModel,
    CameraState,
    CircularMotionModel,
    Figure8MotionModel,
    Scene,
    StraightMotionModel,
    TargetState,
)


def test_target_state_initialization():
    ts = TargetState(position=[10.0, 5.0, 200.0], velocity=[1.0, 0.0, -2.0])
    np.testing.assert_allclose(ts.position, [10.0, 5.0, 200.0])
    np.testing.assert_allclose(ts.velocity, [1.0, 0.0, -2.0])


def test_camera_state_initialization():
    cs = CameraState(position=[1.0, 2.0, 3.0], pan_rad=0.1, tilt_rad=-0.2)
    np.testing.assert_allclose(cs.position, [1.0, 2.0, 3.0])
    assert cs.pan_rad == 0.1
    assert cs.tilt_rad == -0.2


def test_straight_motion_analytical():
    p0 = np.array([0.0, 0.0, 100.0])
    vel = np.array([2.0, -1.0, 5.0])
    motion = StraightMotionModel(velocity=vel)
    scene = Scene(
        target_state=TargetState(position=p0),
        motion_model=motion,
    )

    # Verify initial position preservation
    np.testing.assert_allclose(scene.target_state.position, p0)
    np.testing.assert_allclose(scene.target_state.velocity, vel)

    # Step 1: t = 0 -> t = 0.5
    dt1 = 0.5
    scene.step(dt1)
    expected_p1 = p0 + vel * 0.5
    np.testing.assert_allclose(scene.target_state.position, expected_p1)
    np.testing.assert_allclose(scene.target_state.velocity, vel)

    # Step 2: t = 0.5 -> t = 1.5
    dt2 = 1.0
    scene.step(dt2)
    expected_p2 = p0 + vel * 1.5
    np.testing.assert_allclose(scene.target_state.position, expected_p2)
    np.testing.assert_allclose(scene.target_state.velocity, vel)


def test_circular_motion_initial_state_consistency():
    motion = CircularMotionModel(
        center=[0.0, 0.0, 100.0], radius=10.0, angular_speed=0.5, initial_phase=0.0, plane="XY"
    )
    scene = Scene(motion_model=motion)

    expected_pos, expected_vel = motion._get_pos_vel(0.0)

    assert scene.time == 0.0
    np.testing.assert_allclose(scene.target_state.position, expected_pos, atol=1e-12)
    np.testing.assert_allclose(scene.target_state.velocity, expected_vel, atol=1e-12)


def test_figure8_motion_initial_state_consistency():
    motion = Figure8MotionModel(
        center=[1.0, 2.0, 50.0], amplitude_x=8.0, amplitude_y=4.0, angular_speed=0.3, initial_phase=0.1
    )
    scene = Scene(motion_model=motion)

    expected_pos, expected_vel = motion._get_pos_vel(0.0)

    assert scene.time == 0.0
    np.testing.assert_allclose(scene.target_state.position, expected_pos, atol=1e-12)
    np.testing.assert_allclose(scene.target_state.velocity, expected_vel, atol=1e-12)


def test_zero_timestep():
    scene = Scene(target_state=TargetState(position=[10.0, 10.0, 100.0]))
    initial_pos = scene.target_state.position.copy()
    scene.step(0.0)
    np.testing.assert_allclose(scene.target_state.position, initial_pos)
    assert scene.time == 0.0


def test_invalid_timestep():
    scene = Scene()
    with pytest.raises(ValueError):
        scene.step(-0.1)


def test_circular_motion_analytical():
    center = np.array([5.0, -2.0, 100.0])
    radius = 10.0
    w = 0.5
    phase = 0.2
    motion = CircularMotionModel(
        center=center, radius=radius, angular_speed=w, initial_phase=phase, plane="XY"
    )
    scene = Scene(motion_model=motion)

    # Test position and velocity at t = 2.0
    dt = 2.0
    scene.step(dt)

    phi = phase + w * dt
    expected_pos = np.array([
        center[0] + radius * math.cos(phi),
        center[1] + radius * math.sin(phi),
        center[2],
    ])
    expected_vel = np.array([
        -radius * w * math.sin(phi),
        radius * w * math.cos(phi),
        0.0,
    ])

    np.testing.assert_allclose(scene.target_state.position, expected_pos, atol=1e-12)
    np.testing.assert_allclose(scene.target_state.velocity, expected_vel, atol=1e-12)


def test_figure8_motion_analytical():
    center = np.array([1.0, 2.0, 50.0])
    ax, ay = 8.0, 4.0
    w = 0.3
    phase = 0.1
    motion = Figure8MotionModel(
        center=center, amplitude_x=ax, amplitude_y=ay, angular_speed=w, initial_phase=phase
    )
    scene = Scene(motion_model=motion)

    # Test position and velocity at t = 3.0
    dt = 3.0
    scene.step(dt)

    phi = phase + w * dt
    expected_pos = np.array([
        center[0] + ax * math.sin(phi),
        center[1] + (ay / 2.0) * math.sin(2.0 * phi),
        center[2],
    ])
    expected_vel = np.array([
        ax * w * math.cos(phi),
        ay * w * math.cos(2.0 * phi),
        0.0,
    ])

    np.testing.assert_allclose(scene.target_state.position, expected_pos, atol=1e-12)
    np.testing.assert_allclose(scene.target_state.velocity, expected_vel, atol=1e-12)


def test_random_motion_reproducibility_and_reset():
    motion = BoundedRandomMotionModel(seed=123)
    scene = Scene(motion_model=motion)

    # Generate sequence 1
    seq1_pos = []
    seq1_vel = []
    for _ in range(5):
        scene.step(0.1)
        seq1_pos.append(scene.target_state.position.copy())
        seq1_vel.append(scene.target_state.velocity.copy())

    # Reset model and scene state
    motion.reset(initial_position=[0.0, 0.0, 100.0])
    scene_reset = Scene(motion_model=motion)

    # Generate sequence 2
    for i in range(5):
        scene_reset.step(0.1)
        np.testing.assert_allclose(scene_reset.target_state.position, seq1_pos[i])
        np.testing.assert_allclose(scene_reset.target_state.velocity, seq1_vel[i])


def test_random_motion_validation():
    with pytest.raises(ValueError):
        BoundedRandomMotionModel(max_acceleration=-1.0)

    with pytest.raises(ValueError):
        BoundedRandomMotionModel(max_speed=-0.5)


def test_different_random_seeds():
    scene1 = Scene(motion_model=BoundedRandomMotionModel(seed=42))
    scene2 = Scene(motion_model=BoundedRandomMotionModel(seed=999))

    scene1.step(0.1)
    scene2.step(0.1)
    assert not np.array_equal(scene1.target_state.position, scene2.target_state.position)


def test_scene_time_progression():
    scene = Scene(initial_time=5.0)
    scene.step(0.2)
    assert scene.time == pytest.approx(5.2)
    scene.step(0.3)
    assert scene.time == pytest.approx(5.5)


def test_camera_state_isolation():
    initial_cam_pos = np.array([1.0, 2.0, 3.0])
    scene = Scene(camera_state=CameraState(position=initial_cam_pos, pan_rad=0.5, tilt_rad=-0.1))

    scene.step(1.0)
    np.testing.assert_allclose(scene.camera_state.position, initial_cam_pos)
    assert scene.camera_state.pan_rad == 0.5
    assert scene.camera_state.tilt_rad == -0.1


def test_vector_validation():
    with pytest.raises(ValueError):
        TargetState(position=[1.0, 2.0])  # Only 2D

    with pytest.raises(ValueError):
        CameraState(position=[1.0, 2.0, 3.0, 4.0])  # 4D