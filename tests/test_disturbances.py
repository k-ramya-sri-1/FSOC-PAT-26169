"""Unit tests for core/disturbances.py following docs/TEST_STRATEGY.md."""

import numpy as np
import pytest

from core.disturbances import AtmosphereMode, DisturbanceConfig, DisturbedFrame, ImageDisturbanceModel
from core.scene import Scene, TargetState
from core.sensor import MonochromeSensor, SensorFrame


def test_disturbance_config_validation():
    with pytest.raises(ValueError):
        DisturbanceConfig(salt_pepper_probability=1.5)

    with pytest.raises(ValueError):
        DisturbanceConfig(gaussian_std=-1.0)

    with pytest.raises(ValueError):
        DisturbanceConfig(poisson_strength=-0.5)

    with pytest.raises(ValueError):
        DisturbanceConfig(jitter_x_pixels=-2.0)

    with pytest.raises(ValueError):
        DisturbanceConfig(atmosphere_strength=1.2)

    with pytest.raises(ValueError):
        DisturbanceConfig(low_light_scale=-0.1)


def test_zero_disturbance_identity():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    model = ImageDisturbanceModel(DisturbanceConfig())
    disturbed = model.apply(frame)

    assert isinstance(disturbed, DisturbedFrame)
    assert disturbed.source_frame is frame
    np.testing.assert_array_equal(disturbed.image, frame.image)
    assert disturbed.image.dtype == np.float32


def test_source_image_immutability():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    original_copy = frame.image.copy()

    cfg = DisturbanceConfig(
        salt_pepper_probability=0.1,
        gaussian_std=10.0,
        jitter_x_pixels=5.0,
        atmosphere_mode=AtmosphereMode.HAZE,
        atmosphere_strength=0.5,
    )
    model = ImageDisturbanceModel(cfg)
    _ = model.apply(frame)

    # Clean source frame image must remain untouched
    np.testing.assert_array_equal(frame.image, original_copy)


def test_salt_and_pepper_noise():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(salt_pepper_probability=0.05, seed=42)
    model = ImageDisturbanceModel(cfg)
    disturbed = model.apply(frame)

    assert not np.array_equal(disturbed.image, frame.image)
    assert np.min(disturbed.image) >= 0.0
    assert np.max(disturbed.image) <= 255.0


def test_gaussian_noise():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(gaussian_std=15.0, seed=42)
    model = ImageDisturbanceModel(cfg)
    disturbed = model.apply(frame)

    assert not np.array_equal(disturbed.image, frame.image)
    assert np.min(disturbed.image) >= 0.0
    assert np.max(disturbed.image) <= 255.0


def test_poisson_noise():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(poisson_strength=1.0, seed=42)
    model = ImageDisturbanceModel(cfg)
    disturbed = model.apply(frame)

    assert not np.array_equal(disturbed.image, frame.image)
    assert np.min(disturbed.image) >= 0.0
    assert np.max(disturbed.image) <= 255.0


def test_camera_jitter():
    sensor = MonochromeSensor()
    # Off-center beacon to verify jitter displacement
    scene = Scene(target_state=TargetState(position=[0.5, 0.5, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(jitter_x_pixels=5.0, jitter_y_pixels=5.0, seed=123)
    model = ImageDisturbanceModel(cfg)
    disturbed = model.apply(frame)

    assert disturbed.image.shape == frame.image.shape
    assert not np.array_equal(disturbed.image, frame.image)


@pytest.mark.parametrize(
    "mode",
    [
        AtmosphereMode.CLEAR,
        AtmosphereMode.HAZE,
        AtmosphereMode.FOG,
        AtmosphereMode.RAIN,
        AtmosphereMode.LOW_LIGHT,
    ],
)
def test_atmospheric_modes(mode):
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(atmosphere_mode=mode, atmosphere_strength=0.5, seed=42)
    model = ImageDisturbanceModel(cfg)
    disturbed = model.apply(frame)

    assert disturbed.image.shape == frame.image.shape
    assert disturbed.image.dtype == np.float32
    assert np.min(disturbed.image) >= 0.0
    assert np.max(disturbed.image) <= 255.0

    if mode == AtmosphereMode.CLEAR:
        np.testing.assert_array_equal(disturbed.image, frame.image)
    else:
        assert not np.array_equal(disturbed.image, frame.image)


def test_low_light_scaling():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(low_light_scale=0.3)
    model = ImageDisturbanceModel(cfg)
    disturbed = model.apply(frame)

    np.testing.assert_allclose(disturbed.image, frame.image * 0.3)


def test_determinism_and_reproducibility():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(
        salt_pepper_probability=0.02,
        gaussian_std=5.0,
        poisson_strength=0.5,
        jitter_x_pixels=2.0,
        atmosphere_mode=AtmosphereMode.RAIN,
        atmosphere_strength=0.3,
        seed=100,
    )

    model1 = ImageDisturbanceModel(cfg)
    disturbed1 = model1.apply(frame)

    model2 = ImageDisturbanceModel(cfg)
    disturbed2 = model2.apply(frame)

    # Identical seed -> identical output
    np.testing.assert_array_equal(disturbed1.image, disturbed2.image)

    # Different seed -> different output
    cfg_diff_seed = DisturbanceConfig(
        salt_pepper_probability=0.02,
        gaussian_std=5.0,
        poisson_strength=0.5,
        jitter_x_pixels=2.0,
        atmosphere_mode=AtmosphereMode.RAIN,
        atmosphere_strength=0.3,
        seed=200,
    )
    model_diff = ImageDisturbanceModel(cfg_diff_seed)
    disturbed_diff = model_diff.apply(frame)

    assert not np.array_equal(disturbed1.image, disturbed_diff.image)


def test_reset_reproducibility():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(gaussian_std=10.0, seed=777)
    model = ImageDisturbanceModel(cfg)

    out1 = model.apply(frame)
    model.reset()
    out2 = model.apply(frame)

    np.testing.assert_array_equal(out1.image, out2.image)


def test_metadata_and_ground_truth_preservation():
    sensor = MonochromeSensor()
    scene = Scene(target_state=TargetState(position=[0.0, 0.0, 100.0]))
    frame = sensor.capture(scene)

    cfg = DisturbanceConfig(gaussian_std=10.0, atmosphere_mode=AtmosphereMode.FOG, atmosphere_strength=0.5)
    model = ImageDisturbanceModel(cfg)
    disturbed = model.apply(frame)

    # Ensure ground truth metadata from Stage 1C is strictly preserved and unaltered
    assert disturbed.source_frame.target_visible == frame.target_visible
    assert disturbed.source_frame.projected_pixel == frame.projected_pixel