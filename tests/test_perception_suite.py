from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from benchmark.perception_suite import PerceptionBenchmarkSuite, PerceptionResult
from core.detection import BeaconDetector
from core.geometry import CameraConfig
from core.sensor import SensorConfig


EXPECTED_NAMES = (
    "beacon_size_5",
    "beacon_size_10",
    "beacon_size_15",
    "beacon_size_20",
    "position_center",
    "position_left",
    "position_right",
    "position_top",
    "position_bottom",
    "visibility_near_fov_boundary",
    "visibility_outside_fov",
    "visibility_behind_camera",
)


def test_suite_scenarios_have_expected_order_and_settings():
    suite = PerceptionBenchmarkSuite()
    scenarios = suite.scenarios()

    assert len(scenarios) == len(EXPECTED_NAMES)
    assert tuple(scenario.scenario_name for scenario in scenarios) == EXPECTED_NAMES
    assert tuple(s.beacon_diameter for s in scenarios[:4]) == (5.0, 10.0, 15.0, 20.0)
    assert tuple(s.scenario_name for s in scenarios[4:9]) == (
        "position_center",
        "position_left",
        "position_right",
        "position_top",
        "position_bottom",
    )
    assert tuple(s.scenario_name for s in scenarios[9:]) == (
        "visibility_near_fov_boundary",
        "visibility_outside_fov",
        "visibility_behind_camera",
    )
    assert all(isinstance(s.beacon_diameter, float) for s in scenarios[:4])
    assert all(s.expected_visibility for s in scenarios[:9])
    assert scenarios[9].expected_visibility is True
    assert scenarios[10].expected_visibility is False
    assert scenarios[11].expected_visibility is False


def test_beacon_sizes_use_existing_sensor_config_field():
    suite = PerceptionBenchmarkSuite()

    sensors = tuple(
        SensorConfig(
            camera_config=suite.camera_config,
            beacon_diameter=scenario.beacon_diameter,
            peak_intensity=suite.peak_intensity,
        )
        for scenario in suite.scenarios()[:4]
    )

    assert tuple(sensor.beacon_diameter for sensor in sensors) == (5.0, 10.0, 15.0, 20.0)


def test_run_returns_immutable_repeatable_observed_results():
    suite = PerceptionBenchmarkSuite()
    definitions_before = suite.scenarios()

    first = suite.run()
    second = suite.run()

    assert isinstance(first, tuple)
    assert len(first) == len(EXPECTED_NAMES)
    assert all(isinstance(result, PerceptionResult) for result in first)
    assert first == second
    assert suite.scenarios() == definitions_before
    with pytest.raises(FrozenInstanceError):
        first[0].detected = False


def test_sensor_visibility_and_detector_observations(monkeypatch):
    detected_images = []
    original_detect = BeaconDetector.detect

    def observe_detect(detector, image):
        detected_images.append(image)
        return original_detect(detector, image)

    monkeypatch.setattr(BeaconDetector, "detect", observe_detect)
    results = PerceptionBenchmarkSuite().run()

    assert len(detected_images) == len(EXPECTED_NAMES)
    assert all(image.ndim == 2 for image in detected_images)
    assert all(isinstance(result.detected, bool) for result in results)
    assert all(result.candidate_count >= 0 for result in results)
    assert all(result.detected == (result.candidate_count > 0) for result in results)

    by_name = {result.scenario_name: result for result in results}
    for name in EXPECTED_NAMES[:10]:
        assert by_name[name].target_visible is True
        assert by_name[name].projected_pixel is not None
        assert by_name[name].detected is True

    outside = by_name["visibility_outside_fov"]
    behind = by_name["visibility_behind_camera"]
    for result in (outside, behind):
        assert result.target_visible is False
        assert result.projected_pixel is None
        assert result.detected is False
        assert result.candidate_count == 0
    assert np.count_nonzero(detected_images[10]) == 0
    assert np.count_nonzero(detected_images[11]) == 0

    near_boundary = by_name["visibility_near_fov_boundary"]
    assert near_boundary.projected_pixel[0] > 0.9 * 640
    assert not hasattr(near_boundary, "identity")


def test_suite_uses_geometry_config_and_preserves_scenario_positions():
    camera = CameraConfig(width=800, height=600, fov_x_deg=8.0, fov_y_deg=6.0)
    suite = PerceptionBenchmarkSuite(camera_config=camera)
    results = suite.run()
    by_name = {result.scenario_name: result for result in results}

    assert by_name["position_center"].projected_pixel == pytest.approx((400.0, 300.0))
    assert by_name["position_left"].projected_pixel[0] < 400.0
    assert by_name["position_right"].projected_pixel[0] > 400.0
    assert by_name["position_top"].projected_pixel[1] < 300.0
    assert by_name["position_bottom"].projected_pixel[1] > 300.0
