from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from benchmark.perception_stress import PerceptionStressResult, PerceptionStressSuite
from benchmark.perception_suite import PerceptionBenchmarkSuite
from core.detection import BeaconDetector
from core.disturbances import DisturbanceConfig, ImageDisturbanceModel


DISTURBANCE_ORDER = (
    "clean",
    "gaussian",
    "poisson",
    "salt_pepper",
    "low_light",
    "haze",
    "fog",
    "rain",
)
BEACON_SIZES = (5.0, 10.0, 15.0, 20.0)


def test_primary_matrix_has_32_cases_in_size_then_disturbance_order():
    suite = PerceptionStressSuite(seed=51)
    cases = suite.scenarios()

    assert len(cases) == 32
    assert tuple(case.beacon_diameter for case in cases[::8]) == BEACON_SIZES
    assert tuple(case.disturbance_name for case in cases[:8]) == DISTURBANCE_ORDER
    for size_index, diameter in enumerate(BEACON_SIZES):
        group = cases[size_index * 8 : (size_index + 1) * 8]
        assert all(case.beacon_diameter == diameter for case in group)
        assert tuple(case.disturbance_name for case in group) == DISTURBANCE_ORDER
        assert len({case.scenario_name for case in group}) == 8
    assert all(isinstance(case.disturbance_config, DisturbanceConfig) for case in cases)
    assert tuple(case.scenario_name for case in cases) == tuple(
        f"beacon_size_{int(size)}__{mode}"
        for size in BEACON_SIZES
        for mode in DISTURBANCE_ORDER
    )


def test_matrix_runs_to_32_frozen_results_and_is_deterministic():
    suite = PerceptionStressSuite(seed=123)
    cases_before = suite.scenarios()

    first = suite.run()
    second = suite.run()

    assert isinstance(first, tuple)
    assert len(first) == 32
    assert all(isinstance(result, PerceptionStressResult) for result in first)
    assert first == second
    assert tuple(result.scenario_name for result in first) == tuple(
        case.scenario_name for case in cases_before
    )
    assert suite.scenarios() == cases_before
    with pytest.raises(FrozenInstanceError):
        first[0].detected = False


def test_existing_disturbance_pipeline_and_detector_run_for_each_case(monkeypatch):
    disturbance_calls = []
    detection_images = []
    original_apply = ImageDisturbanceModel.apply
    original_detect = BeaconDetector.detect

    def record_apply(model, sensor_frame):
        disturbed = original_apply(model, sensor_frame)
        disturbance_calls.append((model.config, sensor_frame.image.copy(), disturbed.image.copy()))
        return disturbed

    def record_detect(detector, image):
        detection_images.append(image)
        return original_detect(detector, image)

    monkeypatch.setattr(ImageDisturbanceModel, "apply", record_apply)
    monkeypatch.setattr(BeaconDetector, "detect", record_detect)

    suite = PerceptionStressSuite(seed=37)
    cases = suite.scenarios()
    results = suite.run()

    assert len(disturbance_calls) == 32
    assert len(detection_images) == 32
    assert all(image.ndim == 2 for image in detection_images)
    assert tuple(config for config, _, _ in disturbance_calls) == tuple(
        case.disturbance_config for case in cases
    )
    clean_image, clean_disturbed = disturbance_calls[0][1:]
    np.testing.assert_array_equal(clean_disturbed, clean_image)
    gaussian_image, gaussian_disturbed = disturbance_calls[1][1:]
    assert not np.array_equal(gaussian_disturbed, gaussian_image)
    assert all(result.detected == (result.candidate_count > 0) for result in results)


def test_clean_size_cases_match_stage_2h_perception_results():
    suite = PerceptionStressSuite(seed=7)
    stress_results = suite.run()
    stage2h_results = PerceptionBenchmarkSuite(
        camera_config=suite.camera_config,
        peak_intensity=suite.peak_intensity,
        detection_config=suite.detection_config,
        target_range_m=suite.target_range_m,
    ).run()
    stage2h_by_size = {
        result.beacon_diameter: result
        for result in stage2h_results
        if result.scenario_name.startswith("beacon_size_")
    }

    for stress in stress_results[::8]:
        clean = stage2h_by_size[stress.beacon_diameter]
        assert stress.target_visible == clean.target_visible
        assert stress.projected_pixel == clean.projected_pixel
        assert stress.detected == clean.detected
        assert stress.candidate_count == clean.candidate_count
        assert stress.candidate_centroid == clean.candidate_centroid


def test_suite_configuration_and_scenario_definitions_are_not_mutated():
    suite = PerceptionStressSuite(seed=88)
    camera_before = suite.camera_config
    detector_config_before = suite.detection_config
    scenarios_before = suite.scenarios()

    suite.run()

    assert suite.camera_config is camera_before
    assert suite.detection_config is detector_config_before
    assert suite.scenarios() == scenarios_before
