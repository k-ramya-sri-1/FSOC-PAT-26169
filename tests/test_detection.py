"""
Unit tests for core/detection.py following docs/TEST_STRATEGY.md.
"""

import numpy as np
import pytest

from core.detection import BeaconDetector, DetectionCandidate, DetectionConfig


def test_detection_config_defaults_and_validations():
    cfg = DetectionConfig()
    assert cfg.threshold == 128.0
    assert cfg.min_area == 4.0
    assert cfg.max_area == 1000.0
    assert cfg.max_candidates == 10

    with pytest.raises(ValueError):
        DetectionConfig(threshold=-1.0)

    with pytest.raises(ValueError):
        DetectionConfig(threshold=256.0)

    with pytest.raises(ValueError):
        DetectionConfig(min_area=0.0)

    with pytest.raises(ValueError):
        DetectionConfig(min_area=10.0, max_area=5.0)

    with pytest.raises(ValueError):
        DetectionConfig(max_candidates=0)


def test_detection_candidate_dataclass_and_validation():
    cand = DetectionCandidate(
        candidate_id=1,
        centroid=(100.0, 150.0),
        area=25.0,
        bounding_box=(95, 145, 105, 155),
        peak_intensity=255.0,
        mean_intensity=200.0,
    )
    assert cand.candidate_id == 1
    assert cand.centroid == (100.0, 150.0)
    assert cand.area == 25.0
    assert cand.bounding_box == (95, 145, 105, 155)
    assert cand.peak_intensity == 255.0
    assert cand.mean_intensity == 200.0

    with pytest.raises(ValueError):
        DetectionCandidate(
            candidate_id=0,
            centroid=(100.0, 150.0),
            area=25.0,
            bounding_box=(95, 145, 105, 155),
            peak_intensity=255.0,
            mean_intensity=200.0,
        )

    with pytest.raises(ValueError):
        DetectionCandidate(
            candidate_id=1,
            centroid=(float("nan"), 150.0),
            area=25.0,
            bounding_box=(95, 145, 105, 155),
            peak_intensity=255.0,
            mean_intensity=200.0,
        )

    with pytest.raises(ValueError):
        DetectionCandidate(
            candidate_id=1,
            centroid=(100.0, 150.0),
            area=-5.0,
            bounding_box=(95, 145, 105, 155),
            peak_intensity=255.0,
            mean_intensity=200.0,
        )

    with pytest.raises(ValueError):
        DetectionCandidate(
            candidate_id=1,
            centroid=(100.0, 150.0),
            area=25.0,
            bounding_box=(105, 145, 95, 155),  # xmax <= xmin
            peak_intensity=255.0,
            mean_intensity=200.0,
        )


def test_detector_invalid_image_type():
    detector = BeaconDetector()
    with pytest.raises(ValueError):
        detector.detect(np.zeros((100, 100, 3)))  # 3D array

    with pytest.raises(ValueError):
        detector.detect([1, 2, 3])  # Not a numpy array


def test_detector_detects_synthetic_spot():
    detector = BeaconDetector(DetectionConfig(threshold=100.0))
    img = np.zeros((100, 100), dtype=np.uint8)
    # Draw a 5x5 bright region
    img[40:45, 40:45] = 200

    candidates = detector.detect(img)
    assert len(candidates) == 1
    cand = candidates[0]

    assert cand.candidate_id == 1
    assert 41.5 <= cand.centroid[0] <= 42.5
    assert 41.5 <= cand.centroid[1] <= 42.5
    assert cand.area == pytest.approx(25.0)
    assert cand.peak_intensity == 200.0
    assert cand.mean_intensity == 200.0


def test_detector_ignores_spots_below_threshold_or_outside_area_limits():
    detector = BeaconDetector(
        DetectionConfig(threshold=100.0, min_area=10.0, max_area=50.0)
    )
    img = np.zeros((100, 100), dtype=np.uint8)

    # Spot 1: Below threshold
    img[10:15, 10:15] = 50

    # Spot 2: Too small (area = 4)
    img[30:32, 30:32] = 200

    # Spot 3: Too large (area = 100)
    img[60:70, 60:70] = 200

    # Spot 4: Valid (area = 25)
    img[80:85, 80:85] = 200

    candidates = detector.detect(img)
    assert len(candidates) == 1
    assert candidates[0].candidate_id == 1
    assert 81.5 <= candidates[0].centroid[0] <= 82.5


def test_detector_candidate_limit():
    detector = BeaconDetector(DetectionConfig(threshold=100.0, max_candidates=2))
    img = np.zeros((100, 100), dtype=np.uint8)

    # Create 3 spots
    img[10:15, 10:15] = 200
    img[40:45, 40:45] = 200
    img[70:75, 70:75] = 200

    candidates = detector.detect(img)
    assert len(candidates) == 2
    assert candidates[0].candidate_id == 1
    assert candidates[1].candidate_id == 2


def test_detector_candidate_ids_deterministic_and_independent_per_frame():
    detector = BeaconDetector(DetectionConfig(threshold=100.0))
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:20, 10:20] = 200  # Top-left spot
    img[70:80, 70:80] = 200  # Bottom-right spot

    cands_run1 = detector.detect(img)
    assert len(cands_run1) == 2
    assert cands_run1[0].candidate_id == 1
    assert cands_run1[1].candidate_id == 2

    cands_run2 = detector.detect(img)
    assert len(cands_run2) == 2
    assert cands_run2[0].candidate_id == 1
    assert cands_run2[1].candidate_id == 2