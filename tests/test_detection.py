"""Unit tests for core/detection.py following docs/TEST_STRATEGY.md."""

import numpy as np
import pytest

from core.detection import BeaconDetector, DetectionCandidate, DetectionConfig


def _create_gaussian_spot(
    shape: tuple = (480, 640),
    cx: float = 320.4,
    cy: float = 239.7,
    peak: float = 255.0,
    radius: float = 5.0,
) -> np.ndarray:
    """Helper to render a synthetic Gaussian spot on a black image for test cases."""
    img = np.zeros(shape, dtype=np.float32)
    h, w = shape
    grid_y, grid_x = np.ogrid[:h, :w]
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    sigma = radius / 2.0
    spot = peak * np.exp(-dist_sq / (2.0 * sigma**2))
    img = np.where(spot > 1.0, spot, 0.0).astype(np.float32)
    return img


def test_detection_config_validation():
    with pytest.raises(ValueError):
        DetectionConfig(threshold=-10.0)

    with pytest.raises(ValueError):
        DetectionConfig(threshold=300.0)

    with pytest.raises(ValueError):
        DetectionConfig(min_area=-1.0)

    with pytest.raises(ValueError):
        DetectionConfig(min_area=100.0, max_area=50.0)

    with pytest.raises(ValueError):
        DetectionConfig(min_peak=-5.0)


def test_center_beacon_detection():
    detector = BeaconDetector()
    img = _create_gaussian_spot(cx=320.0, cy=240.0, peak=250.0)

    candidates = detector.detect(img)
    assert len(candidates) == 1

    cand = candidates[0]
    assert isinstance(cand, DetectionCandidate)
    assert cand.centroid_x == pytest.approx(320.0, abs=1e-2)
    assert cand.centroid_y == pytest.approx(240.0, abs=1e-2)
    assert cand.peak_intensity == pytest.approx(250.0, abs=1e-1)


def test_subpixel_localization():
    cfg = DetectionConfig(threshold=50.0, min_area=2.0, min_peak=100.0)
    detector = BeaconDetector(cfg)

    true_x, true_y = 320.4, 239.7
    img = _create_gaussian_spot(cx=true_x, cy=true_y, peak=240.0, radius=5.0)

    candidates = detector.detect(img)
    assert len(candidates) == 1

    cand = candidates[0]
    # Check that intensity-weighted subpixel centroid matches true subpixel center
    assert cand.centroid_x == pytest.approx(true_x, abs=0.05)
    assert cand.centroid_y == pytest.approx(true_y, abs=0.05)

    # Verify that returned centroid preserves subpixel fractional component
    assert cand.centroid_x % 1.0 != 0.0
    assert cand.centroid_y % 1.0 != 0.0

    # Geometric center of bounding box would be an integer/half-integer offset
    bbox_center_x = cand.bbox_x + (cand.bbox_width - 1) / 2.0
    bbox_center_y = cand.bbox_y + (cand.bbox_height - 1) / 2.0
    assert cand.centroid_x != bbox_center_x or cand.centroid_y != bbox_center_y


def test_boundary_beacons():
    detector = BeaconDetector()

    # Near top boundary
    img_top = _create_gaussian_spot(cx=320.0, cy=5.0)
    cands_top = detector.detect(img_top)
    assert len(cands_top) == 1
    assert cands_top[0].centroid_y == pytest.approx(5.0, abs=0.5)

    # Near left boundary
    img_left = _create_gaussian_spot(cx=5.0, cy=240.0)
    cands_left = detector.detect(img_left)
    assert len(cands_left) == 1
    assert cands_left[0].centroid_x == pytest.approx(5.0, abs=0.5)


def test_clipped_beacon_edge():
    detector = BeaconDetector()

    # Render beacon partially clipped at left boundary (center at x=-2.0)
    grid_y, grid_x = np.ogrid[:480, :640]
    dist_sq = (grid_x - (-2.0)) ** 2 + (grid_y - 240.0) ** 2
    img = (220.0 * np.exp(-dist_sq / (2.0 * 2.5**2))).astype(np.float32)

    cands = detector.detect(img)
    assert len(cands) == 1
    # Visible pixels will have centroid near x=0..1
    assert cands[0].centroid_x >= 0.0
    assert cands[0].bbox_x == 0


def test_no_beacon_empty_image():
    detector = BeaconDetector()
    img = np.zeros((480, 640), dtype=np.float32)

    candidates = detector.detect(img)
    assert len(candidates) == 0


def test_small_noise_blob_rejection():
    # Noise blob with area < min_area (e.g., single bright pixel)
    cfg = DetectionConfig(threshold=100.0, min_area=5.0, min_peak=120.0)
    detector = BeaconDetector(cfg)

    img = np.zeros((480, 640), dtype=np.float32)
    img[100, 100] = 200.0  # 1 pixel area = 1.0 < min_area 5.0

    candidates = detector.detect(img)
    assert len(candidates) == 0


def test_oversized_region_rejection():
    # Region with area > max_area
    cfg = DetectionConfig(threshold=100.0, max_area=50.0)
    detector = BeaconDetector(cfg)

    img = np.zeros((480, 640), dtype=np.float32)
    img[100:120, 100:120] = 200.0  # Area = 400 > max_area 50.0

    candidates = detector.detect(img)
    assert len(candidates) == 0


def test_multiple_candidates_detection():
    detector = BeaconDetector()

    spot1 = _create_gaussian_spot(cx=100.0, cy=100.0, peak=200.0)
    spot2 = _create_gaussian_spot(cx=500.0, cy=300.0, peak=250.0)
    img = spot1 + spot2

    candidates = detector.detect(img)
    assert len(candidates) == 2


def test_deterministic_candidate_ordering():
    detector = BeaconDetector()

    # Spot A: lower peak (180.0)
    spot_a = _create_gaussian_spot(cx=100.0, cy=100.0, peak=180.0)
    # Spot B: higher peak (240.0)
    spot_b = _create_gaussian_spot(cx=400.0, cy=400.0, peak=240.0)
    img = spot_a + spot_b

    candidates = detector.detect(img)
    assert len(candidates) == 2
    # Candidate B must be first because of higher peak intensity
    assert candidates[0].peak_intensity > candidates[1].peak_intensity
    assert candidates[0].centroid_x == pytest.approx(400.0, abs=0.5)


def test_candidate_metadata_correctness():
    detector = BeaconDetector()
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:43, 40:43] = 200.0  # 3x3 square

    cands = detector.detect(img)
    assert len(cands) == 1
    c = cands[0]
    assert c.area == 9.0
    assert c.peak_intensity == 200.0
    assert c.mean_intensity == 200.0
    assert c.bbox_x == 40
    assert c.bbox_y == 40
    assert c.bbox_width == 3
    assert c.bbox_height == 3


def test_source_image_immutability():
    detector = BeaconDetector()
    img = _create_gaussian_spot(cx=320.0, cy=240.0, peak=220.0)
    img_copy = img.copy()

    _ = detector.detect(img)

    np.testing.assert_array_equal(img, img_copy)


def test_identical_input_determinism():
    detector = BeaconDetector()
    img = _create_gaussian_spot(cx=320.0, cy=240.0, peak=220.0)

    cands1 = detector.detect(img)
    cands2 = detector.detect(img)

    assert len(cands1) == len(cands2)
    assert cands1[0] == cands2[0]


def test_no_ground_truth_leakage_architectural():
    # Verify BeaconDetector can be instantiated and used independently with raw NumPy image array
    detector = BeaconDetector()
    raw_image = np.zeros((480, 640), dtype=np.float32)
    raw_image[235:245, 315:325] = 255.0

    # No Scene, TargetState, CameraState, or SensorFrame passed or required
    detections = detector.detect(raw_image)
    assert isinstance(detections, list)
    assert len(detections) == 1


def test_invalid_input_rejection():
    detector = BeaconDetector()

    with pytest.raises(ValueError):
        detector.detect(None)  # None input

    with pytest.raises(ValueError):
        detector.detect([1, 2, 3])  # Non-NumPy input

    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10, 3)))  # 3D array

    with pytest.raises(ValueError):
        detector.detect(np.array([]))  # Empty array