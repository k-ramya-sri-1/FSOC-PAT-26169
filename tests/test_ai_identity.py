"""Unit tests for core/ai_identity.py following docs/TEST_STRATEGY.md."""

import numpy as np
import pytest

from core.ai_identity import (
    BeaconFeatures,
    BeaconIdentityClassifier,
    BeaconIdentityConfig,
    BeaconIdentityResult,
)


def _generate_synthetic_dataset(
    n_beacon: int = 50, n_non_beacon: int = 50, seed: int = 42
) -> tuple[list[BeaconFeatures], list[int]]:
    """Helper to generate synthetic observable feature samples for testing."""
    rng = np.random.default_rng(seed)
    features: list[BeaconFeatures] = []
    labels: list[int] = []

    # Class 1: Beacon-like (compact, circular, high intensity)
    for _ in range(n_beacon):
        area = rng.uniform(10.0, 30.0)
        width = rng.uniform(3.0, 6.0)
        height = rng.uniform(3.0, 6.0)
        aspect = width / height
        circularity = rng.uniform(0.8, 1.0)
        peak = rng.uniform(200.0, 255.0)
        mean = rng.uniform(150.0, 220.0)
        std = rng.uniform(10.0, 30.0)
        snr = rng.uniform(15.0, 30.0)

        features.append(
            BeaconFeatures(
                area=area,
                width=width,
                height=height,
                aspect_ratio=aspect,
                circularity=circularity,
                peak_intensity=peak,
                mean_intensity=mean,
                intensity_std=std,
                signal_to_noise_ratio=snr,
            )
        )
        labels.append(1)

    # Class 0: Non-beacon noise/clutter (elongated, low circularity, lower intensity)
    for _ in range(n_non_beacon):
        area = rng.uniform(50.0, 200.0)
        width = rng.uniform(15.0, 40.0)
        height = rng.uniform(2.0, 5.0)
        aspect = width / height
        circularity = rng.uniform(0.1, 0.4)
        peak = rng.uniform(50.0, 120.0)
        mean = rng.uniform(20.0, 80.0)
        std = rng.uniform(1.0, 10.0)
        snr = rng.uniform(1.0, 5.0)

        features.append(
            BeaconFeatures(
                area=area,
                width=width,
                height=height,
                aspect_ratio=aspect,
                circularity=circularity,
                peak_intensity=peak,
                mean_intensity=mean,
                intensity_std=std,
                signal_to_noise_ratio=snr,
            )
        )
        labels.append(0)

    return features, labels


def test_config_validation():
    with pytest.raises(ValueError):
        BeaconIdentityConfig(learning_rate=0.0)

    with pytest.raises(ValueError):
        BeaconIdentityConfig(epochs=0)

    with pytest.raises(ValueError):
        BeaconIdentityConfig(regularization_strength=-0.1)

    with pytest.raises(ValueError):
        BeaconIdentityConfig(classification_threshold=1.5)


def test_beacon_features_validation():
    with pytest.raises(ValueError):
        BeaconFeatures(
            area=0.0,
            width=5.0,
            height=5.0,
            aspect_ratio=1.0,
            circularity=0.9,
            peak_intensity=200.0,
            mean_intensity=150.0,
        )


def test_beacon_features_nan_rejection():
    with pytest.raises(ValueError):
        BeaconFeatures(
            area=float("nan"),
            width=5.0,
            height=5.0,
            aspect_ratio=1.0,
            circularity=0.9,
            peak_intensity=200.0,
            mean_intensity=150.0,
        )


def test_beacon_features_infinite_rejection():
    with pytest.raises(ValueError):
        BeaconFeatures(
            area=10.0,
            width=float("inf"),
            height=5.0,
            aspect_ratio=1.0,
            circularity=0.9,
            peak_intensity=200.0,
            mean_intensity=150.0,
        )


def test_beacon_features_circularity_bounds_rejection():
    # Circularity > 1.0 rejection
    with pytest.raises(ValueError):
        BeaconFeatures(
            area=10.0,
            width=5.0,
            height=5.0,
            aspect_ratio=1.0,
            circularity=1.1,
            peak_intensity=200.0,
            mean_intensity=150.0,
        )

    # Circularity < 0.0 rejection
    with pytest.raises(ValueError):
        BeaconFeatures(
            area=10.0,
            width=5.0,
            height=5.0,
            aspect_ratio=1.0,
            circularity=-0.1,
            peak_intensity=200.0,
            mean_intensity=150.0,
        )


def test_feature_vector_construction():
    feat = BeaconFeatures(
        area=10.0,
        width=3.0,
        height=3.0,
        aspect_ratio=1.0,
        circularity=0.9,
        peak_intensity=200.0,
        mean_intensity=150.0,
        intensity_std=15.0,
        signal_to_noise_ratio=20.0,
    )
    arr = feat.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (9,)
    assert arr[0] == 10.0
    assert arr[4] == 0.9


def test_empty_training_data_rejected():
    clf = BeaconIdentityClassifier()
    with pytest.raises(ValueError):
        clf.fit([], [])


def test_invalid_labels_rejected():
    clf = BeaconIdentityClassifier()
    feats, _ = _generate_synthetic_dataset(10, 10)
    invalid_labels = [0] * 10 + [2] * 10  # Label 2 is invalid

    with pytest.raises(ValueError):
        clf.fit(feats, invalid_labels)


def test_one_class_training_rejected():
    clf = BeaconIdentityClassifier()
    feats, _ = _generate_synthetic_dataset(10, 0)
    single_class_labels = [1] * 10

    with pytest.raises(ValueError):
        clf.fit(feats, single_class_labels)


def test_training_and_classification_separable_data():
    X_train, y_train = _generate_synthetic_dataset(50, 50, seed=100)
    clf = BeaconIdentityClassifier(BeaconIdentityConfig(epochs=1000, learning_rate=0.1))
    clf.fit(X_train, y_train)

    # Class 1 test sample
    beacon_sample = BeaconFeatures(
        area=15.0,
        width=4.0,
        height=4.0,
        aspect_ratio=1.0,
        circularity=0.95,
        peak_intensity=240.0,
        mean_intensity=180.0,
        intensity_std=20.0,
        signal_to_noise_ratio=25.0,
    )

    res = clf.classify(beacon_sample)
    assert isinstance(res, BeaconIdentityResult)
    assert res.is_beacon is True
    assert res.predicted_class == 1
    assert res.probability > 0.5
    assert 0.0 <= res.confidence <= 1.0

    # Class 0 test sample
    non_beacon_sample = BeaconFeatures(
        area=120.0,
        width=30.0,
        height=4.0,
        aspect_ratio=7.5,
        circularity=0.2,
        peak_intensity=80.0,
        mean_intensity=40.0,
        intensity_std=5.0,
        signal_to_noise_ratio=2.0,
    )

    res_non = clf.classify(non_beacon_sample)
    assert res_non.is_beacon is False
    assert res_non.predicted_class == 0
    assert res_non.probability < 0.5


def test_zero_variance_features_handled_safely():
    # Dataset where feature index 0 (area) has zero variance across all samples
    X = np.ones((20, 9), dtype=np.float64)
    X[:10, 1] = 10.0  # Vary second feature
    X[10:, 1] = 1.0
    y = np.array([1] * 10 + [0] * 10)

    clf = BeaconIdentityClassifier()
    clf.fit(X, y)

    prob = clf.predict_proba(X[0])
    assert not np.isnan(prob).any()


def test_nan_infinite_features_rejected():
    clf = BeaconIdentityClassifier()
    X = np.array([[1.0, np.nan, 3.0, 1.0, 0.9, 200.0, 150.0, 10.0, 20.0]])
    y = np.array([1])

    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_dimension_mismatch_rejected():
    X_train, y_train = _generate_synthetic_dataset(10, 10)
    clf = BeaconIdentityClassifier()
    clf.fit(X_train, y_train)

    wrong_dim_features = np.ones((1, 5))
    with pytest.raises(ValueError):
        clf.predict(wrong_dim_features)


def test_input_features_not_mutated():
    X_train, y_train = _generate_synthetic_dataset(10, 10)
    clf = BeaconIdentityClassifier()
    clf.fit(X_train, y_train)

    test_feat = BeaconFeatures(
        area=15.0,
        width=4.0,
        height=4.0,
        aspect_ratio=1.0,
        circularity=0.9,
        peak_intensity=220.0,
        mean_intensity=160.0,
    )

    with pytest.raises(AttributeError):
        test_feat.area = 99.0


def test_classification_threshold_behavior():
    X_train, y_train = _generate_synthetic_dataset(30, 30)

    cfg_high = BeaconIdentityConfig(classification_threshold=0.99)
    clf_high = BeaconIdentityClassifier(cfg_high)
    clf_high.fit(X_train, y_train)

    sample = X_train[0]
    prob = clf_high.predict_proba(sample)[0]

    # If probability is < 0.99, high threshold will classify as 0
    pred = clf_high.predict(sample)[0]
    if prob < 0.99:
        assert pred == 0


def test_determinism():
    X_train, y_train = _generate_synthetic_dataset(20, 20, seed=777)

    clf1 = BeaconIdentityClassifier()
    clf1.fit(X_train, y_train)

    clf2 = BeaconIdentityClassifier()
    clf2.fit(X_train, y_train)

    np.testing.assert_array_equal(clf1.weights, clf2.weights)
    assert clf1.bias == clf2.bias


def test_ground_truth_isolation():
    # Verify BeaconIdentityClassifier works exclusively on BeaconFeatures and NumPy arrays
    X_train, y_train = _generate_synthetic_dataset(20, 20)

    clf = BeaconIdentityClassifier()
    clf.fit(X_train, y_train)

    feat = BeaconFeatures(
        area=15.0,
        width=4.0,
        height=4.0,
        aspect_ratio=1.0,
        circularity=0.9,
        peak_intensity=220.0,
        mean_intensity=160.0,
    )

    # No Scene, TargetState, CameraState, or SensorFrame passed or imported
    res = clf.classify(feat)
    assert isinstance(res, BeaconIdentityResult)


def test_candidate_id_not_used_as_feature():
    # Two identical BeaconFeatures instances must produce identical predictions regardless of external stream ID
    X_train, y_train = _generate_synthetic_dataset(20, 20)
    clf = BeaconIdentityClassifier().fit(X_train, y_train)

    feat1 = BeaconFeatures(
        area=15.0,
        width=4.0,
        height=4.0,
        aspect_ratio=1.0,
        circularity=0.9,
        peak_intensity=220.0,
        mean_intensity=160.0,
    )
    feat2 = BeaconFeatures(
        area=15.0,
        width=4.0,
        height=4.0,
        aspect_ratio=1.0,
        circularity=0.9,
        peak_intensity=220.0,
        mean_intensity=160.0,
    )

    res1 = clf.classify(feat1)
    res2 = clf.classify(feat2)

    assert res1.probability == res2.probability
    assert res1.is_beacon == res2.is_beacon