"""Unit tests for core/kalman.py following docs/TEST_STRATEGY.md."""

import math
import numpy as np
import pytest

from core.kalman import KalmanConfig, KalmanFilter2D, KalmanState


def test_kalman_config_defaults():
    cfg = KalmanConfig()
    assert cfg.process_noise == 1.0
    assert cfg.measurement_noise == 4.0
    assert cfg.initial_position_variance == 100.0
    assert cfg.initial_velocity_variance == 100.0
    assert cfg.max_dt == 1.0


def test_kalman_config_validation():
    with pytest.raises(ValueError):
        KalmanConfig(process_noise=-1.0)

    with pytest.raises(ValueError):
        KalmanConfig(measurement_noise=float("nan"))

    with pytest.raises(ValueError):
        KalmanConfig(max_dt=float("inf"))

    with pytest.raises(ValueError):
        KalmanConfig(initial_position_variance=0.0)


def test_kalman_state_immutability_and_validation():
    state = KalmanState(
        timestamp=1.0, position_x=10.0, position_y=20.0, velocity_x=1.0, velocity_y=2.0
    )
    with pytest.raises(AttributeError):
        state.position_x = 99.0  # Frozen dataclass

    with pytest.raises(ValueError):
        KalmanState(
            timestamp=-1.0,
            position_x=10.0,
            position_y=20.0,
            velocity_x=1.0,
            velocity_y=2.0,
        )

    with pytest.raises(ValueError):
        KalmanState(
            timestamp=1.0,
            position_x=float("nan"),
            position_y=20.0,
            velocity_x=1.0,
            velocity_y=2.0,
        )


def test_uninitialized_filter_behavior():
    kf = KalmanFilter2D()
    assert kf.get_state() is None

    with pytest.raises(ValueError):
        kf.predict(1.0)

    with pytest.raises(ValueError):
        kf.update(1.0, 100.0, 200.0)

    with pytest.raises(ValueError):
        kf.predict_and_update(1.0, 100.0, 200.0)


def test_initialization_at_timestamp_zero():
    kf = KalmanFilter2D()
    state = kf.initialize(
        timestamp=0.0,
        position_x=100.0,
        position_y=200.0,
        velocity_x=10.0,
        velocity_y=-5.0,
    )

    assert state.timestamp == 0.0
    assert state.position_x == 100.0
    assert state.position_y == 200.0
    assert state.velocity_x == 10.0
    assert state.velocity_y == -5.0
    assert kf.get_state() == state


def test_explicit_reinitialization():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=1.0, position_x=10.0, position_y=20.0)

    # Reinitialize at new timestamp and state
    state2 = kf.initialize(timestamp=5.0, position_x=50.0, position_y=60.0)
    assert state2.timestamp == 5.0
    assert state2.position_x == 50.0
    assert state2.position_y == 60.0


def test_reset():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=0.0, position_x=10.0, position_y=10.0)
    assert kf.get_state() is not None

    kf.reset()
    assert kf.get_state() is None

    with pytest.raises(ValueError):
        kf.predict(1.0)

    with pytest.raises(ValueError):
        kf.update(1.0, 10.0, 10.0)

    with pytest.raises(ValueError):
        kf.predict_and_update(1.0, 10.0, 10.0)


def test_constant_velocity_prediction():
    kf = KalmanFilter2D()
    kf.initialize(
        timestamp=0.0,
        position_x=100.0,
        position_y=200.0,
        velocity_x=10.0,
        velocity_y=-5.0,
    )

    state_pred = kf.predict(1.0)
    assert state_pred.timestamp == 1.0
    assert state_pred.position_x == pytest.approx(110.0)
    assert state_pred.position_y == pytest.approx(195.0)
    assert state_pred.velocity_x == pytest.approx(10.0)
    assert state_pred.velocity_y == pytest.approx(-5.0)


def test_measurement_correction_and_smoothing():
    cfg = KalmanConfig(process_noise=0.1, measurement_noise=10.0)
    kf = KalmanFilter2D(cfg)
    kf.initialize(timestamp=0.0, position_x=0.0, position_y=0.0)

    state = kf.update(1.0, 10.0, 0.0)

    # Corrected position should lie between prediction (0) and measurement (10)
    assert 0.0 < state.position_x < 10.0
    assert state.position_y == pytest.approx(0.0)


def test_max_dt_clamping_preserves_timestamp():
    cfg = KalmanConfig(max_dt=1.0)
    kf = KalmanFilter2D(cfg)
    kf.initialize(
        timestamp=0.0,
        position_x=0.0,
        position_y=0.0,
        velocity_x=10.0,
        velocity_y=0.0,
    )

    # Step dt = 5.0, clamped to max_dt = 1.0 for propagation
    state = kf.predict(5.0)
    assert state.timestamp == 5.0
    assert state.position_x == pytest.approx(10.0)  # 0 + 10 * 1.0


def test_update_equal_timestamp_rejection():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=1.0, position_x=10.0, position_y=20.0)

    with pytest.raises(ValueError):
        kf.update(1.0, 10.0, 20.0)


def test_update_backward_timestamp_rejection():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=1.0, position_x=10.0, position_y=20.0)

    with pytest.raises(ValueError):
        kf.update(0.5, 10.0, 20.0)


def test_update_invalid_measurement_rejection():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=1.0, position_x=10.0, position_y=20.0)

    # NaN measurement rejection
    with pytest.raises(ValueError):
        kf.update(2.0, float("nan"), 20.0)

    # Infinite measurement rejection
    with pytest.raises(ValueError):
        kf.update(2.0, 10.0, float("inf"))


def test_predict_and_update_timestamp_rejections():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=1.0, position_x=10.0, position_y=20.0)

    # Equal timestamp rejection
    with pytest.raises(ValueError):
        kf.predict_and_update(1.0, 10.0, 20.0)

    # Backward timestamp rejection
    with pytest.raises(ValueError):
        kf.predict_and_update(0.5, 10.0, 20.0)


def test_predict_and_update_equivalence():
    cfg = KalmanConfig()

    kf_a = KalmanFilter2D(cfg)
    kf_a.initialize(timestamp=0.0, position_x=0.0, position_y=0.0)
    s_a = kf_a.predict_and_update(1.0, 10.0, 10.0)

    kf_b = KalmanFilter2D(cfg)
    kf_b.initialize(timestamp=0.0, position_x=0.0, position_y=0.0)
    s_b = kf_b.update(1.0, 10.0, 10.0)

    assert s_a.timestamp == s_b.timestamp
    assert s_a.position_x == pytest.approx(s_b.position_x)
    assert s_a.position_y == pytest.approx(s_b.position_y)
    assert s_a.velocity_x == pytest.approx(s_b.velocity_x)
    assert s_a.velocity_y == pytest.approx(s_b.velocity_y)


def test_repeated_updates_maintain_finite_state_and_symmetry():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=0.0, position_x=0.0, position_y=0.0)

    for i in range(1, 20):
        state = kf.predict_and_update(float(i), float(i * 10), float(i * 5))
        assert math.isfinite(state.position_x)
        assert math.isfinite(state.position_y)
        assert math.isfinite(state.velocity_x)
        assert math.isfinite(state.velocity_y)

    # Verify covariance symmetry
    assert np.allclose(kf._P, kf._P.T)


def test_deterministic_repeated_sequence():
    cfg = KalmanConfig()

    kf1 = KalmanFilter2D(cfg)
    kf1.initialize(timestamp=0.0, position_x=0.0, position_y=0.0)
    seq1 = [
        kf1.predict_and_update(float(i), float(i * 5), float(i * 2))
        for i in range(1, 10)
    ]

    kf2 = KalmanFilter2D(cfg)
    kf2.initialize(timestamp=0.0, position_x=0.0, position_y=0.0)
    seq2 = [
        kf2.predict_and_update(float(i), float(i * 5), float(i * 2))
        for i in range(1, 10)
    ]

    assert seq1 == seq2


def test_ground_truth_isolation():
    kf = KalmanFilter2D()
    kf.initialize(timestamp=0.0, position_x=10.0, position_y=20.0)
    state = kf.predict_and_update(1.0, 15.0, 22.0)
    assert isinstance(state, KalmanState)