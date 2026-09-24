"""Unit tests for core/prediction.py following docs/TEST_STRATEGY.md."""

import math
import pytest

from core.prediction import (
    PredictionResult,
    PredictionState,
    TargetPositionPredictor,
)


def test_prediction_state_valid_construction():
    state = PredictionState(
        timestamp=10.0,
        position_x=100.0,
        position_y=200.0,
        velocity_x=5.0,
        velocity_y=-2.0,
    )
    assert state.timestamp == 10.0
    assert state.position_x == 100.0
    assert state.position_y == 200.0
    assert state.velocity_x == 5.0
    assert state.velocity_y == -2.0


def test_prediction_state_validations():
    # Negative timestamp
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=-1.0,
            position_x=0.0,
            position_y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
        )

    # NaN timestamp
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=float("nan"),
            position_x=0.0,
            position_y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
        )

    # Infinite timestamp
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=float("inf"),
            position_x=0.0,
            position_y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
        )

    # NaN position_x
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=0.0,
            position_x=float("nan"),
            position_y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
        )

    # NaN position_y
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=0.0,
            position_x=0.0,
            position_y=float("nan"),
            velocity_x=0.0,
            velocity_y=0.0,
        )

    # Infinite position
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=0.0,
            position_x=float("inf"),
            position_y=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
        )

    # NaN velocity
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=0.0,
            position_x=0.0,
            position_y=0.0,
            velocity_x=float("nan"),
            velocity_y=0.0,
        )

    # Infinite velocity_y
    with pytest.raises(ValueError):
        PredictionState(
            timestamp=0.0,
            position_x=0.0,
            position_y=0.0,
            velocity_x=0.0,
            velocity_y=float("-inf"),
        )


def test_prediction_result_validations():
    # Invalid/NaN field
    with pytest.raises(ValueError):
        PredictionResult(
            source_timestamp=1.0,
            prediction_horizon=2.0,
            predicted_timestamp=3.0,
            predicted_position_x=float("nan"),
            predicted_position_y=10.0,
        )

    # Negative source_timestamp
    with pytest.raises(ValueError):
        PredictionResult(
            source_timestamp=-1.0,
            prediction_horizon=2.0,
            predicted_timestamp=1.0,
            predicted_position_x=10.0,
            predicted_position_y=10.0,
        )

    # Negative prediction_horizon
    with pytest.raises(ValueError):
        PredictionResult(
            source_timestamp=1.0,
            prediction_horizon=-0.5,
            predicted_timestamp=0.5,
            predicted_position_x=10.0,
            predicted_position_y=10.0,
        )

    # Negative predicted_timestamp
    with pytest.raises(ValueError):
        PredictionResult(
            source_timestamp=1.0,
            prediction_horizon=1.0,
            predicted_timestamp=-0.1,
            predicted_position_x=10.0,
            predicted_position_y=10.0,
        )


def test_prediction_immutability():
    state = PredictionState(
        timestamp=1.0,
        position_x=10.0,
        position_y=20.0,
        velocity_x=1.0,
        velocity_y=1.0,
    )
    with pytest.raises(AttributeError):
        state.position_x = 50.0

    res = PredictionResult(
        source_timestamp=1.0,
        prediction_horizon=2.0,
        predicted_timestamp=3.0,
        predicted_position_x=12.0,
        predicted_position_y=22.0,
    )
    with pytest.raises(AttributeError):
        res.predicted_position_x = 99.0


def test_basic_constant_velocity_prediction():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=10.0,
        position_x=100.0,
        position_y=200.0,
        velocity_x=5.0,
        velocity_y=-2.0,
    )

    result = predictor.predict(state, prediction_horizon=2.0)

    assert result.source_timestamp == 10.0
    assert result.prediction_horizon == 2.0
    assert result.predicted_timestamp == 12.0
    assert result.predicted_position_x == 110.0
    assert result.predicted_position_y == 196.0


def test_zero_horizon_prediction():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=5.0,
        position_x=50.0,
        position_y=75.0,
        velocity_x=12.0,
        velocity_y=-8.0,
    )

    result = predictor.predict(state, prediction_horizon=0.0)

    assert result.source_timestamp == 5.0
    assert result.prediction_horizon == 0.0
    assert result.predicted_timestamp == 5.0
    assert result.predicted_position_x == 50.0
    assert result.predicted_position_y == 75.0


def test_invalid_horizon_rejections():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=0.0,
        position_x=0.0,
        position_y=0.0,
        velocity_x=1.0,
        velocity_y=1.0,
    )

    # Negative horizon
    with pytest.raises(ValueError):
        predictor.predict(state, -1.0)

    # NaN horizon
    with pytest.raises(ValueError):
        predictor.predict(state, float("nan"))

    # Infinite horizon
    with pytest.raises(ValueError):
        predictor.predict(state, float("inf"))


def test_predictor_rejects_non_prediction_state_input():
    predictor = TargetPositionPredictor()
    with pytest.raises(ValueError):
        predictor.predict({"position_x": 10.0}, 1.0)  # type: ignore


def test_stationary_target():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=0.0,
        position_x=320.0,
        position_y=240.0,
        velocity_x=0.0,
        velocity_y=0.0,
    )

    result = predictor.predict(state, prediction_horizon=100.0)

    assert result.predicted_timestamp == 100.0
    assert result.predicted_position_x == 320.0
    assert result.predicted_position_y == 240.0


def test_negative_positions_and_velocities():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=1.0,
        position_x=-100.0,
        position_y=-200.0,
        velocity_x=-15.0,
        velocity_y=25.0,
    )

    result = predictor.predict(state, prediction_horizon=2.0)

    assert result.predicted_timestamp == 3.0
    assert result.predicted_position_x == -130.0
    assert result.predicted_position_y == -150.0


def test_fractional_position_velocity_and_horizon():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=1.5,
        position_x=10.25,
        position_y=20.75,
        velocity_x=-2.5,
        velocity_y=0.5,
    )

    result = predictor.predict(state, prediction_horizon=0.25)

    assert result.predicted_timestamp == 1.75
    assert result.predicted_position_x == pytest.approx(9.625)
    assert result.predicted_position_y == pytest.approx(20.875)


def test_large_finite_values():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=0.0,
        position_x=1e6,
        position_y=1e6,
        velocity_x=1e4,
        velocity_y=-1e4,
    )

    result = predictor.predict(state, prediction_horizon=100.0)

    assert math.isfinite(result.predicted_position_x)
    assert math.isfinite(result.predicted_position_y)
    assert result.predicted_position_x == 2e6
    assert result.predicted_position_y == 0.0


def test_input_state_immutability_during_prediction():
    predictor = TargetPositionPredictor()
    state = PredictionState(
        timestamp=1.0,
        position_x=10.0,
        position_y=20.0,
        velocity_x=2.0,
        velocity_y=3.0,
    )

    result = predictor.predict(state, prediction_horizon=5.0)

    assert state.timestamp == 1.0
    assert state.position_x == 10.0
    assert state.position_y == 20.0
    assert state.velocity_x == 2.0
    assert state.velocity_y == 3.0
    assert result.predicted_position_x == 20.0


def test_determinism_and_reusability():
    predictor = TargetPositionPredictor()
    state1 = PredictionState(
        timestamp=2.5,
        position_x=100.0,
        position_y=50.0,
        velocity_x=-3.0,
        velocity_y=4.0,
    )
    state2 = PredictionState(
        timestamp=0.0,
        position_x=0.0,
        position_y=0.0,
        velocity_x=10.0,
        velocity_y=10.0,
    )

    # Repeated calls on same instance are deterministic
    res1_a = predictor.predict(state1, prediction_horizon=1.5)
    res1_b = predictor.predict(state1, prediction_horizon=1.5)
    assert res1_a == res1_b

    # Reuse same instance for an independent second prediction
    res2 = predictor.predict(state2, prediction_horizon=3.0)
    assert res2.predicted_position_x == 30.0
    assert res2.predicted_position_y == 30.0


def test_exact_constant_velocity_kinematics():
    predictor = TargetPositionPredictor()
    x, y, vx, vy, dt = 12.3, -45.6, 7.89, -0.12, 3.45
    state = PredictionState(
        timestamp=10.0,
        position_x=x,
        position_y=y,
        velocity_x=vx,
        velocity_y=vy,
    )

    res = predictor.predict(state, dt)

    assert res.predicted_position_x == x + vx * dt
    assert res.predicted_position_y == y + vy * dt


def test_prediction_is_independently_usable():
    state = PredictionState(
        timestamp=0.0,
        position_x=10.0,
        position_y=20.0,
        velocity_x=2.0,
        velocity_y=-1.0,
    )
    predictor = TargetPositionPredictor()
    res = predictor.predict(state, 5.0)

    assert isinstance(res, PredictionResult)
    assert res.predicted_position_x == 20.0
    assert res.predicted_position_y == 15.0