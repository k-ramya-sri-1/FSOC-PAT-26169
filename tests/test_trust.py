"""Unit tests for core/trust.py following docs/TEST_STRATEGY.md."""

import math
import pytest

from core.trust import (
    ModelVisionTrustEvaluator,
    TrustConfig,
    TrustObservation,
    TrustResult,
)


def test_valid_trust_config_defaults():
    cfg = TrustConfig()
    assert cfg.full_trust_distance_px == 5.0
    assert cfg.zero_trust_distance_px == 50.0


def test_trust_config_immutability():
    cfg = TrustConfig()
    with pytest.raises(AttributeError):
        cfg.full_trust_distance_px = 10.0  # type: ignore


def test_trust_config_rejections():
    # Negative full-trust distance
    with pytest.raises(ValueError):
        TrustConfig(full_trust_distance_px=-1.0)

    # Negative zero-trust distance
    with pytest.raises(ValueError):
        TrustConfig(zero_trust_distance_px=-10.0)

    # Zero-trust distance equal to 0
    with pytest.raises(ValueError):
        TrustConfig(zero_trust_distance_px=0.0)

    # Equal thresholds
    with pytest.raises(ValueError):
        TrustConfig(full_trust_distance_px=10.0, zero_trust_distance_px=10.0)

    # Reversed thresholds
    with pytest.raises(ValueError):
        TrustConfig(full_trust_distance_px=20.0, zero_trust_distance_px=10.0)

    # NaN threshold
    with pytest.raises(ValueError):
        TrustConfig(full_trust_distance_px=float("nan"))

    # Infinite threshold
    with pytest.raises(ValueError):
        TrustConfig(zero_trust_distance_px=float("inf"))


def test_valid_trust_observation_and_immutability():
    obs = TrustObservation(
        observed_x=100.0,
        observed_y=200.0,
        predicted_x=105.0,
        predicted_y=204.0,
    )
    assert obs.observed_x == 100.0
    assert obs.predicted_y == 204.0

    with pytest.raises(AttributeError):
        obs.observed_x = 0.0  # type: ignore


def test_trust_observation_rejections():
    # NaN observed_x
    with pytest.raises(ValueError):
        TrustObservation(
            observed_x=float("nan"),
            observed_y=0.0,
            predicted_x=0.0,
            predicted_y=0.0,
        )

    # NaN observed_y
    with pytest.raises(ValueError):
        TrustObservation(
            observed_x=0.0,
            observed_y=float("nan"),
            predicted_x=0.0,
            predicted_y=0.0,
        )

    # Infinite predicted_x
    with pytest.raises(ValueError):
        TrustObservation(
            observed_x=0.0,
            observed_y=0.0,
            predicted_x=float("inf"),
            predicted_y=0.0,
        )

    # Infinite predicted_y
    with pytest.raises(ValueError):
        TrustObservation(
            observed_x=0.0,
            observed_y=0.0,
            predicted_x=0.0,
            predicted_y=float("-inf"),
        )


def test_negative_coordinates_accepted():
    obs = TrustObservation(
        observed_x=-100.0,
        observed_y=-200.0,
        predicted_x=-103.0,
        predicted_y=-204.0,
    )
    evaluator = ModelVisionTrustEvaluator()
    res = evaluator.evaluate(obs)
    assert res.disagreement_px == pytest.approx(5.0)
    assert res.trust_score == 1.0
    assert res.model_preferred is False


def test_zero_distance_case():
    evaluator = ModelVisionTrustEvaluator()
    obs = TrustObservation(
        observed_x=100.0,
        observed_y=200.0,
        predicted_x=100.0,
        predicted_y=200.0,
    )
    res = evaluator.evaluate(obs)
    assert res.disagreement_px == 0.0
    assert res.trust_score == 1.0
    assert res.model_preferred is False


def test_full_trust_boundary():
    evaluator = ModelVisionTrustEvaluator()
    # Distance = 5.0 (dx=3, dy=4 -> hypot=5)
    obs = TrustObservation(
        observed_x=100.0,
        observed_y=200.0,
        predicted_x=103.0,
        predicted_y=204.0,
    )
    res = evaluator.evaluate(obs)
    assert res.disagreement_px == pytest.approx(5.0)
    assert res.trust_score == 1.0
    assert res.model_preferred is False


def test_zero_trust_boundary():
    evaluator = ModelVisionTrustEvaluator()
    # Distance = 50.0 (dx=50, dy=0 -> hypot=50)
    obs = TrustObservation(
        observed_x=100.0,
        observed_y=200.0,
        predicted_x=150.0,
        predicted_y=200.0,
    )
    res = evaluator.evaluate(obs)
    assert res.disagreement_px == pytest.approx(50.0)
    assert res.trust_score == 0.0
    assert res.model_preferred is True


def test_midpoint_linear_trust_calculation():
    evaluator = ModelVisionTrustEvaluator()
    # Distance = 27.5 (midpoint between 5 and 50)
    obs = TrustObservation(
        observed_x=100.0,
        observed_y=200.0,
        predicted_x=127.5,
        predicted_y=200.0,
    )
    res = evaluator.evaluate(obs)
    assert res.disagreement_px == pytest.approx(27.5)
    assert res.trust_score == pytest.approx(0.5)
    assert res.model_preferred is True


def test_known_3_4_5_distance():
    evaluator = ModelVisionTrustEvaluator()
    # Distance = 5.0
    obs = TrustObservation(
        observed_x=10.0,
        observed_y=20.0,
        predicted_x=13.0,
        predicted_y=24.0,
    )
    res = evaluator.evaluate(obs)
    assert res.disagreement_px == pytest.approx(5.0)


def test_known_arbitrary_distance():
    evaluator = ModelVisionTrustEvaluator()
    # Distance = sqrt(10^2 + 20^2) = sqrt(500) ≈ 22.360679775
    obs = TrustObservation(
        observed_x=0.0,
        observed_y=0.0,
        predicted_x=10.0,
        predicted_y=20.0,
    )
    res = evaluator.evaluate(obs)
    expected_dist = math.hypot(10.0, 20.0)
    expected_trust = 1.0 - (expected_dist - 5.0) / (50.0 - 5.0)

    assert res.disagreement_px == pytest.approx(expected_dist)
    assert res.trust_score == pytest.approx(expected_trust)
    assert res.model_preferred is True


def test_model_preferred_flag_thresholds():
    evaluator = ModelVisionTrustEvaluator()

    # Inside full-trust region (dist = 4.0 <= 5.0) -> model_preferred = False
    obs_inside = TrustObservation(
        observed_x=100.0, observed_y=200.0, predicted_x=104.0, predicted_y=200.0
    )
    assert evaluator.evaluate(obs_inside).model_preferred is False

    # Outside full-trust region (dist = 5.1 > 5.0) -> model_preferred = True
    obs_outside = TrustObservation(
        observed_x=100.0, observed_y=200.0, predicted_x=105.1, predicted_y=200.0
    )
    assert evaluator.evaluate(obs_outside).model_preferred is True


def test_determinism_and_evaluator_reuse():
    evaluator = ModelVisionTrustEvaluator()
    obs1 = TrustObservation(
        observed_x=100.0, observed_y=200.0, predicted_x=110.0, predicted_y=200.0
    )
    obs2 = TrustObservation(
        observed_x=0.0, observed_y=0.0, predicted_x=100.0, predicted_y=0.0
    )

    # Repeated calls on same instance produce identical results
    res1_a = evaluator.evaluate(obs1)
    res1_b = evaluator.evaluate(obs1)
    assert res1_a == res1_b

    # Evaluator reuse for independent evaluation
    res2 = evaluator.evaluate(obs2)
    assert res2.disagreement_px == pytest.approx(100.0)
    assert res2.trust_score == 0.0


def test_different_valid_configurations():
    cfg = TrustConfig(full_trust_distance_px=10.0, zero_trust_distance_px=100.0)
    evaluator = ModelVisionTrustEvaluator(cfg)

    # Distance = 10.0 is full trust under custom config
    obs = TrustObservation(
        observed_x=0.0, observed_y=0.0, predicted_x=10.0, predicted_y=0.0
    )
    res = evaluator.evaluate(obs)
    assert res.disagreement_px == 10.0
    assert res.trust_score == 1.0
    assert res.model_preferred is False


def test_result_immutability():
    evaluator = ModelVisionTrustEvaluator()
    obs = TrustObservation(
        observed_x=0.0, observed_y=0.0, predicted_x=1.0, predicted_y=0.0
    )
    res = evaluator.evaluate(obs)

    with pytest.raises(AttributeError):
        res.trust_score = 0.5  # type: ignore


def test_result_values_finiteness_and_trust_score_bounds():
    evaluator = ModelVisionTrustEvaluator()
    coords = [0.0, 10.0, 50.0, 100.0, 1000.0]

    for c in coords:
        obs = TrustObservation(
            observed_x=0.0, observed_y=0.0, predicted_x=c, predicted_y=c
        )
        res = evaluator.evaluate(obs)
        assert math.isfinite(res.disagreement_px)
        assert math.isfinite(res.trust_score)
        assert 0.0 <= res.trust_score <= 1.0


def test_evaluator_rejects_non_trust_observation_input():
    evaluator = ModelVisionTrustEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate({"observed_x": 0.0})  # type: ignore