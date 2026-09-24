"""Unit tests for core/tracking.py following docs/TEST_STRATEGY.md."""

import pytest

from core.tracking import (
    TrackedTargetState,
    TrackingConfig,
    TrackingObservation,
    TrackingState,
    TargetTracker,
)


def _make_obs(
    timestamp: float = 0.0,
    cx: float = 320.0,
    cy: float = 240.0,
    det_conf: float = 0.8,
    ai_conf: float = 0.8,
    ai_beacon: bool = True,
    mod_conf: float = 0.8,
    mod_ver: bool = True,
    cand_id: int = 1,
) -> TrackingObservation:
    """Helper to create valid TrackingObservation instances."""
    return TrackingObservation(
        timestamp=timestamp,
        centroid_x=cx,
        centroid_y=cy,
        detection_confidence=det_conf,
        ai_confidence=ai_conf,
        ai_is_beacon=ai_beacon,
        modulation_confidence=mod_conf,
        modulation_verified=mod_ver,
        candidate_id=cand_id,
    )


def test_independent_distance_thresholds():
    """Verify tracking_max_distance_px and reacquisition_max_distance_px function independently."""
    cfg = TrackingConfig(
        minimum_acquisition_observations=1,
        max_missed_frames=1,
        tracking_max_distance_px=20.0,
        reacquisition_max_distance_px=80.0,
    )
    tracker = TargetTracker(cfg)

    # 1. Establish track at (320, 240)
    tracker.update(_make_obs(timestamp=0.0, cx=320.0, cy=240.0))

    # 2. Candidate at dist 30.0 px (> tracking_max_distance_px 20.0) is treated as missed observation during TRACKING
    state_miss = tracker.update(_make_obs(timestamp=0.1, cx=350.0, cy=240.0))
    assert state_miss.position_x == 320.0
    assert state_miss.missed_frames == 1

    # 3. Transition to LOST
    state_lost = tracker.update(timestamp=0.2)
    assert state_lost.state == TrackingState.LOST

    # 4. Candidate at dist 70.0 px (> tracking 20.0, but <= reacquisition 80.0) is accepted for REACQUIRING
    state_reacq = tracker.update(_make_obs(timestamp=0.3, cx=390.0, cy=240.0))
    assert state_reacq.state == TrackingState.TRACKING
    assert state_reacq.position_x == 390.0


def test_timestamp_zero_initialization():
    """Regression test ensuring timestamp 0.0 calculates valid dt = 1.0 when updated at 1.0."""
    cfg = TrackingConfig(minimum_acquisition_observations=1, velocity_smoothing_factor=1.0)
    tracker = TargetTracker(cfg)

    obs0 = _make_obs(timestamp=0.0, cx=320.0, cy=240.0)
    state0 = tracker.update(obs0)
    assert state0.timestamp == 0.0
    assert state0.position_x == 320.0

    obs1 = _make_obs(timestamp=1.0, cx=330.0, cy=240.0)
    state1 = tracker.update(obs1)
    assert state1.timestamp == 1.0
    assert state1.position_x == 330.0
    assert state1.velocity_x == pytest.approx(10.0)


def test_normal_tracking_spatial_association():
    """Verify distant candidate beyond tracking_max_distance_px does not replace current track."""
    cfg = TrackingConfig(minimum_acquisition_observations=1, tracking_max_distance_px=50.0)
    tracker = TargetTracker(cfg)

    obs1 = _make_obs(timestamp=0.0, cx=320.0, cy=240.0)
    tracker.update(obs1)  # TRACKING at (320, 240)

    # Distant candidate at (600, 240) distance = 280 > 50 px threshold
    obs_distant = _make_obs(timestamp=0.1, cx=600.0, cy=240.0)
    state_after = tracker.update(obs_distant)

    assert state_after.position_x == 320.0
    assert state_after.missed_frames == 1


def test_ai_rejection():
    """Verify observation with ai_is_beacon=False is rejected even with high confidence."""
    cfg = TrackingConfig(minimum_acquisition_observations=1)
    tracker = TargetTracker(cfg)

    obs_rejected = _make_obs(
        timestamp=0.0,
        cx=320.0,
        cy=240.0,
        det_conf=1.0,
        ai_conf=1.0,
        ai_beacon=False,  # Explicit AI rejection
        mod_conf=1.0,
    )

    state = tracker.update(obs_rejected)
    assert state.state == TrackingState.SEARCHING
    assert state.position_x is None


def test_loss_confidence_threshold_transition():
    """Verify decay below loss_confidence_threshold causes transition to LOST."""
    cfg = TrackingConfig(
        minimum_acquisition_observations=1,
        max_missed_frames=10,
        loss_confidence_threshold=0.5,
    )
    tracker = TargetTracker(cfg)

    obs = _make_obs(timestamp=0.0, det_conf=0.6, ai_conf=0.6, mod_conf=0.6)
    state0 = tracker.update(obs)
    assert state0.confidence == pytest.approx(0.6)

    # First miss: confidence decays to 0.6 * 0.7 = 0.42 < loss_confidence_threshold (0.5)
    state_miss = tracker.update(timestamp=0.1)
    assert state_miss.state == TrackingState.LOST


def test_velocity_smoothing_and_clamping():
    cfg = TrackingConfig(
        minimum_acquisition_observations=1,
        velocity_smoothing_factor=0.5,
        tracking_max_distance_px=500.0,
        max_velocity_px_per_second=100.0,
    )
    tracker = TargetTracker(cfg)

    tracker.update(_make_obs(timestamp=0.0, cx=320.0, cy=240.0))

    state = tracker.update(_make_obs(timestamp=1.0, cx=720.0, cy=240.0))
    assert state.velocity_x == pytest.approx(100.0)


def test_zero_or_negative_dt_rejection():
    tracker = TargetTracker()
    tracker.update(_make_obs(timestamp=0.5))

    with pytest.raises(ValueError):
        tracker.update(_make_obs(timestamp=0.5))  # dt = 0.0

    with pytest.raises(ValueError):
        tracker.update(_make_obs(timestamp=0.4))  # dt < 0.0


def test_confidence_bounded_in_zero_one():
    tracker = TargetTracker()
    obs = _make_obs(det_conf=1.0, ai_conf=1.0, mod_conf=1.0)
    state = tracker.update(obs)
    assert 0.0 <= state.confidence <= 1.0


def test_candidate_id_does_not_determine_association():
    cfg = TrackingConfig(minimum_acquisition_observations=1)
    tracker = TargetTracker(cfg)

    tracker.update(_make_obs(timestamp=0.0, cx=320.0, cy=240.0, cand_id=1))
    state = tracker.update(_make_obs(timestamp=0.1, cx=322.0, cy=241.0, cand_id=99))
    assert state.state == TrackingState.TRACKING
    assert state.candidate_id == 99


def test_deterministic_repeated_sequence():
    obs_list = [
        _make_obs(timestamp=0.1 * i, cx=320.0 + i, cy=240.0) for i in range(5)
    ]

    tracker1 = TargetTracker()
    res1 = [tracker1.update(o) for o in obs_list]

    tracker2 = TargetTracker()
    res2 = [tracker2.update(o) for o in obs_list]

    assert res1 == res2


def test_reacquisition_success_and_distant_rejection():
    cfg = TrackingConfig(
        minimum_acquisition_observations=1,
        max_missed_frames=1,
        reacquisition_max_distance_px=30.0,
    )
    tracker = TargetTracker(cfg)

    tracker.update(_make_obs(timestamp=0.0, cx=320.0, cy=240.0))  # TRACKING
    tracker.update(timestamp=0.1)
    tracker.update(timestamp=0.2)  # LOST

    state_far = tracker.update(_make_obs(timestamp=0.3, cx=400.0, cy=240.0))
    assert state_far.state == TrackingState.LOST

    state_near = tracker.update(_make_obs(timestamp=0.4, cx=325.0, cy=240.0))
    assert state_near.state == TrackingState.TRACKING


def test_only_tracking_reports_is_locked():
    tracker = TargetTracker()

    assert tracker.update(timestamp=0.0).is_locked is False

    tracker.update(_make_obs(timestamp=0.1))  # ACQUIRING
    assert tracker.update(_make_obs(timestamp=0.2)).is_locked is True  # TRACKING


def test_input_observation_immutability():
    obs = _make_obs(timestamp=0.1)
    with pytest.raises(AttributeError):
        obs.centroid_x = 999.0


def test_ground_truth_isolation():
    tracker = TargetTracker()
    obs = TrackingObservation(
        timestamp=0.1,
        centroid_x=320.0,
        centroid_y=240.0,
        detection_confidence=0.9,
        ai_confidence=0.85,
        ai_is_beacon=True,
        modulation_confidence=0.95,
        modulation_verified=True,
    )
    res = tracker.update(obs)
    assert isinstance(res, TrackedTargetState)


def test_tracking_config_validation():
    with pytest.raises(ValueError):
        TrackingConfig(acquisition_confidence_threshold=1.5)

    with pytest.raises(ValueError):
        TrackingConfig(tracking_max_distance_px=-10.0)

    with pytest.raises(ValueError):
        TrackingConfig(reacquisition_max_distance_px=float("nan"))


def test_tracked_target_state_validation():
    with pytest.raises(ValueError):
        TrackedTargetState(
            state=TrackingState.SEARCHING,
            timestamp=-1.0,
            position_x=None,
            position_y=None,
            velocity_x=0.0,
            velocity_y=0.0,
            confidence=0.0,
            candidate_id=None,
            missed_frames=0,
            acquisition_count=0,
        )