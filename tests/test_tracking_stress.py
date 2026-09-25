from dataclasses import FrozenInstanceError

import pytest

from benchmark.tracking_stress import TrackingStressHarness
from core.tracking import TrackingState


def test_temporary_detection_loss_is_recorded_without_losing_lock():
    result = TrackingStressHarness().run_temporary_detection_loss()

    assert result.scenario_name == "temporary_detection_loss"
    assert tuple(frame.observation_supplied for frame in result.frames) == (
        True,
        True,
        True,
        False,
        False,
        True,
    )
    assert tuple(frame.tracking.state for frame in result.frames) == (
        TrackingState.ACQUIRING,
        TrackingState.TRACKING,
        TrackingState.TRACKING,
        TrackingState.TRACKING,
        TrackingState.TRACKING,
        TrackingState.TRACKING,
    )
    assert result.frames[3].tracking.missed_frames == 1
    assert result.frames[4].tracking.missed_frames == 2
    assert result.frames[5].tracking.missed_frames == 0
    assert all(frame.kalman is not None for frame in result.frames)
    assert all(frame.prediction is not None for frame in result.frames)


def test_extended_loss_records_actual_lost_transition_frame():
    result = TrackingStressHarness().run_extended_detection_loss()

    lost_frames = [frame for frame in result.frames if frame.tracking.state == TrackingState.LOST]
    assert lost_frames
    assert lost_frames[0].frame_index == 5
    assert lost_frames[0].tracking.missed_frames == 4
    assert result.frames[4].tracking.state == TrackingState.TRACKING
    assert result.frames[5].tracking.state == TrackingState.LOST
    assert result.frames[5].prediction is not None


def test_reacquisition_records_existing_state_sequence():
    result = TrackingStressHarness().run_reacquisition()

    assert tuple(frame.tracking.state for frame in result.frames) == (
        TrackingState.ACQUIRING,
        TrackingState.TRACKING,
        TrackingState.TRACKING,
        TrackingState.LOST,
        TrackingState.REACQUIRING,
        TrackingState.TRACKING,
    )
    assert result.frames[4].tracking.acquisition_count == 1
    assert result.frames[5].tracking.acquisition_count == 2


def test_distant_candidate_is_rejected_then_close_candidate_reacquires():
    result = TrackingStressHarness().run_distant_reacquisition_rejection()

    assert result.frames[3].tracking.state == TrackingState.LOST
    assert result.frames[4].tracking.state == TrackingState.LOST
    assert result.frames[4].tracking.position_x == pytest.approx(320.0)
    assert result.frames[5].tracking.state == TrackingState.REACQUIRING
    assert result.frames[6].tracking.state == TrackingState.TRACKING


def test_highest_ai_probability_candidate_is_selected_then_spatially_gated():
    result = TrackingStressHarness().run_multiple_candidate_selection()
    established = result.frames[1]
    multi_candidate_frame = result.frames[2]

    assert len(multi_candidate_frame.candidate_centroids) == 2
    assert multi_candidate_frame.selected_candidate_centroid == pytest.approx((400.0, 240.0))
    assert multi_candidate_frame.identity is not None
    assert multi_candidate_frame.identity.probability == pytest.approx(0.99)
    assert established.tracking.is_locked is True
    assert multi_candidate_frame.tracking.is_locked is True
    assert multi_candidate_frame.tracking.position_x == pytest.approx(320.0)
    assert multi_candidate_frame.tracking.missed_frames == 1


def test_kalman_method_calls_and_prediction_during_loss_are_observed():
    result = TrackingStressHarness().run_extended_detection_loss()

    missed_frames = [frame for frame in result.frames if not frame.observation_supplied]
    assert all(frame.kalman_calls == ("update",) for frame in missed_frames)
    assert all(frame.kalman is not None for frame in missed_frames)
    assert all(frame.prediction is not None for frame in missed_frames)
    assert all(frame.command is None for frame in result.frames if frame.tracking.state == TrackingState.LOST)


def test_harness_results_are_truth_free_immutable_and_deterministic():
    harness = TrackingStressHarness()
    first = harness.run_all()
    second = harness.run_all()

    assert tuple(result.scenario_name for result in first) == (
        "temporary_detection_loss",
        "extended_detection_loss",
        "reacquisition",
        "distant_reacquisition_rejection",
        "multiple_candidate_selection_vs_gating",
    )
    assert first == second
    assert all(not hasattr(frame, "target_truth") for result in first for frame in result.frames)
    with pytest.raises(FrozenInstanceError):
        first[0].frames[0].observation_supplied = False
