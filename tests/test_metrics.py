from dataclasses import replace

import pytest

from benchmark.config import BenchmarkConfig, MotionType
from benchmark.simulation_runner import SimulationBenchmarkRunner
from core.sensor import SensorFrame
from core.tracking import TrackedTargetState, TrackingState
from metrics.evaluator import (
    approximate_processing_fps,
    acquisition_time_seconds,
    centroid_error_pixels,
    centroid_error_statistics,
    elapsed_simulation_time,
    frame_count,
    lock_retention_percentage,
    lock_statistics,
    pointing_error_pixels,
    processing_statistics,
    processing_times,
    reacquisition_statistics,
    target_loss_percentage,
    truth_unavailable_lock_statistics,
    tracking_availability,
)


def _run_short_benchmark():
    return SimulationBenchmarkRunner(
        BenchmarkConfig(
            scenario_name="metrics",
            duration_seconds=0.1,
            dt=1.0 / 30.0,
            motion_type=MotionType.STRAIGHT,
        )
    ).run()


def _controlled_telemetry(states):
    result = SimulationBenchmarkRunner(
        BenchmarkConfig(
            scenario_name="controlled-metrics",
            duration_seconds=len(states) / 30.0,
            dt=1.0 / 30.0,
            motion_type=MotionType.STRAIGHT,
        )
    ).run()
    frames = []
    for frame, state in zip(result.telemetry_frames, states):
        truth = frame.sensor_frame.projected_pixel
        position_x = None if truth is None else truth[0]
        position_y = None if truth is None else truth[1]
        frames.append(
            replace(
                frame,
                tracking=TrackedTargetState(
                    state=state,
                    timestamp=frame.timestamp,
                    position_x=position_x,
                    position_y=position_y,
                    velocity_x=0.0,
                    velocity_y=0.0,
                    confidence=1.0 if state == TrackingState.TRACKING else 0.0,
                    candidate_id=0 if position_x is not None else None,
                    missed_frames=0,
                    acquisition_count=1 if state == TrackingState.TRACKING else 0,
                ),
            )
        )
    return tuple(frames)


def test_primitive_metrics_and_error_helpers():
    result = _run_short_benchmark()
    telemetry = result.telemetry_frames

    assert frame_count(telemetry) == 3
    assert elapsed_simulation_time(telemetry) == pytest.approx(2.0 / 30.0)
    assert len(processing_times(telemetry)) == 3
    assert approximate_processing_fps(telemetry) > 0.0
    assert 0.0 <= tracking_availability(telemetry) <= 1.0
    stats = lock_statistics(telemetry)
    assert stats.total_frames == 3
    assert 0.0 <= stats.availability_ratio <= 1.0
    assert 0.0 <= stats.lock_ratio <= 1.0
    assert centroid_error_pixels(telemetry[0]) is None
    assert pointing_error_pixels(telemetry[0]) is not None


def test_metrics_do_not_mutate_benchmark_telemetry():
    result = _run_short_benchmark()
    before = tuple(
        (
            frame.timestamp,
            frame.frame_index,
            frame.tracking,
            frame.processing_time_seconds,
        )
        for frame in result.telemetry_frames
    )

    frame_count(result.telemetry_frames)
    elapsed_simulation_time(result.telemetry_frames)
    processing_times(result.telemetry_frames)
    approximate_processing_fps(result.telemetry_frames)
    tracking_availability(result.telemetry_frames)
    lock_statistics(result.telemetry_frames)
    [centroid_error_pixels(frame) for frame in result.telemetry_frames]
    [pointing_error_pixels(frame) for frame in result.telemetry_frames]

    after = tuple(
        (
            frame.timestamp,
            frame.frame_index,
            frame.tracking,
            frame.processing_time_seconds,
        )
        for frame in result.telemetry_frames
    )
    assert after == before


def test_processing_fps_handles_empty_zero_and_invalid_durations():
    assert approximate_processing_fps(()) == 0.0
    result = _run_short_benchmark()
    zero_duration = tuple(
        replace(frame, processing_time_seconds=0.0) for frame in result.telemetry_frames
    )
    assert approximate_processing_fps(zero_duration) == 0.0

    invalid_duration = (replace(result.telemetry_frames[0], processing_time_seconds=-1.0),)
    with pytest.raises(ValueError):
        approximate_processing_fps(invalid_duration)


def test_error_metrics_handle_missing_estimates_safely():
    result = _run_short_benchmark()
    first = result.telemetry_frames[0]
    missing_tracking = replace(
        first,
        tracking=replace(first.tracking, position_x=None, position_y=None),
    )

    assert centroid_error_pixels(missing_tracking) is None
    assert pointing_error_pixels(missing_tracking) is not None


def test_acquisition_time_and_never_acquired_case():
    acquired = _controlled_telemetry(
        [TrackingState.ACQUIRING, TrackingState.TRACKING]
    )
    never_acquired = _controlled_telemetry(
        [TrackingState.SEARCHING, TrackingState.ACQUIRING]
    )

    assert acquisition_time_seconds(acquired) == pytest.approx(1.0 / 30.0)
    assert acquisition_time_seconds(never_acquired) is None
    assert acquisition_time_seconds(()) is None


def test_centroid_error_statistics_mean_max_rms_and_missing_pairs():
    telemetry = list(_controlled_telemetry([TrackingState.TRACKING] * 3))
    for index, frame in enumerate(telemetry):
        truth = frame.sensor_frame.projected_pixel
        telemetry[index] = replace(
            frame,
            tracking=replace(
                frame.tracking,
                position_x=truth[0] + (index + 1),
                position_y=truth[1],
            ),
        )
    telemetry[1] = replace(
        telemetry[1],
        sensor_frame=replace(telemetry[1].sensor_frame, projected_pixel=None),
    )

    stats = centroid_error_statistics(tuple(telemetry))

    assert stats.valid_sample_count == 2
    assert stats.mean_error_pixels == pytest.approx(2.0)
    assert stats.maximum_error_pixels == pytest.approx(3.0)
    assert stats.rms_error_pixels == pytest.approx(5.0 ** 0.5)
    assert centroid_error_statistics(()).valid_sample_count == 0


def test_target_loss_and_lock_retention_percentages():
    telemetry = _controlled_telemetry(
        [
            TrackingState.TRACKING,
            TrackingState.LOST,
            TrackingState.LOST,
            TrackingState.SEARCHING,
        ]
    )

    assert target_loss_percentage(telemetry) == pytest.approx(50.0)
    assert lock_retention_percentage(telemetry) == pytest.approx(25.0)
    assert target_loss_percentage(()) == 0.0
    assert lock_retention_percentage(()) == 0.0


def test_reacquisition_statistics_for_recovery_paths_and_no_event_case():
    telemetry = _controlled_telemetry(
        [
            TrackingState.TRACKING,
            TrackingState.LOST,
            TrackingState.LOST,
            TrackingState.REACQUIRING,
            TrackingState.TRACKING,
        ]
    )
    no_event = _controlled_telemetry(
        [TrackingState.TRACKING, TrackingState.LOST, TrackingState.LOST]
    )
    direct_recovery = _controlled_telemetry(
        [TrackingState.TRACKING, TrackingState.LOST, TrackingState.TRACKING]
    )
    never_tracked = _controlled_telemetry(
        [TrackingState.SEARCHING, TrackingState.LOST, TrackingState.TRACKING]
    )
    multiple_events = _controlled_telemetry(
        [
            TrackingState.TRACKING,
            TrackingState.LOST,
            TrackingState.REACQUIRING,
            TrackingState.TRACKING,
            TrackingState.LOST,
            TrackingState.TRACKING,
        ]
    )

    stats = reacquisition_statistics(telemetry)
    assert len(stats.durations_seconds) == 1
    assert stats.durations_seconds[0] == pytest.approx(3.0 / 30.0)
    assert stats.mean_reacquisition_seconds == pytest.approx(3.0 / 30.0)
    assert stats.maximum_reacquisition_seconds == pytest.approx(3.0 / 30.0)
    direct_stats = reacquisition_statistics(direct_recovery)
    assert len(direct_stats.durations_seconds) == 1
    assert direct_stats.durations_seconds[0] == pytest.approx(1.0 / 30.0)
    assert reacquisition_statistics(never_tracked).durations_seconds == ()
    multiple_stats = reacquisition_statistics(multiple_events)
    assert multiple_stats.durations_seconds == pytest.approx((2.0 / 30.0, 1.0 / 30.0))
    assert multiple_stats.mean_reacquisition_seconds == pytest.approx(1.5 / 30.0)
    assert reacquisition_statistics(no_event).durations_seconds == ()
    assert reacquisition_statistics(no_event).mean_reacquisition_seconds is None


def test_processing_statistics_and_truth_unavailable_lock_evaluation():
    result = _run_short_benchmark()
    telemetry = tuple(
        replace(frame, processing_time_seconds=0.1 * (index + 1))
        for index, frame in enumerate(result.telemetry_frames)
    )
    processing = processing_statistics(telemetry)
    assert processing.frame_count == 3
    assert processing.mean_processing_time_seconds == pytest.approx(0.2)
    assert processing.processing_fps == pytest.approx(5.0)

    locked_without_truth = replace(
        telemetry[0],
        tracking=replace(
            telemetry[0].tracking,
            state=TrackingState.TRACKING,
            position_x=320.0,
            position_y=240.0,
        ),
        sensor_frame=SensorFrame(
            image=telemetry[0].sensor_frame.image,
            target_visible=False,
            projected_pixel=None,
        ),
    )
    truth_unavailable = truth_unavailable_lock_statistics(
        (locked_without_truth, telemetry[1])
    )
    assert truth_unavailable.total_frames == 2
    assert truth_unavailable.truth_unavailable_lock_frames == 1
    assert truth_unavailable.truth_unavailable_lock_percentage == pytest.approx(50.0)
    assert truth_unavailable_lock_statistics(()).truth_unavailable_lock_frames == 0


def test_metric_functions_are_deterministic_and_do_not_mutate_telemetry():
    telemetry = _controlled_telemetry(
        [TrackingState.ACQUIRING, TrackingState.TRACKING, TrackingState.LOST]
    )
    before = tuple((frame.timestamp, frame.tracking) for frame in telemetry)

    outputs = (
        acquisition_time_seconds(telemetry),
        centroid_error_statistics(telemetry),
        target_loss_percentage(telemetry),
        lock_retention_percentage(telemetry),
        reacquisition_statistics(telemetry),
        processing_statistics(telemetry),
        truth_unavailable_lock_statistics(telemetry),
    )
    assert outputs == (
        acquisition_time_seconds(telemetry),
        centroid_error_statistics(telemetry),
        target_loss_percentage(telemetry),
        lock_retention_percentage(telemetry),
        reacquisition_statistics(telemetry),
        processing_statistics(telemetry),
        truth_unavailable_lock_statistics(telemetry),
    )
    assert tuple((frame.timestamp, frame.tracking) for frame in telemetry) == before
