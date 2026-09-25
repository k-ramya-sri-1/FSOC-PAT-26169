from dataclasses import replace

import pytest

from benchmark.config import BenchmarkConfig, MotionType
from benchmark.simulation_runner import SimulationBenchmarkRunner
from metrics.evaluator import (
    approximate_processing_fps,
    centroid_error_pixels,
    elapsed_simulation_time,
    frame_count,
    lock_statistics,
    pointing_error_pixels,
    processing_times,
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
