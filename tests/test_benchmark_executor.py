import pytest

from benchmark.config import BenchmarkConfig, MotionType
from benchmark.executor import BenchmarkExecutor
from benchmark.mp4_runner import MP4BenchmarkConfig, MP4BenchmarkResult
from metrics import evaluator


def _config(motion_type: MotionType, seed: int = 23) -> BenchmarkConfig:
    return BenchmarkConfig(
        scenario_name=f"stage2d-{motion_type.value}",
        seed=seed,
        duration_seconds=0.1,
        dt=1.0 / 30.0,
        motion_type=motion_type,
        circular_radius=0.1,
        figure8_amplitude_x=0.1,
        figure8_amplitude_y=0.1,
    )


def test_executor_runs_one_simulation_and_reports_evaluator_metrics():
    config = _config(MotionType.STRAIGHT)
    report = BenchmarkExecutor().run_simulation(config)
    telemetry = report.simulation_result.telemetry_frames

    assert report.source_type == "simulation"
    assert report.scenario_name == config.scenario_name
    assert report.motion_type == MotionType.STRAIGHT.value
    assert report.seed == config.seed
    assert report.frame_count == len(telemetry)
    assert report.elapsed_simulation_seconds == evaluator.elapsed_simulation_time(telemetry)
    assert report.metrics["acquisition_time_seconds"] == evaluator.acquisition_time_seconds(telemetry)
    assert report.metrics["centroid_error"] == evaluator.centroid_error_statistics(telemetry)
    assert report.metrics["target_loss_percentage"] == evaluator.target_loss_percentage(telemetry)
    assert report.metrics["lock_retention_percentage"] == evaluator.lock_retention_percentage(telemetry)
    assert report.metrics["reacquisition"] == evaluator.reacquisition_statistics(telemetry)
    assert report.metrics["processing"] == evaluator.processing_statistics(telemetry)


def test_executor_runs_all_configured_motion_scenarios_in_order():
    scenarios = tuple(
        _config(motion)
        for motion in (
            MotionType.STRAIGHT,
            MotionType.CIRCULAR,
            MotionType.FIGURE8,
            MotionType.BOUNDED_RANDOM,
        )
    )

    reports = BenchmarkExecutor().run_simulation_suite(scenarios)

    assert tuple(report.motion_type for report in reports) == tuple(
        motion.value
        for motion in (
            MotionType.STRAIGHT,
            MotionType.CIRCULAR,
            MotionType.FIGURE8,
            MotionType.BOUNDED_RANDOM,
        )
    )
    assert all(report.frame_count == 3 for report in reports)


def test_report_deterministic_signature_excludes_processing_wall_time():
    executor = BenchmarkExecutor()
    config = _config(MotionType.BOUNDED_RANDOM, seed=99)
    first = executor.run_simulation(config)
    second = executor.run_simulation(config)

    assert first.deterministic_signature() == second.deterministic_signature()


def test_executor_rejects_invalid_runner_configurations():
    executor = BenchmarkExecutor()

    with pytest.raises(TypeError):
        executor.run_simulation(object())
    with pytest.raises(TypeError):
        executor.run_mp4(object())


def test_mp4_report_keeps_metadata_and_omits_truth_dependent_metrics(monkeypatch):
    mp4_result = MP4BenchmarkResult(
        video_path="recording.mp4",
        scenario_name="mp4-demo",
        opened=True,
        frame_count=0,
        telemetry_frames=(),
        video_fps=30.0,
        reported_frame_count=12,
        frame_width=64,
        frame_height=48,
        timing_source="CAP_PROP_POS_MSEC",
    )

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run(self):
            return mp4_result

    monkeypatch.setattr("benchmark.executor.MP4BenchmarkRunner", FakeRunner)
    report = BenchmarkExecutor().run_mp4(
        MP4BenchmarkConfig(video_path="recording.mp4", scenario_name="mp4-demo")
    )

    assert report.source_type == "mp4"
    assert report.mp4_result is mp4_result
    assert report.frame_count == 0
    assert report.elapsed_simulation_seconds is None
    assert report.metrics["processing"] == evaluator.processing_statistics(())
    assert report.metrics["acquisition_time_seconds"] is None
    assert report.metrics["target_loss_percentage"] == 0.0
    assert report.metrics["lock_retention_percentage"] == 0.0
    assert "centroid_error" not in report.metrics
    assert "truth_unavailable_lock" not in report.metrics