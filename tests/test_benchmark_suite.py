from dataclasses import replace

from benchmark.config import BenchmarkConfig, MotionType
from benchmark.executor import BenchmarkExecutor, BenchmarkReport
from benchmark.suite import BenchmarkSuite


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        scenario_name="base-scenario",
        seed=71,
        duration_seconds=0.1,
        dt=1.0 / 30.0,
        motion_type=MotionType.CIRCULAR,
        target_initial_position=(1.0, 2.0, 100.0),
        target_velocity=(0.2, -0.1, 0.0),
        circular_radius=0.5,
        figure8_amplitude_x=1.5,
        random_max_acceleration=0.75,
    )


def test_suite_runs_four_scenarios_in_required_order():
    reports = BenchmarkSuite(_base_config()).run()

    assert len(reports) == 4
    assert all(isinstance(report, BenchmarkReport) for report in reports)
    assert [report.scenario_name for report in reports] == [
        "straight",
        "circular",
        "figure8",
        "bounded_random",
    ]
    assert [report.motion_type for report in reports] == [
        "straight",
        "circular",
        "figure-8",
        "bounded-random",
    ]
    assert all(report.source_type == "simulation" for report in reports)


def test_suite_delegates_configs_and_preserves_base_values(monkeypatch):
    base_config = _base_config()
    delegated_configs = []

    def fake_run_simulation(executor, config):
        delegated_configs.append(config)
        return BenchmarkReport(
            scenario_name=config.scenario_name,
            source_type="simulation",
            frame_count=0,
            elapsed_simulation_seconds=0.0,
            metrics={},
            motion_type=config.motion_type.value,
            seed=config.seed,
        )

    monkeypatch.setattr(BenchmarkExecutor, "run_simulation", fake_run_simulation)
    reports = BenchmarkSuite(base_config).run()

    assert len(reports) == 4
    assert [config.motion_type for config in delegated_configs] == [
        MotionType.STRAIGHT,
        MotionType.CIRCULAR,
        MotionType.FIGURE8,
        MotionType.BOUNDED_RANDOM,
    ]
    assert [config.scenario_name for config in delegated_configs] == [
        "straight",
        "circular",
        "figure8",
        "bounded_random",
    ]
    for config, expected_name, expected_motion in zip(
        delegated_configs,
        ("straight", "circular", "figure8", "bounded_random"),
        (
            MotionType.STRAIGHT,
            MotionType.CIRCULAR,
            MotionType.FIGURE8,
            MotionType.BOUNDED_RANDOM,
        ),
    ):
        assert config == replace(
            base_config,
            scenario_name=expected_name,
            motion_type=expected_motion,
        )
    assert base_config == _base_config()


def test_same_suite_configuration_produces_same_deterministic_signatures():
    suite = BenchmarkSuite(_base_config())

    first = suite.run()
    second = suite.run()

    assert [report.deterministic_signature() for report in first] == [
        report.deterministic_signature() for report in second
    ]


def test_suite_does_not_mutate_base_configuration():
    base_config = _base_config()
    before = replace(base_config)

    BenchmarkSuite(base_config).run()

    assert base_config == before
