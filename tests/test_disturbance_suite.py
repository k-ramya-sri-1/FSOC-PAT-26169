from dataclasses import replace

from benchmark.config import BenchmarkConfig, MotionType
from benchmark.disturbance_suite import (
    DisturbanceBenchmarkScenario,
    DisturbanceBenchmarkSuite,
)
from benchmark.executor import BenchmarkExecutor, BenchmarkReport
from core.disturbances import AtmosphereMode, DisturbanceConfig


SCENARIO_NAMES = (
    "clean",
    "gaussian",
    "poisson",
    "salt_pepper",
    "camera_jitter",
    "platform_motion",
    "low_light",
    "haze",
    "fog",
    "rain",
)


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        scenario_name="base",
        seed=123,
        duration_seconds=0.2,
        dt=1.0 / 30.0,
        motion_type=MotionType.FIGURE8,
        target_initial_position=(1.0, 2.0, 120.0),
        target_velocity=(0.2, -0.1, 0.0),
    )


def test_suite_exposes_supported_scenarios_in_deterministic_order():
    scenarios = DisturbanceBenchmarkSuite(_base_config()).scenarios()

    assert tuple(scenario.scenario_name for scenario in scenarios) == SCENARIO_NAMES
    assert len({scenario.scenario_name for scenario in scenarios}) == len(scenarios)
    assert all(isinstance(scenario, DisturbanceBenchmarkScenario) for scenario in scenarios)
    assert all(isinstance(scenario.disturbance_config, DisturbanceConfig) for scenario in scenarios)
    assert scenarios == DisturbanceBenchmarkSuite(_base_config()).scenarios()


def test_scenarios_use_existing_disturbance_configuration_fields():
    scenarios = {
        scenario.scenario_name: scenario
        for scenario in DisturbanceBenchmarkSuite(_base_config()).scenarios()
    }

    assert scenarios["clean"].disturbance_config == DisturbanceConfig(seed=123)
    assert scenarios["gaussian"].disturbance_config.gaussian_std == 20.0
    assert scenarios["poisson"].disturbance_config.poisson_strength == 1.0
    assert scenarios["salt_pepper"].disturbance_config.salt_pepper_probability == 0.1
    assert scenarios["camera_jitter"].disturbance_config.jitter_x_pixels == 20.0
    assert scenarios["camera_jitter"].disturbance_config.jitter_y_pixels == 20.0
    assert scenarios["platform_motion"].disturbance_config.jitter_x_pixels == 20.0
    assert "existing camera/platform pixel-jitter" in scenarios["platform_motion"].description
    assert scenarios["low_light"].disturbance_config.atmosphere_mode == AtmosphereMode.LOW_LIGHT
    assert scenarios["haze"].disturbance_config.atmosphere_mode == AtmosphereMode.HAZE
    assert scenarios["fog"].disturbance_config.atmosphere_mode == AtmosphereMode.FOG
    assert scenarios["rain"].disturbance_config.atmosphere_mode == AtmosphereMode.RAIN
    assert all(
        scenarios[name].disturbance_config.atmosphere_strength == 0.5
        for name in ("low_light", "haze", "fog", "rain")
    )


def test_suite_replaces_only_scenario_name_and_disturbance_config(monkeypatch):
    base_config = _base_config()
    delegated = []

    def fake_run_simulation(executor, config):
        delegated.append(config)
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
    reports = DisturbanceBenchmarkSuite(base_config).run()

    assert len(reports) == len(SCENARIO_NAMES)
    assert all(isinstance(report, BenchmarkReport) for report in reports)
    assert all(report.source_type == "simulation" for report in reports)
    assert tuple(report.scenario_name for report in reports) == SCENARIO_NAMES
    assert len(delegated) == len(SCENARIO_NAMES)
    assert tuple(config.scenario_name for config in delegated) == SCENARIO_NAMES
    assert tuple(config.disturbance_config for config in delegated) == tuple(
        scenario.disturbance_config
        for scenario in DisturbanceBenchmarkSuite(base_config).scenarios()
    )
    for config, scenario in zip(delegated, DisturbanceBenchmarkSuite(base_config).scenarios()):
        assert config == replace(
            base_config,
            scenario_name=scenario.scenario_name,
            disturbance_config=scenario.disturbance_config,
        )

    assert base_config == _base_config()


def test_same_suite_produces_identical_report_signatures(monkeypatch):
    def fake_run_simulation(executor, config):
        return BenchmarkReport(
            scenario_name=config.scenario_name,
            source_type="simulation",
            frame_count=5,
            elapsed_simulation_seconds=0.1,
            metrics={"configured_disturbance": config.disturbance_config},
            motion_type=config.motion_type.value,
            seed=config.seed,
        )

    monkeypatch.setattr(BenchmarkExecutor, "run_simulation", fake_run_simulation)
    suite = DisturbanceBenchmarkSuite(_base_config())

    first = suite.run()
    second = suite.run()

    assert [report.deterministic_signature() for report in first] == [
        report.deterministic_signature() for report in second
    ]
