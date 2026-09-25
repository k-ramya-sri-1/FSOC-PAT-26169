from core.scene import BoundedRandomMotionModel, CircularMotionModel, Figure8MotionModel, StraightMotionModel
from benchmark.config import BenchmarkConfig, MotionType
from benchmark.simulation_runner import SimulationBenchmarkRunner


def _short_config(motion_type: MotionType, seed: int = 42) -> BenchmarkConfig:
    return BenchmarkConfig(
        scenario_name=motion_type.value,
        seed=seed,
        duration_seconds=0.1,
        dt=1.0 / 30.0,
        motion_type=motion_type,
        circular_radius=0.1,
        figure8_amplitude_x=0.1,
        figure8_amplitude_y=0.1,
    )


def test_benchmark_configuration_construction_and_motion_selection():
    assert isinstance(_short_config(MotionType.STRAIGHT).create_motion_model(), StraightMotionModel)
    assert isinstance(_short_config(MotionType.CIRCULAR).create_motion_model(), CircularMotionModel)
    assert isinstance(_short_config(MotionType.FIGURE8).create_motion_model(), Figure8MotionModel)
    assert isinstance(_short_config(MotionType.BOUNDED_RANDOM).create_motion_model(), BoundedRandomMotionModel)


def test_runner_executes_and_collects_timestamped_telemetry():
    result = SimulationBenchmarkRunner(_short_config(MotionType.STRAIGHT)).run()

    assert result.scenario_name == MotionType.STRAIGHT.value
    assert result.seed == 42
    assert result.frame_count == 3
    assert len(result.telemetry_frames) == result.frame_count
    assert [frame.timestamp for frame in result.telemetry_frames] == [0.0, 1.0 / 30.0, 2.0 / 30.0]
    assert result.configuration_metadata["motion_type"] == MotionType.STRAIGHT.value


def test_runner_is_deterministic_for_same_configuration():
    config = _short_config(MotionType.BOUNDED_RANDOM, seed=123)

    first = SimulationBenchmarkRunner(config).run()
    second = SimulationBenchmarkRunner(config).run()

    assert first == second
    assert first.deterministic_signature() == second.deterministic_signature()


def test_different_seeds_change_stochastic_benchmark_output():
    first = SimulationBenchmarkRunner(_short_config(MotionType.BOUNDED_RANDOM, seed=1)).run()
    second = SimulationBenchmarkRunner(_short_config(MotionType.BOUNDED_RANDOM, seed=2)).run()

    assert first != second