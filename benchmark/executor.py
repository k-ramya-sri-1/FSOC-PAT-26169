"""Benchmark execution and structured metric aggregation."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from benchmark.config import BenchmarkConfig
from benchmark.mp4_runner import MP4BenchmarkConfig, MP4BenchmarkResult, MP4BenchmarkRunner
from benchmark.simulation_runner import BenchmarkResult, SimulationBenchmarkRunner
from metrics import evaluator


@dataclass(frozen=True)
class BenchmarkReport:
    """Execution metadata, telemetry, and applicable metrics for one run."""

    scenario_name: str
    source_type: str
    frame_count: int
    elapsed_simulation_seconds: Optional[float]
    metrics: Mapping[str, object]
    simulation_result: Optional[BenchmarkResult] = None
    mp4_result: Optional[MP4BenchmarkResult] = None
    motion_type: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None

    def deterministic_signature(self) -> tuple[object, ...]:
        """Return repeatable simulation-report content, excluding wall-clock measurements."""
        return (
            self.scenario_name,
            self.source_type,
            self.frame_count,
            self.elapsed_simulation_seconds,
            self.motion_type,
            self.seed,
            self.error,
            _stable_metrics(self.metrics),
            (
                self.simulation_result.deterministic_signature()
                if self.simulation_result is not None
                else None
            ),
        )


class BenchmarkExecutor:
    """Execute existing simulation or MP4 runners and compose reports."""

    def run_simulation(self, config: BenchmarkConfig) -> BenchmarkReport:
        """Execute one simulation configuration and evaluate its telemetry."""
        if not isinstance(config, BenchmarkConfig):
            raise TypeError("config must be a BenchmarkConfig")
        result = SimulationBenchmarkRunner(config).run()
        return self._simulation_report(config, result)

    def run_simulation_suite(
        self, scenarios: Sequence[BenchmarkConfig]
    ) -> Tuple[BenchmarkReport, ...]:
        """Execute configured simulation scenarios in the supplied order."""
        return tuple(self.run_simulation(scenario) for scenario in scenarios)

    def run_mp4(self, config: MP4BenchmarkConfig) -> BenchmarkReport:
        """Execute a truth-free MP4 run and report only metrics supported by it."""
        if not isinstance(config, MP4BenchmarkConfig):
            raise TypeError("config must be an MP4BenchmarkConfig")
        result = MP4BenchmarkRunner(config).run()
        telemetry = result.telemetry_frames
        metrics = _common_operational_metrics(telemetry)
        return BenchmarkReport(
            scenario_name=result.scenario_name,
            source_type="mp4",
            frame_count=result.frame_count,
            elapsed_simulation_seconds=None,
            metrics=metrics,
            mp4_result=result,
            error=result.error,
        )

    @staticmethod
    def _simulation_report(
        config: BenchmarkConfig,
        result: BenchmarkResult,
    ) -> BenchmarkReport:
        telemetry = result.telemetry_frames
        metrics = _common_operational_metrics(telemetry)
        metrics.update(
            {
                "centroid_error": evaluator.centroid_error_statistics(telemetry),
                "truth_unavailable_lock": evaluator.truth_unavailable_lock_statistics(
                    telemetry
                ),
            }
        )
        return BenchmarkReport(
            scenario_name=result.scenario_name,
            source_type="simulation",
            frame_count=result.frame_count,
            elapsed_simulation_seconds=evaluator.elapsed_simulation_time(telemetry),
            metrics=metrics,
            simulation_result=result,
            motion_type=config.motion_type.value,
            seed=config.seed,
        )


def _common_operational_metrics(telemetry) -> dict[str, object]:
    """Metrics that do not require target ground truth."""
    return {
        "acquisition_time_seconds": evaluator.acquisition_time_seconds(telemetry),
        "target_loss_percentage": evaluator.target_loss_percentage(telemetry),
        "lock_retention_percentage": evaluator.lock_retention_percentage(telemetry),
        "reacquisition": evaluator.reacquisition_statistics(telemetry),
        "processing": evaluator.processing_statistics(telemetry),
        "tracking_availability": evaluator.tracking_availability(telemetry),
        "lock_statistics": evaluator.lock_statistics(telemetry),
    }


def _stable_metrics(metrics: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    """Remove nondeterministic timing fields for deterministic simulation comparison."""
    stable = []
    for key, value in sorted(metrics.items()):
        if key == "processing":
            processing = value
            stable.append(
                (
                    key,
                    (
                        processing.frame_count,
                        None,
                        None,
                    ),
                )
            )
        else:
            stable.append((key, value))
    return tuple(stable)