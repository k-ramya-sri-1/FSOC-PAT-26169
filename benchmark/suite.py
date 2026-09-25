"""Deterministic suite for the required simulation motion scenarios."""

from dataclasses import dataclass, replace
from typing import Tuple

from benchmark.config import BenchmarkConfig, MotionType
from benchmark.executor import BenchmarkExecutor, BenchmarkReport


@dataclass(frozen=True)
class BenchmarkSuite:
    """Run all required simulation motions from a shared base configuration."""

    base_config: BenchmarkConfig

    def __post_init__(self) -> None:
        if not isinstance(self.base_config, BenchmarkConfig):
            raise TypeError("base_config must be a BenchmarkConfig")

    def run(self) -> Tuple[BenchmarkReport, ...]:
        """Execute straight, circular, figure-8, and bounded-random scenarios."""
        scenario_definitions = (
            ("straight", MotionType.STRAIGHT),
            ("circular", MotionType.CIRCULAR),
            ("figure8", MotionType.FIGURE8),
            ("bounded_random", MotionType.BOUNDED_RANDOM),
        )
        executor = BenchmarkExecutor()
        return tuple(
            executor.run_simulation(
                replace(
                    self.base_config,
                    scenario_name=scenario_name,
                    motion_type=motion_type,
                )
            )
            for scenario_name, motion_type in scenario_definitions
        )
