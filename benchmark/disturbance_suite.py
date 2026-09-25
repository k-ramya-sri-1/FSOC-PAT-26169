"""Predefined deterministic disturbance scenarios for simulation benchmarks."""

from dataclasses import dataclass, replace
from typing import Tuple

from benchmark.config import BenchmarkConfig
from benchmark.executor import BenchmarkExecutor, BenchmarkReport
from core.disturbances import AtmosphereMode, DisturbanceConfig


@dataclass(frozen=True)
class DisturbanceBenchmarkScenario:
    """A named disturbance configuration; contains no execution logic."""

    scenario_name: str
    disturbance_config: DisturbanceConfig
    description: str = ""


@dataclass(frozen=True)
class DisturbanceBenchmarkSuite:
    """Build and run predefined disturbance configurations from a base scenario."""

    base_config: BenchmarkConfig

    def __post_init__(self) -> None:
        if not isinstance(self.base_config, BenchmarkConfig):
            raise TypeError("base_config must be a BenchmarkConfig")

    def scenarios(self) -> Tuple[DisturbanceBenchmarkScenario, ...]:
        """Return supported disturbance scenarios in fixed order."""
        seed = self.base_config.seed
        clean = DisturbanceConfig(seed=seed)
        return (
            DisturbanceBenchmarkScenario(
                "clean",
                clean,
                "No additional image disturbance.",
            ),
            DisturbanceBenchmarkScenario(
                "gaussian",
                replace(clean, gaussian_std=20.0),
                "Gaussian image noise at the configured 20-intensity ceiling.",
            ),
            DisturbanceBenchmarkScenario(
                "poisson",
                replace(clean, poisson_strength=1.0),
                "Poisson image noise using the existing strength parameter.",
            ),
            DisturbanceBenchmarkScenario(
                "salt_pepper",
                replace(clean, salt_pepper_probability=0.1),
                "Salt-and-pepper image noise at 10 percent density.",
            ),
            DisturbanceBenchmarkScenario(
                "camera_jitter",
                replace(clean, jitter_x_pixels=20.0, jitter_y_pixels=20.0),
                "Pixel jitter configured to the plus/minus 20-pixel limits.",
            ),
            DisturbanceBenchmarkScenario(
                "platform_motion",
                replace(clean, jitter_x_pixels=20.0, jitter_y_pixels=20.0),
                "Uses the existing camera/platform pixel-jitter model; there is no separate platform-motion model.",
            ),
            DisturbanceBenchmarkScenario(
                "low_light",
                replace(
                    clean,
                    atmosphere_mode=AtmosphereMode.LOW_LIGHT,
                    atmosphere_strength=0.5,
                ),
                "Low-light atmospheric attenuation at strength 0.5.",
            ),
            DisturbanceBenchmarkScenario(
                "haze",
                replace(
                    clean,
                    atmosphere_mode=AtmosphereMode.HAZE,
                    atmosphere_strength=0.5,
                ),
                "Haze atmospheric degradation at strength 0.5.",
            ),
            DisturbanceBenchmarkScenario(
                "fog",
                replace(
                    clean,
                    atmosphere_mode=AtmosphereMode.FOG,
                    atmosphere_strength=0.5,
                ),
                "Fog atmospheric degradation at strength 0.5.",
            ),
            DisturbanceBenchmarkScenario(
                "rain",
                replace(
                    clean,
                    atmosphere_mode=AtmosphereMode.RAIN,
                    atmosphere_strength=0.5,
                ),
                "Rain atmospheric degradation at strength 0.5.",
            ),
        )

    def run(self) -> Tuple[BenchmarkReport, ...]:
        """Execute every predefined disturbance scenario through the executor."""
        executor = BenchmarkExecutor()
        return tuple(
            executor.run_simulation(
                replace(
                    self.base_config,
                    scenario_name=scenario.scenario_name,
                    disturbance_config=scenario.disturbance_config,
                )
            )
            for scenario in self.scenarios()
        )
