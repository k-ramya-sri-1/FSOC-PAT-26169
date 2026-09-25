"""Perception stress matrix combining sensor sizes and existing disturbances."""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

from benchmark.config import BenchmarkConfig
from benchmark.disturbance_suite import DisturbanceBenchmarkSuite
from core.detection import BeaconDetector, DetectionCandidate, DetectionConfig
from core.disturbances import DisturbanceConfig, ImageDisturbanceModel
from core.geometry import CameraConfig
from core.scene import Scene, TargetState
from core.sensor import MonochromeSensor, SensorConfig


_DISTURBANCE_ORDER = (
    "clean",
    "gaussian",
    "poisson",
    "salt_pepper",
    "low_light",
    "haze",
    "fog",
    "rain",
)
_BEACON_SIZES = (5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class PerceptionStressCase:
    """Reproducible input configuration for one size/disturbance pair."""

    scenario_name: str
    beacon_diameter: float
    disturbance_name: str
    disturbance_config: DisturbanceConfig


@dataclass(frozen=True)
class PerceptionStressResult:
    """Observed sensor and detector outputs for one stress-matrix case."""

    scenario_name: str
    beacon_diameter: float
    disturbance_name: str
    target_visible: bool
    projected_pixel: Optional[Tuple[float, float]]
    detected: bool
    candidate_count: int
    candidate_centroid: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class PerceptionStressSuite:
    """Run a deterministic 4-by-8 matrix through existing perception components."""

    camera_config: CameraConfig = field(default_factory=CameraConfig)
    peak_intensity: float = 255.0
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    seed: int = 42
    target_range_m: float = 100.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_range_m) or self.target_range_m <= 0.0:
            raise ValueError("target_range_m must be finite and positive")
        SensorConfig(
            camera_config=self.camera_config,
            beacon_diameter=10.0,
            peak_intensity=self.peak_intensity,
        )

    def scenarios(self) -> Tuple[PerceptionStressCase, ...]:
        """Return all 32 cases in stable beacon-size then disturbance order."""
        existing_disturbances = DisturbanceBenchmarkSuite(
            BenchmarkConfig(seed=self.seed)
        ).scenarios()
        disturbance_by_name = {
            scenario.scenario_name: scenario.disturbance_config
            for scenario in existing_disturbances
        }
        cases = []
        for diameter in _BEACON_SIZES:
            for disturbance_name in _DISTURBANCE_ORDER:
                cases.append(
                    PerceptionStressCase(
                        scenario_name=f"beacon_size_{int(diameter)}__{disturbance_name}",
                        beacon_diameter=diameter,
                        disturbance_name=disturbance_name,
                        disturbance_config=disturbance_by_name[disturbance_name],
                    )
                )
        return tuple(cases)

    def run(self) -> Tuple[PerceptionStressResult, ...]:
        """Capture, disturb, and detect each controlled center-target scenario."""
        detector = BeaconDetector(self.detection_config)
        results = []
        for case in self.scenarios():
            sensor = MonochromeSensor(
                SensorConfig(
                    camera_config=self.camera_config,
                    beacon_diameter=case.beacon_diameter,
                    peak_intensity=self.peak_intensity,
                )
            )
            scene = Scene(
                target_state=TargetState(position=(0.0, 0.0, self.target_range_m))
            )
            sensor_frame = sensor.capture(scene)
            disturbed_frame = ImageDisturbanceModel(case.disturbance_config).apply(
                sensor_frame
            )
            candidates = detector.detect(disturbed_frame.image)
            candidate: Optional[DetectionCandidate] = candidates[0] if candidates else None
            results.append(
                PerceptionStressResult(
                    scenario_name=case.scenario_name,
                    beacon_diameter=case.beacon_diameter,
                    disturbance_name=case.disturbance_name,
                    target_visible=sensor_frame.target_visible,
                    projected_pixel=sensor_frame.projected_pixel,
                    detected=bool(candidates),
                    candidate_count=len(candidates),
                    candidate_centroid=(
                        (candidate.centroid_x, candidate.centroid_y)
                        if candidate is not None
                        else None
                    ),
                )
            )
        return tuple(results)
