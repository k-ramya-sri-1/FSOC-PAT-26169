"""Controlled sensor and detector validation scenarios."""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

from core.detection import BeaconDetector, DetectionCandidate, DetectionConfig
from core.geometry import CameraConfig
from core.scene import Scene, TargetState
from core.sensor import MonochromeSensor, SensorConfig


@dataclass(frozen=True)
class PerceptionScenario:
    """Reproducible sensor input configuration for one perception case."""

    scenario_name: str
    beacon_diameter: float
    target_position: Tuple[float, float, float]
    expected_visibility: bool


@dataclass(frozen=True)
class PerceptionResult:
    """Configured scenario and observed sensor/detector output."""

    scenario_name: str
    beacon_diameter: float
    target_position: Tuple[float, float, float]
    expected_visibility: bool
    target_visible: bool
    projected_pixel: Optional[Tuple[float, float]]
    detected: bool
    candidate_count: int
    candidate_centroid: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class PerceptionBenchmarkSuite:
    """Run controlled perception scenarios using the existing sensor and detector."""

    camera_config: CameraConfig = field(default_factory=CameraConfig)
    peak_intensity: float = 255.0
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    target_range_m: float = 100.0

    def __post_init__(self) -> None:
        if self.target_range_m <= 0.0 or not math.isfinite(self.target_range_m):
            raise ValueError("target_range_m must be finite and positive")
        SensorConfig(camera_config=self.camera_config, peak_intensity=self.peak_intensity)

    def scenarios(self) -> Tuple[PerceptionScenario, ...]:
        """Return the deterministic size, position, and visibility scenario set."""
        distance = self.target_range_m
        half_fov_x = self.camera_config.fov_x_rad / 2.0
        half_fov_y = self.camera_config.fov_y_rad / 2.0
        lateral_offset = distance * math.tan(half_fov_x / 4.0)
        vertical_offset = distance * math.tan(half_fov_y / 4.0)
        near_boundary_x = distance * math.tan(half_fov_x * 0.95)
        outside_fov_x = distance * math.tan(half_fov_x * 1.05)

        size_scenarios = tuple(
            PerceptionScenario(
                scenario_name=f"beacon_size_{diameter}",
                beacon_diameter=float(diameter),
                target_position=(0.0, 0.0, distance),
                expected_visibility=True,
            )
            for diameter in (5, 10, 15, 20)
        )
        position_scenarios = (
            PerceptionScenario("position_center", 10.0, (0.0, 0.0, distance), True),
            PerceptionScenario("position_left", 10.0, (-lateral_offset, 0.0, distance), True),
            PerceptionScenario("position_right", 10.0, (lateral_offset, 0.0, distance), True),
            PerceptionScenario("position_top", 10.0, (0.0, vertical_offset, distance), True),
            PerceptionScenario("position_bottom", 10.0, (0.0, -vertical_offset, distance), True),
        )
        visibility_scenarios = (
            PerceptionScenario(
                "visibility_near_fov_boundary",
                10.0,
                (near_boundary_x, 0.0, distance),
                True,
            ),
            PerceptionScenario(
                "visibility_outside_fov",
                10.0,
                (outside_fov_x, 0.0, distance),
                False,
            ),
            PerceptionScenario(
                "visibility_behind_camera",
                10.0,
                (0.0, 0.0, -distance),
                False,
            ),
        )
        return size_scenarios + position_scenarios + visibility_scenarios

    def run(self) -> Tuple[PerceptionResult, ...]:
        """Run every scenario through the existing sensor and detector APIs."""
        detector = BeaconDetector(self.detection_config)
        results = []
        for scenario in self.scenarios():
            sensor = MonochromeSensor(
                SensorConfig(
                    camera_config=self.camera_config,
                    beacon_diameter=scenario.beacon_diameter,
                    peak_intensity=self.peak_intensity,
                )
            )
            scene = Scene(target_state=TargetState(position=scenario.target_position))
            sensor_frame = sensor.capture(scene)
            candidates = detector.detect(sensor_frame.image)
            first_candidate: Optional[DetectionCandidate] = (
                candidates[0] if candidates else None
            )
            results.append(
                PerceptionResult(
                    scenario_name=scenario.scenario_name,
                    beacon_diameter=scenario.beacon_diameter,
                    target_position=scenario.target_position,
                    expected_visibility=scenario.expected_visibility,
                    target_visible=sensor_frame.target_visible,
                    projected_pixel=sensor_frame.projected_pixel,
                    detected=bool(candidates),
                    candidate_count=len(candidates),
                    candidate_centroid=(
                        (first_candidate.centroid_x, first_candidate.centroid_y)
                        if first_candidate is not None
                        else None
                    ),
                )
            )
        return tuple(results)
