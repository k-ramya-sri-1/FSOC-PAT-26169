"""Deterministic benchmark scenario configuration."""

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Tuple

from core.detection import DetectionConfig
from core.disturbances import DisturbanceConfig
from core.geometry import CameraConfig
from core.gimbal import ControlConfig, GimbalConfig
from core.kalman import KalmanConfig
from core.scene import (
    BoundedRandomMotionModel,
    CircularMotionModel,
    Figure8MotionModel,
    StraightMotionModel,
    TargetMotionModel,
)
from core.sensor import SensorConfig
from core.tracking import TrackingConfig


class MotionType(str, Enum):
    """Mandatory Stage 2A benchmark motion scenarios."""

    STRAIGHT = "straight"
    CIRCULAR = "circular"
    FIGURE8 = "figure-8"
    BOUNDED_RANDOM = "bounded-random"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete deterministic configuration for one simulation benchmark run."""

    scenario_name: str = "default"
    seed: int = 42
    duration_seconds: float = 1.0
    dt: float = 1.0 / 60.0
    motion_type: MotionType = MotionType.STRAIGHT
    target_initial_position: Tuple[float, float, float] = (0.5, 0.2, 100.0)
    target_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    sensor_config: SensorConfig = field(
        default_factory=lambda: SensorConfig(modulation_depth=0.25)
    )
    disturbance_config: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    tracking_config: TrackingConfig = field(
        default_factory=lambda: TrackingConfig(minimum_acquisition_observations=1)
    )
    kalman_config: KalmanConfig = field(default_factory=KalmanConfig)
    gimbal_config: GimbalConfig = field(default_factory=GimbalConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    prediction_horizon: float = 0.0
    circular_center: Tuple[float, float, float] = (0.0, 0.0, 100.0)
    circular_radius: float = 10.0
    circular_angular_speed: float = 0.5
    circular_initial_phase: float = 0.0
    circular_plane: str = "XY"
    figure8_center: Tuple[float, float, float] = (0.0, 0.0, 100.0)
    figure8_amplitude_x: float = 10.0
    figure8_amplitude_y: float = 10.0
    figure8_angular_speed: float = 0.5
    figure8_initial_phase: float = 0.0
    random_max_acceleration: float = 2.0
    random_max_speed: float = 10.0
    random_bounds_min: Tuple[float, float, float] = (-50.0, -50.0, 50.0)
    random_bounds_max: Tuple[float, float, float] = (50.0, 50.0, 150.0)

    def __post_init__(self) -> None:
        if not self.scenario_name:
            raise ValueError("scenario_name must not be empty")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(self.prediction_horizon) or self.prediction_horizon < 0.0:
            raise ValueError("prediction_horizon must be finite and non-negative")
        if not isinstance(self.motion_type, MotionType):
            try:
                object.__setattr__(self, "motion_type", MotionType(self.motion_type))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unsupported motion_type: {self.motion_type}") from exc
        self._validate_vector(self.target_initial_position, "target_initial_position")
        self._validate_vector(self.target_velocity, "target_velocity")

    @staticmethod
    def _validate_vector(value: Tuple[float, float, float], name: str) -> None:
        if len(value) != 3 or not all(math.isfinite(float(component)) for component in value):
            raise ValueError(f"{name} must contain three finite values")

    def create_motion_model(self) -> TargetMotionModel:
        """Create the configured existing Scene motion model."""
        if self.motion_type == MotionType.STRAIGHT:
            return StraightMotionModel(self.target_velocity)
        if self.motion_type == MotionType.CIRCULAR:
            return CircularMotionModel(
                center=self.circular_center,
                radius=self.circular_radius,
                angular_speed=self.circular_angular_speed,
                initial_phase=self.circular_initial_phase,
                plane=self.circular_plane,
            )
        if self.motion_type == MotionType.FIGURE8:
            return Figure8MotionModel(
                center=self.figure8_center,
                amplitude_x=self.figure8_amplitude_x,
                amplitude_y=self.figure8_amplitude_y,
                angular_speed=self.figure8_angular_speed,
                initial_phase=self.figure8_initial_phase,
            )
        return BoundedRandomMotionModel(
            seed=self.seed,
            max_acceleration=self.random_max_acceleration,
            max_speed=self.random_max_speed,
            bounds_min=self.random_bounds_min,
            bounds_max=self.random_bounds_max,
        )

    def to_simulator_config(self):
        """Build a core simulator configuration without changing core behavior."""
        sensor_config = replace(self.sensor_config, camera_config=self.camera_config)
        disturbance_config = replace(self.disturbance_config, seed=self.seed)
        from core.simulator import SimulatorConfig

        return SimulatorConfig(
            dt=self.dt,
            target_position=self.target_initial_position,
            target_velocity=self.target_velocity,
            motion_model=self.create_motion_model(),
            camera_config=self.camera_config,
            sensor_config=sensor_config,
            disturbance_config=disturbance_config,
            detection_config=self.detection_config,
            tracking_config=self.tracking_config,
            kalman_config=self.kalman_config,
            gimbal_config=self.gimbal_config,
            control_config=self.control_config,
            prediction_horizon=self.prediction_horizon,
        )

    def metadata(self) -> dict[str, object]:
        """Return stable primitive metadata suitable for benchmark results."""
        return {
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "dt": self.dt,
            "motion_type": self.motion_type.value,
            "target_initial_position": tuple(self.target_initial_position),
            "target_velocity": tuple(self.target_velocity),
            "camera_width": self.camera_config.width,
            "camera_height": self.camera_config.height,
            "camera_fov_x_deg": self.camera_config.fov_x_deg,
            "camera_fov_y_deg": self.camera_config.fov_y_deg,
        }