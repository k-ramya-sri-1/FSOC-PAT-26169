"""Closed-loop virtual FSOC camera tracking simulator."""

from dataclasses import dataclass, field
import copy
import math
import time
from typing import List, Optional, Tuple

from core.ai_identity import BeaconIdentityResult
from core.confidence import ConfidenceResult
from core.detection import DetectionCandidate, DetectionConfig
from core.disturbances import DisturbanceConfig, DisturbedFrame, ImageDisturbanceModel
from core.gimbal import ControlCommand, ControlConfig, GimbalConfig, GimbalState, VirtualGimbal
from core.geometry import CameraConfig
from core.kalman import KalmanConfig, KalmanState
from core.modulation import ModulationConfig, ModulationResult
from core.prediction import PredictionResult
from core.processing import OperationalProcessor
from core.scene import CameraState, Scene, StraightMotionModel, TargetMotionModel, TargetState
from core.sensor import MonochromeSensor, SensorConfig, SensorFrame
from core.tracking import TrackedTargetState, TrackingConfig
from core.trust import TrustResult


@dataclass
class SimulatorConfig:
    """Configuration for a deterministic closed-loop simulation run."""

    dt: float = 1.0 / 60.0
    target_position: Tuple[float, float, float] = (0.5, 0.2, 100.0)
    target_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion_model: Optional[TargetMotionModel] = None
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    sensor_config: Optional[SensorConfig] = None
    disturbance_config: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    modulation_config: ModulationConfig = field(default_factory=ModulationConfig)
    tracking_config: TrackingConfig = field(default_factory=lambda: TrackingConfig(minimum_acquisition_observations=1))
    kalman_config: KalmanConfig = field(default_factory=KalmanConfig)
    gimbal_config: GimbalConfig = field(default_factory=GimbalConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    prediction_horizon: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.prediction_horizon < 0.0 or not math.isfinite(self.prediction_horizon):
            raise ValueError("prediction_horizon must be finite and non-negative")
        if self.sensor_config is None:
            self.sensor_config = SensorConfig(camera_config=self.camera_config, modulation_depth=0.25)
        elif self.sensor_config.camera_config != self.camera_config:
            self.camera_config = self.sensor_config.camera_config


@dataclass(frozen=True)
class SimulatorFrame:
    """Per-frame observations, estimates, commands, and metrics/debug truth."""

    timestamp: float
    frame_index: int
    target_truth: TargetState
    camera_state: CameraState
    gimbal_state: GimbalState
    sensor_frame: SensorFrame
    disturbed_frame: DisturbedFrame
    candidates: List[DetectionCandidate]
    identity: Optional[BeaconIdentityResult]
    modulation: ModulationResult
    tracking: TrackedTargetState
    kalman: Optional[KalmanState]
    prediction: Optional[PredictionResult]
    confidence: ConfidenceResult
    trust: Optional[TrustResult]
    command: Optional[ControlCommand]
    processing_time_seconds: float


class ClosedLoopSimulator:
    """Orchestrate the existing scene, vision, estimation, and gimbal components."""

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        self.config = config if config is not None else SimulatorConfig()
        motion_model = copy.deepcopy(self.config.motion_model)
        if motion_model is None:
            motion_model = StraightMotionModel(self.config.target_velocity)
        self._initial_target = TargetState(position=self.config.target_position)
        self._initial_camera = CameraState()
        self.scene = Scene(target_state=self._initial_target, camera_state=copy.deepcopy(self._initial_camera), motion_model=motion_model)
        self.sensor = MonochromeSensor(self.config.sensor_config)
        self.disturbances = ImageDisturbanceModel(self.config.disturbance_config)
        self.processor = OperationalProcessor(
            detection_config=self.config.detection_config,
            tracking_config=self.config.tracking_config,
            modulation_config=self.config.modulation_config,
            kalman_config=self.config.kalman_config,
            control_config=self.config.control_config,
            prediction_horizon=self.config.prediction_horizon,
        )
        self.detector = self.processor.detector
        self.identity_classifier = self.processor.identity_classifier
        self.modulation = self.processor.modulation
        self.tracker = self.processor.tracker
        self.kalman = self.processor.kalman
        self.predictor = self.processor.predictor
        self.confidence_fusion = self.processor.confidence_fusion
        self.trust_evaluator = self.processor.trust_evaluator
        self.gimbal = VirtualGimbal(self.config.gimbal_config)
        self.controller = self.processor.controller
        self.reset()

    def reset(self) -> None:
        """Reset all stateful components while preserving the configured motion model."""
        self.scene.time = 0.0
        self.scene.motion_model.reset(self._initial_target.position)
        self.scene.target_state = self.scene.motion_model.get_initial_state(0.0, TargetState(position=self._initial_target.position.copy()))
        self.scene.camera_state = copy.deepcopy(self._initial_camera)
        self.disturbances.reset()
        self.processor.reset()
        self.gimbal.reset(0.0)
        self.frame_index = 0
        self.history: List[SimulatorFrame] = []

    def step(self) -> SimulatorFrame:
        """Advance one timestamped frame through the complete feedback loop."""
        started = time.perf_counter()
        timestamp = self.scene.time
        target_truth = copy.deepcopy(self.scene.target_state)
        camera_state = copy.deepcopy(self.scene.camera_state)
        sensor_frame = self.sensor.capture(self.scene, timestamp)
        disturbed_frame = self.disturbances.apply(sensor_frame)
        operational = self.processor.process_frame(
            disturbed_frame.image,
            timestamp,
            self.frame_index,
        )

        if operational.command is not None:
            gimbal_state = self.gimbal.update(
                operational.command.pan_rate_command_rad_s,
                operational.command.tilt_rate_command_rad_s,
                timestamp + self.config.dt,
            )
            self.scene.camera_state.pan_rad = gimbal_state.pan_angle_rad
            self.scene.camera_state.tilt_rad = gimbal_state.tilt_angle_rad
        else:
            gimbal_state = self.gimbal.get_state()

        self.scene.step(self.config.dt)
        frame = SimulatorFrame(timestamp, self.frame_index, target_truth, camera_state, gimbal_state, sensor_frame, disturbed_frame, operational.candidates, operational.identity, operational.modulation, operational.tracking, operational.kalman, operational.prediction, operational.confidence, operational.trust, operational.command, time.perf_counter() - started)
        self.history.append(frame)
        self.frame_index += 1
        return frame
