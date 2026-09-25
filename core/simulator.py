"""Closed-loop virtual FSOC camera tracking simulator."""

from dataclasses import dataclass, field
import copy
import math
import time
from typing import List, Optional, Tuple

from core.ai_identity import BeaconFeatures, BeaconIdentityClassifier, BeaconIdentityResult
from core.confidence import ConfidenceEvidence, ConfidenceFusion, ConfidenceResult
from core.detection import BeaconDetector, DetectionCandidate, DetectionConfig
from core.disturbances import DisturbanceConfig, DisturbedFrame, ImageDisturbanceModel
from core.gimbal import CoarsePointingController, ControlCommand, ControlConfig, GimbalConfig, GimbalState, VirtualGimbal
from core.geometry import CameraConfig
from core.kalman import KalmanConfig, KalmanFilter2D, KalmanState
from core.modulation import ModulationConfig, ModulationObservation, ModulationResult, ModulationVerifier
from core.prediction import PredictionResult, PredictionState, TargetPositionPredictor
from core.scene import CameraState, Scene, StraightMotionModel, TargetMotionModel, TargetState
from core.sensor import MonochromeSensor, SensorConfig, SensorFrame
from core.tracking import TargetTracker, TrackedTargetState, TrackingConfig, TrackingObservation
from core.trust import ModelVisionTrustEvaluator, TrustObservation, TrustResult


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
        self.detector = BeaconDetector(self.config.detection_config)
        self.identity_classifier = self._make_default_classifier()
        self.modulation = ModulationVerifier(self.config.modulation_config)
        self.tracker = TargetTracker(self.config.tracking_config)
        self.kalman = KalmanFilter2D(self.config.kalman_config)
        self.predictor = TargetPositionPredictor()
        self.confidence_fusion = ConfidenceFusion()
        self.trust_evaluator = ModelVisionTrustEvaluator()
        self.gimbal = VirtualGimbal(self.config.gimbal_config)
        self.controller = CoarsePointingController(self.config.control_config)
        self.reset()

    @staticmethod
    def _make_default_classifier() -> BeaconIdentityClassifier:
        """Fit a deterministic lightweight model from observable beacon-like features."""
        beacon = [
            BeaconFeatures(25.0, 10.0, 10.0, 1.0, 0.8, 240.0, 180.0, 20.0, 12.0),
            BeaconFeatures(45.0, 12.0, 12.0, 1.0, 0.8, 230.0, 160.0, 20.0, 10.0),
            BeaconFeatures(25.0, 10.0, 10.0, 1.0, 0.8, 150.0, 110.0, 20.0, 8.0),
        ]
        clutter = [
            BeaconFeatures(120.0, 30.0, 4.0, 7.5, 0.2, 80.0, 40.0, 5.0, 2.0),
            BeaconFeatures(180.0, 40.0, 5.0, 8.0, 0.1, 60.0, 30.0, 5.0, 1.0),
        ]
        return BeaconIdentityClassifier().fit(beacon + clutter, [1, 1, 1, 0, 0])

    def reset(self) -> None:
        """Reset all stateful components while preserving the configured motion model."""
        self.scene.time = 0.0
        self.scene.motion_model.reset(self._initial_target.position)
        self.scene.target_state = self.scene.motion_model.get_initial_state(0.0, TargetState(position=self._initial_target.position.copy()))
        self.scene.camera_state = copy.deepcopy(self._initial_camera)
        self.disturbances.reset()
        self.modulation.reset()
        self.tracker.reset()
        self.kalman.reset()
        self.gimbal.reset(0.0)
        self.controller.reset(0.0)
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
        candidates = self.detector.detect(disturbed_frame.image)

        identity: Optional[BeaconIdentityResult] = None
        selected_candidate: Optional[DetectionCandidate] = None
        modulation_result = self.modulation.verify()
        observation: Optional[TrackingObservation] = None
        for candidate in candidates:
            features = self._features_from_candidate(candidate)
            candidate_identity = self.identity_classifier.classify(features)
            if identity is None or candidate_identity.probability > identity.probability:
                identity = candidate_identity
                selected_candidate = candidate

        if selected_candidate is not None and identity is not None:
            # Candidate IDs are temporal stream IDs here because DetectionCandidate
            # is an immutable image measurement without an intrinsic ID.
            candidate_id = 0
            self.modulation.add_observation(ModulationObservation(timestamp=timestamp, intensity=selected_candidate.peak_intensity, candidate_id=candidate_id, frame_index=self.frame_index, centroid_x=selected_candidate.centroid_x, centroid_y=selected_candidate.centroid_y))
            modulation_result = self.modulation.verify()
            detection_confidence = min(1.0, selected_candidate.peak_intensity / 255.0)
            observation = TrackingObservation(timestamp=timestamp, centroid_x=selected_candidate.centroid_x, centroid_y=selected_candidate.centroid_y, detection_confidence=detection_confidence, ai_confidence=identity.confidence, ai_is_beacon=identity.is_beacon, modulation_confidence=modulation_result.confidence, modulation_verified=modulation_result.verified, candidate_id=candidate_id)

        tracking = self.tracker.update(observation=observation, timestamp=timestamp)
        kalman_state: Optional[KalmanState] = self.kalman.get_state()
        if tracking.position_x is not None and tracking.position_y is not None:
            if kalman_state is None:
                kalman_state = self.kalman.initialize(timestamp, tracking.position_x, tracking.position_y, tracking.velocity_x, tracking.velocity_y)
            elif timestamp > kalman_state.timestamp:
                kalman_state = self.kalman.update(timestamp, tracking.position_x, tracking.position_y)
        elif kalman_state is not None and timestamp > kalman_state.timestamp:
            kalman_state = self.kalman.predict(timestamp)

        prediction: Optional[PredictionResult] = None
        command: Optional[ControlCommand] = None
        trust: Optional[TrustResult] = None
        if kalman_state is not None:
            prediction = self.predictor.predict(PredictionState(kalman_state.timestamp, kalman_state.position_x, kalman_state.position_y, kalman_state.velocity_x, kalman_state.velocity_y), self.config.prediction_horizon)
            if selected_candidate is not None:
                trust = self.trust_evaluator.evaluate(TrustObservation(selected_candidate.centroid_x, selected_candidate.centroid_y, prediction.predicted_position_x, prediction.predicted_position_y))
            confidence = self.confidence_fusion.fuse(ConfidenceEvidence(tracking.confidence if identity is not None else 0.0, identity.confidence if identity is not None else 0.0, modulation_result.confidence))
            if tracking.is_locked:
                center_x, center_y = self.config.camera_config.cx, self.config.camera_config.cy
                command = self.controller.compute_command(
                    2.0 * center_x - prediction.predicted_position_x,
                    prediction.predicted_position_y,
                    center_x,
                    center_y,
                    timestamp,
                )
        else:
            confidence = self.confidence_fusion.fuse(ConfidenceEvidence(0.0, 0.0, modulation_result.confidence))

        if command is not None:
            gimbal_state = self.gimbal.update(command.pan_rate_command_rad_s, command.tilt_rate_command_rad_s, timestamp + self.config.dt)
            self.scene.camera_state.pan_rad = gimbal_state.pan_angle_rad
            self.scene.camera_state.tilt_rad = gimbal_state.tilt_angle_rad
        else:
            gimbal_state = self.gimbal.get_state()

        self.scene.step(self.config.dt)
        frame = SimulatorFrame(timestamp, self.frame_index, target_truth, camera_state, gimbal_state, sensor_frame, disturbed_frame, candidates, identity, modulation_result, tracking, kalman_state, prediction, confidence, trust, command, time.perf_counter() - started)
        self.history.append(frame)
        self.frame_index += 1
        return frame

    @staticmethod
    def _features_from_candidate(candidate: DetectionCandidate) -> BeaconFeatures:
        width = float(candidate.bbox_width)
        height = float(candidate.bbox_height)
        aspect_ratio = width / height
        perimeter = 2.0 * (width + height)
        circularity = min(1.0, max(0.0, 4.0 * math.pi * candidate.area / (perimeter * perimeter)))
        snr = candidate.peak_intensity / max(candidate.mean_intensity, 1e-6)
        return BeaconFeatures(area=candidate.area, width=width, height=height, aspect_ratio=aspect_ratio, circularity=circularity, peak_intensity=candidate.peak_intensity, mean_intensity=candidate.mean_intensity, intensity_std=0.0, signal_to_noise_ratio=snr)