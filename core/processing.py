"""Shared truth-free operational image-processing pipeline."""

from dataclasses import dataclass
import math
from typing import List, Optional

import numpy as np

from core.ai_identity import BeaconFeatures, BeaconIdentityClassifier, BeaconIdentityResult
from core.confidence import ConfidenceEvidence, ConfidenceFusion, ConfidenceResult
from core.detection import BeaconDetector, DetectionCandidate, DetectionConfig
from core.gimbal import CoarsePointingController, ControlCommand, ControlConfig
from core.kalman import KalmanConfig, KalmanFilter2D, KalmanState
from core.modulation import ModulationConfig, ModulationObservation, ModulationResult, ModulationVerifier
from core.prediction import PredictionResult, PredictionState, TargetPositionPredictor
from core.tracking import TargetTracker, TrackedTargetState, TrackingConfig, TrackingObservation
from core.trust import ModelVisionTrustEvaluator, TrustObservation, TrustResult


@dataclass(frozen=True)
class OperationalFrame:
    """Truth-free output of one operational image-processing step."""

    timestamp: float
    frame_index: int
    candidates: List[DetectionCandidate]
    identity: Optional[BeaconIdentityResult]
    modulation: ModulationResult
    tracking: TrackedTargetState
    kalman: Optional[KalmanState]
    prediction: Optional[PredictionResult]
    confidence: ConfidenceResult
    trust: Optional[TrustResult]
    command: Optional[ControlCommand]


class OperationalProcessor:
    """Run the common detector-to-command pipeline without simulation truth."""

    def __init__(
        self,
        detection_config: Optional[DetectionConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        modulation_config: Optional[ModulationConfig] = None,
        kalman_config: Optional[KalmanConfig] = None,
        control_config: Optional[ControlConfig] = None,
        prediction_horizon: float = 0.0,
    ) -> None:
        self.detector = BeaconDetector(detection_config)
        self.identity_classifier = self._make_default_classifier()
        self.modulation = ModulationVerifier(modulation_config)
        self.tracker = TargetTracker(tracking_config)
        self.kalman = KalmanFilter2D(kalman_config)
        self.predictor = TargetPositionPredictor()
        self.confidence_fusion = ConfidenceFusion()
        self.trust_evaluator = ModelVisionTrustEvaluator()
        self.controller = CoarsePointingController(control_config)
        self.prediction_horizon = prediction_horizon

    @staticmethod
    def _make_default_classifier() -> BeaconIdentityClassifier:
        """Fit the deterministic Stage 1N observable-feature classifier."""
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
        """Reset stateful operational components for a new frame sequence."""
        self.modulation.reset()
        self.tracker.reset()
        self.kalman.reset()
        self.controller.reset(0.0)

    def process_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> OperationalFrame:
        """Process one externally supplied monochrome image."""
        candidates = self.detector.detect(image)
        identity: Optional[BeaconIdentityResult] = None
        selected_candidate: Optional[DetectionCandidate] = None
        modulation = self.modulation.verify()

        for candidate in candidates:
            candidate_identity = self.identity_classifier.classify(
                self.features_from_candidate(candidate)
            )
            if identity is None or candidate_identity.probability > identity.probability:
                identity = candidate_identity
                selected_candidate = candidate

        observation: Optional[TrackingObservation] = None
        if selected_candidate is not None and identity is not None:
            self.modulation.add_observation(
                ModulationObservation(
                    timestamp=timestamp,
                    intensity=selected_candidate.peak_intensity,
                    candidate_id=0,
                    frame_index=frame_index,
                    centroid_x=selected_candidate.centroid_x,
                    centroid_y=selected_candidate.centroid_y,
                )
            )
            modulation = self.modulation.verify()
            observation = TrackingObservation(
                timestamp=timestamp,
                centroid_x=selected_candidate.centroid_x,
                centroid_y=selected_candidate.centroid_y,
                detection_confidence=min(1.0, selected_candidate.peak_intensity / 255.0),
                ai_confidence=identity.confidence,
                ai_is_beacon=identity.is_beacon,
                modulation_confidence=modulation.confidence,
                modulation_verified=modulation.verified,
                candidate_id=0,
            )

        tracking = self.tracker.update(observation=observation, timestamp=timestamp)
        kalman = self.kalman.get_state()
        if tracking.position_x is not None and tracking.position_y is not None:
            if kalman is None:
                kalman = self.kalman.initialize(
                    timestamp,
                    tracking.position_x,
                    tracking.position_y,
                    tracking.velocity_x,
                    tracking.velocity_y,
                )
            elif timestamp > kalman.timestamp:
                kalman = self.kalman.update(
                    timestamp, tracking.position_x, tracking.position_y
                )
        elif kalman is not None and timestamp > kalman.timestamp:
            kalman = self.kalman.predict(timestamp)

        prediction: Optional[PredictionResult] = None
        trust: Optional[TrustResult] = None
        command: Optional[ControlCommand] = None
        if kalman is not None:
            prediction = self.predictor.predict(
                PredictionState(
                    kalman.timestamp,
                    kalman.position_x,
                    kalman.position_y,
                    kalman.velocity_x,
                    kalman.velocity_y,
                ),
                self.prediction_horizon,
            )
            if selected_candidate is not None:
                trust = self.trust_evaluator.evaluate(
                    TrustObservation(
                        observed_x=selected_candidate.centroid_x,
                        observed_y=selected_candidate.centroid_y,
                        predicted_x=prediction.predicted_position_x,
                        predicted_y=prediction.predicted_position_y,
                    )
                )
            confidence = self.confidence_fusion.fuse(
                ConfidenceEvidence(
                    tracking.confidence if identity is not None else 0.0,
                    identity.confidence if identity is not None else 0.0,
                    modulation.confidence,
                )
            )
            if tracking.is_locked:
                height, width = image.shape[:2]
                command = self.controller.compute_command(
                    2.0 * (width / 2.0) - prediction.predicted_position_x,
                    prediction.predicted_position_y,
                    width / 2.0,
                    height / 2.0,
                    timestamp,
                )
        else:
            confidence = self.confidence_fusion.fuse(
                ConfidenceEvidence(0.0, 0.0, modulation.confidence)
            )

        return OperationalFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            candidates=candidates,
            identity=identity,
            modulation=modulation,
            tracking=tracking,
            kalman=kalman,
            prediction=prediction,
            confidence=confidence,
            trust=trust,
            command=command,
        )

    @staticmethod
    def features_from_candidate(candidate: DetectionCandidate) -> BeaconFeatures:
        """Convert one detector measurement into observable classifier features."""
        width = float(candidate.bbox_width)
        height = float(candidate.bbox_height)
        aspect_ratio = width / height
        perimeter = 2.0 * (width + height)
        circularity = min(
            1.0,
            max(0.0, 4.0 * math.pi * candidate.area / (perimeter * perimeter)),
        )
        snr = candidate.peak_intensity / max(candidate.mean_intensity, 1e-6)
        return BeaconFeatures(
            area=candidate.area,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            circularity=circularity,
            peak_intensity=candidate.peak_intensity,
            mean_intensity=candidate.mean_intensity,
            intensity_std=0.0,
            signal_to_noise_ratio=snr,
        )