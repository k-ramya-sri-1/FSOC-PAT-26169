"""Controlled, observational validation scenarios for tracking loss and recovery."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from core.ai_identity import BeaconFeatures, BeaconIdentityResult
from core.confidence import ConfidenceResult
from core.detection import DetectionCandidate, DetectionConfig
from core.gimbal import ControlConfig, ControlCommand
from core.kalman import KalmanConfig, KalmanState
from core.modulation import ModulationConfig
from core.prediction import PredictionResult
from core.processing import OperationalProcessor
from core.tracking import TrackedTargetState, TrackingConfig
from core.trust import TrustResult


@dataclass(frozen=True)
class TrackingStressFrame:
    """Truth-free record of operational outputs for one controlled image frame."""

    frame_index: int
    timestamp: float
    observation_supplied: bool
    candidate_centroids: Tuple[Tuple[float, float], ...]
    selected_candidate_centroid: Optional[Tuple[float, float]]
    identity: Optional[BeaconIdentityResult]
    tracking: TrackedTargetState
    kalman: Optional[KalmanState]
    prediction: Optional[PredictionResult]
    confidence: ConfidenceResult
    trust: Optional[TrustResult]
    command: Optional[ControlCommand]
    kalman_calls: Tuple[str, ...]


@dataclass(frozen=True)
class TrackingStressResult:
    """Ordered outputs from one deterministic diagnostic scenario."""

    scenario_name: str
    frames: Tuple[TrackingStressFrame, ...]


class TrackingStressHarness:
    """Drive the existing OperationalProcessor with controlled candidate sequences."""

    def run_temporary_detection_loss(self) -> TrackingStressResult:
        center = _candidate(320.0, 240.0)
        return self._run_sequence(
            "temporary_detection_loss",
            ((center,), (center,), (center,), (), (), (center,)),
            _tracking_config(max_missed_frames=5),
        )

    def run_extended_detection_loss(self) -> TrackingStressResult:
        center = _candidate(320.0, 240.0)
        return self._run_sequence(
            "extended_detection_loss",
            ((center,), (center,), (), (), (), (), (), ()),
            _tracking_config(max_missed_frames=3),
        )

    def run_reacquisition(self) -> TrackingStressResult:
        center = _candidate(320.0, 240.0)
        near = _candidate(324.0, 240.0)
        return self._run_sequence(
            "reacquisition",
            ((center,), (center,), (), (), (near,), (near,)),
            _tracking_config(max_missed_frames=1),
        )

    def run_distant_reacquisition_rejection(self) -> TrackingStressResult:
        center = _candidate(320.0, 240.0)
        distant = _candidate(400.0, 240.0)
        near = _candidate(325.0, 240.0)
        return self._run_sequence(
            "distant_reacquisition_rejection",
            ((center,), (center,), (), (), (distant,), (near,), (near,)),
            _tracking_config(
                max_missed_frames=1,
                reacquisition_max_distance_px=30.0,
            ),
        )

    def run_multiple_candidate_selection(self) -> TrackingStressResult:
        center = _candidate(320.0, 240.0)
        farther_high_ai = _candidate(400.0, 240.0, peak=250.0)
        nearer_low_ai = _candidate(322.0, 240.0, peak=220.0)

        def controlled_classifier(features: BeaconFeatures) -> BeaconIdentityResult:
            if features.peak_intensity >= 249.0 and features.peak_intensity < 255.0:
                return BeaconIdentityResult(True, 1.0, 0.99, 1)
            if features.peak_intensity < 249.0:
                return BeaconIdentityResult(True, 0.8, 0.80, 1)
            return BeaconIdentityResult(True, 0.9, 0.95, 1)

        return self._run_sequence(
            "multiple_candidate_selection_vs_gating",
            ((center,), (center,), (farther_high_ai, nearer_low_ai)),
            _tracking_config(
                max_missed_frames=5,
                tracking_max_distance_px=40.0,
            ),
            classifier=controlled_classifier,
        )

    def run_all(self) -> Tuple[TrackingStressResult, ...]:
        """Run the five diagnostic scenarios in a fixed deterministic order."""
        return (
            self.run_temporary_detection_loss(),
            self.run_extended_detection_loss(),
            self.run_reacquisition(),
            self.run_distant_reacquisition_rejection(),
            self.run_multiple_candidate_selection(),
        )

    @staticmethod
    def _run_sequence(
        scenario_name: str,
        candidate_frames: Sequence[Sequence[DetectionCandidate]],
        tracking_config: TrackingConfig,
        classifier: Optional[Callable[[BeaconFeatures], BeaconIdentityResult]] = None,
    ) -> TrackingStressResult:
        processor = OperationalProcessor(
            detection_config=DetectionConfig(),
            tracking_config=tracking_config,
            modulation_config=ModulationConfig(),
            kalman_config=KalmanConfig(),
            control_config=ControlConfig(),
        )
        frame_candidates = iter(candidate_frames)
        processor.detector.detect = lambda image: list(next(frame_candidates))
        if classifier is not None:
            processor.identity_classifier.classify = classifier
        else:
            processor.identity_classifier.classify = lambda features: BeaconIdentityResult(
                is_beacon=True,
                confidence=1.0,
                probability=1.0,
                predicted_class=1,
            )

        selected_by_frame = {}
        original_add_observation = processor.modulation.add_observation

        def record_selected_candidate(observation):
            selected_by_frame[observation.frame_index] = (
                observation.centroid_x,
                observation.centroid_y,
            )
            original_add_observation(observation)

        processor.modulation.add_observation = record_selected_candidate

        kalman_calls = []
        active_frame = [0]
        original_initialize = processor.kalman.initialize
        original_update = processor.kalman.update
        original_predict = processor.kalman.predict

        def record_initialize(
            timestamp,
            position_x,
            position_y,
            velocity_x=0.0,
            velocity_y=0.0,
        ):
            kalman_calls.append((active_frame[0], "initialize"))
            return original_initialize(
                timestamp, position_x, position_y, velocity_x, velocity_y
            )

        def record_update(timestamp, position_x, position_y):
            kalman_calls.append((active_frame[0], "update"))
            return original_update(timestamp, position_x, position_y)

        def record_predict(timestamp):
            kalman_calls.append((active_frame[0], "predict"))
            return original_predict(timestamp)

        processor.kalman.initialize = record_initialize
        processor.kalman.update = record_update
        processor.kalman.predict = record_predict

        frames = []
        image = np.zeros((480, 640), dtype=np.float32)
        for frame_index, candidates in enumerate(candidate_frames):
            active_frame[0] = frame_index
            timestamp = frame_index / 10.0
            start_call_index = len(kalman_calls)
            operational = processor.process_frame(image, timestamp, frame_index)
            frame_calls = tuple(
                method
                for called_frame, method in kalman_calls[start_call_index:]
                if called_frame == frame_index
            )
            frames.append(
                TrackingStressFrame(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    observation_supplied=bool(candidates),
                    candidate_centroids=tuple(
                        (candidate.centroid_x, candidate.centroid_y)
                        for candidate in operational.candidates
                    ),
                    selected_candidate_centroid=selected_by_frame.get(frame_index),
                    identity=operational.identity,
                    tracking=operational.tracking,
                    kalman=operational.kalman,
                    prediction=operational.prediction,
                    confidence=operational.confidence,
                    trust=operational.trust,
                    command=operational.command,
                    kalman_calls=frame_calls,
                )
            )
        return TrackingStressResult(scenario_name, tuple(frames))


def _tracking_config(
    max_missed_frames: int,
    tracking_max_distance_px: float = 50.0,
    reacquisition_max_distance_px: float = 50.0,
) -> TrackingConfig:
    return TrackingConfig(
        acquisition_confidence_threshold=0.6,
        tracking_confidence_threshold=0.5,
        loss_confidence_threshold=0.0,
        max_missed_frames=max_missed_frames,
        tracking_max_distance_px=tracking_max_distance_px,
        reacquisition_max_distance_px=reacquisition_max_distance_px,
        minimum_acquisition_observations=2,
    )


def _candidate(
    centroid_x: float,
    centroid_y: float,
    peak: float = 255.0,
) -> DetectionCandidate:
    return DetectionCandidate(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        area=25.0,
        peak_intensity=peak,
        mean_intensity=peak * 0.8,
        bbox_x=int(centroid_x - 2),
        bbox_y=int(centroid_y - 2),
        bbox_width=5,
        bbox_height=5,
    )
