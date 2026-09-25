"""Recorded-video benchmark runner using the common operational tracking core."""

from dataclasses import dataclass, field
from pathlib import Path
import math
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from core.ai_identity import BeaconIdentityResult
from core.confidence import ConfidenceResult
from core.detection import DetectionCandidate, DetectionConfig
from core.gimbal import ControlCommand, ControlConfig
from core.kalman import KalmanConfig, KalmanState
from core.modulation import ModulationConfig, ModulationResult
from core.prediction import PredictionResult
from core.processing import OperationalProcessor
from core.tracking import TrackedTargetState, TrackingConfig
from core.trust import TrustResult


@dataclass(frozen=True)
class MP4BenchmarkConfig:
    """Configuration for one sequential recorded-video evaluation."""

    video_path: Union[str, Path]
    scenario_name: str = "mp4"
    max_frames: Optional[int] = None
    max_duration_seconds: Optional[float] = None
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    tracking_config: TrackingConfig = field(
        default_factory=lambda: TrackingConfig(
            minimum_acquisition_observations=1
        )
    )
    modulation_config: ModulationConfig = field(default_factory=ModulationConfig)
    kalman_config: KalmanConfig = field(default_factory=KalmanConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    prediction_horizon: float = 0.0

    def __post_init__(self) -> None:
        if not self.scenario_name:
            raise ValueError("scenario_name must not be empty")
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError("max_frames must be non-negative when provided")
        if self.max_duration_seconds is not None and self.max_duration_seconds < 0.0:
            raise ValueError("max_duration_seconds must be non-negative when provided")
        if not math.isfinite(self.prediction_horizon) or self.prediction_horizon < 0.0:
            raise ValueError("prediction_horizon must be finite and non-negative")


@dataclass(frozen=True)
class MP4TelemetryFrame:
    """Truth-free operational telemetry for one decoded video frame."""

    timestamp: float
    frame_index: int
    frame_shape: Tuple[int, ...]
    frame_dtype: str
    candidates: Tuple[DetectionCandidate, ...]
    identity: Optional[BeaconIdentityResult]
    modulation: ModulationResult
    tracking: TrackedTargetState
    kalman: Optional[KalmanState]
    prediction: Optional[PredictionResult]
    confidence: ConfidenceResult
    trust: Optional[TrustResult]
    command: Optional[ControlCommand]
    processing_time_seconds: float


@dataclass(frozen=True)
class MP4BenchmarkResult:
    """Result of a recorded-video run, including safe failure information."""

    video_path: str
    scenario_name: str
    opened: bool
    frame_count: int
    telemetry_frames: Tuple[MP4TelemetryFrame, ...]
    video_fps: Optional[float]
    reported_frame_count: Optional[int]
    frame_width: Optional[int]
    frame_height: Optional[int]
    timing_source: str
    error: Optional[str] = None


class MP4BenchmarkRunner:
    """Read video frames sequentially through the existing operational components."""

    def __init__(self, config: MP4BenchmarkConfig) -> None:
        if not isinstance(config, MP4BenchmarkConfig):
            raise TypeError("config must be an MP4BenchmarkConfig")
        self.config = config
        self.capture: Optional[cv2.VideoCapture] = None

    def run(self) -> MP4BenchmarkResult:
        """Process frames until EOF, a configured limit, or an unrecoverable read error."""
        path = str(self.config.video_path)
        capture = cv2.VideoCapture(path)
        self.capture = capture
        telemetry = []
        error: Optional[str] = None

        try:
            if not capture.isOpened():
                return self._result(capture, path, False, telemetry, "Unable to open video")

            video_fps = _valid_positive(capture.get(cv2.CAP_PROP_FPS))
            reported_frame_count = _valid_nonnegative_int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            frame_width = _valid_nonnegative_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = _valid_nonnegative_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            timing_source = "CAP_PROP_POS_MSEC" if video_fps is not None else "frame_index_fallback"
            previous_timestamp: Optional[float] = None
            frame_index = 0
            processor = self._create_processor()

            while True:
                if self.config.max_frames is not None and frame_index >= self.config.max_frames:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp, timestamp_source = self._timestamp(
                    capture, frame_index, video_fps, previous_timestamp
                )
                if timestamp_source != timing_source:
                    timing_source = timestamp_source
                if (
                    self.config.max_duration_seconds is not None
                    and timestamp > self.config.max_duration_seconds
                ):
                    break
                telemetry.append(self._process_frame(processor, frame, frame_index, timestamp))
                previous_timestamp = timestamp
                frame_index += 1

            return MP4BenchmarkResult(
                video_path=path,
                scenario_name=self.config.scenario_name,
                opened=True,
                frame_count=len(telemetry),
                telemetry_frames=tuple(telemetry),
                video_fps=video_fps,
                reported_frame_count=reported_frame_count,
                frame_width=frame_width,
                frame_height=frame_height,
                timing_source=timing_source,
            )
        except Exception as exc:
            error = str(exc)
            return self._result(capture, path, True, telemetry, error)
        finally:
            capture.release()

    def _create_processor(self) -> OperationalProcessor:
        """Create the shared operational processor without scene or sensor truth."""
        return OperationalProcessor(
            detection_config=self.config.detection_config,
            tracking_config=self.config.tracking_config,
            modulation_config=self.config.modulation_config,
            kalman_config=self.config.kalman_config,
            control_config=self.config.control_config,
            prediction_horizon=self.config.prediction_horizon,
        )

    def _process_frame(
        self,
        processor: OperationalProcessor,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> MP4TelemetryFrame:
        started = time.perf_counter()
        image = _to_monochrome(frame)
        operational = processor.process_frame(image, timestamp, frame_index)

        return MP4TelemetryFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            frame_shape=tuple(frame.shape),
            frame_dtype=str(frame.dtype),
            candidates=tuple(operational.candidates),
            identity=operational.identity,
            modulation=operational.modulation,
            tracking=operational.tracking,
            kalman=operational.kalman,
            prediction=operational.prediction,
            confidence=operational.confidence,
            trust=operational.trust,
            command=operational.command,
            processing_time_seconds=time.perf_counter() - started,
        )

    def _timestamp(
        self,
        capture: cv2.VideoCapture,
        frame_index: int,
        video_fps: Optional[float],
        previous_timestamp: Optional[float],
    ) -> Tuple[float, str]:
        timestamp_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
        if math.isfinite(timestamp_msec) and timestamp_msec >= 0.0:
            timestamp = timestamp_msec / 1000.0
            if previous_timestamp is None or timestamp > previous_timestamp:
                return timestamp, "CAP_PROP_POS_MSEC"
        if video_fps is not None:
            timestamp = frame_index / video_fps
            if previous_timestamp is None or timestamp > previous_timestamp:
                return timestamp, "FPS/frame-index fallback"
        step = 1.0 / video_fps if video_fps is not None else 1.0
        timestamp = (previous_timestamp + step) if previous_timestamp is not None else 0.0
        return timestamp, "frame-index fallback"

    def _result(
        self,
        capture: cv2.VideoCapture,
        path: str,
        opened: bool,
        telemetry: list[MP4TelemetryFrame],
        error: str,
    ) -> MP4BenchmarkResult:
        fps = _valid_positive(capture.get(cv2.CAP_PROP_FPS))
        return MP4BenchmarkResult(
            video_path=path,
            scenario_name=self.config.scenario_name,
            opened=opened,
            frame_count=len(telemetry),
            telemetry_frames=tuple(telemetry),
            video_fps=fps,
            reported_frame_count=None,
            frame_width=None,
            frame_height=None,
            timing_source="unavailable",
            error=error,
        )


def _to_monochrome(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported video frame shape: {frame.shape}")


def _valid_positive(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) and value > 0.0 else None


def _valid_nonnegative_int(value: float) -> Optional[int]:
    return int(value) if math.isfinite(value) and value >= 0.0 else None