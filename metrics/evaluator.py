"""Pure evaluation helpers for Stage 2A benchmark telemetry."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

from core.simulator import SimulatorFrame


@dataclass(frozen=True)
class LockStatistics:
    """Basic tracking availability and lock counts for a telemetry sequence."""

    total_frames: int
    available_frames: int
    locked_frames: int
    availability_ratio: float
    lock_ratio: float


@dataclass(frozen=True)
class ErrorStatistics:
    """Aggregate Euclidean tracking error over valid truth/estimate pairs."""

    valid_sample_count: int
    mean_error_pixels: Optional[float]
    maximum_error_pixels: Optional[float]
    rms_error_pixels: Optional[float]


@dataclass(frozen=True)
class ReacquisitionStatistics:
    """Durations from entering LOST to the next locked frame."""

    durations_seconds: Tuple[float, ...]
    mean_reacquisition_seconds: Optional[float]
    maximum_reacquisition_seconds: Optional[float]


@dataclass(frozen=True)
class ProcessingStatistics:
    """Processing-time and wall-clock throughput measurements."""

    frame_count: int
    mean_processing_time_seconds: Optional[float]
    processing_fps: float


@dataclass(frozen=True)
class TruthUnavailableLockStatistics:
    """Evaluation indicator, not proof of false lock.

    It identifies locked frames for which simulation truth is unavailable. This
    is not sufficient by itself to establish a false lock.
    """

    total_frames: int
    truth_unavailable_lock_frames: int
    truth_unavailable_lock_percentage: float


def frame_count(telemetry: Sequence[SimulatorFrame]) -> int:
    """Return the number of processed telemetry frames."""
    return len(telemetry)


def elapsed_simulation_time(telemetry: Sequence[SimulatorFrame]) -> float:
    """Return elapsed timestamp span, or zero for empty/single-frame telemetry."""
    if len(telemetry) < 2:
        return 0.0
    return max(0.0, float(telemetry[-1].timestamp - telemetry[0].timestamp))


def processing_times(telemetry: Sequence[SimulatorFrame]) -> Tuple[float, ...]:
    """Return measured wall-clock processing durations without modifying telemetry."""
    return tuple(float(frame.processing_time_seconds) for frame in telemetry)


def approximate_processing_fps(telemetry: Sequence[SimulatorFrame]) -> float:
    """Calculate processed frames per measured wall-clock second.

    Empty telemetry and zero-duration measurements return zero rather than raising.
    """
    durations = processing_times(telemetry)
    if any(not math.isfinite(duration) or duration < 0.0 for duration in durations):
        raise ValueError("processing durations must be finite and non-negative")
    total_duration = sum(durations)
    if not durations or total_duration <= 0.0:
        return 0.0
    return len(durations) / total_duration


def acquisition_time_seconds(
    telemetry: Sequence[SimulatorFrame],
) -> Optional[float]:
    """Return time from the first telemetry timestamp to the first locked frame."""
    if not telemetry:
        return None
    start_timestamp = telemetry[0].timestamp
    for frame in telemetry:
        if frame.tracking.is_locked:
            return float(frame.timestamp - start_timestamp)
    return None


def centroid_error_pixels(frame: SimulatorFrame) -> Optional[float]:
    """Measure estimated centroid error against the sensor's evaluation truth pixel."""
    truth_pixel = frame.sensor_frame.projected_pixel
    tracked = frame.tracking
    if truth_pixel is None or tracked.position_x is None or tracked.position_y is None:
        return None
    return math.hypot(
        tracked.position_x - truth_pixel[0],
        tracked.position_y - truth_pixel[1],
    )


def pointing_error_pixels(frame: SimulatorFrame) -> Optional[float]:
    """Measure true beacon displacement from the image center for evaluation."""
    truth_pixel = frame.sensor_frame.projected_pixel
    if truth_pixel is None:
        return None
    height, width = frame.sensor_frame.image.shape
    return math.hypot(truth_pixel[0] - width / 2.0, truth_pixel[1] - height / 2.0)


def tracking_availability(telemetry: Sequence[SimulatorFrame]) -> float:
    """Return the fraction of frames with an estimated tracked position."""
    if not telemetry:
        return 0.0
    available = sum(
        frame.tracking.position_x is not None and frame.tracking.position_y is not None
        for frame in telemetry
    )
    return available / len(telemetry)


def centroid_error_statistics(
    telemetry: Sequence[SimulatorFrame],
) -> ErrorStatistics:
    """Aggregate centroid error for frames with truth pixels and estimates."""
    errors = tuple(
        error
        for frame in telemetry
        if (error := centroid_error_pixels(frame)) is not None
    )
    if not errors:
        return ErrorStatistics(0, None, None, None)
    return ErrorStatistics(
        valid_sample_count=len(errors),
        mean_error_pixels=sum(errors) / len(errors),
        maximum_error_pixels=max(errors),
        rms_error_pixels=math.sqrt(sum(error * error for error in errors) / len(errors)),
    )


def target_loss_percentage(telemetry: Sequence[SimulatorFrame]) -> float:
    """Return LOST-state frames divided by all telemetry frames, as a percentage."""
    if not telemetry:
        return 0.0
    lost_frames = sum(frame.tracking.state.value == "LOST" for frame in telemetry)
    return 100.0 * lost_frames / len(telemetry)


def lock_retention_percentage(telemetry: Sequence[SimulatorFrame]) -> float:
    """Return the percentage of frames whose tracker reports ``is_locked``."""
    if not telemetry:
        return 0.0
    locked_frames = sum(frame.tracking.is_locked for frame in telemetry)
    return 100.0 * locked_frames / len(telemetry)


def reacquisition_statistics(
    telemetry: Sequence[SimulatorFrame],
) -> ReacquisitionStatistics:
    """Measure recovery intervals after a prior locked/tracking state.

    When telemetry exposes LOST followed by REACQUIRING, the interval ends at
    the first subsequent locked frame. If the recovery state is not exposed,
    LOST-to-next-locked is retained as an evaluation approximation.
    """
    had_tracking = False
    loss_timestamp: Optional[float] = None
    durations = []
    for frame in telemetry:
        is_lost = frame.tracking.state.value == "LOST"
        if frame.tracking.is_locked:
            if loss_timestamp is not None:
                duration = float(frame.timestamp - loss_timestamp)
                if duration >= 0.0:
                    durations.append(duration)
                loss_timestamp = None
            had_tracking = True
        elif is_lost and had_tracking and loss_timestamp is None:
            loss_timestamp = frame.timestamp

    duration_values = tuple(durations)
    if not duration_values:
        return ReacquisitionStatistics((), None, None)
    return ReacquisitionStatistics(
        durations_seconds=duration_values,
        mean_reacquisition_seconds=sum(duration_values) / len(duration_values),
        maximum_reacquisition_seconds=max(duration_values),
    )


def processing_statistics(
    telemetry: Sequence[SimulatorFrame],
) -> ProcessingStatistics:
    """Return aggregate processing duration and measured wall-clock FPS."""
    durations = processing_times(telemetry)
    return ProcessingStatistics(
        frame_count=len(durations),
        mean_processing_time_seconds=(sum(durations) / len(durations) if durations else None),
        processing_fps=approximate_processing_fps(telemetry),
    )


def truth_unavailable_lock_statistics(
    telemetry: Sequence[SimulatorFrame],
) -> TruthUnavailableLockStatistics:
    """Count locked frames where simulation truth is unavailable.

    This metric identifies locked frames for which simulation truth is
    unavailable. It is NOT sufficient by itself to establish a false lock.
    """
    unavailable_locks = sum(
        frame.tracking.is_locked and frame.sensor_frame.projected_pixel is None
        for frame in telemetry
    )
    total = len(telemetry)
    return TruthUnavailableLockStatistics(
        total_frames=total,
        truth_unavailable_lock_frames=unavailable_locks,
        truth_unavailable_lock_percentage=(100.0 * unavailable_locks / total if total else 0.0),
    )


def lock_statistics(telemetry: Sequence[SimulatorFrame]) -> LockStatistics:
    """Return basic availability and locked-frame statistics."""
    total = len(telemetry)
    available = sum(
        frame.tracking.position_x is not None and frame.tracking.position_y is not None
        for frame in telemetry
    )
    locked = sum(frame.tracking.is_locked for frame in telemetry)
    if total == 0:
        return LockStatistics(0, 0, 0, 0.0, 0.0)
    return LockStatistics(
        total_frames=total,
        available_frames=available,
        locked_frames=locked,
        availability_ratio=available / total,
        lock_ratio=locked / total,
    )