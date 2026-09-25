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