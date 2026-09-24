"""Authoritative Target Tracking and State Machine for FSOC-PAT-26169.

This module implements the target tracking layer, maintaining track continuity,
observed velocity estimation, confidence fusion, missed-frame processing, and deterministic
state machine transitions (SEARCHING, ACQUIRING, TRACKING, LOST, REACQUIRING).

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass, field
import enum
import math
from typing import Optional


@enum.unique
class TrackingState(enum.Enum):
    """Target tracking state machine states."""

    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


@dataclass(frozen=True)
class TrackingObservation:
    """Immutable candidate observation received by the target tracker."""

    timestamp: float
    centroid_x: float
    centroid_y: float
    detection_confidence: float
    ai_confidence: float
    ai_is_beacon: bool
    modulation_confidence: float
    modulation_verified: bool
    candidate_id: int = 0

    def __post_init__(self) -> None:
        """Validate observation fields and numeric integrity."""
        numeric_fields = [
            ("timestamp", self.timestamp),
            ("centroid_x", self.centroid_x),
            ("centroid_y", self.centroid_y),
            ("detection_confidence", self.detection_confidence),
            ("ai_confidence", self.ai_confidence),
            ("modulation_confidence", self.modulation_confidence),
        ]
        for name, val in numeric_fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")

        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")

        for conf_name, conf_val in [
            ("detection_confidence", self.detection_confidence),
            ("ai_confidence", self.ai_confidence),
            ("modulation_confidence", self.modulation_confidence),
        ]:
            if not (0.0 <= conf_val <= 1.0):
                raise ValueError(f"{conf_name} must be in range [0, 1], got {conf_val}")

        if not isinstance(self.candidate_id, int) or self.candidate_id < 0:
            raise ValueError(
                f"candidate_id must be a non-negative integer, got {self.candidate_id}"
            )

        if not isinstance(self.ai_is_beacon, bool):
            raise ValueError(f"ai_is_beacon must be a boolean, got {self.ai_is_beacon}")

        if not isinstance(self.modulation_verified, bool):
            raise ValueError(
                f"modulation_verified must be a boolean, got {self.modulation_verified}"
            )


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration parameters for target tracking state machine and filters."""

    acquisition_confidence_threshold: float = 0.6
    tracking_confidence_threshold: float = 0.5
    loss_confidence_threshold: float = 0.25
    max_missed_frames: int = 5
    tracking_max_distance_px: float = 50.0
    reacquisition_max_distance_px: float = 50.0
    minimum_acquisition_observations: int = 2
    velocity_smoothing_factor: float = 0.5
    max_velocity_px_per_second: float = 1000.0

    def __post_init__(self) -> None:
       """Validate configuration hyper-parameters."""
       for conf_name, conf_val in [
           ("acquisition_confidence_threshold", self.acquisition_confidence_threshold),
           ("tracking_confidence_threshold", self.tracking_confidence_threshold),
           ("loss_confidence_threshold", self.loss_confidence_threshold),
        ]:
           if not (0.0 <= conf_val <= 1.0):
               raise ValueError(f"{conf_name} must be in range [0, 1], got {conf_val}")

       if self.max_missed_frames < 1:
           raise ValueError(f"max_missed_frames must be >= 1, got {self.max_missed_frames}")

       if not math.isfinite(self.tracking_max_distance_px) or self.tracking_max_distance_px <= 0.0:
           raise ValueError(
               f"tracking_max_distance_px must be positive and finite, got {self.tracking_max_distance_px}"
           )

       if not math.isfinite(self.reacquisition_max_distance_px) or self.reacquisition_max_distance_px <= 0.0:
           raise ValueError(
               f"reacquisition_max_distance_px must be positive and finite, got {self.reacquisition_max_distance_px}"
           )

       if self.minimum_acquisition_observations < 1:
           raise ValueError(
               f"minimum_acquisition_observations must be >= 1, got {self.minimum_acquisition_observations}"
           )

       if not (0.0 <= self.velocity_smoothing_factor <= 1.0):
           raise ValueError(
               f"velocity_smoothing_factor must be in [0, 1], got {self.velocity_smoothing_factor}"
           )

       if (
           not math.isfinite(self.max_velocity_px_per_second)
           or self.max_velocity_px_per_second <= 0.0
       ):
           raise ValueError(
               f"max_velocity_px_per_second must be positive and finite, "
               f"got {self.max_velocity_px_per_second}"
           )

@dataclass(frozen=True)
class TrackedTargetState:
    """Immutable output snapshot representing current tracker state."""

    state: TrackingState
    timestamp: float
    position_x: Optional[float]
    position_y: Optional[float]
    velocity_x: float
    velocity_y: float
    confidence: float
    candidate_id: Optional[int]
    missed_frames: int
    acquisition_count: int
    is_locked: bool = field(init=False)

    def __post_init__(self) -> None:
        """Validate tracked state fields and compute derived parameters."""
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"timestamp must be finite and >= 0, got {self.timestamp}")

        if self.position_x is not None:
            if not math.isfinite(self.position_x):
                raise ValueError("position_x must be finite when present")
            if self.position_y is None or not math.isfinite(self.position_y):
                raise ValueError("position_y must be finite when position_x is present")

        if not (math.isfinite(self.velocity_x) and math.isfinite(self.velocity_y)):
            raise ValueError("velocity components must be finite")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        if self.candidate_id is not None and (
            not isinstance(self.candidate_id, int) or self.candidate_id < 0
        ):
            raise ValueError("candidate_id must be a non-negative integer when present")

        if self.missed_frames < 0:
            raise ValueError(f"missed_frames must be >= 0, got {self.missed_frames}")

        if self.acquisition_count < 0:
            raise ValueError(f"acquisition_count must be >= 0, got {self.acquisition_count}")

        # Derived property: only TRACKING state is locked
        object.__setattr__(self, "is_locked", self.state == TrackingState.TRACKING)


class TargetTracker:
    """Authoritative target tracker maintaining state machine and target track state."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config if config is not None else TrackingConfig()
        self._state: TrackingState = TrackingState.SEARCHING
        self._last_timestamp: Optional[float] = None
        self._position_x: Optional[float] = None
        self._position_y: Optional[float] = None
        self._velocity_x: float = 0.0
        self._velocity_y: float = 0.0
        self._confidence: float = 0.0
        self._candidate_id: Optional[int] = None
        self._missed_frames: int = 0
        self._acquisition_count: int = 0

    def update(
        self, observation: Optional[TrackingObservation] = None, timestamp: Optional[float] = None
    ) -> TrackedTargetState:
        """Update tracker state given a new observation or missed frame."""
        if observation is not None:
            if timestamp is not None and timestamp != observation.timestamp:
                raise ValueError("Explicit timestamp disagrees with observation.timestamp")
            current_time = observation.timestamp
        elif timestamp is not None:
            current_time = timestamp
        else:
            raise ValueError("Must provide either observation or explicit timestamp for update")

        if not math.isfinite(current_time) or current_time < 0.0:
            raise ValueError(f"Timestamp must be finite and >= 0, got {current_time}")

        if self._last_timestamp is not None:
            if current_time <= self._last_timestamp:
                raise ValueError(
                    f"Timestamps must be strictly monotonic: current {current_time} <= last {self._last_timestamp}"
                )

        if observation is None:
            self._handle_missed_frame(current_time)
        else:
            self._handle_observation(observation)

        self._last_timestamp = current_time

        return TrackedTargetState(
            state=self._state,
            timestamp=self._last_timestamp,
            position_x=self._position_x,
            position_y=self._position_y,
            velocity_x=self._velocity_x,
            velocity_y=self._velocity_y,
            confidence=self._confidence,
            candidate_id=self._candidate_id,
            missed_frames=self._missed_frames,
            acquisition_count=self._acquisition_count,
        )

    def reset(self) -> None:
        """Reset internal tracking state machine to initial SEARCHING state."""
        self._state = TrackingState.SEARCHING
        self._last_timestamp = None
        self._position_x = None
        self._position_y = None
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._confidence = 0.0
        self._candidate_id = None
        self._missed_frames = 0
        self._acquisition_count = 0

    def _fuse_confidence(self, obs: TrackingObservation) -> float:
        """Compute weighted evidence fusion confidence score."""
        fused = (
            0.4 * obs.detection_confidence
            + 0.3 * obs.ai_confidence
            + 0.3 * obs.modulation_confidence
        )
        return float(min(1.0, max(0.0, fused)))

    def _handle_observation(self, obs: TrackingObservation) -> None:
        """Process valid incoming observation according to current tracking state."""
        fused_conf = self._fuse_confidence(obs)

        # AI rejection: if ai_is_beacon is False, observation cannot be accepted as a valid beacon
        is_credible = (
            obs.ai_is_beacon
            and (fused_conf >= self.config.acquisition_confidence_threshold)
        )

        dt = obs.timestamp - self._last_timestamp if self._last_timestamp is not None else 0.0

        if self._state == TrackingState.SEARCHING:
            if is_credible:
                self._state = TrackingState.ACQUIRING
                self._position_x = obs.centroid_x
                self._position_y = obs.centroid_y
                self._velocity_x = 0.0
                self._velocity_y = 0.0
                self._confidence = fused_conf
                self._candidate_id = obs.candidate_id
                self._missed_frames = 0
                self._acquisition_count = 1

                if self._acquisition_count >= self.config.minimum_acquisition_observations:
                    self._state = TrackingState.TRACKING
            else:
                self._confidence = 0.0

        elif self._state == TrackingState.ACQUIRING:
            if is_credible:
                self._update_velocity_and_position(obs.centroid_x, obs.centroid_y, dt)
                self._confidence = fused_conf
                self._candidate_id = obs.candidate_id
                self._missed_frames = 0
                self._acquisition_count += 1

                if self._acquisition_count >= self.config.minimum_acquisition_observations:
                    self._state = TrackingState.TRACKING
            else:
                self._state = TrackingState.SEARCHING
                self._position_x = None
                self._position_y = None
                self._velocity_x = 0.0
                self._velocity_y = 0.0
                self._confidence = 0.0
                self._candidate_id = None
                self._missed_frames = 0
                self._acquisition_count = 0

        elif self._state == TrackingState.TRACKING:
            # Spatial plausibility check against tracking_max_distance_px
            is_spatially_plausible = True
            if self._position_x is not None and self._position_y is not None:
                dist = math.hypot(
                    obs.centroid_x - self._position_x, obs.centroid_y - self._position_y
                )
                if dist > self.config.tracking_max_distance_px:
                    is_spatially_plausible = False

            valid_track_obs = (
                obs.ai_is_beacon
                and is_spatially_plausible
                and (fused_conf >= self.config.tracking_confidence_threshold)
            )

            if valid_track_obs:
                self._update_velocity_and_position(obs.centroid_x, obs.centroid_y, dt)
                self._confidence = fused_conf
                self._candidate_id = obs.candidate_id
                self._missed_frames = 0
            else:
                self._apply_miss_decay()

        elif self._state == TrackingState.LOST:
            if is_credible and self._position_x is not None and self._position_y is not None:
                dist = math.hypot(
                    obs.centroid_x - self._position_x, obs.centroid_y - self._position_y
                )
                if dist <= self.config.reacquisition_max_distance_px:
                    self._state = TrackingState.REACQUIRING
                    self._update_velocity_and_position(obs.centroid_x, obs.centroid_y, dt)
                    self._confidence = fused_conf
                    self._candidate_id = obs.candidate_id
                    self._missed_frames = 0
                    self._acquisition_count = 1

                    if self._acquisition_count >= self.config.minimum_acquisition_observations:
                        self._state = TrackingState.TRACKING

        elif self._state == TrackingState.REACQUIRING:
            if is_credible and self._position_x is not None and self._position_y is not None:
                dist = math.hypot(
                    obs.centroid_x - self._position_x, obs.centroid_y - self._position_y
                )
                if dist <= self.config.reacquisition_max_distance_px:
                    self._update_velocity_and_position(obs.centroid_x, obs.centroid_y, dt)
                    self._confidence = fused_conf
                    self._candidate_id = obs.candidate_id
                    self._missed_frames = 0
                    self._acquisition_count += 1

                    if self._acquisition_count >= self.config.minimum_acquisition_observations:
                        self._state = TrackingState.TRACKING
                else:
                    self._state = TrackingState.LOST
                    self._apply_miss_decay()
            else:
                self._state = TrackingState.LOST
                self._apply_miss_decay()

    def _handle_missed_frame(self, current_time: float) -> None:
        """Process a missed observation frame update."""
        if self._state in (TrackingState.TRACKING, TrackingState.REACQUIRING, TrackingState.LOST):
            self._apply_miss_decay()
        elif self._state == TrackingState.ACQUIRING:
            self._state = TrackingState.SEARCHING
            self._position_x = None
            self._position_y = None
            self._velocity_x = 0.0
            self._velocity_y = 0.0
            self._confidence = 0.0
            self._candidate_id = None
            self._missed_frames = 0
            self._acquisition_count = 0

    def _apply_miss_decay(self) -> None:
        """Increment missed frame count, decay confidence, and transition to LOST if threshold breached."""
        self._missed_frames += 1
        self._confidence = float(max(0.0, self._confidence * 0.7))

        if (
            self._missed_frames > self.config.max_missed_frames
            or self._confidence < self.config.loss_confidence_threshold
        ):
            self._state = TrackingState.LOST

    def _update_velocity_and_position(self, new_x: float, new_y: float, dt: float) -> None:
        """Update position and apply exponential smoothing to observed velocity estimate."""
        if dt > 0.0 and self._position_x is not None and self._position_y is not None:
            raw_vx = (new_x - self._position_x) / dt
            raw_vy = (new_y - self._position_y) / dt

            alpha = self.config.velocity_smoothing_factor
            smooth_vx = alpha * raw_vx + (1.0 - alpha) * self._velocity_x
            smooth_vy = alpha * raw_vy + (1.0 - alpha) * self._velocity_y

            # Clamp velocity magnitude to max_velocity_px_per_second
            speed = math.hypot(smooth_vx, smooth_vy)
            if speed > self.config.max_velocity_px_per_second and speed > 0.0:
                scale = self.config.max_velocity_px_per_second / speed
                smooth_vx *= scale
                smooth_vy *= scale

            self._velocity_x = smooth_vx
            self._velocity_y = smooth_vy

        self._position_x = new_x
        self._position_y = new_y