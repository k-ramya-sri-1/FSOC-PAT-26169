"""Authoritative Virtual Gimbal Actuator and Coarse Pointing Controller for FSOC-PAT-26169.

This module provides two standalone components:
1. VirtualGimbal: A physical/virtual rate-controlled pan-tilt actuator enforcing rate limits,
   angular limits, and continuous integration over variable time steps.
2. CoarsePointingController: A coarse proportional controller that computes pan/tilt rate commands
   from image-plane tracking errors according to camera sign conventions.

Follows docs/ARCHITECTURE.md, docs/COORDINATE_CONVENTION.md, docs/INTERFACES.md,
docs/DEVELOPMENT_RULES.md, docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class GimbalConfig:
    """Immutable configuration container for VirtualGimbal limits and initial conditions."""

    max_pan_angle_rad: float = math.radians(45.0)
    max_tilt_angle_rad: float = math.radians(45.0)
    max_pan_rate_rad_s: float = math.radians(10.0)
    max_tilt_rate_rad_s: float = math.radians(10.0)
    initial_pan_angle_rad: float = 0.0
    initial_tilt_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        """Validate numeric limits and initial state bounds."""
        fields = [
            ("max_pan_angle_rad", self.max_pan_angle_rad),
            ("max_tilt_angle_rad", self.max_tilt_angle_rad),
            ("max_pan_rate_rad_s", self.max_pan_rate_rad_s),
            ("max_tilt_rate_rad_s", self.max_tilt_rate_rad_s),
            ("initial_pan_angle_rad", self.initial_pan_angle_rad),
            ("initial_tilt_angle_rad", self.initial_tilt_angle_rad),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.max_pan_angle_rad <= 0.0:
            raise ValueError(f"max_pan_angle_rad must be > 0, got {self.max_pan_angle_rad}")
        if self.max_tilt_angle_rad <= 0.0:
            raise ValueError(f"max_tilt_angle_rad must be > 0, got {self.max_tilt_angle_rad}")
        if self.max_pan_rate_rad_s <= 0.0:
            raise ValueError(f"max_pan_rate_rad_s must be > 0, got {self.max_pan_rate_rad_s}")
        if self.max_tilt_rate_rad_s <= 0.0:
            raise ValueError(f"max_tilt_rate_rad_s must be > 0, got {self.max_tilt_rate_rad_s}")

        if not (-self.max_pan_angle_rad <= self.initial_pan_angle_rad <= self.max_pan_angle_rad):
            raise ValueError(
                f"initial_pan_angle_rad ({self.initial_pan_angle_rad}) outside limits "
                f"[-{self.max_pan_angle_rad}, {self.max_pan_angle_rad}]"
            )
        if not (-self.max_tilt_angle_rad <= self.initial_tilt_angle_rad <= self.max_tilt_angle_rad):
            raise ValueError(
                f"initial_tilt_angle_rad ({self.initial_tilt_angle_rad}) outside limits "
                f"[-{self.max_tilt_angle_rad}, {self.max_tilt_angle_rad}]"
            )


@dataclass(frozen=True)
class GimbalState:
    """Immutable snapshot of the current state of VirtualGimbal."""

    timestamp: float
    pan_angle_rad: float
    tilt_angle_rad: float
    pan_rate_rad_s: float
    tilt_rate_rad_s: float

    def __post_init__(self) -> None:
        """Validate numeric integrity and non-negativity of timestamp."""
        fields = [
            ("timestamp", self.timestamp),
            ("pan_angle_rad", self.pan_angle_rad),
            ("tilt_angle_rad", self.tilt_angle_rad),
            ("pan_rate_rad_s", self.pan_rate_rad_s),
            ("tilt_rate_rad_s", self.tilt_rate_rad_s),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0, got {self.timestamp}")


class VirtualGimbal:
    """Standalone rate-controlled virtual pan-tilt actuator."""

    def __init__(self, config: Optional[GimbalConfig] = None) -> None:
        self.config = config if config is not None else GimbalConfig()
        self._last_timestamp: Optional[float] = None
        self._pan_angle: float = 0.0
        self._tilt_angle: float = 0.0
        self._pan_rate: float = 0.0
        self._tilt_rate: float = 0.0
        self.reset(timestamp=0.0)

    def reset(self, timestamp: float = 0.0) -> None:
        """Reset internal gimbal state to initial configuration values."""
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"timestamp must be finite and >= 0.0, got {timestamp}")

        self._last_timestamp = float(timestamp)
        self._pan_angle = float(self.config.initial_pan_angle_rad)
        self._tilt_angle = float(self.config.initial_tilt_angle_rad)
        self._pan_rate = 0.0
        self._tilt_rate = 0.0

    def get_state(self) -> GimbalState:
        """Return the current immutable snapshot of the gimbal state."""
        return GimbalState(
            timestamp=float(self._last_timestamp if self._last_timestamp is not None else 0.0),
            pan_angle_rad=float(self._pan_angle),
            tilt_angle_rad=float(self._tilt_angle),
            pan_rate_rad_s=float(self._pan_rate),
            tilt_rate_rad_s=float(self._tilt_rate),
        )

    def update(
        self,
        pan_rate_command_rad_s: float,
        tilt_rate_command_rad_s: float,
        timestamp: float,
    ) -> GimbalState:
        """Update gimbal orientation given rate commands and a strictly monotonic timestamp."""
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"timestamp must be a finite float >= 0.0, got {timestamp}")

        if self._last_timestamp is None:
            raise ValueError("Gimbal must be initialized via reset prior to update")

        if timestamp <= self._last_timestamp:
            raise ValueError(
                f"Timestamp must be strictly monotonic: current {timestamp} <= last {self._last_timestamp}"
            )

        for name, val in [
            ("pan_rate_command_rad_s", pan_rate_command_rad_s),
            ("tilt_rate_command_rad_s", tilt_rate_command_rad_s),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        dt = float(timestamp - self._last_timestamp)

        # 1. Apply actuator rate saturation
        sat_pan_rate = max(
            -self.config.max_pan_rate_rad_s,
            min(self.config.max_pan_rate_rad_s, float(pan_rate_command_rad_s)),
        )
        sat_tilt_rate = max(
            -self.config.max_tilt_rate_rad_s,
            min(self.config.max_tilt_rate_rad_s, float(tilt_rate_command_rad_s)),
        )

        # 2. Integrate state forward
        next_pan_angle = self._pan_angle + sat_pan_rate * dt
        next_tilt_angle = self._tilt_angle + sat_tilt_rate * dt

        # 3. Apply angular saturation limits
        clamped_pan_angle = max(
            -self.config.max_pan_angle_rad,
            min(self.config.max_pan_angle_rad, next_pan_angle),
        )
        clamped_tilt_angle = max(
            -self.config.max_tilt_angle_rad,
            min(self.config.max_tilt_angle_rad, next_tilt_angle),
        )

        # 4. Record actual applied rates (recalculated if clamped by limits)
        actual_pan_rate = (clamped_pan_angle - self._pan_angle) / dt if dt > 0.0 else 0.0
        actual_tilt_rate = (clamped_tilt_angle - self._tilt_angle) / dt if dt > 0.0 else 0.0

        self._pan_angle = clamped_pan_angle
        self._tilt_angle = clamped_tilt_angle
        self._pan_rate = actual_pan_rate
        self._tilt_rate = actual_tilt_rate
        self._last_timestamp = float(timestamp)

        return self.get_state()


@dataclass(frozen=True)
class ControlConfig:
    """Immutable configuration container for CoarsePointingController parameters."""

    pan_gain_rad_s_per_px: float = 0.001
    tilt_gain_rad_s_per_px: float = 0.001
    max_pan_rate_rad_s: float = math.radians(10.0)
    max_tilt_rate_rad_s: float = math.radians(10.0)
    deadband_px: float = 0.0

    def __post_init__(self) -> None:
        """Validate controller parameters."""
        fields = [
            ("pan_gain_rad_s_per_px", self.pan_gain_rad_s_per_px),
            ("tilt_gain_rad_s_per_px", self.tilt_gain_rad_s_per_px),
            ("max_pan_rate_rad_s", self.max_pan_rate_rad_s),
            ("max_tilt_rate_rad_s", self.max_tilt_rate_rad_s),
            ("deadband_px", self.deadband_px),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.pan_gain_rad_s_per_px < 0.0:
            raise ValueError(f"pan_gain_rad_s_per_px must be >= 0, got {self.pan_gain_rad_s_per_px}")
        if self.tilt_gain_rad_s_per_px < 0.0:
            raise ValueError(f"tilt_gain_rad_s_per_px must be >= 0, got {self.tilt_gain_rad_s_per_px}")
        if self.max_pan_rate_rad_s <= 0.0:
            raise ValueError(f"max_pan_rate_rad_s must be > 0, got {self.max_pan_rate_rad_s}")
        if self.max_tilt_rate_rad_s <= 0.0:
            raise ValueError(f"max_tilt_rate_rad_s must be > 0, got {self.max_tilt_rate_rad_s}")
        if self.deadband_px < 0.0:
            raise ValueError(f"deadband_px must be >= 0, got {self.deadband_px}")


@dataclass(frozen=True)
class ControlCommand:
    """Immutable output container for controller rate commands."""

    timestamp: float
    pan_rate_command_rad_s: float
    tilt_rate_command_rad_s: float

    def __post_init__(self) -> None:
        """Validate numeric integrity and non-negativity of timestamp."""
        fields = [
            ("timestamp", self.timestamp),
            ("pan_rate_command_rad_s", self.pan_rate_command_rad_s),
            ("tilt_rate_command_rad_s", self.tilt_rate_command_rad_s),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0, got {self.timestamp}")


class CoarsePointingController:
    """Proportional image-error controller for coarse pan-tilt alignment."""

    def __init__(self, config: Optional[ControlConfig] = None) -> None:
        self.config = config if config is not None else ControlConfig()
        self._last_timestamp: Optional[float] = None
        self.reset(timestamp=0.0)

    def reset(self, timestamp: float = 0.0) -> None:
        """Reset controller timing state."""
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"timestamp must be a finite float >= 0.0, got {timestamp}")

        self._last_timestamp = None  # Allow first call to establish timestamp

    def compute_command(
        self,
        target_x: float,
        target_y: float,
        image_center_x: float,
        image_center_y: float,
        timestamp: float,
    ) -> ControlCommand:
        """Compute pan/tilt rate command from image-plane tracking error."""
        coords = [
            ("target_x", target_x),
            ("target_y", target_y),
            ("image_center_x", image_center_x),
            ("image_center_y", image_center_y),
            ("timestamp", timestamp),
        ]
        for name, val in coords:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0, got {timestamp}")

        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                f"Timestamp must be strictly monotonic: current {timestamp} <= last {self._last_timestamp}"
            )

        error_x = float(target_x - image_center_x)
        error_y = float(target_y - image_center_y)

        # 1. Apply independent X/Y deadband
        eff_error_x = 0.0 if abs(error_x) <= self.config.deadband_px else error_x
        eff_error_y = 0.0 if abs(error_y) <= self.config.deadband_px else error_y

        # 2. Compute proportional rate commands according to sign convention
        # pan_command  = -Kp_pan  * error_x
        # tilt_command = +Kp_tilt * error_y
        raw_pan_command = -self.config.pan_gain_rad_s_per_px * eff_error_x
        raw_tilt_command = self.config.tilt_gain_rad_s_per_px * eff_error_y

        # 3. Apply controller command saturation
        sat_pan_command = max(
            -self.config.max_pan_rate_rad_s,
            min(self.config.max_pan_rate_rad_s, raw_pan_command),
        )
        sat_tilt_command = max(
            -self.config.max_tilt_rate_rad_s,
            min(self.config.max_tilt_rate_rad_s, raw_tilt_command),
        )

        self._last_timestamp = float(timestamp)

        return ControlCommand(
            timestamp=float(timestamp),
            pan_rate_command_rad_s=float(sat_pan_command),
            tilt_rate_command_rad_s=float(sat_tilt_command),
        )