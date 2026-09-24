"""Authoritative 2D Kalman Filter for FSOC-PAT-26169.

This module provides a robust, reusable 2D constant-velocity Kalman filter for
image-plane target tracking and state estimation.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KalmanConfig:
    """Configuration parameters for 2D Kalman Filter."""

    process_noise: float = 1.0
    measurement_noise: float = 4.0
    initial_position_variance: float = 100.0
    initial_velocity_variance: float = 100.0
    max_dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration hyper-parameters."""
        fields = [
            ("process_noise", self.process_noise),
            ("measurement_noise", self.measurement_noise),
            ("initial_position_variance", self.initial_position_variance),
            ("initial_velocity_variance", self.initial_velocity_variance),
            ("max_dt", self.max_dt),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(
                    f"{name} must be a strictly positive finite float, got {val}"
                )


@dataclass(frozen=True)
class KalmanState:
    """Immutable state estimate output representation."""

    timestamp: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float

    def __post_init__(self) -> None:
        """Validate state fields and numeric integrity."""
        fields = [
            ("timestamp", self.timestamp),
            ("position_x", self.position_x),
            ("position_y", self.position_y),
            ("velocity_x", self.velocity_x),
            ("velocity_y", self.velocity_y),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0, got {self.timestamp}")


class KalmanFilter2D:
    """Independent 2D Constant-Velocity Kalman Filter."""

    def __init__(self, config: Optional[KalmanConfig] = None) -> None:
        self.config = config if config is not None else KalmanConfig()
        self._timestamp: Optional[float] = None
        self._x: Optional[np.ndarray] = None  # State vector [x, y, vx, vy]^T
        self._P: Optional[np.ndarray] = None  # Covariance matrix 4x4
        self._H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self._R = np.eye(2, dtype=np.float64) * float(self.config.measurement_noise)

    def reset(self) -> None:
        """Reset internal filter state to uninitialized."""
        self._timestamp = None
        self._x = None
        self._P = None

    def initialize(
        self,
        timestamp: float,
        position_x: float,
        position_y: float,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
    ) -> KalmanState:
        """Initialize filter state vector, covariance, and timestamp."""
        self._validate_numeric("timestamp", timestamp, non_negative=True)
        self._validate_numeric("position_x", position_x)
        self._validate_numeric("position_y", position_y)
        self._validate_numeric("velocity_x", velocity_x)
        self._validate_numeric("velocity_y", velocity_y)

        self._timestamp = float(timestamp)
        self._x = np.array(
            [position_x, position_y, velocity_x, velocity_y], dtype=np.float64
        )
        self._P = np.diag(
            [
                float(self.config.initial_position_variance),
                float(self.config.initial_position_variance),
                float(self.config.initial_velocity_variance),
                float(self.config.initial_velocity_variance),
            ]
        ).astype(np.float64)

        return self._make_state()

    def predict(self, timestamp: float) -> KalmanState:
        """Predict filter state forward to the target timestamp."""
        self._ensure_initialized()
        self._validate_timestamp_sequence(timestamp)

        actual_dt = float(timestamp) - self._timestamp
        dt = min(actual_dt, float(self.config.max_dt))

        self._predict_internal(dt)
        self._timestamp = float(timestamp)

        return self._make_state()

    def update(
        self,
        timestamp: float,
        position_x: float,
        position_y: float,
    ) -> KalmanState:
        """Perform a prediction and measurement correction at a strictly increasing timestamp."""
        self._ensure_initialized()
        self._validate_timestamp_sequence(timestamp)
        self._validate_numeric("position_x", position_x)
        self._validate_numeric("position_y", position_y)

        actual_dt = float(timestamp) - self._timestamp
        dt = min(actual_dt, float(self.config.max_dt))

        self._predict_internal(dt)
        self._timestamp = float(timestamp)

        z = np.array([position_x, position_y], dtype=np.float64)
        self._update_internal(z)

        return self._make_state()

    def predict_and_update(
        self,
        timestamp: float,
        position_x: float,
        position_y: float,
    ) -> KalmanState:
        """Convenience single call executing prediction and update at a strictly increasing timestamp."""
        return self.update(timestamp, position_x, position_y)

    def get_state(self) -> Optional[KalmanState]:
        """Return current state snapshot if initialized, otherwise None."""
        if self._timestamp is None or self._x is None:
            return None
        return self._make_state()

    def _predict_internal(self, dt: float) -> None:
        """Perform internal state propagation and process noise covariance addition."""
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = float(self.config.process_noise)

        Q = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._symmetrize_P()

    def _update_internal(self, z: np.ndarray) -> None:
        """Perform measurement correction using Joseph form covariance update."""
        y = z - (self._H @ self._x)
        S = self._H @ self._P @ self._H.T + self._R

        # Compute Kalman gain via linear solve: K = P H^T S^-1 => K^T = S^-1 H P^T
        K = np.linalg.solve(S, self._H @ self._P).T

        self._x = self._x + K @ y

        I_KH = np.eye(4, dtype=np.float64) - (K @ self._H)
        self._P = (I_KH @ self._P @ I_KH.T) + (K @ self._R @ K.T)
        self._symmetrize_P()

    def _symmetrize_P(self) -> None:
        """Maintain numerical symmetry of the state covariance matrix."""
        if self._P is not None:
            self._P = 0.5 * (self._P + self._P.T)

    def _make_state(self) -> KalmanState:
        """Create an immutable KalmanState snapshot from current internal vectors."""
        return KalmanState(
            timestamp=float(self._timestamp),
            position_x=float(self._x[0]),
            position_y=float(self._x[1]),
            velocity_x=float(self._x[2]),
            velocity_y=float(self._x[3]),
        )

    def _ensure_initialized(self) -> None:
        """Raise ValueError if the filter is not yet initialized."""
        if self._timestamp is None or self._x is None or self._P is None:
            raise ValueError("Kalman filter must be initialized before operation")

    def _validate_timestamp_sequence(self, timestamp: float) -> None:
        """Ensure sequential timestamps are finite, non-negative, and strictly increasing."""
        self._validate_numeric("timestamp", timestamp, non_negative=True)
        if timestamp <= self._timestamp:
            raise ValueError(
                f"New timestamp ({timestamp}) must be strictly greater than previous timestamp ({self._timestamp})"
            )

    @staticmethod
    def _validate_numeric(
        name: str, val: float, non_negative: bool = False
    ) -> None:
        """Validate that input values are finite floats."""
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{name} must be a finite float, got {val}")
        if non_negative and val < 0.0:
            raise ValueError(f"{name} must be non-negative, got {val}")