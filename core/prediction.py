"""Authoritative Target Position Prediction for FSOC-PAT-26169.

This module provides a pure, deterministic 2D target-position prediction component
using a constant-velocity kinematic model.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PredictionState:
    """Immutable input state representing target position and velocity estimate."""

    timestamp: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float

    def __post_init__(self) -> None:
        """Validate input state numeric bounds and finite values."""
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


@dataclass(frozen=True)
class PredictionResult:
    """Immutable output snapshot representing predicted future position."""

    source_timestamp: float
    prediction_horizon: float
    predicted_timestamp: float
    predicted_position_x: float
    predicted_position_y: float

    def __post_init__(self) -> None:
        """Validate prediction result fields and numeric bounds."""
        fields = [
            ("source_timestamp", self.source_timestamp),
            ("prediction_horizon", self.prediction_horizon),
            ("predicted_timestamp", self.predicted_timestamp),
            ("predicted_position_x", self.predicted_position_x),
            ("predicted_position_y", self.predicted_position_y),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.source_timestamp < 0.0:
            raise ValueError(
                f"source_timestamp must be >= 0.0, got {self.source_timestamp}"
            )
        if self.prediction_horizon < 0.0:
            raise ValueError(
                f"prediction_horizon must be >= 0.0, got {self.prediction_horizon}"
            )
        if self.predicted_timestamp < 0.0:
            raise ValueError(
                f"predicted_timestamp must be >= 0.0, got {self.predicted_timestamp}"
            )


class TargetPositionPredictor:
    """Authoritative standalone predictor using constant-velocity extrapolation."""

    def predict(
        self, state: PredictionState, prediction_horizon: float
    ) -> PredictionResult:
        """Predict future target position for a given prediction horizon.

        Args:
            state: Current estimated PredictionState.
            prediction_horizon: Future time delta (dt_future) >= 0.0.

        Returns:
            PredictionResult containing future timestamp and predicted coordinates.
        """
        if not isinstance(state, PredictionState):
            raise ValueError(f"Expected PredictionState instance, got {type(state)}")

        if (
            not isinstance(prediction_horizon, (int, float))
            or not math.isfinite(prediction_horizon)
        ):
            raise ValueError(
                f"prediction_horizon must be a finite float, got {prediction_horizon}"
            )

        if prediction_horizon < 0.0:
            raise ValueError(
                f"prediction_horizon must be non-negative, got {prediction_horizon}"
            )

        dt = float(prediction_horizon)
        pred_x = float(state.position_x + state.velocity_x * dt)
        pred_y = float(state.position_y + state.velocity_y * dt)
        pred_timestamp = float(state.timestamp + dt)

        return PredictionResult(
            source_timestamp=float(state.timestamp),
            prediction_horizon=dt,
            predicted_timestamp=pred_timestamp,
            predicted_position_x=pred_x,
            predicted_position_y=pred_y,
        )