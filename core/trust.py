"""Authoritative Model-vs-Vision Trust Evaluation for FSOC-PAT-26169.

This module provides a pure, deterministic trust-evaluation component that measures
the spatial agreement (Euclidean disagreement) between visual observations and model
predictions in 2D image-plane coordinates.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class TrustConfig:
    """Immutable configuration container for trust evaluation thresholds."""

    full_trust_distance_px: float = 5.0
    zero_trust_distance_px: float = 50.0

    def __post_init__(self) -> None:
        """Validate configuration threshold bounds and relationships."""
        fields = [
            ("full_trust_distance_px", self.full_trust_distance_px),
            ("zero_trust_distance_px", self.zero_trust_distance_px),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.full_trust_distance_px < 0.0:
            raise ValueError(
                f"full_trust_distance_px must be >= 0.0, got {self.full_trust_distance_px}"
            )

        if self.zero_trust_distance_px <= 0.0:
            raise ValueError(
                f"zero_trust_distance_px must be > 0.0, got {self.zero_trust_distance_px}"
            )

        if self.zero_trust_distance_px <= self.full_trust_distance_px:
            raise ValueError(
                f"zero_trust_distance_px ({self.zero_trust_distance_px}) must be strictly greater than "
                f"full_trust_distance_px ({self.full_trust_distance_px})"
            )


@dataclass(frozen=True)
class TrustObservation:
    """Immutable input container representing observed and predicted coordinates."""

    observed_x: float
    observed_y: float
    predicted_x: float
    predicted_y: float

    def __post_init__(self) -> None:
        """Validate input coordinate finiteness."""
        fields = [
            ("observed_x", self.observed_x),
            ("observed_y", self.observed_y),
            ("predicted_x", self.predicted_x),
            ("predicted_y", self.predicted_y),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")


@dataclass(frozen=True)
class TrustResult:
    """Immutable output snapshot representing spatial disagreement and trust evaluation."""

    observed_x: float
    observed_y: float
    predicted_x: float
    predicted_y: float
    disagreement_px: float
    trust_score: float
    model_preferred: bool

    def __post_init__(self) -> None:
        """Validate output fields and ranges."""
        fields = [
            ("observed_x", self.observed_x),
            ("observed_y", self.observed_y),
            ("predicted_x", self.predicted_x),
            ("predicted_y", self.predicted_y),
            ("disagreement_px", self.disagreement_px),
            ("trust_score", self.trust_score),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")

        if self.disagreement_px < 0.0:
            raise ValueError(
                f"disagreement_px must be non-negative, got {self.disagreement_px}"
            )

        if not (0.0 <= float(self.trust_score) <= 1.0):
            raise ValueError(
                f"trust_score must be in range [0.0, 1.0], got {self.trust_score}"
            )

        if not isinstance(self.model_preferred, bool):
            raise ValueError(
                f"model_preferred must be a boolean, got {type(self.model_preferred)}"
            )


class ModelVisionTrustEvaluator:
    """Authoritative standalone evaluator for visual observation vs model prediction trust."""

    def __init__(self, config: Optional[TrustConfig] = None) -> None:
        self.config = config if config is not None else TrustConfig()

    def evaluate(self, observation: TrustObservation) -> TrustResult:
        """Evaluate Euclidean spatial disagreement and compute linear trust score."""
        if not isinstance(observation, TrustObservation):
            raise ValueError(
                f"Expected TrustObservation instance, got {type(observation)}"
            )

        dx = float(observation.observed_x - observation.predicted_x)
        dy = float(observation.observed_y - observation.predicted_y)
        disagreement = float(math.hypot(dx, dy))

        full_dist = float(self.config.full_trust_distance_px)
        zero_dist = float(self.config.zero_trust_distance_px)

        if disagreement <= full_dist:
            trust_score = 1.0
        elif disagreement >= zero_dist:
            trust_score = 0.0
        else:
            trust_score = 1.0 - (disagreement - full_dist) / (zero_dist - full_dist)

        model_preferred = bool(disagreement > full_dist)

        return TrustResult(
            observed_x=float(observation.observed_x),
            observed_y=float(observation.observed_y),
            predicted_x=float(observation.predicted_x),
            predicted_y=float(observation.predicted_y),
            disagreement_px=disagreement,
            trust_score=float(trust_score),
            model_preferred=model_preferred,
        )