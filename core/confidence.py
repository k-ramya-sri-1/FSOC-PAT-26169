"""Authoritative Confidence Fusion for FSOC-PAT-26169.

This module provides a pure, deterministic evidence-fusion component that combines
independent confidence scores from detection, AI beacon identification, and modulation
verification using a weighted arithmetic mean.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class ConfidenceEvidence:
    """Immutable input evidence container representing independent confidence scores."""

    detection_confidence: float
    ai_confidence: float
    modulation_confidence: float

    def __post_init__(self) -> None:
        """Validate input confidence values."""
        fields = [
            ("detection_confidence", self.detection_confidence),
            ("ai_confidence", self.ai_confidence),
            ("modulation_confidence", self.modulation_confidence),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {val}")


@dataclass(frozen=True)
class ConfidenceResult:
    """Immutable result container representing input evidence and fused score."""

    detection_confidence: float
    ai_confidence: float
    modulation_confidence: float
    fused_confidence: float

    def __post_init__(self) -> None:
        """Validate output confidence values."""
        fields = [
            ("detection_confidence", self.detection_confidence),
            ("ai_confidence", self.ai_confidence),
            ("modulation_confidence", self.modulation_confidence),
            ("fused_confidence", self.fused_confidence),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {val}")


@dataclass(frozen=True)
class ConfidenceFusionConfig:
    """Immutable configuration container for fusion weights."""

    detection_weight: float = 0.4
    ai_weight: float = 0.3
    modulation_weight: float = 0.3

    def __post_init__(self) -> None:
        """Validate weight non-negativity and total sum constraint."""
        fields = [
            ("detection_weight", self.detection_weight),
            ("ai_weight", self.ai_weight),
            ("modulation_weight", self.modulation_weight),
        ]
        for name, val in fields:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val}")
            if float(val) < 0.0 or float(val) > 1.0:
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {val}")

        total_weight = (
            float(self.detection_weight)
            + float(self.ai_weight)
            + float(self.modulation_weight)
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"Fusion weights must sum to 1.0 within tolerance, got sum {total_weight}"
            )


class ConfidenceFusion:
    """Authoritative standalone evidence-fusion component."""

    def __init__(self, config: Optional[ConfidenceFusionConfig] = None) -> None:
        self.config = config if config is not None else ConfidenceFusionConfig()

    def fuse(self, evidence: ConfidenceEvidence) -> ConfidenceResult:
        """Combine input confidence scores into a single fused score using weighted arithmetic mean."""
        if not isinstance(evidence, ConfidenceEvidence):
            raise ValueError(f"Expected ConfidenceEvidence instance, got {type(evidence)}")

        fused = (
            self.config.detection_weight * evidence.detection_confidence
            + self.config.ai_weight * evidence.ai_confidence
            + self.config.modulation_weight * evidence.modulation_confidence
        )

        return ConfidenceResult(
            detection_confidence=float(evidence.detection_confidence),
            ai_confidence=float(evidence.ai_confidence),
            modulation_confidence=float(evidence.modulation_confidence),
            fused_confidence=float(fused),
        )