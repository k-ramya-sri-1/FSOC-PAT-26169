"""Authoritative AI Beacon Identity Classification for FSOC-PAT-26169.

This module evaluates observable candidate feature vectors derived from detection outputs
and uses a lightweight, deterministic Logistic Regression classifier to evaluate beacon identity confidence.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class BeaconFeatures:
    """Immutable feature vector representation derived from candidate observations."""

    area: float
    width: float
    height: float
    aspect_ratio: float
    circularity: float
    peak_intensity: float
    mean_intensity: float
    intensity_std: float = 0.0
    signal_to_noise_ratio: float = 0.0

    def __post_init__(self) -> None:
        """Validate numeric feature bounds and numerical integrity."""
        # 1. Finite value validation for all numeric features
        numeric_fields = [
            ("area", self.area),
            ("width", self.width),
            ("height", self.height),
            ("aspect_ratio", self.aspect_ratio),
            ("circularity", self.circularity),
            ("peak_intensity", self.peak_intensity),
            ("mean_intensity", self.mean_intensity),
            ("intensity_std", self.intensity_std),
            ("signal_to_noise_ratio", self.signal_to_noise_ratio),
        ]
        for name, val in numeric_fields:
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")

        # 2. Strict physical feature bounds
        if self.area <= 0.0:
            raise ValueError(f"area must be strictly positive, got {self.area}")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("width and height must be strictly positive")
        if self.aspect_ratio <= 0.0:
            raise ValueError(f"aspect_ratio must be strictly positive, got {self.aspect_ratio}")
        if not (0.0 <= self.circularity <= 1.0):
            raise ValueError(f"circularity must be in [0.0, 1.0], got {self.circularity}")
        if self.peak_intensity < 0.0 or self.mean_intensity < 0.0:
            raise ValueError("intensities must be non-negative")
        if self.intensity_std < 0.0 or self.signal_to_noise_ratio < 0.0:
            raise ValueError("std and SNR must be non-negative")

    def to_array(self) -> np.ndarray:
        """Convert features to a 1D NumPy floating-point array."""
        return np.array(
            [
                self.area,
                self.width,
                self.height,
                self.aspect_ratio,
                self.circularity,
                self.peak_intensity,
                self.mean_intensity,
                self.intensity_std,
                self.signal_to_noise_ratio,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BeaconIdentityConfig:
    """Configuration parameters for Logistic Regression training and inference."""

    learning_rate: float = 0.05
    epochs: int = 1000
    regularization_strength: float = 0.001
    classification_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration hyper-parameters."""
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be strictly positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be strictly positive, got {self.epochs}")
        if self.regularization_strength < 0.0:
            raise ValueError(f"regularization_strength must be non-negative, got {self.regularization_strength}")
        if not (0.0 <= self.classification_threshold <= 1.0):
            raise ValueError(f"classification_threshold must be in [0, 1], got {self.classification_threshold}")


@dataclass(frozen=True)
class BeaconIdentityResult:
    """Result of AI beacon identity classification."""

    is_beacon: bool
    confidence: float
    probability: float
    predicted_class: int


class BeaconIdentityClassifier:
    """Lightweight NumPy-based Logistic Regression classifier for AI beacon identification."""

    def __init__(self, config: Optional[BeaconIdentityConfig] = None) -> None:
        self.config = config if config is not None else BeaconIdentityConfig()
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(
        self,
        features: Union[List[BeaconFeatures], np.ndarray],
        labels: Union[List[int], np.ndarray],
    ) -> "BeaconIdentityClassifier":
        """Fit the Logistic Regression model on training features and binary labels using gradient descent."""
        X_raw = self._parse_features_input(features)
        y = np.asarray(labels, dtype=np.float64)

        if X_raw.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Training data arrays cannot be empty")
        if X_raw.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch: {X_raw.shape[0]} feature samples vs {y.shape[0]} labels")

        # Validate numerical integrity
        if np.isnan(X_raw).any() or np.isinf(X_raw).any():
            raise ValueError("Training features contain NaN or Infinite values")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("Training labels contain NaN or Infinite values")

        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("Labels must be binary integers in {0, 1}")
        if len(unique_labels) < 2:
            raise ValueError("Training requires at least two classes (0 and 1)")

        n_samples, n_features = X_raw.shape

        # Store feature standardization parameters (z-score)
        self.feature_mean = np.mean(X_raw, axis=0)
        self.feature_std = np.std(X_raw, axis=0)
        # Avoid division by zero for zero-variance features
        self.feature_std[self.feature_std == 0.0] = 1.0

        X = (X_raw - self.feature_mean) / self.feature_std

        # Deterministic zero initialization
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        lr = self.config.learning_rate
        reg = self.config.regularization_strength

        # Gradient descent optimization
        for _ in range(self.config.epochs):
            z = np.dot(X, self.weights) + self.bias
            p = self._sigmoid(z)

            error = p - y
            dw = (np.dot(X.T, error) / n_samples) + (reg * self.weights)
            db = float(np.sum(error) / n_samples)

            self.weights -= lr * dw
            self.bias -= lr * db

        self._is_fitted = True
        return self

    def predict_proba(self, features: Union[BeaconFeatures, List[BeaconFeatures], np.ndarray]) -> np.ndarray:
        """Compute estimated probabilities of being a beacon for input features."""
        if not self._is_fitted or self.weights is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Classifier must be fitted prior to running inference")

        X_raw = self._parse_features_input(features)
        if X_raw.shape[1] != len(self.weights):
            raise ValueError(f"Feature dimension mismatch: expected {len(self.weights)}, got {X_raw.shape[1]}")

        if np.isnan(X_raw).any() or np.isinf(X_raw).any():
            raise ValueError("Input features contain NaN or Infinite values")

        X = (X_raw - self.feature_mean) / self.feature_std
        z = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(z)
        return probs

    def predict(self, features: Union[BeaconFeatures, List[BeaconFeatures], np.ndarray]) -> np.ndarray:
        """Predict binary class labels (1 for beacon, 0 for non-beacon)."""
        probs = self.predict_proba(features)
        return (probs >= self.config.classification_threshold).astype(int)

    def classify(self, feature: BeaconFeatures) -> BeaconIdentityResult:
        """Classify a single BeaconFeatures instance and return structured result."""
        if not isinstance(feature, BeaconFeatures):
            raise ValueError(f"Expected BeaconFeatures instance, got {type(feature)}")

        prob = float(self.predict_proba(feature)[0])
        pred_class = 1 if prob >= self.config.classification_threshold else 0
        is_beacon = bool(pred_class == 1)

        # Confidence metric derived from distance to decision boundary
        confidence = float(np.clip(2.0 * abs(prob - 0.5), 0.0, 1.0))

        return BeaconIdentityResult(
            is_beacon=is_beacon,
            confidence=confidence,
            probability=prob,
            predicted_class=pred_class,
        )

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _parse_features_input(
        features: Union[BeaconFeatures, List[BeaconFeatures], np.ndarray]
    ) -> np.ndarray:
        """Helper to parse supported feature input types into a 2D float64 NumPy array."""
        if isinstance(features, BeaconFeatures):
            return features.to_array().reshape(1, -1)
        elif isinstance(features, list):
            if not features:
                raise ValueError("Input feature list cannot be empty")
            if all(isinstance(f, BeaconFeatures) for f in features):
                return np.vstack([f.to_array() for f in features])
            else:
                arr = np.asarray(features, dtype=np.float64)
                if arr.ndim == 1:
                    return arr.reshape(1, -1)
                elif arr.ndim == 2:
                    return arr
                else:
                    raise ValueError(f"Array features must be 1D or 2D, got {arr.ndim}D")
        elif isinstance(features, np.ndarray):
            if features.size == 0:
                raise ValueError("Feature array cannot be empty")
            arr = features.astype(np.float64)
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            elif arr.ndim == 2:
                return arr
            else:
                raise ValueError(f"Array features must be 1D or 2D, got {arr.ndim}D")
        else:
            raise ValueError(f"Unsupported feature input type: {type(features)}")