"""Authoritative 15 Hz Beacon Modulation Verification for FSOC-PAT-26169.

This module evaluates temporal candidate observation sequences produced by the detection pipeline
to verify whether an observed intensity signal contains the required fixed 15.0 Hz optical beacon modulation.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np

# Fixed reference beacon modulation frequency required by the project specification
EXPECTED_BEACON_FREQUENCY_HZ: float = 15.0


@dataclass(frozen=True)
class ModulationObservation:
    """Immutable temporal observation extracted from candidate detection observations."""

    timestamp: float
    intensity: float
    candidate_id: int = 0
    signal_strength: Optional[float] = None
    frame_index: Optional[int] = None
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate observation fields."""
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")


@dataclass(frozen=True)
class ModulationConfig:
    """Configuration parameters for 15 Hz beacon modulation verification."""

    frequency_tolerance_hz: float = 2.0
    minimum_samples: int = 10
    minimum_duration_seconds: float = 0.3
    minimum_confidence: float = 0.6
    max_history_length: int = 200

    def __post_init__(self) -> None:
        """Validate modulation configuration parameters."""
        if self.frequency_tolerance_hz <= 0.0:
            raise ValueError(
                f"frequency_tolerance_hz must be strictly positive, got {self.frequency_tolerance_hz}"
            )
        if self.minimum_samples < 2:
            raise ValueError(
                f"minimum_samples must be at least 2, got {self.minimum_samples}"
            )
        if self.minimum_duration_seconds <= 0.0:
            raise ValueError(
                f"minimum_duration_seconds must be strictly positive, got {self.minimum_duration_seconds}"
            )
        if not (0.0 <= self.minimum_confidence <= 1.0):
            raise ValueError(
                f"minimum_confidence must be in [0, 1], got {self.minimum_confidence}"
            )
        if self.max_history_length < self.minimum_samples or self.max_history_length <= 0:
            raise ValueError(
                f"max_history_length ({self.max_history_length}) must be positive and >= minimum_samples ({self.minimum_samples})"
            )


@dataclass(frozen=True)
class ModulationResult:
    """Result of temporal 15 Hz beacon modulation verification."""

    verified: bool
    confidence: float
    estimated_frequency_hz: Optional[float]
    expected_frequency_hz: float
    sample_count: int
    duration_seconds: float
    frequency_error_hz: Optional[float] = None


class ModulationVerifier:
    """Evaluates temporal candidate intensity sequences for the fixed 15 Hz beacon modulation signature."""

    def __init__(self, config: Optional[ModulationConfig] = None) -> None:
        self.config = config if config is not None else ModulationConfig()
        self._observations: List[ModulationObservation] = []
        self._candidate_id: Optional[int] = None

    def add_observation(self, observation: ModulationObservation) -> None:
        """Add a temporal observation to history with candidate stream isolation and timestamp validation."""
        if not isinstance(observation, ModulationObservation):
            raise ValueError(
                f"Expected ModulationObservation instance, got {type(observation)}"
            )

        if self._candidate_id is None:
            self._candidate_id = observation.candidate_id
        elif observation.candidate_id != self._candidate_id:
            raise ValueError(
                f"Candidate ID mismatch: expected {self._candidate_id}, got {observation.candidate_id}"
            )

        if self._observations:
            last_ts = self._observations[-1].timestamp
            if observation.timestamp == last_ts:
                raise ValueError(
                    f"Duplicate timestamp encountered: {observation.timestamp}"
                )
            if observation.timestamp < last_ts:
                raise ValueError(
                    f"Non-monotonic/decreasing timestamp sequence: {observation.timestamp} < {last_ts}"
                )

        self._observations.append(observation)

        # Enforce history window bound while retaining newest observations
        if len(self._observations) > self.config.max_history_length:
            self._observations = self._observations[-self.config.max_history_length :]

    def reset(self) -> None:
        """Clear temporal observation history and candidate stream state."""
        self._observations.clear()
        self._candidate_id = None

    def verify(self) -> ModulationResult:
        """Analyze temporal observation history and evaluate against required 15.0 Hz beacon frequency."""
        sample_count = len(self._observations)
        expected_freq = EXPECTED_BEACON_FREQUENCY_HZ

        if sample_count < self.config.minimum_samples:
            return ModulationResult(
                verified=False,
                confidence=0.0,
                estimated_frequency_hz=None,
                expected_frequency_hz=expected_freq,
                sample_count=sample_count,
                duration_seconds=0.0 if sample_count == 0 else self._observations[-1].timestamp - self._observations[0].timestamp,
            )

        timestamps = np.array([obs.timestamp for obs in self._observations], dtype=np.float64)
        intensities = np.array([obs.intensity for obs in self._observations], dtype=np.float64)

        duration = float(timestamps[-1] - timestamps[0])
        if duration < self.config.minimum_duration_seconds or duration <= 0.0:
            return ModulationResult(
                verified=False,
                confidence=0.0,
                estimated_frequency_hz=None,
                expected_frequency_hz=expected_freq,
                sample_count=sample_count,
                duration_seconds=duration,
            )

        # Calculate sampling characteristics from timestamps
        intervals = np.diff(timestamps)
        mean_dt = float(np.mean(intervals))
        if mean_dt <= 0.0:
            return ModulationResult(
                verified=False,
                confidence=0.0,
                estimated_frequency_hz=None,
                expected_frequency_hz=expected_freq,
                sample_count=sample_count,
                duration_seconds=duration,
            )

        fs = 1.0 / mean_dt
        nyquist_freq = fs / 2.0

        # Nyquist safety check: ensure required 15 Hz is below usable temporal bandwidth
        if expected_freq >= nyquist_freq:
            return ModulationResult(
                verified=False,
                confidence=0.0,
                estimated_frequency_hz=None,
                expected_frequency_hz=expected_freq,
                sample_count=sample_count,
                duration_seconds=duration,
            )

        # Remove DC component / mean intensity
        mean_val = float(np.mean(intensities))
        ac_signal = intensities - mean_val

        signal_std = float(np.std(ac_signal))
        if signal_std < 1e-6:
            # Constant or zero-amplitude signal
            return ModulationResult(
                verified=False,
                confidence=0.0,
                estimated_frequency_hz=0.0,
                expected_frequency_hz=expected_freq,
                sample_count=sample_count,
                duration_seconds=duration,
                frequency_error_hz=expected_freq,
            )

        # Spectral frequency estimation via classical FFT analysis
        est_freq, spectral_purity = self._estimate_dominant_frequency(
            timestamps, ac_signal, nyquist_freq
        )

        if est_freq is None:
            return ModulationResult(
                verified=False,
                confidence=0.0,
                estimated_frequency_hz=None,
                expected_frequency_hz=expected_freq,
                sample_count=sample_count,
                duration_seconds=duration,
            )

        freq_err = abs(est_freq - expected_freq)

        # Calculate deterministic verification confidence score
        if freq_err <= self.config.frequency_tolerance_hz:
            freq_score = 1.0 - (freq_err / self.config.frequency_tolerance_hz)
            confidence = float(np.clip(0.5 * freq_score + 0.5 * spectral_purity, 0.0, 1.0))
        else:
            confidence = 0.0

        verified = (
            freq_err <= self.config.frequency_tolerance_hz
            and confidence >= self.config.minimum_confidence
        )

        return ModulationResult(
            verified=verified,
            confidence=confidence,
            estimated_frequency_hz=est_freq,
            expected_frequency_hz=expected_freq,
            sample_count=sample_count,
            duration_seconds=duration,
            frequency_error_hz=freq_err,
        )

    def _estimate_dominant_frequency(
        self,
        timestamps: np.ndarray,
        ac_signal: np.ndarray,
        nyquist_freq: float,
    ) -> Tuple[Optional[float], float]:
        """Estimate dominant AC signal modulation frequency and spectral purity."""
        n_grid = len(timestamps)
        duration = float(timestamps[-1] - timestamps[0])

        # Resample onto a uniform temporal grid using linear interpolation
        uniform_t = np.linspace(timestamps[0], timestamps[-1], n_grid)
        uniform_signal = np.interp(uniform_t, timestamps, ac_signal)

        # Apply Hanning window to minimize spectral leakage
        window = np.hanning(n_grid)
        windowed_signal = uniform_signal * window

        fft_vals = np.fft.rfft(windowed_signal)
        fft_freqs = np.fft.rfftfreq(n_grid, d=duration / (n_grid - 1))
        fft_mag = np.abs(fft_vals)

        # Filter out DC residual (< 0.5 Hz) and frequencies above Nyquist limit
        valid_mask = (fft_freqs >= 0.5) & (fft_freqs <= nyquist_freq)
        if not np.any(valid_mask):
            return None, 0.0

        masked_freqs = fft_freqs[valid_mask]
        masked_mag = fft_mag[valid_mask]

        max_idx = np.argmax(masked_mag)
        peak_mag = masked_mag[max_idx]
        total_mag = np.sum(masked_mag)

        if total_mag <= 0.0:
            return None, 0.0

        estimated_freq = float(masked_freqs[max_idx])
        spectral_purity = float(peak_mag / total_mag)

        return estimated_freq, spectral_purity