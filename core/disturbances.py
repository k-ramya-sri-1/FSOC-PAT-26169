"""Authoritative Sensor Disturbances and Environmental Degradation for FSOC-PAT-26169.

This module applies configurable noise, platform jitter, and atmospheric degradation
effects to clean synthetic SensorFrame instances produced by Stage 1C.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, docs/DEVELOPMENT_RULES.md,
docs/TEST_STRATEGY.md, and docs/PS_REQUIREMENTS.md.
"""

from dataclasses import dataclass, field
import enum
from typing import Optional

import numpy as np

from core.sensor import SensorFrame


class AtmosphereMode(str, enum.Enum):
    """Supported atmospheric degradation modes."""

    CLEAR = "clear"
    HAZE = "haze"
    FOG = "fog"
    RAIN = "rain"
    LOW_LIGHT = "low_light"


@dataclass(frozen=True)
class DisturbanceConfig:
    """Configuration parameters for sensor noise and environmental disturbances."""

    salt_pepper_probability: float = 0.0
    gaussian_std: float = 0.0
    poisson_strength: float = 0.0
    jitter_x_pixels: float = 0.0
    jitter_y_pixels: float = 0.0
    atmosphere_mode: AtmosphereMode = AtmosphereMode.CLEAR
    atmosphere_strength: float = 0.0
    low_light_scale: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate disturbance configuration parameters."""
        if not (0.0 <= self.salt_pepper_probability <= 1.0):
            raise ValueError(
                f"salt_pepper_probability must be in [0, 1], got {self.salt_pepper_probability}"
            )
        if self.gaussian_std < 0.0:
            raise ValueError(
                f"gaussian_std must be non-negative, got {self.gaussian_std}"
            )
        if self.poisson_strength < 0.0:
            raise ValueError(
                f"poisson_strength must be non-negative, got {self.poisson_strength}"
            )
        if self.jitter_x_pixels < 0.0 or self.jitter_y_pixels < 0.0:
            raise ValueError("jitter parameters must be non-negative")
        if not (0.0 <= self.atmosphere_strength <= 1.0):
            raise ValueError(
                f"atmosphere_strength must be in [0, 1], got {self.atmosphere_strength}"
            )
        if not (0.0 <= self.low_light_scale <= 1.0):
            raise ValueError(
                f"low_light_scale must be in [0, 1], got {self.low_light_scale}"
            )


@dataclass(frozen=True)
class DisturbedFrame:
    """Container for the degraded sensor image and its original source frame."""

    image: np.ndarray
    source_frame: SensorFrame

    def __post_init__(self) -> None:
        """Validate disturbed frame output array."""
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 2:
            raise ValueError("Disturbed image must be a 2D NumPy array")


class ImageDisturbanceModel:
    """Applies environmental degradation, jitter, and noise to clean SensorFrames."""

    def __init__(self, config: Optional[DisturbanceConfig] = None) -> None:
        self.config = config if config is not None else DisturbanceConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def reset(self) -> None:
        """Reset internal pseudo-random number generator to configured seed."""
        self.rng = np.random.default_rng(self.config.seed)

    def apply(self, frame: SensorFrame) -> DisturbedFrame:
        """Apply configured disturbances to clean sensor frame in deterministic order."""
        # Ensure source frame image remains immutable by working on a float32 copy
        img = frame.image.astype(np.float32, copy=True)

        # 1. Atmospheric degradation
        img = self._apply_atmosphere(img)

        # 2. Low-light scaling
        if self.config.low_light_scale < 1.0:
            img = img * self.config.low_light_scale

        # 3. Camera / Platform jitter (pixel translation)
        if self.config.jitter_x_pixels > 0.0 or self.config.jitter_y_pixels > 0.0:
            img = self._apply_jitter(img)

        # 4. Poisson noise
        if self.config.poisson_strength > 0.0:
            img = self._apply_poisson_noise(img)

        # 5. Gaussian noise
        if self.config.gaussian_std > 0.0:
            noise = self.rng.normal(0.0, self.config.gaussian_std, size=img.shape)
            img = img + noise

        # 6. Salt-and-pepper noise
        if self.config.salt_pepper_probability > 0.0:
            img = self._apply_salt_pepper_noise(img)

        # Final clipping to valid monochrome range [0.0, 255.0]
        np.clip(img, 0.0, 255.0, out=img)

        return DisturbedFrame(image=img, source_frame=frame)

    def _apply_atmosphere(self, img: np.ndarray) -> np.ndarray:
        """Apply atmospheric attenuation/veil approximations."""
        mode = self.config.atmosphere_mode
        alpha = self.config.atmosphere_strength

        if mode == AtmosphereMode.CLEAR or alpha == 0.0:
            return img

        if mode == AtmosphereMode.HAZE:
            # Contrast reduction with additive atmospheric background veil
            veil = 30.0 * alpha
            return img * (1.0 - 0.4 * alpha) + veil

        if mode == AtmosphereMode.FOG:
            # Stronger contrast attenuation and heavy veil
            veil = 80.0 * alpha
            return img * (1.0 - 0.7 * alpha) + veil

        if mode == AtmosphereMode.RAIN:
            # Attenuation + deterministic synthetic rain streaks
            attenuated = img * (1.0 - 0.3 * alpha)
            h, w = img.shape
            num_streaks = int(50 * alpha)
            if num_streaks > 0:
                y_coords = self.rng.integers(0, max(1, h - 10), size=num_streaks)
                x_coords = self.rng.integers(0, w, size=num_streaks)
                for x, y in zip(x_coords, y_coords):
                    length = int(self.rng.integers(5, 12))
                    y_end = min(h, y + length)
                    attenuated[y:y_end, x] += 40.0 * alpha
            return attenuated

        if mode == AtmosphereMode.LOW_LIGHT:
            # Direct uniform signal attenuation
            return img * (1.0 - 0.8 * alpha)

        return img

    def _apply_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply image-plane pixel translation jitter."""
        dx = self.rng.uniform(-self.config.jitter_x_pixels, self.config.jitter_x_pixels)
        dy = self.rng.uniform(-self.config.jitter_y_pixels, self.config.jitter_y_pixels)

        shift_x = int(round(dx))
        shift_y = int(round(dy))

        if shift_x == 0 and shift_y == 0:
            return img

        h, w = img.shape
        shifted = np.zeros_like(img)

        # Compute overlap slices
        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(h, h + shift_y)

        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        dst_x_start = max(0, shift_x)
        dst_x_end = min(w, w + shift_x)

        if (
            src_y_start < src_y_end
            and src_x_start < src_x_end
            and dst_y_start < dst_y_end
            and dst_x_start < dst_x_end
        ):
            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]

        return shifted

    def _apply_poisson_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply Poisson photon-counting noise approximation.

        Scales pixel intensities to expected photon counts based on poisson_strength,
        samples from Poisson distribution, and converts back to intensity units.
        """
        scale = self.config.poisson_strength
        # Avoid zero/negative values in Poisson expectation
        lambda_arr = np.maximum(img * scale, 0.0)
        poisson_counts = self.rng.poisson(lambda_arr).astype(np.float32)
        return poisson_counts / scale

    def _apply_salt_pepper_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply impulse salt-and-pepper noise."""
        p = self.config.salt_pepper_probability
        mask = self.rng.uniform(0.0, 1.0, size=img.shape)

        # Half salt (255.0), half pepper (0.0)
        salt_mask = mask < (p / 2.0)
        pepper_mask = (mask >= (p / 2.0)) & (mask < p)

        img[salt_mask] = 255.0
        img[pepper_mask] = 0.0
        return img