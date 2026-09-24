"""Unit tests for core/modulation.py following docs/TEST_STRATEGY.md."""

import math
import numpy as np
import pytest

from core.modulation import (
    EXPECTED_BEACON_FREQUENCY_HZ,
    ModulationConfig,
    ModulationObservation,
    ModulationResult,
    ModulationVerifier,
)


def _generate_synthetic_observations(
    frequency_hz: float = 15.0,
    amplitude: float = 50.0,
    baseline: float = 100.0,
    duration: float = 1.0,
    fps: float = 100.0,
    noise_std: float = 0.0,
    candidate_id: int = 1,
    seed: int = 42,
    jitter_dt: float = 0.0,
) -> list[ModulationObservation]:
    """Helper to generate synthetic temporal intensity observations for testing."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration * fps) + 1
    obs = []

    for i in range(n_samples):
        t = i / fps
        if jitter_dt > 0.0 and 0 < i < n_samples - 1:
            t += rng.uniform(-jitter_dt, jitter_dt)

        noise = rng.normal(0.0, noise_std) if noise_std > 0.0 else 0.0
        intensity = baseline + amplitude * math.sin(2.0 * math.pi * frequency_hz * t) + noise
        obs.append(
            ModulationObservation(
                timestamp=t,
                intensity=float(intensity),
                candidate_id=candidate_id,
            )
        )

    obs.sort(key=lambda x: x.timestamp)
    return obs


def test_modulation_config_validation():
    with pytest.raises(ValueError):
        ModulationConfig(frequency_tolerance_hz=-1.0)

    with pytest.raises(ValueError):
        ModulationConfig(minimum_samples=1)

    with pytest.raises(ValueError):
        ModulationConfig(minimum_duration_seconds=-0.5)

    with pytest.raises(ValueError):
        ModulationConfig(minimum_confidence=1.5)

    with pytest.raises(ValueError):
        ModulationConfig(minimum_samples=20, max_history_length=10)


def test_default_config():
    cfg = ModulationConfig()
    assert EXPECTED_BEACON_FREQUENCY_HZ == 15.0
    assert cfg.frequency_tolerance_hz == 2.0


def test_empty_history_returns_unverified():
    verifier = ModulationVerifier()
    res = verifier.verify()

    assert isinstance(res, ModulationResult)
    assert res.verified is False
    assert res.confidence == 0.0
    assert res.estimated_frequency_hz is None
    assert res.expected_frequency_hz == 15.0
    assert res.sample_count == 0


def test_insufficient_samples():
    cfg = ModulationConfig(minimum_samples=10)
    verifier = ModulationVerifier(cfg)

    for i in range(5):
        verifier.add_observation(ModulationObservation(timestamp=i * 0.01, intensity=100.0))

    res = verifier.verify()
    assert res.verified is False
    assert res.sample_count == 5


def test_insufficient_duration():
    cfg = ModulationConfig(minimum_samples=5, minimum_duration_seconds=1.0)
    verifier = ModulationVerifier(cfg)

    for i in range(5):
        verifier.add_observation(ModulationObservation(timestamp=i * 0.05, intensity=100.0))

    res = verifier.verify()
    assert res.verified is False
    assert res.duration_seconds < 1.0


def test_clean_15hz_synthetic_signal_recognized():
    verifier = ModulationVerifier()
    obs = _generate_synthetic_observations(frequency_hz=15.0, fps=100.0, duration=1.0)
    for o in obs:
        verifier.add_observation(o)

    res = verifier.verify()
    assert res.verified is True
    assert res.expected_frequency_hz == 15.0
    assert res.estimated_frequency_hz == pytest.approx(15.0, abs=1.0)
    assert res.confidence >= verifier.config.minimum_confidence


def test_10hz_signal_rejected_as_15hz_beacon():
    verifier = ModulationVerifier()
    obs = _generate_synthetic_observations(frequency_hz=10.0, fps=100.0, duration=1.0)
    for o in obs:
        verifier.add_observation(o)

    res = verifier.verify()
    assert res.verified is False
    assert res.estimated_frequency_hz == pytest.approx(10.0, abs=1.0)
    assert res.expected_frequency_hz == 15.0


def test_constant_intensity_rejected():
    verifier = ModulationVerifier()
    for i in range(101):
        verifier.add_observation(ModulationObservation(timestamp=i * 0.01, intensity=100.0))

    res = verifier.verify()
    assert res.verified is False
    assert res.confidence == 0.0


def test_noisy_15hz_signal_detectable():
    verifier = ModulationVerifier()
    obs = _generate_synthetic_observations(
        frequency_hz=15.0, fps=100.0, duration=1.0, noise_std=5.0
    )
    for o in obs:
        verifier.add_observation(o)

    res = verifier.verify()
    assert res.verified is True
    assert res.estimated_frequency_hz == pytest.approx(15.0, abs=1.0)


def test_irregular_timestamps_handled():
    verifier = ModulationVerifier()
    obs = _generate_synthetic_observations(
        frequency_hz=15.0, fps=100.0, duration=1.0, jitter_dt=0.001
    )
    for o in obs:
        verifier.add_observation(o)

    res = verifier.verify()
    assert res.verified is True


def test_duplicate_timestamps_rejected():
    verifier = ModulationVerifier()
    verifier.add_observation(ModulationObservation(timestamp=0.1, intensity=100.0))

    with pytest.raises(ValueError):
        verifier.add_observation(ModulationObservation(timestamp=0.1, intensity=110.0))


def test_non_monotonic_timestamps_rejected():
    verifier = ModulationVerifier()
    verifier.add_observation(ModulationObservation(timestamp=0.2, intensity=100.0))

    with pytest.raises(ValueError):
        verifier.add_observation(ModulationObservation(timestamp=0.1, intensity=110.0))


def test_reset_clears_verifier_state():
    verifier = ModulationVerifier()
    obs = _generate_synthetic_observations(frequency_hz=15.0, fps=100.0, duration=1.0)
    for o in obs:
        verifier.add_observation(o)

    assert len(verifier._observations) > 0
    verifier.reset()
    assert len(verifier._observations) == 0
    assert verifier._candidate_id is None


def test_repeated_verification_determinism():
    verifier = ModulationVerifier()
    obs = _generate_synthetic_observations(frequency_hz=15.0, fps=100.0, duration=1.0)
    for o in obs:
        verifier.add_observation(o)

    res1 = verifier.verify()
    res2 = verifier.verify()

    assert res1 == res2


def test_input_observation_immutability():
    obs = ModulationObservation(timestamp=0.1, intensity=100.0, candidate_id=1)

    with pytest.raises(AttributeError):
        obs.intensity = 120.0  # Cannot modify frozen dataclass


def test_candidate_id_isolation():
    verifier = ModulationVerifier()
    verifier.add_observation(ModulationObservation(timestamp=0.1, intensity=100.0, candidate_id=1))

    with pytest.raises(ValueError):
        verifier.add_observation(ModulationObservation(timestamp=0.2, intensity=100.0, candidate_id=2))


def test_undersampled_nyquist_violation_rejected():
    # Sampling at 20 FPS gives Nyquist = 10 Hz. Required 15 Hz cannot be resolved.
    verifier = ModulationVerifier()
    obs = _generate_synthetic_observations(frequency_hz=15.0, fps=20.0, duration=1.0)
    for o in obs:
        verifier.add_observation(o)

    res = verifier.verify()
    assert res.verified is False


def test_ground_truth_isolation():
    # Verify ModulationVerifier operates exclusively on ModulationObservation instances
    verifier = ModulationVerifier()
    for i in range(101):
        t = i * 0.01
        intensity = 100.0 + 50.0 * math.sin(2.0 * math.pi * 15.0 * t)
        verifier.add_observation(ModulationObservation(timestamp=t, intensity=intensity))

    res = verifier.verify()
    assert isinstance(res, ModulationResult)
    assert res.verified is True
    assert res.expected_frequency_hz == 15.0