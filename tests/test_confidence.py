"""Unit tests for core/confidence.py following docs/TEST_STRATEGY.md."""

import math
import pytest

from core.confidence import (
    ConfidenceEvidence,
    ConfidenceFusion,
    ConfidenceFusionConfig,
    ConfidenceResult,
)


def test_valid_evidence_construction():
    ev = ConfidenceEvidence(
        detection_confidence=0.8,
        ai_confidence=0.7,
        modulation_confidence=0.9,
    )
    assert ev.detection_confidence == 0.8
    assert ev.ai_confidence == 0.7
    assert ev.modulation_confidence == 0.9


def test_evidence_invalid_value_rejections():
    # Below 0
    with pytest.raises(ValueError):
        ConfidenceEvidence(detection_confidence=-0.1, ai_confidence=0.5, modulation_confidence=0.5)

    # Above 1
    with pytest.raises(ValueError):
        ConfidenceEvidence(detection_confidence=0.5, ai_confidence=1.1, modulation_confidence=0.5)

    # NaN
    with pytest.raises(ValueError):
        ConfidenceEvidence(
            detection_confidence=0.5, ai_confidence=0.5, modulation_confidence=float("nan")
        )

    # Positive Infinity
    with pytest.raises(ValueError):
        ConfidenceEvidence(
            detection_confidence=float("inf"), ai_confidence=0.5, modulation_confidence=0.5
        )

    # Negative Infinity
    with pytest.raises(ValueError):
        ConfidenceEvidence(
            detection_confidence=0.5, ai_confidence=float("-inf"), modulation_confidence=0.5
        )


def test_evidence_immutability():
    ev = ConfidenceEvidence(detection_confidence=0.8, ai_confidence=0.7, modulation_confidence=0.9)
    with pytest.raises(AttributeError):
        ev.detection_confidence = 0.5  # type: ignore


def test_valid_result_construction_and_immutability():
    res = ConfidenceResult(
        detection_confidence=0.8,
        ai_confidence=0.7,
        modulation_confidence=0.9,
        fused_confidence=0.79,
    )
    assert res.fused_confidence == 0.79
    with pytest.raises(AttributeError):
        res.fused_confidence = 1.0  # type: ignore


def test_config_defaults_and_immutability():
    cfg = ConfidenceFusionConfig()
    assert cfg.detection_weight == 0.4
    assert cfg.ai_weight == 0.3
    assert cfg.modulation_weight == 0.3
    with pytest.raises(AttributeError):
        cfg.detection_weight = 0.5  # type: ignore


def test_config_invalid_weight_rejections():
    # Negative weight
    with pytest.raises(ValueError):
        ConfidenceFusionConfig(detection_weight=-0.1, ai_weight=0.6, modulation_weight=0.5)

    # Weight greater than 1
    with pytest.raises(ValueError):
        ConfidenceFusionConfig(detection_weight=1.2, ai_weight=0.0, modulation_weight=0.0)

    # NaN weight
    with pytest.raises(ValueError):
        ConfidenceFusionConfig(detection_weight=float("nan"), ai_weight=0.5, modulation_weight=0.5)

    # Infinite weight
    with pytest.raises(ValueError):
        ConfidenceFusionConfig(
            detection_weight=0.4, ai_weight=float("inf"), modulation_weight=0.3
        )

    # Weights sum != 1.0
    with pytest.raises(ValueError):
        ConfidenceFusionConfig(detection_weight=0.5, ai_weight=0.5, modulation_weight=0.5)


def test_fusion_all_zero():
    fusion = ConfidenceFusion()
    ev = ConfidenceEvidence(detection_confidence=0.0, ai_confidence=0.0, modulation_confidence=0.0)
    res = fusion.fuse(ev)
    assert res.fused_confidence == 0.0


def test_fusion_all_one():
    fusion = ConfidenceFusion()
    ev = ConfidenceEvidence(detection_confidence=1.0, ai_confidence=1.0, modulation_confidence=1.0)
    res = fusion.fuse(ev)
    assert res.fused_confidence == 1.0


def test_fusion_detection_only():
    fusion = ConfidenceFusion()
    ev = ConfidenceEvidence(detection_confidence=1.0, ai_confidence=0.0, modulation_confidence=0.0)
    res = fusion.fuse(ev)
    assert res.fused_confidence == pytest.approx(0.4)


def test_fusion_ai_only():
    fusion = ConfidenceFusion()
    ev = ConfidenceEvidence(detection_confidence=0.0, ai_confidence=1.0, modulation_confidence=0.0)
    res = fusion.fuse(ev)
    assert res.fused_confidence == pytest.approx(0.3)


def test_fusion_modulation_only():
    fusion = ConfidenceFusion()
    ev = ConfidenceEvidence(detection_confidence=0.0, ai_confidence=0.0, modulation_confidence=1.0)
    res = fusion.fuse(ev)
    assert res.fused_confidence == pytest.approx(0.3)


def test_fusion_mixed_evidence():
    fusion = ConfidenceFusion()
    ev = ConfidenceEvidence(detection_confidence=0.8, ai_confidence=0.6, modulation_confidence=0.9)
    res = fusion.fuse(ev)
    # Expected: 0.4*0.8 + 0.3*0.6 + 0.3*0.9 = 0.32 + 0.18 + 0.27 = 0.77
    assert res.fused_confidence == pytest.approx(0.77)


def test_fusion_determinism_and_reusability():
    fusion = ConfidenceFusion()
    ev1 = ConfidenceEvidence(detection_confidence=0.5, ai_confidence=0.5, modulation_confidence=0.5)
    ev2 = ConfidenceEvidence(detection_confidence=0.9, ai_confidence=0.1, modulation_confidence=0.4)

    # Repeated calls produce identical result
    res1_a = fusion.fuse(ev1)
    res1_b = fusion.fuse(ev1)
    assert res1_a == res1_b

    # Instance reusability
    res2 = fusion.fuse(ev2)
    assert res2.fused_confidence == pytest.approx(0.4 * 0.9 + 0.3 * 0.1 + 0.3 * 0.4)


def test_custom_valid_configuration():
    cfg = ConfidenceFusionConfig(detection_weight=0.5, ai_weight=0.25, modulation_weight=0.25)
    fusion = ConfidenceFusion(cfg)
    ev = ConfidenceEvidence(detection_confidence=1.0, ai_confidence=0.0, modulation_confidence=0.0)
    res = fusion.fuse(ev)
    assert res.fused_confidence == pytest.approx(0.5)


def test_fusion_rejects_non_evidence_input():
    fusion = ConfidenceFusion()
    with pytest.raises(ValueError):
        fusion.fuse({"detection_confidence": 0.5})  # type: ignore


def test_fused_confidence_bounded_in_zero_one():
    fusion = ConfidenceFusion()
    ev = ConfidenceEvidence(
        detection_confidence=0.999, ai_confidence=0.999, modulation_confidence=0.999
    )
    res = fusion.fuse(ev)
    assert 0.0 <= res.fused_confidence <= 1.0