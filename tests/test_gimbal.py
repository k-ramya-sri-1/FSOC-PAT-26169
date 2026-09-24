"""Unit tests for core/gimbal.py following docs/TEST_STRATEGY.md."""

import math
import pytest

from core.gimbal import (
    CoarsePointingController,
    ControlCommand,
    ControlConfig,
    GimbalConfig,
    GimbalState,
    VirtualGimbal,
)


# ============================================================================
# 1. GimbalConfig Tests
# ============================================================================

def test_gimbal_config_defaults():
    cfg = GimbalConfig()
    assert cfg.max_pan_angle_rad == pytest.approx(math.radians(45.0))
    assert cfg.max_tilt_angle_rad == pytest.approx(math.radians(45.0))
    assert cfg.max_pan_rate_rad_s == pytest.approx(math.radians(10.0))
    assert cfg.max_tilt_rate_rad_s == pytest.approx(math.radians(10.0))
    assert cfg.initial_pan_angle_rad == 0.0
    assert cfg.initial_tilt_angle_rad == 0.0


def test_gimbal_config_valid_custom():
    cfg = GimbalConfig(
        max_pan_angle_rad=1.0,
        max_tilt_angle_rad=0.8,
        max_pan_rate_rad_s=0.2,
        max_tilt_rate_rad_s=0.15,
        initial_pan_angle_rad=0.1,
        initial_tilt_angle_rad=-0.1,
    )
    assert cfg.max_pan_angle_rad == 1.0
    assert cfg.max_tilt_angle_rad == 0.8
    assert cfg.max_pan_rate_rad_s == 0.2
    assert cfg.max_tilt_rate_rad_s == 0.15
    assert cfg.initial_pan_angle_rad == 0.1
    assert cfg.initial_tilt_angle_rad == -0.1


def test_gimbal_config_zero_negative_angles_rejected():
    with pytest.raises(ValueError):
        GimbalConfig(max_pan_angle_rad=0.0)

    with pytest.raises(ValueError):
        GimbalConfig(max_pan_angle_rad=-0.5)

    with pytest.raises(ValueError):
        GimbalConfig(max_tilt_angle_rad=0.0)

    with pytest.raises(ValueError):
        GimbalConfig(max_tilt_angle_rad=-0.5)


def test_gimbal_config_zero_negative_rates_rejected():
    with pytest.raises(ValueError):
        GimbalConfig(max_pan_rate_rad_s=0.0)

    with pytest.raises(ValueError):
        GimbalConfig(max_pan_rate_rad_s=-0.1)

    with pytest.raises(ValueError):
        GimbalConfig(max_tilt_rate_rad_s=0.0)

    with pytest.raises(ValueError):
        GimbalConfig(max_tilt_rate_rad_s=-0.1)


def test_gimbal_config_non_finite_rejected():
    # NaN
    with pytest.raises(ValueError):
        GimbalConfig(max_pan_angle_rad=float("nan"))

    # Positive infinity
    with pytest.raises(ValueError):
        GimbalConfig(max_tilt_angle_rad=float("inf"))

    # Negative infinity
    with pytest.raises(ValueError):
        GimbalConfig(max_pan_rate_rad_s=float("-inf"))


def test_gimbal_config_initial_angles_outside_limits_rejected():
    with pytest.raises(ValueError):
        GimbalConfig(max_pan_angle_rad=0.5, initial_pan_angle_rad=0.6)

    with pytest.raises(ValueError):
        GimbalConfig(max_pan_angle_rad=0.5, initial_pan_angle_rad=-0.6)

    with pytest.raises(ValueError):
        GimbalConfig(max_tilt_angle_rad=0.5, initial_tilt_angle_rad=0.6)

    with pytest.raises(ValueError):
        GimbalConfig(max_tilt_angle_rad=0.5, initial_tilt_angle_rad=-0.6)


# ============================================================================
# 2. GimbalState Tests
# ============================================================================

def test_gimbal_state_valid_construction_and_immutability():
    state = GimbalState(
        timestamp=1.5,
        pan_angle_rad=0.1,
        tilt_angle_rad=-0.2,
        pan_rate_rad_s=0.05,
        tilt_rate_rad_s=-0.05,
    )
    assert state.timestamp == 1.5
    assert state.pan_angle_rad == 0.1
    assert state.tilt_angle_rad == -0.2
    assert state.pan_rate_rad_s == 0.05
    assert state.tilt_rate_rad_s == -0.05

    with pytest.raises(AttributeError):
        state.pan_angle_rad = 0.5  # type: ignore


def test_gimbal_state_negative_timestamp_rejected():
    with pytest.raises(ValueError):
        GimbalState(
            timestamp=-0.1,
            pan_angle_rad=0.0,
            tilt_angle_rad=0.0,
            pan_rate_rad_s=0.0,
            tilt_rate_rad_s=0.0,
        )


def test_gimbal_state_non_finite_numeric_fields_rejected():
    with pytest.raises(ValueError):
        GimbalState(
            timestamp=float("nan"),
            pan_angle_rad=0.0,
            tilt_angle_rad=0.0,
            pan_rate_rad_s=0.0,
            tilt_rate_rad_s=0.0,
        )

    with pytest.raises(ValueError):
        GimbalState(
            timestamp=1.0,
            pan_angle_rad=float("inf"),
            tilt_angle_rad=0.0,
            pan_rate_rad_s=0.0,
            tilt_rate_rad_s=0.0,
        )


# ============================================================================
# 3. VirtualGimbal Tests
# ============================================================================

def test_virtual_gimbal_initial_state():
    cfg = GimbalConfig(initial_pan_angle_rad=0.1, initial_tilt_angle_rad=-0.1)
    gimbal = VirtualGimbal(cfg)

    s0 = gimbal.get_state()
    assert s0.timestamp == 0.0
    assert s0.pan_angle_rad == 0.1
    assert s0.tilt_angle_rad == -0.1
    assert s0.pan_rate_rad_s == 0.0
    assert s0.tilt_rate_rad_s == 0.0


def test_virtual_gimbal_reset():
    cfg = GimbalConfig(initial_pan_angle_rad=0.1, initial_tilt_angle_rad=-0.1)
    gimbal = VirtualGimbal(cfg)

    gimbal.update(0.05, 0.05, timestamp=1.0)
    gimbal.reset(timestamp=5.0)

    s_reset = gimbal.get_state()
    assert s_reset.timestamp == 5.0
    assert s_reset.pan_angle_rad == 0.1
    assert s_reset.tilt_angle_rad == -0.1
    assert s_reset.pan_rate_rad_s == 0.0
    assert s_reset.tilt_rate_rad_s == 0.0


def test_virtual_gimbal_pan_and_tilt_integration():
    gimbal = VirtualGimbal()
    rate = math.radians(2.0)  # 2 deg/s

    # Positive pan, negative tilt integration over dt=1.0s
    s1 = gimbal.update(rate, -rate, timestamp=1.0)
    assert s1.pan_angle_rad == pytest.approx(rate)
    assert s1.tilt_angle_rad == pytest.approx(-rate)

    # Negative pan, positive tilt integration over dt=1.0s
    s2 = gimbal.update(-rate, rate, timestamp=2.0)
    assert s2.pan_angle_rad == pytest.approx(0.0)
    assert s2.tilt_angle_rad == pytest.approx(0.0)


def test_virtual_gimbal_variable_timestep_integration():
    gimbal = VirtualGimbal()
    rate = 0.1  # rad/s

    # Step 1: dt = 0.5s -> pan = 0.05 rad
    s1 = gimbal.update(rate, 0.0, timestamp=0.5)
    assert s1.pan_angle_rad == pytest.approx(0.05)

    # Step 2: dt = 2.0s -> pan = 0.05 + 0.20 = 0.25 rad
    s2 = gimbal.update(rate, 0.0, timestamp=2.5)
    assert s2.pan_angle_rad == pytest.approx(0.25)


def test_virtual_gimbal_rate_saturations():
    max_rate = math.radians(10.0)
    cfg = GimbalConfig(max_pan_rate_rad_s=max_rate, max_tilt_rate_rad_s=max_rate)
    gimbal = VirtualGimbal(cfg)

    # Positive pan and negative tilt rate saturation
    s1 = gimbal.update(math.radians(20.0), math.radians(-20.0), timestamp=1.0)
    assert s1.pan_angle_rad == pytest.approx(max_rate)
    assert s1.tilt_angle_rad == pytest.approx(-max_rate)
    assert s1.pan_rate_rad_s == pytest.approx(max_rate)
    assert s1.tilt_rate_rad_s == pytest.approx(-max_rate)

    # Negative pan and positive tilt rate saturation
    s2 = gimbal.update(math.radians(-20.0), math.radians(20.0), timestamp=2.0)
    assert s2.pan_angle_rad == pytest.approx(0.0)
    assert s2.tilt_angle_rad == pytest.approx(0.0)
    assert s2.pan_rate_rad_s == pytest.approx(-max_rate)
    assert s2.tilt_rate_rad_s == pytest.approx(max_rate)


def test_virtual_gimbal_angle_limits_and_actual_rates():
    max_angle = math.radians(5.0)
    max_rate = math.radians(10.0)
    cfg = GimbalConfig(
        max_pan_angle_rad=max_angle,
        max_tilt_angle_rad=max_angle,
        max_pan_rate_rad_s=max_rate,
        max_tilt_rate_rad_s=max_rate,
    )
    gimbal = VirtualGimbal(cfg)

    # Upper pan limit & lower tilt limit (1 second update -> clamped at 5 deg)
    s1 = gimbal.update(max_rate, -max_rate, timestamp=1.0)
    assert s1.pan_angle_rad == pytest.approx(max_angle)
    assert s1.tilt_angle_rad == pytest.approx(-max_angle)
    assert s1.pan_rate_rad_s == pytest.approx(max_angle)  # Actual applied rate 5 deg/s
    assert s1.tilt_rate_rad_s == pytest.approx(-max_angle)

    # Lower pan limit & upper tilt limit
    gimbal.reset(timestamp=0.0)
    s2 = gimbal.update(-max_rate, max_rate, timestamp=1.0)
    assert s2.pan_angle_rad == pytest.approx(-max_angle)
    assert s2.tilt_angle_rad == pytest.approx(max_angle)
    assert s2.pan_rate_rad_s == pytest.approx(-max_angle)
    assert s2.tilt_rate_rad_s == pytest.approx(max_angle)


def test_virtual_gimbal_independent_pan_tilt_limits():
    cfg = GimbalConfig(
        max_pan_angle_rad=math.radians(10.0),
        max_tilt_angle_rad=math.radians(2.0),
    )
    gimbal = VirtualGimbal(cfg)

    rate = math.radians(5.0)
    s1 = gimbal.update(rate, rate, timestamp=1.0)

    assert s1.pan_angle_rad == pytest.approx(math.radians(5.0))
    assert s1.tilt_angle_rad == pytest.approx(math.radians(2.0))  # Clamped to 2 deg


def test_virtual_gimbal_timestamp_validations():
    gimbal = VirtualGimbal()

    # Equal timestamp rejection
    with pytest.raises(ValueError):
        gimbal.update(0.0, 0.0, timestamp=0.0)

    # Backward timestamp rejection
    with pytest.raises(ValueError):
        gimbal.update(0.0, 0.0, timestamp=-0.5)

    # NaN timestamp rejection
    with pytest.raises(ValueError):
        gimbal.update(0.0, 0.0, timestamp=float("nan"))

    # Infinite timestamp rejection
    with pytest.raises(ValueError):
        gimbal.update(0.0, 0.0, timestamp=float("inf"))


def test_virtual_gimbal_command_validations():
    gimbal = VirtualGimbal()

    # NaN command rejection
    with pytest.raises(ValueError):
        gimbal.update(float("nan"), 0.0, timestamp=1.0)

    # Infinite command rejection
    with pytest.raises(ValueError):
        gimbal.update(0.0, float("inf"), timestamp=1.0)


def test_virtual_gimbal_determinism():
    gimbal1 = VirtualGimbal()
    gimbal2 = VirtualGimbal()

    s1_a = gimbal1.update(0.1, -0.05, timestamp=0.5)
    s1_b = gimbal1.update(-0.02, 0.01, timestamp=1.0)

    s2_a = gimbal2.update(0.1, -0.05, timestamp=0.5)
    s2_b = gimbal2.update(-0.02, 0.01, timestamp=1.0)

    assert s1_a == s2_a
    assert s1_b == s2_b


# ============================================================================
# 4. ControlConfig Tests
# ============================================================================

def test_control_config_valid_defaults_and_custom():
    cfg_default = ControlConfig()
    assert cfg_default.pan_gain_rad_s_per_px == 0.001
    assert cfg_default.tilt_gain_rad_s_per_px == 0.001
    assert cfg_default.deadband_px == 0.0

    cfg_custom = ControlConfig(
        pan_gain_rad_s_per_px=0.002,
        tilt_gain_rad_s_per_px=0.003,
        max_pan_rate_rad_s=0.1,
        max_tilt_rate_rad_s=0.1,
        deadband_px=2.0,
    )
    assert cfg_custom.pan_gain_rad_s_per_px == 0.002
    assert cfg_custom.deadband_px == 2.0


def test_control_config_rejections():
    # Negative gain rejected
    with pytest.raises(ValueError):
        ControlConfig(pan_gain_rad_s_per_px=-0.001)

    # Negative deadband rejected
    with pytest.raises(ValueError):
        ControlConfig(deadband_px=-1.0)

    # Zero max rate rejected
    with pytest.raises(ValueError):
        ControlConfig(max_pan_rate_rad_s=0.0)

    # Non-finite value rejected
    with pytest.raises(ValueError):
        ControlConfig(pan_gain_rad_s_per_px=float("nan"))


def test_control_config_explicit_non_finite_rejections():
    # Positive infinity for pan_gain_rad_s_per_px
    with pytest.raises(ValueError):
        ControlConfig(pan_gain_rad_s_per_px=float("inf"))

    # Negative infinity for tilt_gain_rad_s_per_px
    with pytest.raises(ValueError):
        ControlConfig(tilt_gain_rad_s_per_px=float("-inf"))

    # Positive infinity for max_pan_rate_rad_s
    with pytest.raises(ValueError):
        ControlConfig(max_pan_rate_rad_s=float("inf"))

    # Negative infinity for max_tilt_rate_rad_s
    with pytest.raises(ValueError):
        ControlConfig(max_tilt_rate_rad_s=float("-inf"))

    # Positive infinity for deadband_px
    with pytest.raises(ValueError):
        ControlConfig(deadband_px=float("inf"))


# ============================================================================
# 5. CoarsePointingController Tests
# ============================================================================

def test_controller_zero_error_produces_zero_command():
    ctrl = CoarsePointingController()
    cmd = ctrl.compute_command(
        target_x=320.0,
        target_y=240.0,
        image_center_x=320.0,
        image_center_y=240.0,
        timestamp=0.1,
    )
    assert cmd.pan_rate_command_rad_s == 0.0
    assert cmd.tilt_rate_command_rad_s == 0.0


def test_controller_sign_conventions_and_proportional_scaling():
    ctrl = CoarsePointingController(
        ControlConfig(pan_gain_rad_s_per_px=0.001, tilt_gain_rad_s_per_px=0.001)
    )

    # Target right of center (target_x > center_x) -> error_x > 0 -> pan_command < 0
    cmd_right = ctrl.compute_command(350.0, 240.0, 320.0, 240.0, timestamp=0.1)
    assert cmd_right.pan_rate_command_rad_s == pytest.approx(-0.03)

    # Target left of center (target_x < center_x) -> error_x < 0 -> pan_command > 0
    cmd_left = ctrl.compute_command(290.0, 240.0, 320.0, 240.0, timestamp=0.2)
    assert cmd_left.pan_rate_command_rad_s == pytest.approx(0.03)

    # Target below center (target_y > center_y) -> error_y > 0 -> tilt_command > 0
    cmd_below = ctrl.compute_command(320.0, 270.0, 320.0, 240.0, timestamp=0.3)
    assert cmd_below.tilt_rate_command_rad_s == pytest.approx(0.03)

    # Target above center (target_y < center_y) -> error_y < 0 -> tilt_command < 0
    cmd_above = ctrl.compute_command(320.0, 210.0, 320.0, 240.0, timestamp=0.4)
    assert cmd_above.tilt_rate_command_rad_s == pytest.approx(-0.03)


def test_controller_command_saturations():
    max_rate = 0.05
    cfg = ControlConfig(
        pan_gain_rad_s_per_px=0.001,
        tilt_gain_rad_s_per_px=0.001,
        max_pan_rate_rad_s=max_rate,
        max_tilt_rate_rad_s=max_rate,
    )
    ctrl = CoarsePointingController(cfg)

    # Error 100px -> raw command = 0.10 rad/s -> saturated to 0.05 rad/s
    cmd = ctrl.compute_command(420.0, 140.0, 320.0, 240.0, timestamp=0.1)
    assert cmd.pan_rate_command_rad_s == pytest.approx(-max_rate)
    assert cmd.tilt_rate_command_rad_s == pytest.approx(-max_rate)


def test_controller_deadband_behavior():
    cfg = ControlConfig(
        pan_gain_rad_s_per_px=0.001,
        tilt_gain_rad_s_per_px=0.001,
        deadband_px=5.0,
    )
    ctrl = CoarsePointingController(cfg)

    # Error X = 4px (<= 5px deadband) -> pan = 0
    # Error Y = 10px (> 5px deadband) -> tilt active
    cmd1 = ctrl.compute_command(324.0, 250.0, 320.0, 240.0, timestamp=0.1)
    assert cmd1.pan_rate_command_rad_s == 0.0
    assert cmd1.tilt_rate_command_rad_s == pytest.approx(0.01)

    # Exact deadband boundary (Error X = 5.0px <= 5.0px) -> pan = 0
    cmd2 = ctrl.compute_command(325.0, 240.0, 320.0, 240.0, timestamp=0.2)
    assert cmd2.pan_rate_command_rad_s == 0.0

    # Just outside deadband boundary (Error X = 5.1px > 5.0px) -> pan active
    cmd3 = ctrl.compute_command(325.1, 240.0, 320.0, 240.0, timestamp=0.3)
    assert cmd3.pan_rate_command_rad_s == pytest.approx(-0.0051)


def test_controller_timestamp_validations_and_reset():
    ctrl = CoarsePointingController()

    cmd = ctrl.compute_command(320.0, 240.0, 320.0, 240.0, timestamp=1.0)
    assert cmd.timestamp == 1.0

    # Equal timestamp rejection
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, 240.0, 320.0, 240.0, timestamp=1.0)

    # Backward timestamp rejection
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, 240.0, 320.0, 240.0, timestamp=0.5)

    # NaN timestamp rejection
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, 240.0, 320.0, 240.0, timestamp=float("nan"))

    # Reset allows new timestamp sequence
    ctrl.reset(timestamp=0.0)
    cmd2 = ctrl.compute_command(320.0, 240.0, 320.0, 240.0, timestamp=0.2)
    assert cmd2.timestamp == 0.2


def test_controller_infinite_timestamp_rejections():
    ctrl = CoarsePointingController()

    # Positive infinity timestamp
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, 240.0, 320.0, 240.0, timestamp=float("inf"))

    # Negative infinity timestamp
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, 240.0, 320.0, 240.0, timestamp=float("-inf"))


def test_controller_coordinate_validations():
    ctrl = CoarsePointingController()

    # NaN coordinate rejection
    with pytest.raises(ValueError):
        ctrl.compute_command(float("nan"), 240.0, 320.0, 240.0, timestamp=0.1)

    # Positive infinity target_x
    with pytest.raises(ValueError):
        ctrl.compute_command(float("inf"), 240.0, 320.0, 240.0, timestamp=0.1)

    # Negative infinity target_y
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, float("-inf"), 320.0, 240.0, timestamp=0.1)

    # Positive infinity image_center_x
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, 240.0, float("inf"), 240.0, timestamp=0.1)

    # Negative infinity image_center_y
    with pytest.raises(ValueError):
        ctrl.compute_command(320.0, 240.0, 320.0, float("-inf"), timestamp=0.1)


def test_controller_determinism():
    ctrl1 = CoarsePointingController()
    ctrl2 = CoarsePointingController()

    cmd1 = ctrl1.compute_command(350.0, 220.0, 320.0, 240.0, timestamp=0.1)
    cmd2 = ctrl2.compute_command(350.0, 220.0, 320.0, 240.0, timestamp=0.1)

    assert cmd1 == cmd2