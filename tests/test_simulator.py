import numpy as np
import pytest

from core.scene import StraightMotionModel
from core.ai_identity import BeaconIdentityResult
from core.detection import DetectionCandidate
from core.simulator import ClosedLoopSimulator, SimulatorConfig
from core.tracking import TrackingState


def test_simulator_initializes_and_steps_through_pipeline():
    simulator = ClosedLoopSimulator()

    simulator.step()
    frame = simulator.step()

    assert frame.frame_index == 1
    assert frame.timestamp == simulator.config.dt
    assert frame.sensor_frame.image.ndim == 2
    assert frame.candidates
    assert frame.identity is not None
    assert frame.tracking is not None
    assert frame.kalman is not None
    assert frame.prediction is not None
    assert frame.command is not None


def test_multiple_frames_advance_time_and_target_motion():
    simulator = ClosedLoopSimulator(
        SimulatorConfig(
            dt=1.0 / 30.0,
            target_velocity=(1.0, 0.0, 0.0),
            target_position=(0.0, 0.0, 100.0),
        )
    )

    frames = [simulator.step() for _ in range(4)]

    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp for frame in frames] == [0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0]
    assert frames[-1].target_truth.position[0] > frames[0].target_truth.position[0]


def test_closed_loop_gimbal_changes_camera_and_affects_next_frame():
    simulator = ClosedLoopSimulator()

    first = simulator.step()
    second = simulator.step()
    third = simulator.step()

    assert second.gimbal_state.pan_angle_rad != first.gimbal_state.pan_angle_rad
    assert second.gimbal_state.tilt_angle_rad != first.gimbal_state.tilt_angle_rad
    assert second.sensor_frame.projected_pixel is not None
    assert third.sensor_frame.projected_pixel is not None
    assert third.sensor_frame.projected_pixel != second.sensor_frame.projected_pixel
    assert simulator.history[2].camera_state.pan_rad == second.gimbal_state.pan_angle_rad
    assert simulator.history[2].camera_state.tilt_rad == second.gimbal_state.tilt_angle_rad


def test_real_temporal_modulation_reaches_verification():
    simulator = ClosedLoopSimulator()

    frames = [simulator.step() for _ in range(40)]

    assert frames[-1].modulation.sample_count == 40
    assert frames[-1].modulation.verified is True
    assert frames[-1].modulation.estimated_frequency_hz == pytest.approx(15.0)


def test_pipeline_handles_temporary_target_loss_without_crashing():
    simulator = ClosedLoopSimulator()
    simulator.step()
    simulator.step()

    simulator.scene.target_state.position = np.array([0.0, 0.0, -10.0])
    lost_frame = simulator.step()

    assert lost_frame.candidates == []
    assert lost_frame.identity is None
    assert lost_frame.tracking.state in {
        TrackingState.SEARCHING,
        TrackingState.TRACKING,
        TrackingState.LOST,
    }
    assert lost_frame.kalman is not None


def test_reset_is_deterministic_and_preserves_motion_model():
    config = SimulatorConfig(
        target_position=(0.0, 0.0, 100.0),
        motion_model=StraightMotionModel((0.5, 0.0, 0.0)),
    )
    simulator = ClosedLoopSimulator(config)
    first_run = [simulator.step() for _ in range(3)]

    simulator.reset()
    second_run = [simulator.step() for _ in range(3)]

    for first, second in zip(first_run, second_run):
        assert first.timestamp == second.timestamp
        np.testing.assert_array_equal(first.sensor_frame.image, second.sensor_frame.image)
        np.testing.assert_array_equal(first.target_truth.position, second.target_truth.position)
        assert first.gimbal_state == second.gimbal_state
        assert first.tracking == second.tracking


def test_operational_boundaries_receive_observations_not_scene_truth():
    simulator = ClosedLoopSimulator()
    received = []
    original_detect = simulator.detector.detect

    def record_detection(image):
        received.append(image)
        return original_detect(image)

    simulator.detector.detect = record_detection
    simulator.step()
    frame = simulator.step()

    assert received
    assert isinstance(received[0], np.ndarray)
    assert frame.identity is not None
    assert frame.kalman is not None
    assert frame.prediction is not None
    assert frame.command is not None
    assert frame.command.timestamp == frame.timestamp


def test_selected_candidate_is_shared_by_modulation_and_trust():
    simulator = ClosedLoopSimulator()
    candidates = [
        DetectionCandidate(300.0, 240.0, 20.0, 150.0, 120.0, 298, 238, 5, 5),
        DetectionCandidate(350.0, 240.0, 20.0, 250.0, 200.0, 348, 238, 5, 5),
    ]
    simulator.detector.detect = lambda image: candidates
    simulator.identity_classifier.classify = lambda feature: BeaconIdentityResult(
        is_beacon=feature.peak_intensity > 200.0,
        confidence=0.9 if feature.peak_intensity > 200.0 else 0.1,
        probability=0.95 if feature.peak_intensity > 200.0 else 0.05,
        predicted_class=1 if feature.peak_intensity > 200.0 else 0,
    )

    frame = simulator.step()

    assert frame.modulation.sample_count == 1
    assert frame.trust is not None
    assert frame.trust.observed_x == candidates[1].centroid_x
    assert frame.tracking.candidate_id == 0