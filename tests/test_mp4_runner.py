from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmark.mp4_runner import MP4BenchmarkConfig, MP4BenchmarkRunner
from core.modulation import ModulationVerifier
from core.processing import OperationalProcessor
from core.simulator import ClosedLoopSimulator


def _write_video(path: Path, frame_count: int = 5) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (64, 48),
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV MP4 encoder is unavailable")
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + index, 24), 4, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_valid_mp4_processes_frames_in_order_with_monotonic_timestamps(tmp_path):
    video_path = _write_video(tmp_path / "sequence.mp4", frame_count=5)

    result = MP4BenchmarkRunner(
        MP4BenchmarkConfig(video_path=video_path, scenario_name="sequence")
    ).run()

    assert result.opened is True
    assert result.error is None
    assert result.frame_count == 5
    assert [frame.frame_index for frame in result.telemetry_frames] == list(range(5))
    timestamps = [frame.timestamp for frame in result.telemetry_frames]
    assert timestamps == sorted(timestamps)
    assert result.frame_width == 64
    assert result.frame_height == 48
    assert result.video_fps is not None
    assert all(frame.processing_time_seconds >= 0.0 for frame in result.telemetry_frames)


def test_mp4_frame_limit_and_eof_are_clean(tmp_path):
    video_path = _write_video(tmp_path / "limited.mp4", frame_count=4)

    result = MP4BenchmarkRunner(
        MP4BenchmarkConfig(video_path=video_path, max_frames=2)
    ).run()

    assert result.opened is True
    assert result.frame_count == 2
    assert [frame.frame_index for frame in result.telemetry_frames] == [0, 1]


def test_invalid_and_unreadable_video_paths_are_handled_cleanly(tmp_path):
    missing = MP4BenchmarkRunner(
        MP4BenchmarkConfig(video_path=tmp_path / "missing.mp4")
    ).run()
    invalid_file = tmp_path / "invalid.mp4"
    invalid_file.write_text("not a video", encoding="ascii")
    unreadable = MP4BenchmarkRunner(
        MP4BenchmarkConfig(video_path=invalid_file)
    ).run()

    assert missing.opened is False
    assert missing.frame_count == 0
    assert missing.error is not None
    assert unreadable.opened is False
    assert unreadable.frame_count == 0
    assert unreadable.error is not None


def test_mp4_telemetry_has_no_synthetic_truth_and_reuses_pipeline(tmp_path, monkeypatch):
    video_path = _write_video(tmp_path / "pipeline.mp4", frame_count=3)
    detector_calls = []
    original_detect = __import__("core.detection", fromlist=["BeaconDetector"]).BeaconDetector.detect

    def record_detect(detector, image):
        detector_calls.append(image)
        return original_detect(detector, image)

    monkeypatch.setattr(
        __import__("core.detection", fromlist=["BeaconDetector"]).BeaconDetector,
        "detect",
        record_detect,
    )
    result = MP4BenchmarkRunner(MP4BenchmarkConfig(video_path=video_path)).run()

    assert len(detector_calls) == result.frame_count
    assert all(isinstance(image, np.ndarray) and image.ndim == 2 for image in detector_calls)
    assert all(not hasattr(frame, "target_truth") for frame in result.telemetry_frames)
    assert all(not hasattr(frame, "projected_pixel") for frame in result.telemetry_frames)


def test_modulation_receives_observed_temporal_candidates(tmp_path, monkeypatch):
    video_path = _write_video(tmp_path / "modulation.mp4", frame_count=4)
    observations = []
    original_add = ModulationVerifier.add_observation

    def record_observation(verifier, observation):
        observations.append(observation)
        return original_add(verifier, observation)

    monkeypatch.setattr(ModulationVerifier, "add_observation", record_observation)
    result = MP4BenchmarkRunner(MP4BenchmarkConfig(video_path=video_path)).run()

    assert len(observations) == sum(
        bool(frame.candidates) for frame in result.telemetry_frames
    )
    assert [observation.frame_index for observation in observations] == sorted(
        observation.frame_index for observation in observations
    )
    assert all(observation.candidate_id == 0 for observation in observations)
    assert all(frame.modulation.verified is False for frame in result.telemetry_frames)


def test_videocapture_is_released_after_error(monkeypatch):
    import benchmark.mp4_runner as mp4_runner

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

        def read(self):
            raise RuntimeError("controlled read failure")

        def release(self):
            self.released = True

    fake_capture = FakeCapture()
    monkeypatch.setattr(mp4_runner.cv2, "VideoCapture", lambda path: fake_capture)

    result = MP4BenchmarkRunner(MP4BenchmarkConfig(video_path="controlled.mp4")).run()

    assert result.opened is True
    assert result.error == "controlled read failure"
    assert fake_capture.released is True


def test_simulation_and_mp4_share_operational_processor(tmp_path, monkeypatch):
    video_path = _write_video(tmp_path / "shared-processor.mp4", frame_count=1)
    calls = []
    original_process = OperationalProcessor.process_frame

    def record_process(processor, image, timestamp, frame_index):
        calls.append(type(processor))
        return original_process(processor, image, timestamp, frame_index)

    monkeypatch.setattr(OperationalProcessor, "process_frame", record_process)
    ClosedLoopSimulator().step()
    runner = MP4BenchmarkRunner(MP4BenchmarkConfig(video_path=video_path))
    result = runner.run()

    assert result.frame_count == 1
    assert len(calls) >= 2
    assert all(processor_type is OperationalProcessor for processor_type in calls)
    assert not hasattr(runner, "_features_from_candidate")