"""Simulation benchmark execution using the existing closed-loop simulator."""

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Tuple

from core.simulator import ClosedLoopSimulator, SimulatorFrame

from benchmark.config import BenchmarkConfig


@dataclass(frozen=True, eq=False)
class BenchmarkResult:
    """Collected telemetry and stable metadata for one benchmark execution."""

    scenario_name: str
    seed: int
    duration_seconds: float
    frame_count: int
    telemetry_frames: Tuple[SimulatorFrame, ...]
    configuration_metadata: Mapping[str, object]

    def deterministic_signature(self) -> tuple[object, ...]:
        """Return deterministic benchmark content, excluding wall-clock timings."""
        return (
            self.scenario_name,
            self.seed,
            self.duration_seconds,
            self.frame_count,
            tuple(_frame_signature(frame) for frame in self.telemetry_frames),
            tuple(sorted(self.configuration_metadata.items())),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BenchmarkResult):
            return NotImplemented
        return self.deterministic_signature() == other.deterministic_signature()


class SimulationBenchmarkRunner:
    """Run a configured simulation without adding operational tracking logic."""

    def __init__(self, config: BenchmarkConfig) -> None:
        if not isinstance(config, BenchmarkConfig):
            raise TypeError("config must be a BenchmarkConfig")
        self.config = config
        self.simulator: Optional[ClosedLoopSimulator] = None

    def run(self) -> BenchmarkResult:
        """Reset and execute the configured number of simulation frames."""
        simulator = ClosedLoopSimulator(self.config.to_simulator_config())
        simulator.reset()
        self.simulator = simulator

        frame_count = math.ceil(self.config.duration_seconds / self.config.dt)
        telemetry = tuple(simulator.step() for _ in range(frame_count))
        return BenchmarkResult(
            scenario_name=self.config.scenario_name,
            seed=self.config.seed,
            duration_seconds=self.config.duration_seconds,
            frame_count=len(telemetry),
            telemetry_frames=telemetry,
            configuration_metadata=self.config.metadata(),
        )


def _frame_signature(frame: SimulatorFrame) -> tuple[object, ...]:
    """Extract deterministic telemetry fields without comparing image arrays or timing."""
    target = frame.target_truth
    camera = frame.camera_state
    gimbal = frame.gimbal_state
    sensor = frame.sensor_frame
    candidates = tuple(
        (
            candidate.centroid_x,
            candidate.centroid_y,
            candidate.area,
            candidate.peak_intensity,
            candidate.mean_intensity,
            candidate.bbox_x,
            candidate.bbox_y,
            candidate.bbox_width,
            candidate.bbox_height,
        )
        for candidate in frame.candidates
    )
    identity = None if frame.identity is None else (
        frame.identity.is_beacon,
        frame.identity.confidence,
        frame.identity.probability,
        frame.identity.predicted_class,
    )
    tracking = frame.tracking
    modulation = frame.modulation
    confidence = frame.confidence
    kalman = None if frame.kalman is None else (
        frame.kalman.timestamp,
        frame.kalman.position_x,
        frame.kalman.position_y,
        frame.kalman.velocity_x,
        frame.kalman.velocity_y,
    )
    prediction = None if frame.prediction is None else (
        frame.prediction.source_timestamp,
        frame.prediction.prediction_horizon,
        frame.prediction.predicted_timestamp,
        frame.prediction.predicted_position_x,
        frame.prediction.predicted_position_y,
    )
    trust = None if frame.trust is None else (
        frame.trust.observed_x,
        frame.trust.observed_y,
        frame.trust.predicted_x,
        frame.trust.predicted_y,
        frame.trust.disagreement_px,
        frame.trust.trust_score,
        frame.trust.model_preferred,
    )
    command = None if frame.command is None else (
        frame.command.timestamp,
        frame.command.pan_rate_command_rad_s,
        frame.command.tilt_rate_command_rad_s,
    )
    return (
        frame.timestamp,
        frame.frame_index,
        tuple(target.position),
        tuple(target.velocity),
        tuple(camera.position),
        camera.pan_rad,
        camera.tilt_rad,
        gimbal.timestamp,
        gimbal.pan_angle_rad,
        gimbal.tilt_angle_rad,
        gimbal.pan_rate_rad_s,
        gimbal.tilt_rate_rad_s,
        sensor.target_visible,
        sensor.projected_pixel,
        candidates,
        identity,
        (
            modulation.verified,
            modulation.confidence,
            modulation.estimated_frequency_hz,
            modulation.sample_count,
            modulation.duration_seconds,
            modulation.frequency_error_hz,
        ),
        (
            tracking.state.value,
            tracking.timestamp,
            tracking.position_x,
            tracking.position_y,
            tracking.velocity_x,
            tracking.velocity_y,
            tracking.confidence,
            tracking.candidate_id,
            tracking.missed_frames,
            tracking.acquisition_count,
            tracking.is_locked,
        ),
        kalman,
        prediction,
        (
            confidence.detection_confidence,
            confidence.ai_confidence,
            confidence.modulation_confidence,
            confidence.fused_confidence,
        ),
        trust,
        command,
    )