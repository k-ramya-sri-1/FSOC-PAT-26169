"""Authoritative Scene State Foundation for FSOC-PAT-26169.

This module provides the physical world state representation and target motion
models for simulation and benchmarking.

Follows docs/ARCHITECTURE.md, docs/INTERFACES.md, and docs/COORDINATE_CONVENTION.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Tuple, Union

import numpy as np


def _validate_3d_vector(vec: Union[np.ndarray, Tuple[float, float, float], list], name: str) -> np.ndarray:
    """Validate and return a 3D float NumPy vector."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector of shape (3,), got shape {arr.shape}")
    return arr


@dataclass
class TargetState:
    """Target state in world coordinates (+X right, +Y up, +Z forward)."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 100.0], dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))

    def __post_init__(self) -> None:
        self.position = _validate_3d_vector(self.position, "Target position")
        self.velocity = _validate_3d_vector(self.velocity, "Target velocity")


@dataclass
class CameraState:
    """Camera state in world coordinates and orientation."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    pan_rad: float = 0.0
    tilt_rad: float = 0.0

    def __post_init__(self) -> None:
        self.position = _validate_3d_vector(self.position, "Camera position")


class TargetMotionModel(ABC):
    """Abstract base class for deterministic target motion models."""

    @abstractmethod
    def get_initial_state(self, initial_time: float, current_state: TargetState) -> TargetState:
        """Return the initial target state consistent with the model at initial_time."""
        pass

    @abstractmethod
    def reset(self, initial_position: np.ndarray) -> None:
        """Reset motion model internal temporal/initialization state."""
        pass

    @abstractmethod
    def update(self, current_state: TargetState, t: float, dt: float) -> TargetState:
        """Calculate next target state given current time t and step dt."""
        pass


class StraightMotionModel(TargetMotionModel):
    """Straight-line uniform linear motion model: position(t) = initial_position + velocity * t."""

    def __init__(self, velocity: Union[np.ndarray, Tuple[float, float, float], list]):
        self.velocity = _validate_3d_vector(velocity, "Velocity")
        self.initial_position: np.ndarray = np.array([0.0, 0.0, 100.0], dtype=float)

    def get_initial_state(self, initial_time: float, current_state: TargetState) -> TargetState:
        self.initial_position = current_state.position.copy()
        pos = self.initial_position + self.velocity * initial_time
        return TargetState(position=pos, velocity=self.velocity.copy())

    def reset(self, initial_position: np.ndarray) -> None:
        """Reset model with explicit initial position while preserving configured velocity."""
        self.initial_position = _validate_3d_vector(initial_position, "Initial position")

    def update(self, current_state: TargetState, t: float, dt: float) -> TargetState:
        if dt == 0.0:
            return TargetState(position=current_state.position.copy(), velocity=self.velocity.copy())

        next_time = t + dt
        new_pos = self.initial_position + self.velocity * next_time
        return TargetState(position=new_pos, velocity=self.velocity.copy())


class CircularMotionModel(TargetMotionModel):
    """Deterministic circular motion trajectory around a center point in 3D world space."""

    def __init__(
        self,
        center: Union[np.ndarray, Tuple[float, float, float], list] = (0.0, 0.0, 100.0),
        radius: float = 10.0,
        angular_speed: float = 0.5,
        initial_phase: float = 0.0,
        plane: str = "XY",
    ):
        self.center = _validate_3d_vector(center, "Center")
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self.radius = float(radius)
        self.angular_speed = float(angular_speed)
        self.initial_phase = float(initial_phase)
        if plane not in ("XY", "XZ", "YZ"):
            raise ValueError(f"Plane must be 'XY', 'XZ', or 'YZ', got '{plane}'")
        self.plane = plane

    def _get_pos_vel(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        phase = self.initial_phase + self.angular_speed * t
        cos_p, sin_p = math.cos(phase), math.sin(phase)
        w = self.angular_speed
        r = self.radius

        pos = self.center.copy()
        vel = np.zeros(3, dtype=float)

        if self.plane == "XY":
            pos[0] += r * cos_p
            pos[1] += r * sin_p
            vel[0] = -r * w * sin_p
            vel[1] = r * w * cos_p
        elif self.plane == "XZ":
            pos[0] += r * cos_p
            pos[2] += r * sin_p
            vel[0] = -r * w * sin_p
            vel[2] = r * w * cos_p
        elif self.plane == "YZ":
            pos[1] += r * cos_p
            pos[2] += r * sin_p
            vel[1] = -r * w * sin_p
            vel[2] = r * w * cos_p

        return pos, vel

    def get_initial_state(self, initial_time: float, current_state: TargetState) -> TargetState:
        pos, vel = self._get_pos_vel(initial_time)
        return TargetState(position=pos, velocity=vel)

    def reset(self, initial_position: np.ndarray) -> None:
        """Validate supplied initial position and preserve configured parametric trajectory parameters."""
        _validate_3d_vector(initial_position, "Initial position")

    def update(self, current_state: TargetState, t: float, dt: float) -> TargetState:
        if dt == 0.0:
            return TargetState(position=current_state.position.copy(), velocity=current_state.velocity.copy())

        next_time = t + dt
        new_pos, new_vel = self._get_pos_vel(next_time)
        return TargetState(position=new_pos, velocity=new_vel)


class Figure8MotionModel(TargetMotionModel):
    """Deterministic Figure-8 (Lissajous) trajectory in 3D world space."""

    def __init__(
        self,
        center: Union[np.ndarray, Tuple[float, float, float], list] = (0.0, 0.0, 100.0),
        amplitude_x: float = 10.0,
        amplitude_y: float = 10.0,
        angular_speed: float = 0.5,
        initial_phase: float = 0.0,
    ):
        self.center = _validate_3d_vector(center, "Center")
        if amplitude_x <= 0 or amplitude_y <= 0:
            raise ValueError("Amplitudes must be positive")
        self.amplitude_x = float(amplitude_x)
        self.amplitude_y = float(amplitude_y)
        self.angular_speed = float(angular_speed)
        self.initial_phase = float(initial_phase)

    def _get_pos_vel(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        w = self.angular_speed
        phase = self.initial_phase + w * t
        ax, ay = self.amplitude_x, self.amplitude_y

        sin_p, cos_p = math.sin(phase), math.cos(phase)
        sin_2p, cos_2p = math.sin(2.0 * phase), math.cos(2.0 * phase)

        pos = self.center.copy()
        pos[0] += ax * sin_p
        pos[1] += (ay / 2.0) * sin_2p

        vel = np.zeros(3, dtype=float)
        vel[0] = ax * w * cos_p
        vel[1] = ay * w * cos_2p

        return pos, vel

    def get_initial_state(self, initial_time: float, current_state: TargetState) -> TargetState:
        pos, vel = self._get_pos_vel(initial_time)
        return TargetState(position=pos, velocity=vel)

    def reset(self, initial_position: np.ndarray) -> None:
        """Validate supplied initial position and preserve configured parametric trajectory parameters."""
        _validate_3d_vector(initial_position, "Initial position")

    def update(self, current_state: TargetState, t: float, dt: float) -> TargetState:
        if dt == 0.0:
            return TargetState(position=current_state.position.copy(), velocity=current_state.velocity.copy())

        next_time = t + dt
        new_pos, new_vel = self._get_pos_vel(next_time)
        return TargetState(position=new_pos, velocity=new_vel)


class BoundedRandomMotionModel(TargetMotionModel):
    """Deterministic bounded random motion model using explicit random seed or Generator."""

    def __init__(
        self,
        seed: int = 42,
        max_acceleration: float = 2.0,
        max_speed: float = 10.0,
        bounds_min: Union[np.ndarray, Tuple[float, float, float], list] = (-50.0, -50.0, 50.0),
        bounds_max: Union[np.ndarray, Tuple[float, float, float], list] = (50.0, 50.0, 150.0),
    ):
        if max_acceleration < 0:
            raise ValueError(f"max_acceleration must be non-negative, got {max_acceleration}")
        if max_speed < 0:
            raise ValueError(f"max_speed must be non-negative, got {max_speed}")

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.max_acceleration = float(max_acceleration)
        self.max_speed = float(max_speed)
        self.bounds_min = _validate_3d_vector(bounds_min, "Bounds min")
        self.bounds_max = _validate_3d_vector(bounds_max, "Bounds max")

        if np.any(self.bounds_min >= self.bounds_max):
            raise ValueError("bounds_min must be strictly less than bounds_max in all dimensions")

    def get_initial_state(self, initial_time: float, current_state: TargetState) -> TargetState:
        return TargetState(position=current_state.position.copy(), velocity=current_state.velocity.copy())

    def reset(self, initial_position: np.ndarray) -> None:
        """Reset pseudo-random number generator to original seed for reproducible trajectory sequence."""
        _validate_3d_vector(initial_position, "Initial position")
        self.rng = np.random.default_rng(self.seed)

    def update(self, current_state: TargetState, t: float, dt: float) -> TargetState:
        if dt == 0.0:
            return TargetState(position=current_state.position.copy(), velocity=current_state.velocity.copy())

        acc = self.rng.uniform(-self.max_acceleration, self.max_acceleration, size=3)
        new_vel = current_state.velocity + acc * dt

        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed and speed > 0:
            new_vel = (new_vel / speed) * self.max_speed

        new_pos = current_state.position + new_vel * dt

        for i in range(3):
            if new_pos[i] < self.bounds_min[i]:
                new_pos[i] = self.bounds_min[i]
                new_vel[i] = abs(new_vel[i])
            elif new_pos[i] > self.bounds_max[i]:
                new_pos[i] = self.bounds_max[i]
                new_vel[i] = -abs(new_vel[i])

        return TargetState(position=new_pos, velocity=new_vel)


class Scene:
    """Scene container owning camera state, target state, motion model, and simulation time."""

    def __init__(
        self,
        target_state: Union[TargetState, None] = None,
        camera_state: Union[CameraState, None] = None,
        motion_model: Union[TargetMotionModel, None] = None,
        initial_time: float = 0.0,
    ):
        self.time = float(initial_time)
        self.camera_state = camera_state if camera_state is not None else CameraState()
        init_target = target_state if target_state is not None else TargetState()

        if motion_model is None:
            self.motion_model: TargetMotionModel = StraightMotionModel(velocity=[0.0, 0.0, 0.0])
        else:
            self.motion_model = motion_model

        self.motion_model.reset(init_target.position)
        self.target_state = self.motion_model.get_initial_state(self.time, init_target)

    def step(self, dt: float) -> None:
        """Advance simulation time by dt and update target motion state."""
        if dt < 0.0:
            raise ValueError(f"Timestep dt must be non-negative, got {dt}")
        if dt == 0.0:
            return

        next_target_state = self.motion_model.update(self.target_state, self.time, dt)
        self.target_state = next_target_state
        self.time += dt