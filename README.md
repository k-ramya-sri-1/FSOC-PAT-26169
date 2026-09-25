# FSOC-PAT-26169

## AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals

An AI-assisted virtual camera tracking system for autonomous coarse alignment of mobile Free Space Optical Communication (FSOC) terminals.

The system is designed as a software simulation and benchmarking platform for detecting, identifying, continuously tracking, and maintaining lock on a moving optical beacon under realistic sensor, platform, atmospheric, and camera disturbances.

---

## Project Objective

The system autonomously:

1. Simulates a configurable virtual FSOC environment.
2. Generates a moving beacon target.
3. Simulates a virtual monochrome focal plane array (FPA).
4. Detects the beacon using computer vision.
5. Estimates the beacon centroid with subpixel accuracy.
6. Identifies and verifies the designated beacon.
7. Estimates target motion.
8. Predicts future target position.
9. Controls a virtual pan-tilt camera.
10. Maintains continuous target lock.
11. Handles target loss and reacquisition.
12. Measures and reports tracking performance.

---

## Target Motion Models

Mandatory motion models:

- Straight Line
- Circular
- Figure-8
- Random

Additional models planned:

- Spiral
- Sinusoidal
- User-defined trajectories

---

## Sensor and Disturbance Models

The simulation will support:

### Sensor Noise

- Gaussian noise
- Poisson noise
- Salt-and-pepper noise

### Camera Disturbances

- Camera jitter
- Platform motion
- Sensor noise
- Processing latency

### Atmospheric Conditions

- Clear
- Haze
- Fog
- Rain
- Low-light conditions

---

## Tracking Pipeline

```text
Virtual Scene
      |
      v
3D Target & Motion
      |
      v
Camera Geometry
      |
      v
Virtual Sensor
      |
      v
Disturbance Models
      |
      v
Beacon Detection
      |
      v
Subpixel Centroid
      |
      v
AI Identification
      |
      v
Temporal / Modulation Verification
      |
      v
Tracking + Kalman Estimation
      |
      v
Target Prediction
      |
      v
Evidence & Trust Fusion
      |
      v
Predictive Gimbal Controller
      |
      v
Virtual Camera

## Closed-Loop Simulator

The Stage 1N integration is exposed through `core.simulator.ClosedLoopSimulator`.
Create a simulator with `SimulatorConfig`, call `step()` for timestamped telemetry,
and call `reset()` to reproduce the configured run. Ground truth is retained in
telemetry for evaluation; detection and control operate on simulated observations.