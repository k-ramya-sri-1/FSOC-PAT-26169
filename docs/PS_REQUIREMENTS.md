# FSOC-PAT-26169 — Problem Statement Requirements

## 1. Purpose

This document converts the official FSOC-PAT-26169 problem statement into an implementation and validation checklist.

The purpose is to ensure that every required capability is explicitly mapped to:

- A system component
- A configuration parameter
- A test
- A measurable metric
- A benchmark where applicable
- Evidence for the final demonstration

This document is a requirements contract.

Implementation decisions must not silently remove or weaken mandatory requirements.

---

# 2. Problem Statement Objective

The system must implement an AI-based virtual camera tracking system for coarse alignment of mobile Free Space Optical Communication (FSOC) terminals.

The software must autonomously:

1. Detect the designated optical beacon.
2. Identify the intended target.
3. Continuously track the moving target.
4. Control/reposition the virtual camera viewport to maintain alignment.
5. Operate under configurable disturbances and motion conditions.
6. Provide real-time tracking and performance information.

The system is a virtual/simulation-based coarse pointing and tracking system.

It must not claim physical hardware control or live spacecraft/aircraft operation unless such functionality is actually implemented and demonstrated.

---

# 3. Mandatory Functional Requirements

## FR-01 — Configurable Virtual Environment

The system shall provide a configurable virtual environment.

Configuration should support, where applicable:

- Scene dimensions
- Camera configuration
- Target configuration
- Target trajectory
- Environment conditions
- Noise
- Platform motion
- Camera jitter
- Simulation duration
- Random seed

---

## FR-02 — Moving Target

The system shall support one or more moving targets.

At least one designated beacon target is mandatory.

The target shall have configurable properties including:

- Initial position
- Motion model
- Size
- Brightness/intensity
- Beacon characteristics

---

## FR-03 — Movable Virtual Camera

The system shall include a movable virtual camera.

The camera must have configurable:

- Resolution
- Field of view
- Position/orientation
- Update rate
- Pan limits
- Tilt limits
- Pan rate
- Tilt rate

---

## FR-04 — Automatic Beacon Detection

The system shall automatically detect the designated beacon from the camera image.

The detector should operate on sensor/image information rather than simulation ground truth.

The detection result should provide at least:

- Detection status
- Beacon centroid
- Candidate information
- Detection confidence where applicable

---

## FR-05 — Target Identification

When multiple candidate objects exist, the system shall identify the designated target.

Identification may combine:

- Appearance
- Spatial consistency
- Motion consistency
- Modulation signature
- Model/ephemeris prior
- AI classification

The implementation must not assume that the brightest object is always the correct beacon.

---

## FR-06 — Continuous Computer-Vision Tracking

After acquisition, the system shall continuously track the target.

Tracking must maintain temporal association between observations.

The system should support:

- Prediction
- Filtering
- Candidate association
- Temporary detection loss
- Reacquisition

---

## FR-07 — Automatic Camera Repositioning

The system shall automatically reposition/control the virtual camera to maintain target alignment.

The control system shall use available estimated target information rather than ground-truth target position.

The controller must respect configured camera/gimbal limits.

---

## FR-08 — Disturbance Handling

The system shall support configurable disturbances.

Required disturbance categories include:

- Atmospheric effects
- Platform motion
- Camera motion/jitter
- Image noise

These disturbances must affect the simulated observation in a manner that can be evaluated by the tracking system.

---

## FR-09 — Real-Time Statistics

The application shall provide real-time system statistics.

At minimum, telemetry should be capable of exposing:

- Detection status
- Tracking state
- Target position
- Camera position/orientation
- Tracking error
- Confidence
- Trust where implemented
- FPS
- Processing time
- Acquisition time
- Target-loss state

---

# 4. Camera Requirements

## CAM-01 — Screen Size

The problem statement specifies a screen size of at least:

```text
2000 × 2000 pixels
The implementation must support the required virtual screen/environment size.

The camera viewport may remain smaller than the complete virtual environment.

CAM-02 — Monochrome FPA

The simulated sensor shall represent a monochrome focal-plane-array style observation.

The image-processing pipeline should therefore operate on grayscale intensity information.

CAM-03 — Default Resolution

Default camera resolution:

640 × 480 pixels

The implementation should keep resolution configurable.

CAM-04 — Default Field of View

Default camera field of view:

Horizontal FOV: 4°
Vertical FOV:   3°

FOV must remain configurable.

CAM-05 — Camera Update Rate

The virtual camera shall update at a rate of at least:

30 Hz

The benchmark should record the actual update/processing rate.

CAM-06 — Initial Camera Center

The system shall support configuration of the initial camera center/orientation.

The initial state must not silently use target ground truth.

5. Target Requirements
TGT-01 — Beacon Spot

The target shall be represented by a beacon/spot suitable for optical detection.

TGT-02 — Target Count

At least one beacon target shall be supported.

Multiple targets/candidates should be supported for identification and false-lock testing.

TGT-03 — Beacon Size

Default beacon size:

5–20 pixels

The implementation should permit configurable beacon size within the required operating range.

TGT-04 — Beacon Position

The target shall support:

User-defined initial position
Randomized initial position

Randomized scenarios must use recorded random seeds for reproducibility.

6. Mandatory Target Motion

The system shall support all mandatory target trajectories.

MOT-01 — Straight Line

The target shall move along a straight trajectory.

MOT-02 — Circular

The target shall move along a circular trajectory.

MOT-03 — Figure-8

The target shall move along a figure-8 trajectory.

MOT-04 — Random

The target shall support randomized motion.

Random motion must be reproducible when a seed is specified.

7. Optional Target Motion

The architecture may additionally support:

Spiral
Sinusoidal
User-defined trajectory

Optional motion models must not compromise the mandatory motion models.

8. Pan/Tilt Requirements
PT-01 — Maximum Pan Rate

Maximum pan rate shall be configurable within the problem-statement operating requirement.

Target range:

5–10°/s
PT-02 — Maximum Tilt Rate

Maximum tilt rate shall be configurable within the problem-statement operating requirement.

Target range:

5–10°/s
PT-03 — Control Update Interval

The pan/tilt control system shall update at a minimum frequency corresponding to:

20 Hz

or faster.

The actual update rate must be measured.

9. Performance Requirements
PERF-01 — Acquisition Time

Target acquisition shall occur within:

≤ 2 seconds

Acquisition time must be measured from the documented acquisition start condition to the documented valid-lock condition.

PERF-02 — Tracking Error

The benchmark target is:

≤ 10 pixels

The implementation must clearly define which tracking-error metric is used.

The report must distinguish:

Centroid error
Pointing error

They must not be reported as if they were the same metric.

PERF-03 — Target Loss

Target loss shall remain below:

< 5%

The calculation method must be documented.

Recommended definition:

Target Loss (%) =
    lost/evaluable frames
    --------------------- × 100
PERF-04 — Reacquisition

After temporary target loss, the system should reacquire the target within:

≤ 1 second

The benchmark must record:

Loss time
Reacquisition time
Reacquisition duration
Whether the reacquired lock is stable
PERF-05 — Processing Rate

The system shall achieve at least:

≥ 20 FPS

for the required benchmark configuration.

Processing FPS must be measured rather than assumed from the camera update rate.

10. Disturbance Requirements
DIST-01 — Salt-and-Pepper Noise

The system shall support salt-and-pepper noise.

The problem statement specifies approximately:

10%

as the required disturbance level.

The implementation must make the actual configured density visible in the benchmark configuration.

DIST-02 — Gaussian Noise

The system shall support Gaussian noise.

DIST-03 — Poisson Noise

The system shall support Poisson noise.

DIST-04 — Maximum Noise Standard Deviation

The system shall support noise up to the problem-statement limit:

20 pixels

The exact interpretation of this parameter must be documented by the implementation.

DIST-05 — Camera Jitter

The system shall support camera jitter up to:

±20 pixels/frame

The actual configured value must be recorded during benchmarks.

DIST-06 — Platform Motion

The system shall support platform motion up to:

±20 pixels/frame

where applicable to the simulated configuration.

11. Atmospheric Conditions

The system shall support configurable atmospheric/environmental conditions including:

Clear
Haze
Fog
Rain
Low light

The atmospheric model should affect the simulated observation rather than directly modifying tracker state.

12. Benchmark-1 Requirements

The system must support the primary simulation benchmark.

Benchmark-1 should cover:

Required scenario execution
Target trajectories
Beacon detection
Centroiding
Tracking
Camera alignment
Performance measurement
Disturbance handling

The benchmark output must be machine-readable and human-readable.

13. Benchmark-1 Metrics

The benchmark must report, at minimum:

Scenario
Duration
Frames
FPS
Acquisition Time
Mean Centroid Error
Maximum Centroid Error
Mean Pointing Error
Maximum Pointing Error
Lock Retention
Target Loss
Reacquisition Time
Average Processing Time
Maximum Processing Time

Where applicable, also report:

Detection Rate
False-Lock Count
False-Lock Duration
Prediction Error
AI Confidence
Trust
14. Benchmark-2 Requirements

The system shall support evaluation using a recorded MP4 video.

The benchmark video is intended to evaluate the tracking pipeline from recorded camera imagery.

The system shall:

Read the MP4 sequentially.
Process frames at their recorded resolution/frame sequence.
Detect the beacon.
Track the beacon.
Produce performance measurements.
Produce benchmark logs.

The recorded-video benchmark must not depend on live PTZ hardware.

15. MP4 Benchmark Integrity

For MP4 evaluation:

The tracker must not receive target ground truth.
Ground-truth annotations, when available, may be used only for evaluation.
The processing pipeline must use the same authoritative detection/tracking implementation as the simulator.
Frame order must be preserved.
No frames may be silently skipped without being reported.
Input video properties must be recorded.

Required video metadata:

Filename
Resolution
FPS
Frame Count
Duration
16. AI/CV Requirements

The project shall combine computer vision and AI where appropriate.

The system architecture should separate:

Computer Vision

Responsible for extracting visual evidence such as:

Candidate blobs
Centroids
Intensity
Shape
Spatial information
AI Identification

Responsible for assisting with target identification/classification when useful.

Tracking

Responsible for temporal association and state estimation.

Prediction

Responsible for estimating short-term future target position.

Control

Responsible for moving the virtual camera/gimbal.

AI must not be used as a replacement for every deterministic subsystem.

17. Ground-Truth Isolation

Ground truth may exist internally in the simulator for evaluation.

However:

GROUND TRUTH → METRICS

is permitted.

The following is prohibited:

GROUND TRUTH → DETECTOR
GROUND TRUTH → TRACKER
GROUND TRUTH → PREDICTOR
GROUND TRUTH → CONTROLLER

unless the information represents a legitimate sensor observation or configured prior that would exist in the actual system.

This separation is mandatory for credible benchmark results.

18. Simulation and Video Architecture

The simulation pipeline and MP4 pipeline should share the same core tracking engine.

Preferred architecture:

                 ┌──────────────────┐
                 │  Simulation      │
                 │  Sensor Frames   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │                  │
                 │ Common Tracking  │
                 │      Core        │
                 │                  │
                 └────────┬─────────┘
                          ▲
                          │
                 ┌────────┴─────────┐
                 │                  │
                 │ MP4 Video Frames │
                 │                  │
                 └──────────────────┘

This prevents the project from having one tracker for simulation and another tracker for recorded-video evaluation.

19. Standalone Requirement

The final project should be capable of being packaged as a standalone application.

The final deliverable should include:

Executable/application package
Source code
Required configuration
Documentation
User instructions
Performance logs

Packaging details may be finalized after the core system is stable.

20. Technical Report Requirements

The technical report should document:

Problem statement
System architecture
Coordinate system
Camera model
Sensor model
Target model
Detection algorithm
AI identification
Modulation verification
Tracking algorithm
Prediction
Confidence/trust model
Gimbal/control model
Disturbance simulation
Benchmark methodology
Results
Failure cases
Limitations
Innovation
Future improvements

The report should reflect the actual implemented system.

21. User Manual Requirements

The user manual should explain:

Installation
Dependencies
Starting the application
Configuration
Running simulation
Selecting target motion
Configuring disturbances
Running benchmarks
Running MP4 evaluation
Understanding telemetry
Interpreting performance metrics
Troubleshooting
22. Performance Log Requirements

The performance log must contain at least:

Duration
FPS
Acquisition Time
Average Tracking Error
Maximum Tracking Error
Lock Retention
Processing Time

Additional metrics should be included where useful.

23. Architecture Requirement

The final architecture should maintain clear separation between:

Scene
   ↓
Geometry
   ↓
Sensor
   ↓
Detection
   ↓
Identification
   ↓
Modulation Verification
   ↓
Tracking
   ↓
Prediction
   ↓
Confidence / Trust
   ↓
Control
   ↓
Virtual Gimbal / Camera
   ↓
New Sensor Frame

This forms the closed-loop tracking system.

24. UI Requirement

The UI is a monitoring and configuration layer.

The UI may display:

Camera view
Target position
Tracking state
FPS
Error
Confidence
Trust
Gimbal position
Scenario
Disturbance configuration
Benchmark results
Event log

The UI must not contain a second authoritative tracking algorithm.

The authoritative tracking logic belongs to the Python core.

25. Required Validation Scenarios

At minimum, final validation should include:

Scenario 1 — Straight Line
Target: Straight Line
Environment: Clear
Noise: Configurable nominal level
Camera: Default configuration
Scenario 2 — Circular
Target: Circular
Environment: Clear
Scenario 3 — Figure-8
Target: Figure-8
Environment: Clear
Scenario 4 — Random
Target: Random
Environment: Clear
Scenario 5 — Noise Stress

Test:

Gaussian
Poisson
Salt-and-pepper
Scenario 6 — Camera Jitter

Test configured camera jitter including the required operating limit.

Scenario 7 — Platform Motion

Test configured platform motion including the required operating limit.

Scenario 8 — Atmospheric Degradation

Test:

Haze
Fog
Rain
Low light
Scenario 9 — Target Loss

Temporarily remove/occlude the target and evaluate:

Loss detection
Coasting/search
Reacquisition
Re-lock stability
Scenario 10 — False-Lock Stress

Introduce bright distractors and evaluate whether the system maintains the correct target association.

Scenario 11 — MP4 Benchmark

Run the complete recorded-video benchmark and generate the required performance log.

26. Compliance Matrix

The final project should maintain a requirement-to-evidence matrix.

Recommended format:

ID	Requirement	Implementation	Test	Metric	Evidence
FR-01	Configurable environment	core/scene.py	Scenario tests	Configuration validity	Benchmark JSON
FR-02	Moving target	core/scene.py	Motion tests	Trajectory validity	Test results
FR-03	Movable camera	core/gimbal.py	Gimbal tests	Angular motion	Test results
FR-04	Beacon detection	core/detection.py	Detection tests	Detection rate	Benchmark
FR-05	Target identification	core/ai_identity.py	False-lock tests	Identification accuracy	Benchmark
FR-06	Continuous tracking	core/tracking.py	Tracking tests	Lock retention	Benchmark
FR-07	Camera repositioning	core/control.py	Closed-loop tests	Pointing error	Benchmark
FR-08	Disturbances	core/disturbances.py	Stress tests	Performance degradation	Stress report
FR-09	Real-time statistics	app/ + ui/	Integration tests	FPS/latency	Telemetry
CAM-01	Screen size	core/scene.py	Camera test	Dimensions	Test result
CAM-02	Monochrome sensor	core/sensor.py	Sensor test	Image format	Test result
CAM-03	640×480 default	core/sensor.py	Camera test	Resolution	Config
CAM-04	4°×3° default FOV	core/geometry.py	Geometry test	FOV	Test result
CAM-05	≥30 Hz camera update	core/sensor.py	Performance test	Update rate	Benchmark
TGT-01	Beacon spot	core/scene.py	Target test	Beacon validity	Test result
TGT-03	5–20 px target	core/scene.py	Target test	Size	Config
MOT-01	Straight line	core/scene.py	Scenario test	Tracking metrics	Benchmark
MOT-02	Circular	core/scene.py	Scenario test	Tracking metrics	Benchmark
MOT-03	Figure-8	core/scene.py	Scenario test	Tracking metrics	Benchmark
MOT-04	Random	core/scene.py	Scenario test	Tracking metrics	Benchmark
PT-01	Pan rate	core/gimbal.py	Gimbal test	°/s	Test result
PT-02	Tilt rate	core/gimbal.py	Gimbal test	°/s	Test result
PT-03	≥20 Hz control	core/control.py	Performance test	Control rate	Benchmark
PERF-01	≤2 s acquisition	benchmark/	Acquisition test	Seconds	Benchmark
PERF-02	≤10 px tracking error	benchmark/	Tracking test	Pixels	Benchmark
PERF-03	<5% target loss	benchmark/	Retention test	Percentage	Benchmark
PERF-04	≤1 s reacquisition	benchmark/	Reacquisition test	Seconds	Benchmark
PERF-05	≥20 FPS	benchmark/	Performance test	FPS	Benchmark
DIST-01	Salt-and-pepper	core/disturbances.py	Noise test	Noise level	Stress report
DIST-02	Gaussian	core/disturbances.py	Noise test	Noise statistics	Stress report
DIST-03	Poisson	core/disturbances.py	Noise test	Noise statistics	Stress report
DIST-05	Camera jitter	core/disturbances.py	Jitter test	px/frame	Stress report
DIST-06	Platform motion	core/disturbances.py	Motion test	px/frame	Stress report
B2	MP4 benchmark	benchmark/mp4_runner.py	MP4 test	FPS/errors	MP4 report

---

# 27. Requirement Status Rules

Each requirement must eventually have one of the following statuses:

```text
NOT_STARTED
IN_PROGRESS
IMPLEMENTED
TESTED
VERIFIED
BLOCKED

IMPLEMENTED must not be treated as equivalent to VERIFIED.

A requirement becomes VERIFIED only when:

The implementation exists.
A corresponding test exists.
The test passes.
The relevant metric/evidence is recorded.
28. Important Scope Rule

The project should prioritize satisfying the mandatory problem-statement requirements before adding optional advanced features.

Advanced features may be added only when they:

Have a clear purpose.
Do not compromise mandatory requirements.
Are testable.
Can be explained technically.
Do not introduce duplicate authoritative implementations.
Have measurable evidence where performance claims are made.
29. Final Compliance Gate

Before declaring the project complete, verify:

 All mandatory functional requirements implemented.
 All mandatory camera requirements verified.
 All mandatory target requirements verified.
 All four mandatory target motions verified.
 Pan/tilt limits verified.
 Camera/control update rates verified.
 Acquisition-time requirement tested.
 Tracking-error requirement tested.
 Target-loss requirement tested.
 Reacquisition requirement tested.
 Processing-FPS requirement tested.
 Required noise models tested.
 Camera jitter tested.
 Platform motion tested.
 Atmospheric conditions tested.
 False-lock behavior tested.
 MP4 benchmark completed.
 Performance log generated.
 Technical report completed.
 User manual completed.
 Source code documented.
 Final compliance matrix completed.
30. Requirement Interpretation Principle

The official problem statement is the source of truth for mandatory requirements.

Where an implementation decision is not explicitly specified by the problem statement, the project architecture may define a reasonable engineering solution.

Such implementation decisions must be documented rather than presented as requirements from the problem statement.

The project must clearly distinguish:

Problem Statement Requirement
        ↓
Engineering Design Decision
        ↓
Implementation
        ↓
Test
        ↓
Measured Evidence

This separation ensures that the final system can demonstrate both compliance and engineering reasoning.