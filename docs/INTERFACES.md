# FSOC-PAT-26169 Module Interfaces

## 1. Purpose

This document defines the conceptual interfaces between the major components of FSOC-PAT-26169.

It establishes what information each subsystem provides, what information it may consume, and which responsibilities belong to each module.

Implementations may use Python classes, dataclasses, typed dictionaries, protocols, or other suitable structures, but the responsibilities and information flow defined here must be preserved.

---

## 2. General Interface Rules

Every module must have a clear responsibility.

Modules must:

- Receive only the information required for their responsibility.
- Avoid accessing unrelated internal state.
- Avoid duplicating functionality owned by another module.
- Use typed and documented interfaces wherever practical.
- Avoid hidden global state.
- Avoid circular dependencies.
- Preserve deterministic behavior when given deterministic inputs.

A module must not obtain information through an undocumented side channel simply to make an algorithm perform better.

---

## 3. Scene → Geometry

### Provider

`core/scene.py`

### Consumer

`core/geometry.py`

### Scene provides

- Target world position
- Target velocity
- Target motion state
- Target identity/configuration
- Camera world position
- Camera orientation
- Relevant scene configuration

### Geometry provides

- Relative target position
- Camera-frame target position
- Line-of-sight direction
- Target range/depth
- Projected pixel coordinates
- Visibility status
- FOV status
- Angular position

Geometry must not modify the target state.

---

## 4. Geometry → Sensor

### Provider

`core/geometry.py`

### Consumer

`core/sensor.py`

### Geometry provides

- Projected target position
- Visibility information
- Camera-frame position
- Line-of-sight direction
- Range/depth
- FPA location when visible

### Sensor uses this information to

- Generate the synthetic FPA image.
- Render the beacon.
- Apply configured sensor characteristics.

The sensor must not use target ground truth to perform detection.

The sensor may use scene truth to generate the synthetic image because the sensor is the simulation source.

---

## 5. Sensor → Disturbances

### Provider

`core/sensor.py`

### Consumer

`core/disturbances.py` and `core/atmosphere.py`

The sensor produces a clean synthetic image.

The disturbance/environment layer transforms that image according to configured conditions.

Possible inputs include:

- Clean FPA frame
- Noise configuration
- Camera jitter configuration
- Platform disturbance configuration
- Atmospheric condition
- Random seed

The resulting disturbed frame is the only image provided to the operational detection pipeline.

---

## 6. Disturbances → Detection

### Provider

`core/disturbances.py` and `core/atmosphere.py`

### Consumer

`core/detection.py`

Detection receives:

- Image/frame
- Image dimensions
- Timestamp
- Sensor configuration
- Detection configuration

Detection must not receive:

- True target pixel position
- True target velocity
- True target identity
- True target state
- Ground-truth trajectory

This is essential to prevent ground-truth leakage.

---

## 7. Detection → AI Identity

### Provider

`core/detection.py`

### Consumer

`core/ai_identity.py`

Detection provides candidate measurements such as:

- Candidate centroid
- Candidate bounding region
- Area
- Width
- Height
- Shape/circularity
- Intensity statistics
- Signal-to-noise ratio
- Appearance features
- Candidate timestamp

AI identity returns information such as:

- Classification
- Identity confidence
- Candidate confidence
- Optional class probabilities
- Optional rejection/ambiguity status

The AI module must not receive ground-truth target information.

---

## 8. Detection → Modulation Verification

### Provider

`core/detection.py`

### Consumer

`core/modulation.py`

Detection provides temporally associated candidate observations.

Possible information includes:

- Candidate identifier
- Candidate centroid
- Candidate intensity
- Candidate signal strength
- Timestamp
- Frame index

The modulation subsystem evaluates temporal behavior.

The implementation should support verification of the required 15 Hz beacon signature.

---

## 9. Detection + AI + Modulation → Tracking

### Providers

- `core/detection.py`
- `core/ai_identity.py`
- `core/modulation.py`

### Consumer

`core/tracking.py`

Tracking receives candidate evidence from the preceding stages.

Candidate evidence may include:

```text
Candidate
├── centroid
├── shape
├── intensity
├── SNR
├── detection confidence
├── AI confidence
├── modulation confidence
└── timestamp
Tracking determines:

Candidate association
Track identity
Tracking state
Lock status
Track confidence
Measurement acceptance/rejection
Target-loss status
Reacquisition status

Tracking must not use ground truth.

10. Tracking → Kalman Filter
Provider

core/tracking.py

Consumer

core/kalman.py

Tracking provides accepted measurements and timing information.

Possible measurement information:

Pixel position
Angular position
Measurement timestamp
Measurement covariance or measurement-quality estimate

The Kalman filter provides:

Estimated state
Estimated velocity
State covariance
Predicted state
Uncertainty information
Innovation/residual information
Measurement acceptance/rejection information

The exact state vector must be documented by the implementation.

11. Kalman → Prediction
Provider

core/kalman.py

Consumer

core/prediction.py

Prediction receives:

Current estimated state
Estimated velocity
State covariance
Timestamp
Requested prediction horizon

Prediction produces:

Predicted target state
Predicted target position
Predicted target velocity
Prediction uncertainty
Prediction timestamp

Prediction must account for processing/control latency when configured.

12. Tracking + Prediction → Confidence and Trust
Providers
core/tracking.py
core/prediction.py
core/ai_identity.py
core/modulation.py
Consumers
core/confidence.py
core/trust.py

Evidence may include:

Detection confidence
AI confidence
Modulation confidence
Temporal consistency
Motion consistency
Prediction residual
Kalman uncertainty
SNR
Candidate persistence
Track age
Recent measurement quality

Confidence and trust calculations must remain interpretable.

They must not silently use ground truth.

13. Trust → Tracking
Provider

core/trust.py

Consumer

core/tracking.py

Trust information may influence:

Measurement acceptance
Track confidence
State transitions
Candidate selection
Reacquisition decisions

Trust must not bypass the fundamental tracking state machine.

A high-confidence but physically inconsistent candidate must not automatically become a valid lock.

14. Tracking + Prediction → Control
Providers
core/tracking.py
core/prediction.py
core/geometry.py
Consumer

core/control.py

Control receives:

Current estimated target direction
Predicted target direction
Pointing error
Camera orientation
Gimbal state
Target motion estimate
Prediction horizon
Timing information
Control configuration

Control produces:

Desired pan command
Desired tilt command
Desired angular velocity
Optional feed-forward component
Control diagnostics

Control must operate on estimated/predicted state, not ground truth.

15. Control → Gimbal
Provider

core/control.py

Consumer

core/gimbal.py

Control provides requested camera/gimbal motion.

The gimbal applies:

Pan limits
Tilt limits
Maximum angular velocity
Optional acceleration limits
Gimbal dynamics
Command latency
Platform disturbance

The gimbal returns:

Actual pan
Actual tilt
Actual angular velocity
Applied command
Saturation status
Gimbal state

The gimbal must not instantly teleport to the commanded orientation if physical dynamics are configured.

16. Gimbal → Geometry
Provider

core/gimbal.py

Consumer

core/geometry.py

The realized gimbal orientation becomes the camera orientation used for the next frame.

This creates the closed-loop relationship:

Target
   ↓
Geometry
   ↓
Sensor
   ↓
Detection
   ↓
Tracking
   ↓
Prediction
   ↓
Control
   ↓
Gimbal
   ↓
Camera Orientation
   └──────────────→ Geometry

The gimbal must provide its realized state rather than only the commanded state.

17. Scene → Simulator
Provider

core/scene.py

Consumer

core/simulator.py

The simulator coordinates:

Scene configuration
Target motion
Camera state
Sensor generation
Disturbances
Detection
Tracking
Prediction
Control
Gimbal dynamics
Metrics

The simulator is an orchestrator.

It must not duplicate the algorithms implemented by individual modules.

18. Simulation → Metrics

The simulation may provide both:

Operational estimates
Estimated target position
Estimated velocity
Estimated centroid
Tracking state
Camera state
Gimbal state
Confidence
Processing timing
Evaluation-only ground truth
True target position
True target velocity
True target pixel position
True target identity
True camera state

Ground truth may be used by metrics to calculate errors.

Ground truth must never be passed backward into the operational pipeline.

19. MP4 Input → Common Core

The MP4 benchmark must feed frames into the same operational detection/tracking pipeline used by the simulation.

The benchmark must not create:

A second detector
A second tracker
A second prediction algorithm
A second controller

The intended architecture is:

Simulation Source ──┐
                    |
                    v
               Common Core
                    ^
                    |
MP4 Video Source ───┘

Only the input source changes.

20. Metrics Interfaces

Metrics may consume:

Estimated state
Ground-truth state
Camera state
Gimbal state
Timestamps
Processing durations
Tracking state history
Detection history
Prediction history

Metrics must calculate and report measurable quantities such as:

Acquisition time
Centroid error
Pointing error
Average tracking error
Maximum tracking error
Lock retention
Target loss
Reacquisition time
Processing FPS
Processing time
False-lock events

Metrics must not modify the operational tracking state.

21. UI Telemetry Interface

The Python engine is the authoritative source of runtime state.

The UI receives telemetry such as:

Timestamp
Frame index
FPS
Processing time
Target state
Tracking state
Estimated centroid
Detection confidence
AI confidence
Modulation confidence
Overall tracking confidence
Pointing error
Pan angle
Tilt angle
Gimbal velocity
Prediction position
Disturbance state
Acquisition time
Lock retention
Target-loss status
Reacquisition status

The UI may visualize this information.

The UI must not independently calculate or replace the authoritative tracking state.

22. UI → Application

The UI may send commands such as:

Start simulation
Stop simulation
Pause simulation
Reset simulation
Select motion model
Configure target
Configure camera
Configure disturbances
Start benchmark
Select MP4 input
Request performance report

The application layer validates commands before passing them to the appropriate subsystem.

23. Application Layer

app/ is responsible for:

Starting the system
Managing runtime configuration
Connecting the engine to the UI
Providing telemetry
Handling user commands
Starting benchmarks
Managing application lifecycle

The application layer must not contain duplicate tracking mathematics.

24. Configuration Interface

Configuration should be centralized rather than scattered across modules.

Configuration may include:

Camera
Width
Height
Horizontal FOV
Vertical FOV
Initial position
Initial pan
Initial tilt
Update rate
Target
Beacon size
Initial position
Initial velocity
Motion model
Motion parameters
Disturbances
Noise type
Noise strength
Jitter
Platform motion
Atmospheric condition
Disturbance severity
Tracking
Acquisition thresholds
Lock thresholds
Loss thresholds
Reacquisition thresholds
Prediction horizon
Control
Pan limits
Tilt limits
Maximum angular velocity
Control gains
Latency parameters
25. Error Handling

Interfaces must handle invalid or unavailable information explicitly.

Examples:

No detection
    ↓
No valid measurement

Invalid geometry
    ↓
Visibility/projection failure

Low-confidence candidate
    ↓
Candidate may be rejected

Temporary target loss
    ↓
COASTING / REACQUIRING

Permanent target loss
    ↓
LOST

A missing measurement must not be replaced with ground truth.

26. Timing

Every measurement and state estimate should have an associated timestamp.

The system must support:

Variable processing time
Variable frame intervals
Camera update timing
Control timing
Measurement latency
Prediction horizon

Algorithms that require dt must calculate it from timestamps rather than assuming a fixed timestep unless the interface explicitly guarantees one.

27. Interface Stability

Once a module interface is implemented and used by other modules:

Do not silently change field names.
Do not silently change units.
Do not silently change coordinate conventions.
Do not silently change return semantics.

Interface changes must update:

Implementation
Tests
Documentation
Dependent modules
28. Units

All interfaces must explicitly document units.

Recommended conventions:

Position      -> meters
Velocity      -> meters/second
Angles        -> radians internally
Display angle -> degrees where appropriate
Pixel         -> pixels
Time          -> seconds
FPS           -> frames/second
Angular rate  -> radians/second internally

Conversions must happen at well-defined boundaries.

29. Ground Truth Isolation

The following modules must never receive ground truth:

core/detection.py
core/ai_identity.py
core/modulation.py
core/tracking.py
core/kalman.py
core/prediction.py
core/trust.py
core/confidence.py
core/control.py
core/gimbal.py

Ground truth is allowed only for:

Simulation generation
Metrics
Evaluation
Debug visualization
Benchmark reporting
30. Interface Design Principle

The final system should behave as one coherent closed-loop system:

WORLD
  ↓
TARGET
  ↓
CAMERA GEOMETRY
  ↓
SENSOR
  ↓
DISTURBANCES
  ↓
DETECTION
  ↓
AI / MODULATION
  ↓
TRACKING
  ↓
STATE ESTIMATION
  ↓
PREDICTION
  ↓
TRUST / CONFIDENCE
  ↓
CONTROL
  ↓
GIMBAL
  ↓
CAMERA
  ↓
WORLD

Every module must have a clear place in this loop.

No module may bypass the architecture simply to obtain better benchmark results.