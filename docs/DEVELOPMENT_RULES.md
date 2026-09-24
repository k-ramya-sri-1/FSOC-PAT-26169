# FSOC-PAT-26169 Development Rules

## 1. Purpose

This document defines the development rules for FSOC-PAT-26169.

These rules apply to:

- Human developers
- AI coding assistants
- Automated coding agents
- Contributors
- Future maintainers

The objective is to keep the project physically consistent, testable, reproducible, maintainable, and compliant with the problem statement.

---

## 2. Rule 1 — One Authoritative Implementation

There must be exactly one authoritative implementation of each major subsystem.

Never create duplicate implementations of:

- Camera geometry
- Target motion
- Sensor projection
- Beacon detection
- Tracking
- Kalman filtering
- Prediction
- Gimbal dynamics
- Control
- Metrics

If an existing subsystem already provides the required functionality, reuse it rather than creating a second implementation.

---

## 3. Rule 2 — Respect Module Responsibilities

Each module must perform only the responsibilities assigned to it in:

```text
docs/ARCHITECTURE.md
docs/INTERFACES.md
A module must not silently absorb functionality belonging to another subsystem.

For example:

geometry.py

must not contain tracking logic.

tracking.py

must not implement its own camera projection equations.

control.py

must not access ground truth.

4. Rule 3 — No Ground Truth Leakage

Ground truth must never enter the operational tracking pipeline.

Ground truth may be used only for:

Metrics
Evaluation
Benchmarking
Debug visualization
Performance reporting

Ground truth must never be supplied to:

Detector
AI classifier
Modulation verifier
Tracker
Kalman filter measurement update
Prediction system
Confidence system
Trust system
Controller
Gimbal controller

The purpose of this rule is to ensure that benchmark results represent the actual capability of the tracking system.

5. Rule 4 — Physics Before AI

The physical and mathematical foundation must be correct before adding advanced AI.

The development order should generally follow:

Correct Geometry
       ↓
Correct Scene
       ↓
Correct Sensor
       ↓
Correct Disturbances
       ↓
Correct Detection
       ↓
Correct Centroiding
       ↓
Correct Tracking
       ↓
Prediction
       ↓
State Estimation
       ↓
AI Enhancement
       ↓
Evidence / Trust Fusion
       ↓
Control Optimization

AI must not be used to hide or compensate for an incorrect mathematical model.

6. Rule 5 — Tests Are Required

Every meaningful new feature must include appropriate tests.

A feature is not considered complete merely because:

The program starts.
The UI displays something.
A demonstration appears visually correct.
An AI coding assistant reports success.

A feature is complete only after its behavior has been tested.

7. Rule 6 — Unit Tests First for Mathematical Components

Mathematical components should be tested independently before integration.

Examples include:

Coordinate transformations
Pinhole projection
Pixel-to-angle conversion
Motion equations
Noise generation
Kalman prediction
Kalman update
Prediction
Gimbal dynamics
Control calculations

Known analytical cases should be used wherever practical.

8. Rule 7 — Integration Testing

After individual modules pass their unit tests, they must be tested together.

The intended integration chain is:

Scene
  ↓
Geometry
  ↓
Sensor
  ↓
Disturbances
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
Camera

Integration testing must verify that data is transferred correctly between modules.

9. Rule 8 — Regression Protection

Previously working functionality must continue to work after changes.

Before modifying an existing subsystem:

Inspect its current implementation.
Inspect its tests.
Identify dependent modules.
Make the smallest reasonable change.
Run affected tests.
Run the broader regression suite.
Review the resulting diff.

Do not assume that a local fix cannot affect another subsystem.

10. Rule 9 — Performance Matters

Every algorithm must consider:

Processing time
Memory usage
FPS
Numerical stability
Scalability
Runtime overhead

A more complex algorithm is not automatically a better algorithm.

Any advanced technique added to the system should have a measurable reason for being included.

11. Rule 10 — Reproducibility

Simulation and benchmark runs should support deterministic execution.

Randomized components should support an explicit random seed.

Benchmark configurations should record:

Random seed
Camera configuration
Target configuration
Motion model
Disturbance configuration
Algorithm configuration
Duration
Frame rate

This allows results to be reproduced.

12. Rule 11 — Configuration Over Hard-Coding

Important system parameters should be configurable.

Examples:

Camera resolution
FOV
Target size
Target position
Motion parameters
Noise strength
Platform disturbance
Camera jitter
Atmospheric condition
Tracking thresholds
Control parameters

Default values must correspond to the project's documented requirements.

Avoid scattering configuration constants throughout unrelated modules.

13. Rule 12 — Units Must Be Explicit

Every physical quantity must have a clearly defined unit.

Recommended internal conventions:

Position       -> meters
Velocity       -> meters/second
Acceleration   -> meters/second²
Angles         -> radians
Angular rate   -> radians/second
Time           -> seconds
Pixel position -> pixels
FPS            -> frames/second

Degrees may be used for user-interface display, but internal mathematical calculations should use radians where appropriate.

A value must not change units silently between modules.

14. Rule 13 — No Unsupported Claims

Documentation, demonstrations, reports, and presentations must not claim capabilities that are not actually implemented and verified.

Do not claim:

Real satellite control
Real optical terminal control
Real hardware gimbal operation
Live satellite ephemeris
Physical laser/FSO transmission
Real-world autonomous spacecraft control

unless such functionality is actually implemented and experimentally demonstrated.

The core system is a software/virtual simulation.

15. Rule 14 — Benchmark Integrity

Benchmark results must be generated from the actual implementation.

Do not:

Modify results manually.
Hide failed scenarios.
Use ground truth to improve tracking.
Create benchmark-only tracking logic.
Report simulated values as measured values.
Claim a requirement passes without running the corresponding test.

If a requirement fails, record the failure and fix the implementation or clearly document the limitation.

16. Rule 15 — Simulation and MP4 Must Share the Core

The simulation path and MP4 path must use the same operational:

Detector
Centroid estimator
AI identity system
Modulation verifier
Tracker
Kalman filter
Predictor
Confidence system
Trust system
Controller

Only the input source should differ.

Simulation ──┐
             ├──> Common Tracking Core
MP4 Video ───┘

This prevents benchmark-specific behavior from appearing only in one mode.

17. Rule 16 — UI Is Not the Tracking Engine

The Mission Control UI is a visualization and control interface.

The UI must not create a second implementation of:

Detection
Tracking
Kalman filtering
Prediction
Gimbal dynamics
Control
Simulation physics

The Python engine is the authoritative source of runtime state.

18. Rule 17 — AI Coding Agents Must Follow the Repository Contracts

Any AI coding assistant or coding agent used on this project must first read the relevant documentation.

At minimum, it should inspect:

README.md
docs/ARCHITECTURE.md
docs/INTERFACES.md
docs/DEVELOPMENT_RULES.md

For geometry-related work, it must also read:

docs/COORDINATE_CONVENTION.md

For testing-related work, it must also read:

docs/TEST_STRATEGY.md

For problem-statement compliance work, it must also read:

docs/PS_REQUIREMENTS.md
19. Rule 18 — AI Coding Agents Must Stay Within Scope

When an AI coding agent is asked to implement a specific file or subsystem, it must:

Inspect the relevant existing files.
Read the relevant documentation.
Understand existing interfaces.
Modify only the requested files unless another change is explicitly necessary.
Preserve existing public interfaces.
Add or update tests.
Avoid unnecessary dependencies.
Avoid unrelated refactoring.
Report all changed files.
Report tests executed.
Report test results.
Explain any assumptions.

The agent must not silently rewrite unrelated parts of the repository.

20. Rule 19 — AI-Generated Code Must Be Reviewed

AI-generated code is not automatically considered correct.

Before committing AI-generated changes:

AI generates code
       ↓
Inspect files
       ↓
Run tests
       ↓
Review git diff
       ↓
Check architecture
       ↓
Check interfaces
       ↓
Check performance
       ↓
Commit

The developer remains responsible for the final code.

21. Rule 20 — Small Commits

Each meaningful subsystem should have its own commit whenever practical.

Examples:

Initialize project foundation

Add project architecture contracts

Implement camera geometry

Add geometry unit tests

Implement target motion models

Add sensor model

Implement disturbance models

Implement beacon detection

Small commits make debugging and rollback easier.

22. Rule 21 — Never Commit Untested Major Changes

Before committing a major subsystem:

Run its unit tests.
Run relevant integration tests.
Inspect the diff.
Verify no unrelated files changed.
Confirm the documented interface is preserved.
23. Rule 22 — Dependency Discipline

Do not add a library simply because it is convenient.

Before adding a dependency:

Confirm it is actually required.
Check whether the standard library or existing dependency can perform the task.
Consider runtime performance.
Consider installation complexity.
Update requirements.txt.
Test installation in a clean environment where practical.
24. Rule 23 — Error Handling

Failures must be explicit.

Do not silently convert invalid states into plausible values.

Examples:

Invalid camera configuration
        ↓
Clear configuration error

Target behind camera
        ↓
Not visible

No detection
        ↓
No measurement

Temporary target loss
        ↓
COASTING / REACQUIRING

Permanent target loss
        ↓
LOST

A missing measurement must never be replaced with ground truth.

25. Rule 24 — Logging and Metrics

Important runtime events should be observable.

Examples:

Acquisition
Lock
Target loss
Reacquisition
False candidate
Confidence change
Gimbal saturation
Processing slowdown
Benchmark completion

Performance metrics must be generated from actual measurements.

26. Rule 25 — Avoid Premature Optimization

The development order should prioritize correctness first.

Use:

Correctness
   ↓
Testing
   ↓
Measurement
   ↓
Optimization

Do not optimize an algorithm before measuring its actual performance.

27. Rule 26 — Advanced Features Require Evidence

Advanced techniques such as:

CNN-based identification
Adaptive prediction
Multi-evidence fusion
Advanced filtering
Model-based trust
Latency compensation

should only be added when their purpose is clearly defined.

Each advanced feature should answer:

What problem does it solve?
Why is it needed?
What is the computational cost?
How will its effect be measured?
What happens if it fails?
28. Rule 27 — No Feature Bloat

The goal is not to include the largest possible number of algorithms.

A feature should be included because it:

Satisfies a requirement.
Improves robustness.
Improves measurable performance.
Improves explainability.
Improves benchmark coverage.
Provides meaningful innovation.

Every added feature increases system complexity and testing requirements.

29. Rule 28 — Documentation Must Match Implementation

Whenever behavior changes significantly:

Update documentation.
Update tests.
Update configuration documentation.
Update benchmark expectations where necessary.

Never allow documentation to claim behavior that the implementation does not provide.

30. Rule 29 — Final Verification

Before declaring the system complete, verify:

Unit Tests
     +
Integration Tests
     +
Regression Tests
     +
PS Compliance Tests
     +
MP4 Benchmark
     +
Stress Tests
     +
Performance Logs
     +
Documentation

All final results must be based on actual executed tests and recorded measurements.

31. Core Development Principle

The project should be developed according to:

Understand
   ↓
Design
   ↓
Implement
   ↓
Test
   ↓
Measure
   ↓
Review
   ↓
Commit
   ↓
Integrate

Never reverse this process by adding complex code first and attempting to understand or test it afterward.