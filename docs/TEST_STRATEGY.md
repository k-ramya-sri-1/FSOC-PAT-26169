# FSOC-PAT-26169 — Test Strategy

## 1. Purpose

This document defines the testing strategy for the FSOC-PAT-26169 system.

The purpose of testing is to verify that the system:

- Detects the designated optical beacon.
- Correctly identifies the intended target.
- Continuously tracks the target.
- Estimates target motion.
- Predicts target position when required.
- Controls the virtual camera/gimbal to maintain alignment.
- Handles disturbances and temporary target loss.
- Meets the functional and quantitative requirements of the problem statement.
- Produces reproducible and auditable performance results.

Testing must validate both individual components and the complete closed-loop system.

---

# 2. Testing Principles

The following principles are mandatory.

### 2.1 Deterministic Testing

Tests should use fixed random seeds whenever randomness is involved.

A test must be reproducible when executed with the same:

- Configuration
- Input
- Random seed
- Software version

Example:

```python
rng = np.random.default_rng(42)
Tests must not depend on uncontrolled random behavior.

2.2 No Ground-Truth Leakage

Ground-truth target information may be used for:

Generating simulation scenarios
Generating benchmark annotations
Calculating evaluation metrics
Validating the tracker

Ground truth must NOT be provided to the detection, identification, tracking, prediction, confidence, or control algorithms as an input during normal operation.

The tracking pipeline must operate only on information that would realistically be available from the camera/sensor and configured system priors.

2.3 Test the Actual Production Pipeline

Tests must exercise the same implementation used by the final application.

A separate simplified implementation must not be created only for passing tests.

Simulation and MP4/video evaluation should use the same core detection and tracking modules wherever technically possible.

2.4 Test Individual Components Before Integration

Each major component must have unit tests before relying on it in the complete system.

Required areas include:

Geometry
Camera projection
Sensor/noise generation
Beacon detection
Beacon identification
Modulation verification
Kalman filtering
Target association
Tracking state machine
Prediction
Confidence estimation
Trust estimation
Gimbal dynamics
Control
Metrics
3. Test Levels

Testing is divided into four levels.

3.1 Unit Tests

Unit tests verify one function or class in isolation.

Examples:

Pixel-to-angle conversion
Angle-to-pixel projection
Bounding-box calculation
Centroid calculation
Noise generation
Kalman prediction
Kalman update
Gimbal rate limiting
Confidence calculation

Unit tests should be fast and deterministic.

3.2 Integration Tests

Integration tests verify that multiple modules work together correctly.

Examples:

Sensor → Detector
Detector → Identifier
Detector → Tracker
Tracker → Predictor
Predictor → Controller
Controller → Gimbal
Complete detection → tracking → control loop

Integration tests should verify both data compatibility and physical/algorithmic behavior.

3.3 Scenario Tests

Scenario tests execute the system under predefined motion and disturbance conditions.

Mandatory target trajectories:

Straight Line
Circular
Figure-8
Random

Additional trajectories may include:

Spiral
Sinusoidal
User-defined trajectory

Scenario tests must record:

Acquisition time
Tracking error
Lock state
Target loss
Reacquisition time
Processing FPS
Detector latency
End-to-end latency
Camera/gimbal state
Confidence
Trust
Prediction error
3.4 Requirement/Benchmark Tests

Requirement tests directly map implementation behavior to the official problem-statement requirements.

Examples:

Camera resolution requirement
Camera update-rate requirement
Target size requirement
Motion-model requirement
Pan/tilt rate requirement
Acquisition-time requirement
Tracking-error requirement
Target-loss requirement
Reacquisition requirement
Processing-FPS requirement
Disturbance requirements

These tests should produce machine-readable results that can later be converted into a compliance report.

4. Test Categories
4.1 Geometry Tests

Verify coordinate transformations.

Required checks:

Camera coordinates are internally consistent.
Pixel center corresponds to zero angular error.
Positive horizontal angular error maps to the documented image direction.
Positive vertical angular error maps to the documented image direction.
Projection and inverse projection are consistent within numerical tolerance.
FOV configuration is respected.
Camera orientation affects projection correctly.

Example property:

angle → pixel → angle

should approximately reproduce the original angle within numerical tolerance.

5. Camera and Sensor Tests

Verify:

Configurable resolution.
Default resolution of 640×480.
Configurable FOV.
Default FOV of 4°×3°.
Camera update rate.
Monochrome image generation.
Beacon rendering.
Background generation.
Sensor noise.

Sensor tests must separately verify:

Gaussian noise
Poisson noise
Salt-and-pepper noise

Noise parameters must be configurable.

6. Detection Tests

Detection tests verify that the detector can locate the beacon under different conditions.

Test conditions should include:

Clean image
Gaussian noise
Poisson noise
Salt-and-pepper noise
Low light
Haze
Fog
Rain
Platform motion
Camera jitter
Multiple bright objects
Partial occlusion
Temporary target disappearance

The detector should return structured information such as:

DetectionResult
    detected
    centroid
    bounding_box
    area
    intensity
    confidence

Detection failure must be represented explicitly.

A missing detection must not be interpreted as a valid coordinate.

7. Beacon Identification Tests

If multiple bright candidates exist, the identification stage must distinguish the designated beacon from unrelated bright objects.

Tests should include:

Correct beacon
Bright distractor
Multiple distractors
Similar-size distractor
Moving distractor
Temporary false candidate
Beacon plus noise blobs

Identification should use the configured evidence sources.

Possible evidence includes:

Spatial consistency
Appearance
Motion consistency
Modulation signature
Prior information
AI classifier confidence

No single weak feature should automatically force a lock.

8. Modulation Verification Tests

The mandatory beacon modulation signature is 15 Hz.

The modulation module must be tested using:

Correct 15 Hz signal
Slightly noisy 15 Hz signal
Incorrect frequency
Missing modulation
Intermittent modulation
Multiple frequencies
Insufficient observation duration

The result must include whether the expected modulation signature is verified.

9. Tracking Tests

Tracking tests verify continuous target association.

The tracker must be tested with:

Smooth motion
Rapid motion
Direction changes
Temporary detection loss
False detections
Multiple candidates
No detections
Reappearing target
Distractor crossing target path

The tracker state machine must correctly handle states such as:

SEARCHING
ACQUIRING
TRACKING
COASTING
REACQUIRING
LOST

The exact states used by the implementation must remain consistent with the project architecture.

10. Kalman Filter Tests

The Kalman filter must be tested independently.

Required checks:

Prediction without measurement.
Update with valid measurement.
Response to noisy measurements.
Handling of missing measurements.
Covariance growth during prediction-only periods.
Covariance reduction after valid measurements.
Stable behavior under constant velocity.
Stable behavior during acceleration.

The filter must not receive simulation ground truth.

11. Prediction Tests

Prediction should be evaluated independently from tracking.

Required tests include:

Constant velocity.
Constant acceleration where supported.
Direction changes.
Temporary measurement loss.
Short-term prediction.
Longer prediction horizon.

Metrics should include:

Prediction Error
Prediction Horizon
Mean Prediction Error
Maximum Prediction Error

Prediction must not be presented as exact knowledge of future target position.

12. Trust and Confidence Tests

Confidence and trust are different concepts and must be tested separately.

Confidence

Confidence describes how strongly the current observation supports the target hypothesis.

Possible evidence:

Detection quality
AI classification probability
Modulation verification
Spatial consistency
Temporal consistency
Trust

Trust describes how much the system should rely on a particular information source or model at the current time.

Trust may change based on:

Detection quality
Prediction consistency
Model residual
Repeated false detections
Track stability
Environmental conditions

Tests must verify that confidence and trust respond appropriately to degraded evidence.

13. Gimbal and Control Tests

The controller must respect configured physical limits.

Required checks:

Maximum pan rate.
Maximum tilt rate.
Position limits where configured.
Control update interval.
Smooth response.
No impossible instantaneous movement.
Correct response direction.
Correct handling of prediction.
Correct behavior when target is lost.

The controller must not directly use simulation ground truth.

It may use:

Current detected position
Filtered state
Predicted position
Confidence/trust
Camera geometry
Configured limits
14. Disturbance Tests

The system must be tested under configurable disturbances.

14.1 Image Noise

Required:

Gaussian noise
Poisson noise
Salt-and-pepper noise

The configured maximum noise parameters must be validated.

14.2 Camera Jitter

Test random camera movement within the configured limit.

Default maximum:

±20 pixels/frame

The exact configured value must be recorded in the benchmark output.

14.3 Platform Motion

Test target/camera relative motion caused by platform movement.

The configured maximum should remain within the problem-statement limit.

14.4 Atmospheric Conditions

Test at least:

Clear
Haze
Fog
Rain
Low light

Atmospheric effects should be implemented as configurable sensor/scene effects rather than hidden modifications to tracker state.

15. Target Motion Tests

Each mandatory trajectory must be tested independently.

15.1 Straight Line

Verify:

Acquisition
Continuous tracking
Prediction
Camera alignment
Error limits
15.2 Circular

Verify:

Continuous angular motion
Prediction during curvature
Tracking through changing direction
15.3 Figure-8

Verify:

Direction changes
Crossing region behavior
Candidate association
Prediction stability
15.4 Random

Verify:

Robustness to unpredictable motion
Track retention
Reacquisition
Control stability
16. Target Loss and Reacquisition Tests

Target loss must be explicitly tested.

Test sequence:

Target visible
        ↓
Tracking
        ↓
Target disappears
        ↓
COASTING / SEARCH
        ↓
Target returns
        ↓
REACQUISITION
        ↓
TRACKING

Required measurements:

Time until target loss is declared.
Duration of coast/search.
Reacquisition time.
Whether false lock occurs.
Final tracking stability.

The requirement target is:

Reacquisition ≤ 1 second

where applicable to the benchmark configuration.

17. False-Lock Tests

False-lock testing is mandatory because a tracking system must not simply follow the brightest object.

Test cases should include:

Bright stationary distractor.
Bright moving distractor.
Distractor crossing beacon.
Multiple equal-intensity blobs.
Beacon temporarily hidden.
Beacon modulation absent.
Distractor with similar size.
Distractor entering predicted region.

Record:

False-lock occurrence.
False-lock duration.
Confidence.
Trust.
Candidate scores.
Rejection reason where available.
18. Performance Tests

Performance testing must measure actual processing time.

Required measurements:

Total Frames
Processed Frames
Processing FPS
Average Frame Time
Maximum Frame Time
Detector Time
Tracker Time
Prediction Time
Control Time
End-to-End Time

The processing pipeline should achieve at least:

20 FPS

for the required benchmark configuration.

The camera update rate requirement and processing FPS requirement must be reported separately.

19. Acquisition Tests

Acquisition time begins when the system starts searching for the designated target and ends when the system establishes a valid tracking lock according to the documented lock criteria.

Target requirement:

Acquisition ≤ 2 seconds

Tests must record:

Start Time
First Valid Detection
Lock Time
Acquisition Time

A detection alone must not automatically be counted as successful acquisition if the architecture requires additional confirmation.

20. Tracking Error Tests

Tracking error must be clearly defined.

The project must distinguish between:

Centroid Error

Difference between detected beacon centroid and ground-truth beacon centroid.

centroid_error_px =
    distance(detected_centroid, ground_truth_centroid)
Pointing Error

Difference between target position and camera optical-axis center.

pointing_error_px =
    distance(target_centroid, image_center)

These metrics must never be mixed.

Reports must clearly label which error is being measured.

The benchmark requirement is:

Tracking error ≤ 10 pixels

The exact definition used for compliance must be stated in the benchmark report.

21. Target Loss / Lock Retention Tests

Lock retention must be measured over the complete test duration.

Example:

Lock Retention (%) =
    locked_frames / evaluable_frames × 100

Target-loss percentage:

Target Loss (%) =
    lost_frames / evaluable_frames × 100

The benchmark requirement is:

Target loss < 5%

The report must specify:

Total frames
Locked frames
Lost frames
Frames excluded from evaluation, if any
Reason for exclusions
22. MP4 Benchmark Tests

The system must support evaluation of recorded MP4 video.

The MP4 pipeline must:

Read the video.
Preserve original frame order.
Process frames sequentially.
Run the same detection/tracking core used by the simulator.
Produce per-frame metrics.
Produce aggregate metrics.
Avoid PTZ camera control in the recorded-video benchmark.

The benchmark must not use video ground truth as an input to the tracker.

If ground-truth annotations exist, they are used only for evaluation.

23. Test Data Integrity

Every benchmark should record:

Scenario Name
Configuration
Random Seed
Video Filename
Video Resolution
Video FPS
Frame Count
Software Version
Git Commit
Timestamp

Where practical, benchmark results should be reproducible from these values.

24. Regression Tests

Every bug discovered during development should result in a regression test when practical.

Example:

If a previous implementation failed to transition from TRACKING to COASTING during target occlusion, a regression test must reproduce that condition and verify the expected state transition.

Regression tests prevent previously solved failures from returning.

25. Test Naming Convention

Use descriptive names.

Examples:

test_pixel_angle_round_trip()
test_camera_projection_center()
test_gaussian_noise_statistics()
test_detector_finds_clean_beacon()
test_detector_handles_salt_pepper_noise()
test_tracker_enters_coasting_after_loss()
test_tracker_reacquires_target()
test_gimbal_respects_pan_rate_limit()
test_straight_line_tracking()
test_circular_tracking()
test_figure8_tracking()
test_random_motion_tracking()

Test names should describe behavior rather than implementation details.

26. Test Tolerances

Numerical tests must define explicit tolerances.

Avoid:

assert result == expected

for floating-point calculations unless exact equality is guaranteed.

Prefer:

assert abs(result - expected) < tolerance

or:

np.testing.assert_allclose(actual, expected, atol=tolerance)

Tolerances must be justified by the expected numerical precision and system behavior.

27. Automated Test Execution

The standard test command should be:

pytest

Coverage may be generated using:

pytest --cov=core --cov=metrics --cov=benchmark

The project should maintain automated tests for all critical components.

28. Benchmark Result Format

Benchmark results should be machine-readable.

Preferred formats:

JSON
CSV

A benchmark result should include at least:

scenario
seed
duration
frames
fps
acquisition_time
mean_centroid_error
max_centroid_error
mean_pointing_error
max_pointing_error
lock_retention
target_loss
reacquisition_time
average_processing_time
max_processing_time
pass/fail status

Additional metrics may be included where useful.

29. Requirement Compliance

The benchmark system must not silently mark a requirement as PASS.

Each requirement should have:

Requirement
Measured Value
Threshold
Units
Pass/Fail
Evidence

Example:

Requirement: Processing FPS
Measured: 31.7
Threshold: 20
Units: FPS
Status: PASS
Evidence: benchmark/results/run_001.json

The compliance engine must calculate the status from measured values.

30. Failure Envelope Testing

The system should be tested beyond nominal conditions to identify where performance degrades.

Parameters may include:

Noise level
Camera jitter
Platform motion
Target speed
Target acceleration
Target size
Target brightness
Atmospheric degradation
Occlusion duration
Distractor count

The objective is not to hide failures.

The objective is to determine:

Where does the system work?
Where does performance degrade?
Where does tracking fail?
Why does it fail?

Results should be visualized as a failure envelope or operating envelope.

31. Ablation Testing

Advanced components should be evaluated through ablation experiments.

Examples:

Baseline detector
Baseline + AI identification
Baseline + modulation verification
Baseline + Kalman filter
Baseline + prediction
Baseline + trust
Baseline + all components

Ablation testing helps determine whether each advanced component provides measurable value.

No advanced feature should be claimed as beneficial without supporting measurements.

32. AI-Specific Testing

If AI/ML is used, tests must verify:

Dataset/input consistency.
Training/inference separation.
No ground-truth leakage.
Reproducible training where applicable.
Class balance where applicable.
Confidence behavior.
False-positive behavior.
False-negative behavior.
Inference latency.

AI should support the tracking system rather than replacing deterministic physics and control logic where those mechanisms are required.

33. Explainability Tests

For AI-assisted decisions, the system should expose useful evidence such as:

Detection confidence
Candidate score
Modulation result
Prediction residual
Spatial consistency
Temporal consistency
Trust value
Final decision

The purpose is to make false locks and rejected candidates explainable during debugging and demonstration.

34. Test Artifacts

Test execution should produce organized artifacts.

Recommended structure:

artifacts/
├── test-results/
├── benchmark-results/
├── logs/
├── plots/
├── videos/
└── reports/

Generated artifacts should not be committed to Git unless they are intentionally selected documentation assets.

35. Continuous Integration

The project should eventually run automated tests through CI.

Minimum CI checks:

Python syntax/import validation.
Unit tests.
Integration tests.
Regression tests.
Basic benchmark smoke test.
Requirements/configuration validation.

Long-running stress tests may be separated from every commit and executed as scheduled/manual CI jobs.

36. Minimum Test Gate Before Major Milestones

Before declaring a milestone complete:

All unit tests pass.
All relevant integration tests pass.
Regression tests pass.
No known critical failure is ignored.
Benchmark results are generated.
Configuration is recorded.
Git commit is recorded.
Documentation matches the implementation.
37. Final Validation Gate

Before the final SIH/ISRO demonstration:

Functional
 Beacon detection works.
 Beacon identification works.
 Continuous tracking works.
 Camera control works.
 Required motion modes work.
 Required disturbances work.
 Target loss and reacquisition work.
 Real-time telemetry works.
Quantitative
 Acquisition benchmark completed.
 Tracking-error benchmark completed.
 Target-loss benchmark completed.
 Reacquisition benchmark completed.
 Processing-FPS benchmark completed.
 MP4 benchmark completed.
Engineering
 Unit tests pass.
 Integration tests pass.
 Regression tests pass.
 Reproducibility verified.
 No ground-truth leakage.
 No duplicate authoritative tracker.
 Configuration documented.
 Benchmark artifacts preserved.
Documentation
 Architecture documented.
 Coordinate convention documented.
 Interfaces documented.
 Development rules documented.
 Test strategy documented.
 Problem-statement requirements mapped.
 User manual prepared.
 Technical report prepared.
 Performance log prepared.
38. Core Testing Principle

The final system must not be considered successful merely because it produces a visually convincing demonstration.

Success requires evidence that the system:

Performs the required functions.
Uses a valid closed-loop architecture.
Handles realistic disturbances.
Maintains tracking without ground-truth assistance.
Recovers from temporary target loss.
Meets measurable performance requirements.
Produces reproducible benchmark results.
Can explain important tracking decisions and failures.
Has automated tests protecting critical behavior.
Can be independently evaluated from recorded data.

A successful demonstration shows that the system works.
A successful test strategy provides evidence of how well, when, and under what conditions it works.