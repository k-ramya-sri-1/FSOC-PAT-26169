# FSOC-PAT-26169 Coordinate Convention

## 1. Purpose

This document defines the coordinate systems, camera model, image coordinates, projection equations, visibility rules, and angle calculations used throughout FSOC-PAT-26169.

All modules must follow these conventions.

No module may silently introduce a different axis convention or projection model.

---

## 2. World Coordinate System

FSOC-PAT-26169 uses a right-handed Cartesian world coordinate system.

The world axes are:

```text
        +Y
         ^
         |
         |
         +--------> +X
        /
       /
     +Z
```

The exact scene orientation must remain consistent throughout the project.

World coordinates describe the positions and motion of objects in the virtual environment.

---

## 3. Camera Coordinate System

The virtual camera uses a right-handed camera coordinate system:

```text
+X = right
+Y = up
+Z = forward
```

Therefore, a target directly in front of the camera has:

```text
Xc = 0
Yc = 0
Zc > 0
```

A target behind the camera has:

```text
Zc <= 0
```

and must not be considered visible.

---

## 4. Camera Orientation

The camera orientation is represented using pan and tilt angles.

The implementation must define the rotation convention explicitly and use it consistently across:

- world-to-camera transformation
- camera-to-world transformation
- target line-of-sight calculation
- pixel projection
- angular error calculation
- gimbal control

No other module may silently introduce a different rotation convention.

---

## 5. Default FPA Resolution

The default virtual focal plane array (FPA) resolution is:

```text
width  = 640 pixels
height = 480 pixels
```

The implementation must support configurable sensor resolution.

The resolution must not be hard-coded into unrelated modules.

---

## 6. Image Coordinate System

The image coordinate origin is at the top-left corner:

```text
(0, 0)
```

The image center is:

```text
cx = width / 2
cy = height / 2
```

For the default 640 × 480 FPA:

```text
cx = 320
cy = 240
```

Pixel x increases toward the right.

Pixel y increases downward.

Therefore:

```text
Target to the right  -> u > cx
Target to the left   -> u < cx
Target above         -> v < cy
Target below         -> v > cy
```

---

## 7. Default Field of View

The default camera field of view is:

```text
Horizontal FOV = 4 degrees
Vertical FOV   = 3 degrees
```

The implementation must support configurable horizontal and vertical FOV.

Angles must be converted to radians internally whenever required by mathematical functions.

---

## 8. Pinhole Camera Model

The camera projection must use a physically meaningful pinhole-camera model.

For sensor width `W`, sensor height `H`, horizontal FOV `FOVx`, and vertical FOV `FOVy`:

```text
fx = W / (2 * tan(FOVx / 2))

fy = H / (2 * tan(FOVy / 2))
```

where the FOV values are expressed in radians during calculation.

The principal point is:

```text
cx = W / 2
cy = H / 2
```

---

## 9. World-to-Camera Transformation

A target position in world coordinates is:

```text
Pw = [Xw, Yw, Zw]
```

The camera has:

```text
Pc = [Xcam, Ycam, Zcam]
```

The first step is to calculate the relative position:

```text
Prelative = Pw - Pc
```

The relative position must then be transformed using the camera orientation to obtain:

```text
Pc_camera = [Xc, Yc, Zc]
```

The exact rotation implementation must be centralized in `core/geometry.py`.

No other module may independently implement a different world-to-camera transformation.

---

## 10. Pinhole Projection

For a target in camera coordinates:

```text
Xc
Yc
Zc
```

where:

```text
Zc > 0
```

the pixel coordinates are:

```text
u = fx * (Xc / Zc) + cx

v = cy - fy * (Yc / Zc)
```

The negative sign in the vertical equation is required because camera +Y is upward while image pixel v increases downward.

---

## 11. Visibility

A target is potentially visible only when:

```text
Zc > 0
```

The target must also lie within the configured horizontal and vertical field of view.

The implementation must therefore perform both:

1. Depth/forward visibility check.
2. Angular/FOV visibility check.

A target outside the FOV must be reported as not visible.

A target behind the camera must be reported as not visible.

---

## 12. FOV Boundary Handling

The implementation must correctly handle targets:

- well inside the FOV
- close to the horizontal FOV boundary
- close to the vertical FOV boundary
- exactly on the boundary
- outside the FOV

Boundary behavior must be deterministic and documented.

---

## 13. Pixel Bounds

For an image of width `W` and height `H`, valid pixel coordinates are:

```text
0 <= u < W
0 <= v < H
```

A projected target outside these bounds is outside the physical FPA even if its angular position is otherwise valid.

The geometry layer must clearly distinguish:

```text
Inside FOV
```

from:

```text
Inside FPA
```

because a target may be geometrically visible but outside the finite sensor array.

---

## 14. Pixel-to-Angle Conversion

For a measured pixel position `(u, v)`:

```text
dx = u - cx
dy = cy - v
```

The corresponding angular offsets must be calculated using the camera model.

For horizontal angular displacement:

```text
angle_x = atan(dx / fx)
```

For vertical angular displacement:

```text
angle_y = atan(dy / fy)
```

The implementation must not use arbitrary pixel-to-degree scaling.

---

## 15. Line-of-Sight Representation

The geometry layer should support conversion between:

```text
3D target position
```

and:

```text
line-of-sight direction
```

The normalized camera-frame line-of-sight vector is:

```text
L = [Xc, Yc, Zc] / ||[Xc, Yc, Zc]||
```

This representation will later support:

- target tracking
- prediction
- angular error calculation
- gimbal control

---

## 16. Camera Center

When the target is exactly on the camera optical axis:

```text
Xc = 0
Yc = 0
Zc > 0
```

the projected position must be:

```text
u = cx
v = cy
```

For the default FPA:

```text
u = 320
v = 240
```

This is a mandatory geometry test.

---

## 17. Directional Sanity Checks

The following relationships must hold:

```text
Target moves right
        ↓
pixel u increases

Target moves left
        ↓
pixel u decreases

Target moves upward
        ↓
pixel v decreases

Target moves downward
        ↓
pixel v increases
```

These checks must be covered by automated tests.

---

## 18. Numerical Safety

Geometry functions must:

- Avoid division by zero.
- Handle very small positive `Zc`.
- Reject `Zc <= 0`.
- Validate FOV values.
- Validate sensor dimensions.
- Produce deterministic results.
- Avoid unnecessary loss of numerical precision.
- Handle targets near FOV boundaries correctly.

Invalid camera configurations should produce clear errors rather than silently generating invalid coordinates.

---

## 19. Configurability

The following must be configurable:

- Sensor width
- Sensor height
- Horizontal FOV
- Vertical FOV
- Camera position
- Camera orientation

The default configuration remains:

```text
640 × 480
4° × 3°
```

---

## 20. Single Source of Truth

`core/geometry.py` is the only authoritative implementation for:

- World-to-camera transformation
- Camera-to-world transformation
- Camera orientation mathematics
- Pinhole projection
- Pixel-to-angle conversion
- FOV checks
- Line-of-sight calculations

No other module may duplicate these equations.

Other modules must call the geometry API.

---

## 21. Required Geometry Tests

The geometry implementation must eventually include automated tests for:

1. Target directly ahead.
2. Target to the right.
3. Target to the left.
4. Target above.
5. Target below.
6. Target behind the camera.
7. Target outside horizontal FOV.
8. Target outside vertical FOV.
9. Target close to FOV boundary.
10. Target exactly at the optical center.
11. Configurable sensor resolution.
12. Configurable FOV.
13. Pixel-to-angle conversion.
14. World-to-camera transformation.
15. Camera orientation changes.
16. Numerical safety near `Zc = 0`.

---

## 22. Mathematical Consistency Rule

Every subsystem that requires target direction, pixel position, angular error, or camera orientation must use the geometry implementation defined here.

The project must never contain multiple competing camera models.

The mathematical camera model must be validated before higher-level tracking, AI, prediction, or control algorithms are integrated.