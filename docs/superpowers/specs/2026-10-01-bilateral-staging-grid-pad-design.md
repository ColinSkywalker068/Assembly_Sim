# Bilateral Fragment Staging and Generic Assembly Grid Pad

## Purpose

Restore the workcell topology established by the skull demonstration while
keeping the generated-fragment pipeline dataset-neutral.  A generic scene must
contain a central assembly surface and stage fragments in two front-to-back
lanes, one reachable from each FANUC arm.

The current radial AABB packing is unsuitable: it clusters all pieces near the
first free locations around the central exclusion rather than communicating
which arm owns each fragment.

## Reference geometry from the skull demo

The default skull configuration is the authoritative spatial reference:

| Item | Demo value | Generic interpretation |
| --- | --- | --- |
| Table top | `z = 0.750 m` | Resting/assembly surface height |
| Central plate footprint | `0.512 x 0.416 m` | Generic assembly-pad footprint |
| Central plate world center | `(0.000, 0.100) m` | Generic pad center |
| Left fragment lane | `x = -0.420 m` | Left-arm staging lane |
| Right fragment lane | `x = +0.420 m` | Right-arm staging lane |
| Demo lane direction | varying `y` at fixed `x` | Front-to-back rows, perpendicular to arm-to-arm axis |
| Separation | `0.040 m` | Minimum AABB staging gap |

The generic pad is not a LEGO asset: it has no studs and no dependency on
`plate.npz`.  Its top is flush with the table top and it has a 9.6-mm deep
static box collider below that surface.  The visible top receives a square
grid at the existing 40-mm scene grid step.  It remains an assembly target,
not a required support for staged fragments.

## Generated staging contract

Schema-version-2 layouts remain the source of all fragment poses.  During
preprocessing the scene workcell settings define a bilateral placement spec:

1. Order fragments deterministically by source fragment name.
2. Assign the first `ceil(N / 2)` fragments to the left lane and the remainder
   to the right lane.  Thus three fragments produce a `2 + 1` layout.
3. Keep each fragment's lane X coordinate fixed at the lane center.  Vary only
   Y, packing its AABB in a front-to-back line with the configured gap.
4. Set Z so each fragment's local lower bound rests on the table top.
5. Reject a sample with an explicit capacity error when a lane cannot fit its
   assigned pieces within the table bounds, rather than silently moving them
   around the pad or into the opposite workspace.

The pad footprint is a staging exclusion.  Lane placements must remain inside
the usable table bounds, outside the pad (including the staging gap), outside
robot-base exclusions, and mutually non-overlapping.  The algorithm records
its bilateral spec, assignment, and resulting poses in `layout.json`; Isaac
continues to consume those stored poses without knowing a dataset convention.

## Scene contract

The scene configuration gains a generic `assembly_pad` description under the
environment/placement settings: center, size, thickness, grid step, and lane
centers.  Generic layouts always receive the pad.  The legacy skull path keeps
its existing `plate.npz` visual and collider unchanged.

The generic USD hierarchy adds a deterministic
`/World/Environment/AssemblyPad` visual and static collider.  The stage
manifest exposes that path separately from the legacy plate path so validation
can distinguish the two assets.  Materials, physics, cameras, robot controls,
and reset behaviour remain unchanged.

## Validation

Tests will establish that:

- the generic manifest contains an assembly-pad path and legacy manifests
  retain their plate path;
- the generic pad has a flat visual grid and a static collision box at the
  demo-reference location and top height;
- bilateral staging places three representative fragments as two fixed-X
  positions in one Y lane and one in the opposite Y lane;
- the placement is deterministic, rests on the table, avoids the pad and
  bases, and detects lane-capacity exhaustion;
- the Breaking Bad example is regenerated, launches with three fragments, and
  yields updated agent and wrist-camera inspection images.

## Scope limits

This change does not add assembly planning, goals to robot control, grasp
planning, learned policies, or dataset-specific placement rules.
