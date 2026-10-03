# Workcell presets

All presets share the same floor, worktable, grid assembly pad, lighting,
materials, physics, camera conventions, and coordinate frame. Presets select
robots; they do not redefine the environment.

| Preset | Robots | Canonical use |
| --- | --- | --- |
| `dual_arm` | left and right | dual-arm tasks and storyboard |
| `single_arm_left` | left only | future single-arm tasks at arm0's pose |
| `single_arm_right` | right only | future single-arm tasks at arm1's pose |

The single-arm robot configurations are exactly the corresponding objects from
the dual-arm preset. Load a preset with:

```python
from core.workcell.config import load_workcell_preset

config = load_workcell_preset("single_arm_left")
```

The agent camera is also preset-aware. Dual-arm scenes retain the canonical
long-edge view at `(0.0, -2.10, 3.44)`. A single-arm scene uses an end-on view
from the side opposite its robot. The right-arm preset uses
`(-1.040404, 0.0, 3.880667)` and aims at Point A, the midpoint
between its base center and the nearest assembly-tape centerline. Its
camera-to-Point-A distance is `3.5 m`; the left-arm preset mirrors the pose.
This tighter framing keeps the complete enabled arm, target, and staging area
in view. Wrist cameras remain mounted to their corresponding robot flanges.

Tasks may select a preset but may not override core-owned environment, physics,
render, robot, or camera fields. `validate_task_workcell_boundary()` enforces
that rule for task configuration.

`core.scene.builder.build_workcell()` produces the invariant scene and no
task objects. A task then authors its own content into that stage and wraps the
core controller only where task reset behavior is needed.
