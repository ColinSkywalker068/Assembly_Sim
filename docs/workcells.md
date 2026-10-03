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

Tasks may select a preset but may not override core-owned environment, physics,
render, robot, or camera fields. `validate_task_workcell_boundary()` enforces
that rule for task configuration.

`core.scene.builder.build_workcell()` produces the invariant scene and no
task objects. A task then authors its own content into that stage and wraps the
core controller only where task reset behavior is needed.
