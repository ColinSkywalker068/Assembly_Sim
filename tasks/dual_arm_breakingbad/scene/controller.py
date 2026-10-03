"""Task controller that adds fragment reset to the core robot controller."""

from __future__ import annotations


def fragment_reset_poses(layout, names):
    poses = tuple(layout.initial_pose(name) for name in names)
    return (
        tuple(pose["position"] for pose in poses),
        tuple(pose["orientation_wxyz"] for pose in poses),
    )


class AssemblySceneController:
    def __init__(self, handles):
        if handles.workcell.controller is None:
            raise RuntimeError("core workcell controller is not initialized")
        self.handles = handles
        self.workcell_controller = handles.workcell.controller

    @property
    def active_robot_name(self):
        return self.workcell_controller.active_robot_name

    def toggle_robot(self):
        return self.workcell_controller.toggle_robot()

    def jog(self, joint_index, delta_rad):
        return self.workcell_controller.jog(joint_index, delta_rad)

    def home(self):
        return self.workcell_controller.home()

    def command_gripper(self, open_fraction):
        return self.workcell_controller.command_gripper(open_fraction)

    def reset(self) -> None:
        self.workcell_controller.reset()
        if not self.handles.fragments:
            return
        import numpy as np
        from isaacsim.core.prims import RigidPrim

        names = tuple(fragment.name for fragment in self.handles.fragments)
        positions, orientations = fragment_reset_poses(self.handles.layout, names)
        view = RigidPrim(
            prim_paths_expr=[fragment.root_path for fragment in self.handles.fragments],
            name="assembly_fragments",
        )
        view.initialize()
        view.set_world_poses(
            positions=np.asarray(positions, dtype=float),
            orientations=np.asarray(orientations, dtype=float),
        )
        view.set_velocities(np.zeros((len(names), 6), dtype=float))
