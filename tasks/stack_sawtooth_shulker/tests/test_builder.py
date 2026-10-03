from types import SimpleNamespace

from tasks.stack_sawtooth_shulker.assets import FragmentHandle
from tasks.stack_sawtooth_shulker.config import load_task_config
from tasks.stack_sawtooth_shulker.validation import expected_task_manifest


class FakeApp:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakeWorld:
    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append("world_reset")

    def step(self, render=False):
        self.events.append(("world_step", render))


class FakeRobot:
    def __init__(self, events):
        self.events = events

    def initialize_dofs(self):
        self.events.append("initialize_dofs")


class FakeController:
    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append("controller_reset")


def _fake_fragment_handle(fragment):
    root = f"/World/Task/Fragments/{fragment.name}"
    count = len(fragment.local_cells)
    return FragmentHandle(
        fragment.name,
        root,
        tuple(f"{root}/Visuals/Voxel_{index:03d}" for index in range(count)),
        tuple(f"{root}/Colliders/Voxel_{index:03d}" for index in range(count)),
        1.0,
    )


def test_builder_composes_core_before_task_and_reinitializes_after_authoring(monkeypatch):
    from tasks.stack_sawtooth_shulker import builder

    config = load_task_config()
    events = []
    app = FakeApp()
    workcell = SimpleNamespace(
        config=config.workcell,
        app=app,
        world=FakeWorld(events),
        stage=object(),
        cameras=object(),
        robots={"right": FakeRobot(events)},
        controller=FakeController(events),
    )

    monkeypatch.setattr(builder, "load_task_config", lambda: config)
    monkeypatch.setattr(
        builder,
        "build_workcell",
        lambda workcell_config, headless, stream: (
            events.append(("build_workcell", workcell_config.preset_name, headless, stream))
            or workcell
        ),
    )
    monkeypatch.setattr(
        builder, "define_task_roots", lambda stage: events.append("define_task_roots")
    )
    monkeypatch.setattr(
        builder,
        "author_target_metadata",
        lambda stage, task_config: events.append("author_metadata"),
    )
    monkeypatch.setattr(
        builder,
        "author_fragment",
        lambda stage, workcell_config, task_config, fragment: (
            events.append(("author_fragment", fragment.name))
            or _fake_fragment_handle(fragment)
        ),
    )
    monkeypatch.setattr(
        builder,
        "stabilize_initial_fragments",
        lambda stage, task_config: events.append("stabilize_fragments"),
    )

    handles = builder.build_stack_sawtooth_scene(headless=True, stream=False)

    assert events[:5] == [
        ("build_workcell", "single_arm_right", True, False),
        "define_task_roots",
        "author_metadata",
        ("author_fragment", "FragmentA"),
        ("author_fragment", "FragmentB"),
    ]
    assert events[5:] == [
        "world_reset",
        "initialize_dofs",
        "controller_reset",
        ("world_step", False),
        "stabilize_fragments",
        ("world_step", False),
        ("world_step", False),
        ("world_step", False),
    ]
    assert handles.config is config
    assert handles.workcell is workcell
    assert handles.app is app
    assert handles.world is workcell.world
    assert handles.stage is workcell.stage
    assert handles.cameras is workcell.cameras
    assert tuple(fragment.name for fragment in handles.fragments) == (
        "FragmentA",
        "FragmentB",
    )


def test_builder_closes_core_app_when_task_authoring_fails(monkeypatch):
    from tasks.stack_sawtooth_shulker import builder

    config = load_task_config()
    app = FakeApp()
    workcell = SimpleNamespace(
        config=config.workcell,
        app=app,
        world=FakeWorld([]),
        stage=object(),
        cameras=object(),
        robots={},
        controller=FakeController([]),
    )
    monkeypatch.setattr(builder, "load_task_config", lambda: config)
    monkeypatch.setattr(builder, "build_workcell", lambda *args, **kwargs: workcell)
    monkeypatch.setattr(builder, "define_task_roots", lambda stage: None)
    monkeypatch.setattr(
        builder,
        "author_target_metadata",
        lambda *args: (_ for _ in ()).throw(RuntimeError("metadata failed")),
    )

    try:
        builder.build_stack_sawtooth_scene()
    except RuntimeError as exc:
        assert str(exc) == "metadata failed"
    else:
        raise AssertionError("builder did not propagate task-authoring failure")
    assert app.close_calls == 1


class FakePath:
    def __init__(self, value):
        self.pathString = value


class FakePrim:
    def __init__(self, path, valid=True, type_name="Xform", children=()):
        self.path = path
        self.valid = valid
        self.type_name = type_name
        self.children = tuple(children)

    def IsValid(self):
        return self.valid

    def GetChildren(self):
        return self.children

    def GetTypeName(self):
        return self.type_name

    def GetPath(self):
        return FakePath(self.path)


class FakeStage:
    def __init__(self, paths, extra_fragment=None):
        self.prims = {path: FakePrim(path) for path in paths}
        fragments_root = "/World/Task/Fragments"
        children = [
            self.prims[path]
            for path in (
                "/World/Task/Fragments/FragmentA",
                "/World/Task/Fragments/FragmentB",
            )
            if path in self.prims
        ]
        if extra_fragment:
            children.append(FakePrim(extra_fragment))
        self.prims[fragments_root] = FakePrim(fragments_root, children=children)

    def GetPrimAtPath(self, path):
        return self.prims.get(path, FakePrim(path, valid=False))


def _task_paths(manifest):
    paths = [manifest.task_root, manifest.metadata_path, manifest.fragments_root]
    for fragment in manifest.fragments:
        paths.extend(
            (
                fragment.root_path,
                fragment.visuals_root,
                fragment.colliders_root,
                *fragment.visual_paths,
                *fragment.collider_paths,
            )
        )
    return tuple(paths)


def _validation_handles(stage):
    config = load_task_config()
    manifest = expected_task_manifest(config.workcell, config)
    return SimpleNamespace(
        config=config,
        workcell=SimpleNamespace(config=config.workcell, stage=stage),
        stage=stage,
        manifest=manifest,
        fragments=tuple(_fake_fragment_handle(fragment) for fragment in config.fragments),
    )


def test_scene_validation_accepts_complete_expected_structure(monkeypatch):
    from tasks.stack_sawtooth_shulker import validation

    config = load_task_config()
    manifest = expected_task_manifest(config.workcell, config)
    handles = _validation_handles(FakeStage(_task_paths(manifest)))
    monkeypatch.setattr(validation, "validate_workcell_stage", lambda workcell: ())
    monkeypatch.setattr(validation, "_live_contract_failures", lambda handles: ())

    report = validation.validate_stack_sawtooth_scene(handles)

    assert report.ok is True
    assert report.missing_paths == ()
    assert report.extra_fragment_paths == ()
    assert report.contract_failures == ()
    assert sum(len(fragment.visual_paths) for fragment in handles.manifest.fragments) == 27
    assert sum(len(fragment.collider_paths) for fragment in handles.manifest.fragments) == 27


def test_scene_validation_reports_missing_extra_and_live_contract_failures(monkeypatch):
    from tasks.stack_sawtooth_shulker import validation

    config = load_task_config()
    manifest = expected_task_manifest(config.workcell, config)
    paths = list(_task_paths(manifest))
    missing = paths.pop()
    extra = "/World/Task/Fragments/Unexpected"
    handles = _validation_handles(FakeStage(paths, extra_fragment=extra))
    monkeypatch.setattr(
        validation, "validate_workcell_stage", lambda workcell: ("/World/MissingCore",)
    )
    monkeypatch.setattr(
        validation,
        "_live_contract_failures",
        lambda handles: ("FragmentA mass mismatch",),
    )

    report = validation.validate_stack_sawtooth_scene(handles)

    assert report.ok is False
    assert report.missing_paths == ("/World/MissingCore", missing)
    assert report.extra_fragment_paths == (extra,)
    assert report.contract_failures == ("FragmentA mass mismatch",)
