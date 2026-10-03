from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _subcommands(parser):
    action = next(action for action in parser._actions if action.dest == "command")
    return action.choices


def test_cli_shape_and_external_output_boundary(tmp_path):
    from tasks.stack_sawtooth_shulker.__main__ import build_parser
    from tasks.stack_sawtooth_shulker.runtime import require_external_output

    parser = build_parser()
    assert tuple(_subcommands(parser)) == ("build", "launch")
    with pytest.raises(SystemExit):
        parser.parse_args(["build"])
    launch_options = {
        option
        for action in _subcommands(parser)["launch"]._actions
        for option in action.option_strings
    }
    assert not ({"--record", "--output", "--motion", "--control"} & launch_options)
    for local in (REPO_ROOT, REPO_ROOT / "generated" / "scene.usda"):
        with pytest.raises(ValueError, match="outside the repository"):
            require_external_output(local, REPO_ROOT)
    external = tmp_path / "scene.usda"
    assert require_external_output(external, REPO_ROOT) == external.resolve()


class _FakeApp:
    def __init__(self, running=(True, False)):
        self.running = iter(running)
        self.close_calls = 0

    def is_running(self):
        return next(self.running)

    def close(self):
        self.close_calls += 1


class _FakeWorld:
    def __init__(self, error=None):
        self.steps = []
        self.error = error

    def step(self, *, render):
        self.steps.append(render)
        if self.error:
            raise self.error


class _FakeLayer:
    def __init__(self, error=None):
        self.exports = []
        self.error = error

    def Export(self, path):
        self.exports.append(path)
        if self.error:
            raise self.error


def _handles(*, report_ok=True, export_error=None):
    app = _FakeApp()
    world = _FakeWorld()
    layer = _FakeLayer(export_error)
    handles = SimpleNamespace(
        app=app,
        world=world,
        stage=SimpleNamespace(GetRootLayer=lambda: layer),
    )
    report = SimpleNamespace(ok=report_ok, contract_failures=("bad",))
    return handles, report, layer


def test_build_exports_once_and_closes(tmp_path, monkeypatch):
    import tasks.stack_sawtooth_shulker.__main__ as cli

    handles, report, layer = _handles()
    monkeypatch.setattr(cli, "build_stack_sawtooth_scene", lambda: handles)
    monkeypatch.setattr(cli, "validate_stack_sawtooth_scene", lambda value: report)
    output = tmp_path / "new" / "scene.usda"

    assert cli.main(["build", "--output-usd", str(output)]) == 0
    assert output.parent.is_dir()
    assert layer.exports == [str(output.resolve())]
    assert handles.app.close_calls == 1


def test_build_does_not_export_invalid_scene(tmp_path, monkeypatch):
    import tasks.stack_sawtooth_shulker.__main__ as cli

    handles, report, layer = _handles(report_ok=False)
    monkeypatch.setattr(cli, "build_stack_sawtooth_scene", lambda: handles)
    monkeypatch.setattr(cli, "validate_stack_sawtooth_scene", lambda value: report)

    with pytest.raises(RuntimeError, match="scene validation failed"):
        cli.main(["build", "--output-usd", str(tmp_path / "scene.usda")])
    assert layer.exports == []
    assert handles.app.close_calls == 1


def test_build_propagates_export_error_and_closes(tmp_path, monkeypatch):
    import tasks.stack_sawtooth_shulker.__main__ as cli

    handles, report, _ = _handles(export_error=OSError("export failed"))
    monkeypatch.setattr(cli, "build_stack_sawtooth_scene", lambda: handles)
    monkeypatch.setattr(cli, "validate_stack_sawtooth_scene", lambda value: report)

    with pytest.raises(OSError, match="export failed"):
        cli.main(["build", "--output-usd", str(tmp_path / "scene.usda")])
    assert handles.app.close_calls == 1


def test_viewer_only_renders_until_app_stops_and_closes(monkeypatch):
    import tasks.stack_sawtooth_shulker.runtime as runtime

    handles, _, _ = _handles()
    monkeypatch.setattr(runtime, "build_stack_sawtooth_scene", lambda headless: handles)

    assert runtime.run_viewer() == 0
    assert handles.world.steps == [True]
    assert handles.app.close_calls == 1


def test_viewer_hidden_smoke_limit_and_failure_cleanup(monkeypatch):
    import tasks.stack_sawtooth_shulker.runtime as runtime

    handles, _, _ = _handles()
    handles.app = _FakeApp(running=(True, True, True))
    monkeypatch.setattr(runtime, "build_stack_sawtooth_scene", lambda headless: handles)
    assert runtime.run_viewer(smoke_frames=2) == 0
    assert handles.world.steps == [True, True]
    assert handles.app.close_calls == 1

    failing, _, _ = _handles()
    failing.world = _FakeWorld(RuntimeError("step failed"))
    monkeypatch.setattr(runtime, "build_stack_sawtooth_scene", lambda headless: failing)
    with pytest.raises(RuntimeError, match="step failed"):
        runtime.run_viewer()
    assert failing.app.close_calls == 1
    with pytest.raises(ValueError, match="non-negative"):
        runtime.run_viewer(smoke_frames=-1)
