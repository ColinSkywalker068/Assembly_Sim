from pathlib import Path

import pytest

from tasks.demo_render_source.__main__ import build_parser
from tasks.demo_render_source.paths import require_external_output


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cli_exposes_the_four_storyboard_workflows():
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )

    assert tuple(subparsers.choices) == ("prepare", "choreograph", "render", "run")


@pytest.mark.parametrize(
    "argv",
    (
        ["prepare", "--input", "case.glb"],
        ["choreograph", "--layout", "layout.json", "--probe", "probe.json"],
        ["render", "--choreography", "choreography.json"],
        ["run", "--input", "case.glb"],
    ),
)
def test_write_commands_require_explicit_external_outputs(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_generated_output_inside_repository_is_rejected(tmp_path):
    output = REPO_ROOT / "tasks" / "demo_render_source" / "generated"

    with pytest.raises(ValueError, match="outside the repository"):
        require_external_output(output, REPO_ROOT)


def test_external_output_is_resolved(tmp_path):
    output = tmp_path / "render"

    assert require_external_output(output, REPO_ROOT) == output.resolve()
