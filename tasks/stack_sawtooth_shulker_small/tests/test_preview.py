import json
from types import SimpleNamespace

from PIL import Image

from tasks.stack_sawtooth_shulker_small import preview
from tasks.stack_sawtooth_shulker_small.dataset import generate_manifest


def test_preview_pipeline_produces_all_conditions_and_two_sheets(tmp_path, monkeypatch):
    applied, guides, closed = [], [], []
    handles = SimpleNamespace(
        workcell=SimpleNamespace(config=SimpleNamespace(camera_resolution=(640, 480))),
        stage=object(), world=SimpleNamespace(step=lambda **kw: None),
        cameras=SimpleNamespace(path=lambda name: name),
        app=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(preview, '_build_preview_scene', lambda config: handles)
    monkeypatch.setattr(preview, '_apply_preview_config', lambda handles, config: applied.append(config))
    monkeypatch.setattr(preview, 'author_preview_guides', lambda stage, specs: guides.append(specs))
    def capture(camera, path, resolution):
        Image.new('RGB', resolution, (90, 100, 110)).save(path)
    monkeypatch.setattr(preview, '_capture_preview_rgb', capture)
    manifest = generate_manifest(20261005)
    result = preview.render_dataset_preview(manifest, tmp_path, timestamp='20261005T190000')
    assert len(applied) == len(guides) == 50
    assert len(result.image_paths) == 100
    assert closed == [True]
    assert json.loads(result.summary_path.read_text())['status'] == 'complete'
    for path in (result.agent_contact_sheet, result.wrist_contact_sheet):
        with Image.open(path) as image:
            assert image.size == (3200, 4800)
