import json
from types import SimpleNamespace

import pytest

from tasks.stack_sawtooth_shulker_small.collection import collect_manifest
from tasks.stack_sawtooth_shulker_small.dataset import generate_manifest, write_manifest


def test_batch_attempts_50_and_keeps_failures_without_retry(tmp_path, monkeypatch):
    manifest_path = write_manifest(generate_manifest(42), tmp_path/'input.json')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: '')
    attempted = []
    def run(args, **kwargs):
        if args == ['nvidia-smi']:
            return SimpleNamespace(returncode=0)
        output = __import__('pathlib').Path(args[args.index('--output')+1])
        demo_id = args[args.index('--demo-id')+1]
        attempted.append(demo_id)
        output.mkdir()
        failed = demo_id in ('demo_002', 'demo_027')
        (output/'status.json').write_text(json.dumps({'demo_id': demo_id,
                                                     'status': 'failed' if failed else 'success'}))
        return SimpleNamespace(returncode=1 if failed else 0)
    monkeypatch.setattr('subprocess.run', run)
    result = collect_manifest(manifest_path, tmp_path/'output')
    summary = json.loads((result/'summary.json').read_text())
    assert attempted == [f'demo_{i:03d}' for i in range(50)]
    assert summary['attempted'] == 50
    assert summary['successful'] == 48


def test_collection_requires_explicit_single_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(RuntimeError, match='exactly one GPU'):
        collect_manifest('unused', 'unused')
