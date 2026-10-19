"""The CRX + YAM port's DexAgent checkout (baselines/dexagent/bench/checkout.py): pinned commit + port patch + generated
bench, verified. Needs the DexAgent clone (DEXAGENT_SOURCE) and skips without it."""
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv
from tasks.dual_arm_crag_skull.baselines.dexagent.bench import checkout

SOURCE = Path(dexenv.dexagent_source())
needs_source = pytest.mark.skipif(not (SOURCE / '.git').exists(), reason=f'no DexAgent clone at {SOURCE}')
S0_LIMITS = {'max_joint_velocity_rad_s': [2.0943951023931953] * 2 + [3.141592653589793] * 4,
             'max_joint_acceleration_rad_s2': 20.0, 'servo_hz': 60}


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    return checkout.prepare_checkout(SOURCE, tmp_path_factory.mktemp('checkout') / 'DexAgent', S0_LIMITS)


def test_the_patch_leaves_setup_alone_and_reads_as_a_git_diff():
    body = checkout.patch_body()
    files = [line.split(' b/')[1] for line in body.splitlines() if line.startswith('diff --git ')]
    assert files and not any(name.startswith('setup/') for name in files)
    assert 'tool_library/common/bench.py' in files and 'tool_library/common/rigid_robot.py' in files


@needs_source
def test_a_prepared_checkout_is_the_commit_the_patch_and_the_bench(prepared):
    assert prepared.commit == checkout.DEXAGENT_COMMIT
    assert (prepared.path / 'setup' / 'robot' / 'crx10ial.xml').is_file()
    assert not (prepared.path / 'setup' / 'robot' / 'sharpa_left.xml').exists()  # no stale Sharpa hand to fall back on
    assert checkout.verify_checkout(prepared.path, S0_LIMITS) == prepared


@needs_source
@pytest.mark.parametrize('target', ['tool_library/common/hand.py', 'setup/setup.yaml', 'tool_library/stray.py'])
def test_any_change_to_a_checkout_is_refused(prepared, target, tmp_path):
    path = prepared.path / target
    original = path.read_bytes() if path.exists() else None
    try:
        path.write_bytes((original or b'') + b'\n# changed\n')
        with pytest.raises(ValueError):
            checkout.verify_checkout(prepared.path, S0_LIMITS)
    finally:
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)
    checkout.verify_checkout(prepared.path, S0_LIMITS)


@needs_source
def test_a_bench_for_other_s0_limits_is_refused(prepared):
    with pytest.raises(ValueError, match='S0 limits'):
        checkout.verify_checkout(prepared.path, {**S0_LIMITS, 'max_joint_acceleration_rad_s2': 5.0})


@needs_source
def test_the_patched_grasp_generator_finds_jaw_chords_in_the_objects_material(prepared, tmp_path):
    script = textwrap.dedent('''
        import json, sys
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from tool_library.skills.generate_rigid_grasps import chords_for_jaws
        block = np.mgrid[-0.05:0.0501:0.002, -0.03:0.0301:0.002, 0:0.0401:0.002].reshape(3, -1).T  # 100 x 60 x 40 mm
        centre = np.array([0.0, 0.0, 0.02])
        chords = chords_for_jaws(block, centre, np.eye(3), lambda f: centre + [0, 0, (2 * f - 1) * 0.02], [0.6],
                                 0.09, 0.0075)
        print(json.dumps([dict(c[3], far=list(c[1]), near=list(c[2])) for c in chords]))
    ''')
    out = subprocess.run([dexenv.dexagent_python(), '-B', '-c', script, str(prepared.path)], capture_output=True,
                         text=True, timeout=600, env=dexenv.scrubbed_env(tmp_path))
    assert out.returncode == 0, out.stderr[-2000:]
    chords = json.loads(out.stdout)
    assert chords and all(c['width_m'] <= 0.09 for c in chords)
    assert not any(c['yaw_deg'] == 0.0 for c in chords)  # 100 mm across, wider than the jaws open
    first = chords[0]
    assert first['com_offset_m'] < 1e-3 and abs(first['far'][2] - 0.024) < 1e-9  # through the centre, at 60 %
    offsets = [c['com_offset_m'] for c in chords]
    assert offsets == sorted(offsets) and len({(c['height_frac'], c['yaw_deg']) for c in chords}) == len(chords)
