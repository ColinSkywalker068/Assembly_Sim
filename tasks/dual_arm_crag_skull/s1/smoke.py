"""PASS/FAIL verdict for the S1 GPU smoke run. Synthetic pick/detour data (synthetic_pick_detour_v2): this
checks the code path, the conditioning on arm and fragment, end-of-skill coverage, and that diffusion
keeps both demo modes where regression cannot. It says nothing about skull assembly.

Criteria v3, fixed before the runs of 2026-10-18 after the S1 code review (all must hold; the
`multimodal.test` metrics come from held-out episodes, 8 samples per window):
  C1  pytest on the GPU node: exit 0, junit counts tests == expected, failures == errors == skipped == 0.
  C2  both trainings finished; checkpoint reload reproduces samples (max abs diff <= 1e-5).
  C3  diffusion place_first (carry about to start, mode unobservable): obstacle_hit_rate <= 0.15,
      left_fraction_of_clear in [0.25, 0.75], episodes_with_both_modes >= 0.5, final_error_m_median <= 0.02.
  C4  regression place_first obstacle_hit_rate >= diffusion place_first obstacle_hit_rate + 0.15.
  C5  DDIM-16 batch-1 median latency <= 100 ms (one 10 Hz policy step).
  C6  deployment-path inference check (buffers holding only past samples -> live_sample -> ActionChunk):
      every attempt yields a contract-valid chunk that is executable at the measured latency.
  C7  diffusion pick_first: chosen_fragment_rate >= 0.9 (two identical-looking fragments; only the
      fragment-pose conditioning says which).
  C8  diffusion place_mid (mode visible in the history): mode_agreement_of_clear >= 0.9, obstacle_hit_rate <= 0.15.
  C9  diffusion place_late (future past the segment end, hold padding): final_error_m_median <= 0.02 and
      final_width_m_median >= 0.07 (released; open width 0.08 m).
  C10 diffusion arms: inactive_motion_m_median <= 0.01 and active_motion_m_median >= 0.05 (raw predictions,
      before S1 forces the hold).
History: v1 (C1-C5) judged run 20219996, v2 (C1-C6) judged 20221507 on synthetic_detour_v1; both superseded.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
    return {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}


def _get(metrics, *keys):
    for key in keys:
        metrics = metrics.get(key, {}) if isinstance(metrics, dict) else {}
    return metrics or {}


def _between(value, low, high):
    return value is not None and low <= value <= high


def checks(diffusion, regression, junit, pytest_exit, expected_tests):
    first, base = _get(diffusion, 'multimodal', 'test', 'place_first'), _get(regression, 'multimodal', 'test', 'place_first')
    pick, mid = _get(diffusion, 'multimodal', 'test', 'pick_first'), _get(diffusion, 'multimodal', 'test', 'place_mid')
    late, arms = _get(diffusion, 'multimodal', 'test', 'place_late'), _get(diffusion, 'multimodal', 'test', 'arms')
    chunks = _get(diffusion, 'inference_check')
    return {
        'C1_pytest': pytest_exit == 0 and junit['tests'] == expected_tests
        and junit['failures'] == junit['errors'] == junit['skipped'] == 0,
        'C2_trained_and_reloadable': all(_get(m, 'checkpoint').get('reload_max_abs_diff', 1.0) <= 1e-5
                                         for m in (diffusion, regression)),
        'C3_diffusion_modes': bool(first) and first['obstacle_hit_rate'] <= 0.15
        and _between(first['left_fraction_of_clear'], 0.25, 0.75)
        and first['episodes_with_both_modes'] >= 0.5 and first['final_error_m_median'] <= 0.02,
        'C4_regression_averages_modes': bool(first and base)
        and base['obstacle_hit_rate'] >= first['obstacle_hit_rate'] + 0.15,
        'C5_latency': _get(diffusion, 'latency_batch1', 'ddim_16').get('median_ms', float('inf')) <= 100.0,
        'C6_inference_chunks': bool(chunks) and chunks.get('attempted', 0) > 0
        and chunks.get('chunks') == chunks.get('attempted') and not chunks.get('not_executable'),
        'C7_pick_conditioned_on_fragment': bool(pick) and pick['chosen_fragment_rate'] >= 0.9,
        'C8_mid_window_follows_mode': bool(mid) and _between(mid['mode_agreement_of_clear'], 0.9, 1.0)
        and mid['obstacle_hit_rate'] <= 0.15,
        'C9_late_window_releases_at_goal': bool(late) and late['final_error_m_median'] <= 0.02
        and late['final_width_m_median'] >= 0.07,
        'C10_only_requested_arm_moves': bool(arms) and arms['inactive_motion_m_median'] <= 0.01
        and arms['active_motion_m_median'] >= 0.05,
    }


def verdict(diffusion, regression, junit, pytest_exit, expected_tests):
    result = checks(diffusion, regression, junit, pytest_exit, expected_tests)
    return {'pass': all(result.values()), 'checks': result, 'criteria_version': 3,
            'pytest': {'exit': pytest_exit, **junit, 'expected_tests': expected_tests},
            'diffusion_test': _get(diffusion, 'multimodal', 'test'), 'regression_test': _get(regression, 'multimodal', 'test'),
            'ddim16_latency': _get(diffusion, 'latency_batch1', 'ddim_16'), 'inference_check': _get(diffusion, 'inference_check')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--diffusion', type=Path, required=True)
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--junit', type=Path, required=True)
    parser.add_argument('--pytest-exit', type=int, required=True)
    parser.add_argument('--expected-tests', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(p.read_text()) if p.exists() else {}  # noqa: E731
    result = verdict(load(args.diffusion), load(args.regression),
                     junit_counts(args.junit) if args.junit.exists() else
                     {'tests': 0, 'failures': 0, 'errors': 1, 'skipped': 0},
                     args.pytest_exit, args.expected_tests)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    (args.output.parent / ('PASS' if result['pass'] else 'FAIL')).write_text(json.dumps(result['checks']) + '\n')
    print(json.dumps(result['checks']))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
