import copy
import hashlib
import json
import unittest
from pathlib import Path

from core.robots.linear_4310 import ASSET_DIR
from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG, readiness, validate
from tasks.dual_arm_crag_skull.workspace import repo_path


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_scaffold_is_valid_but_not_ready(self):
        validate(self.config)
        result = readiness(self.config)
        self.assertFalse(result['simulator_ready'])
        self.assertFalse(result['hardware_ready'])
        self.assertIn('workcell.calibration_file', result['missing_configuration'])
        self.assertIn('s2.program_path', result['missing_configuration'])
        self.assertEqual(self.config['s2']['planner'], 'aspire_program')
        self.assertEqual(self.config['s2']['execution_mode'], 'skills')

    def test_gripper_is_the_vendored_i2rt_linear_4310(self):
        workcell = self.config['workcell']
        self.assertEqual(workcell['gripper_model'], 'I2RT linear_4310')
        self.assertEqual(repo_path(workcell['gripper_asset']), ASSET_DIR / 'linear_4310.xml')
        self.assertNotIn('workcell.gripper_model', readiness(self.config)['missing_configuration'])
        # flange_T_tcp needs the adapter plate, which is not designed yet.
        self.assertIsNone(self.config['s0']['flange_T_tcp'])

    def test_contact_phase_parameters_are_explicit_and_reported_missing(self):
        missing = readiness(self.config)['missing_configuration']
        for key in ('contact_table_clearance_m', 'contact_region_radius_m', 'contact_max_speed_m_s',
                    'finger_opening_margin_m'):
            with self.subTest(key=key):
                self.assertIn(key, self.config['s0'])
                self.assertIsNone(self.config['s0'][key])
                self.assertIn(f's0.{key}', missing)

    def test_invalid_contracts_are_rejected(self):
        for section, key, value in [
            ('dataset', 'minimum_fragments', 5),
            ('s1', 'action_dim', 14),
            ('s1', 'execution_steps', 17),
            ('s0', 'servo_hz', 59),
            ('s0', 'physics_hz', 0),
            ('s0', 'hardware_execution_enabled', True),
        ]:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.config)
                cfg[section][key] = value
                with self.assertRaises(ValueError):
                    validate(cfg)

    def test_invalid_planner_settings_are_rejected(self):
        for key, value in [
            ('planner', 'unknown'),
            ('execution_mode', 'mixed'),
            ('program_path', ''),
            ('program_path', 3),
            ('effort', 'extreme'),
            ('server_fallbacks', [{'model': 'claude-opus-4-8'}]),
            ('max_turns', 0),
            ('max_turns', 2.5),
            ('max_planning_s', 0),
            ('timeouts_s', {**self.config['s2']['timeouts_s'], 'pick': 0}),
            ('completion_defaults', {**self.config['s2']['completion_defaults'], 'pick': {'type': 'observed'}}),
            ('completion_defaults', {k: v for k, v in self.config['s2']['completion_defaults'].items() if k != 'retreat'}),
        ]:
            with self.subTest(key=key, value=value):
                cfg = copy.deepcopy(self.config)
                cfg['s2'][key] = value
                with self.assertRaises(ValueError):
                    validate(cfg)

    def test_missing_planner_key_is_a_value_error(self):
        del self.config['s2']['timeouts_s']
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_duplicate_or_missing_camera_is_rejected(self):
        self.config['cameras'] = ['agent', 'agent', 'right_wrist']
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_baseline_settings_are_checked(self):
        result = readiness(self.config)
        self.assertIn('DexAgent baseline live run', result['pending_implementation'])
        self.assertIn('ASPIRE baseline live run', result['pending_implementation'])
        for path, value in [
            (('seed_splits', 'selection'), [120, 130]),
            (('seed_splits', 'heldout'), [1.5, 100]),
            (('initial_condition', 'max_yaw_deg'), 270.0),
            (('initial_condition', 'attempts'), 0),
            (('success_tolerances_from',), 's2.completion_defaults.pick'),
            (('success_tolerances_from',), 's1.place'),
            (('dexagent', 'upstream_commit'), 'e29bdad'),
            (('dexagent', 'upstream_commit'), 'g' * 40),
            (('aspire', 'live_run_verified'), 'no'),
            (('initial_condition', 'max_shift_m'), 2.0),
            (('initial_condition', 'gap_m'), 2.0),
            (('initial_condition', 'max_yaw_rad'), 0.5),
            (('success_tolerances_from',), None),
            (('success_tolerances_from',), 's2.completion_defaults.align'),
            (('heldout_claims_dir',), '/tmp/claims'),
            (('heldout_claims_dir',), '../claims'),
            (('heldout_claims_dir',), None),
        ]:
            with self.subTest(path=path, value=value):
                cfg = copy.deepcopy(self.config)
                section = cfg['baselines']
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = value
                with self.assertRaises(ValueError):
                    validate(cfg)

    def test_aspire_settings_match_the_pinned_checkout_and_patch(self):
        from tasks.dual_arm_crag_skull.baselines.aspire import checkout
        aspire = self.config['baselines']['aspire']
        self.assertEqual(aspire['upstream_commit'], checkout.PINNED_COMMIT)
        self.assertEqual(repo_path(aspire['checkout_path']), checkout.DEFAULT_DESTINATION)
        self.assertEqual((DEFAULT_CONFIG.parents[1] / aspire['patch']).resolve(), checkout.PATCH)
        self.assertEqual(aspire['patch_sha256'], hashlib.sha256(checkout.PATCH.read_bytes()).hexdigest())

    def test_dexagent_settings_match_the_code_defaults(self):
        import dataclasses
        from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv, export, runner
        dexagent = self.config['baselines']['dexagent']
        trial = {f.name: f.default for f in dataclasses.fields(runner.TrialConfig)}
        options = export.ExportOptions()
        self.assertEqual(dexagent['upstream_commit'], export.DEXAGENT_COMMIT)
        self.assertEqual((repo_path(dexagent['source_clone']), repo_path(dexagent['venv_python'])),
                         (Path(dexenv.DEFAULT_SOURCE), Path(dexenv.DEFAULT_PYTHON)))
        self.assertEqual((dexagent['variant'], dexagent['anchor_mode'], tuple(dexagent['pad_xy_rig_m'])),
                         (export.VARIANT, options.anchor_mode, options.pad_xy))
        self.assertEqual((dexagent['model'], dexagent['effort'], dexagent['max_tokens'], dexagent['cache_ttl']),
                         (trial['model'], trial['effort'], trial['max_tokens'], trial['cache_ttl']))
        self.assertNotIn('budget_usd_per_trial', dexagent)  # one shared pool instead (llm_budget)

    def test_live_baseline_runs_share_one_budget_pool(self):
        from tasks.dual_arm_crag_skull.baselines.budget import configured_pool_dir, configured_total_usd
        budget = self.config['baselines']['llm_budget']
        self.assertEqual(configured_total_usd(self.config), 100.0)  # decided 2026-10-19 for DexAgent and ASPIRE together
        self.assertEqual(configured_pool_dir(self.config), repo_path(budget['pool_dir']))
        for bad in (0, -5, float('inf'), True, '100'):
            config = json.loads(json.dumps(self.config))
            config['baselines']['llm_budget']['total_usd'] = bad
            with self.assertRaises(ValueError):
                validate(config)
        config = json.loads(json.dumps(self.config))
        config['baselines']['llm_budget']['pool_dir'] = '/tmp/elsewhere'
        with self.assertRaises(ValueError):
            validate(config)

    def test_the_protective_stop_is_part_of_the_protocol(self):
        rule = self.config['s0']['protective_stop']
        self.assertEqual((rule['tracking_error_rad'], rule['periods'], rule['speed_factor']), (0.01, 3, 1.1))
        for key, bad in (('tracking_error_rad', 0), ('periods', 0), ('speed_factor', 1.0), ('periods', None)):
            config = json.loads(json.dumps(self.config))
            config['s0']['protective_stop'][key] = bad
            with self.assertRaises(ValueError):
                validate(config)
        config = json.loads(json.dumps(self.config))
        del config['s0']['protective_stop']
        with self.assertRaises(ValueError):
            validate(config)

    def test_missing_baseline_section_is_a_value_error(self):
        del self.config['baselines']['seed_splits']
        with self.assertRaises(ValueError):
            validate(self.config)


if __name__ == '__main__':
    unittest.main()
