import copy
import json
import unittest

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG, readiness, validate


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_scaffold_is_valid_but_not_ready(self):
        validate(self.config)
        result = readiness(self.config)
        self.assertFalse(result['simulator_ready'])
        self.assertFalse(result['hardware_ready'])
        self.assertIn('workcell.gripper_model', result['missing_configuration'])

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

    def test_duplicate_or_missing_camera_is_rejected(self):
        self.config['cameras'] = ['agent', 'agent', 'right_wrist']
        with self.assertRaises(ValueError):
            validate(self.config)


if __name__ == '__main__':
    unittest.main()
