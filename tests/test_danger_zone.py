import copy
import unittest

import numpy as np

from experiments.run_sweep import (
    _json_ready,
    classify_danger_zone,
    generate_joint_danger_zone_data,
)
from src.utils import load_config


class DangerZoneTests(unittest.TestCase):
    def test_definition_is_directional_and_material(self):
        delta_dp = np.array([0.20, 0.00, 0.03, -0.03, 0.00])
        delta_tpr = np.array([-0.10, 0.06, 0.07, 0.06, 0.04])

        actual = classify_danger_zone(delta_dp, delta_tpr)

        np.testing.assert_array_equal(
            actual,
            np.array([False, True, True, True, False]),
        )

    def test_joint_sweep_has_expected_shape_and_zero_reference(self):
        config = copy.deepcopy(load_config())
        config['data']['num_samples'] = 500
        config['shifts']['group_shift']['num_steps'] = 3
        config['shifts']['label_shift']['num_steps'] = 2
        config['danger_zone']['num_seeds'] = 2

        results = generate_joint_danger_zone_data(config, seed=12)

        self.assertEqual(
            np.asarray(results['metrics']['delta_dp']).shape,
            (2, 3),
        )
        self.assertEqual(
            np.asarray(results['danger_probability']).shape,
            (2, 3),
        )
        self.assertAlmostEqual(
            float(results['metrics']['delta_dp'][0, 0]), 0.0
        )
        self.assertAlmostEqual(
            float(results['metrics']['delta_tpr_gap_clean'][0, 0]), 0.0
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_danger_zone(np.zeros(2), np.zeros(3))

    def test_numpy_boolean_is_json_ready(self):
        self.assertIs(_json_ready(np.bool_(True)), True)


if __name__ == '__main__':
    unittest.main()
