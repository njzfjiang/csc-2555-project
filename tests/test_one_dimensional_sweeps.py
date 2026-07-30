import copy
import unittest

import numpy as np

from experiments.run_sweep import (
    METHOD_KEYS,
    METRIC_KEYS,
    _empty_shift_results,
    _one_dimensional_seed_values,
    aggregate_one_dimensional_results,
)
from src.utils import load_config


class OneDimensionalSweepTests(unittest.TestCase):
    def _repetition(self, offset):
        repetition = {
            shift: _empty_shift_results()
            for shift in ('group_shift', 'covariate_shift', 'label_shift')
        }
        for shift_result in repetition.values():
            for method in METHOD_KEYS:
                for metric in METRIC_KEYS:
                    shift_result[method][metric] = [offset, offset + 2.0]
            diagnostics = shift_result['diagnostics']
            for key in ('ess', 'ess_fraction', 'weight_min', 'weight_max'):
                diagnostics[key] = [offset, offset + 2.0]
            diagnostics['thresholds'] = [
                {'0': offset + 0.1, '1': offset + 0.2},
                {'0': offset + 0.3, '1': offset + 0.4},
            ]
        return repetition

    def test_configured_seed_values_use_coupled_stride(self):
        config = copy.deepcopy(load_config())
        config['one_dimensional_sweeps']['num_seeds'] = 3
        config['one_dimensional_sweeps']['seed_stride'] = 1000

        self.assertEqual(
            _one_dimensional_seed_values(config, 42),
            [42, 1042, 2042],
        )

    def test_aggregation_preserves_mean_std_and_samples(self):
        seed_results = [self._repetition(1.0), self._repetition(3.0)]
        aggregated = aggregate_one_dimensional_results(
            seed_results,
            [12, 1012],
        )

        baseline = aggregated['group_shift']['baseline']
        np.testing.assert_allclose(baseline['dp'], [2.0, 4.0])
        np.testing.assert_allclose(
            aggregated['group_shift']['metric_std']['baseline']['dp'],
            [1.0, 1.0],
        )
        self.assertEqual(
            np.asarray(
                aggregated['group_shift']['metric_samples']['baseline']['dp']
            ).shape,
            (2, 2),
        )
        self.assertEqual(
            aggregated['one_dimensional_metadata']['seeds'],
            [12, 1012],
        )
        self.assertAlmostEqual(
            aggregated['group_shift']['diagnostics']['thresholds'][0]['0'],
            2.1,
        )

    def test_invalid_seed_count_is_rejected(self):
        config = copy.deepcopy(load_config())
        config['one_dimensional_sweeps']['num_seeds'] = 0
        with self.assertRaises(ValueError):
            _one_dimensional_seed_values(config, 42)


if __name__ == '__main__':
    unittest.main()
