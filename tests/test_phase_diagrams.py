import unittest

from experiments.plot_phase_diagrams import (
    _available_phase_metrics,
    _metric_std_for_method,
    _method_available,
)


class PhaseDiagramMetricTests(unittest.TestCase):
    def _results(self, include_tpr_gap):
        metrics = {
            'dp': [0.1],
            'eo': [0.2],
            'ece_gap': [0.3],
        }
        if include_tpr_gap:
            metrics['tpr_gap'] = [0.15]
        return {
            shift: {'baseline': dict(metrics)}
            for shift in ('group_shift', 'covariate_shift', 'label_shift')
        }

    def test_schema_v3_includes_tpr_gap(self):
        metrics = _available_phase_metrics(
            self._results(include_tpr_gap=True),
            ('group_shift', 'covariate_shift', 'label_shift'),
        )

        self.assertEqual(
            [metric for metric, _ in metrics],
            ['dp', 'eo', 'tpr_gap', 'ece_gap'],
        )

    def test_older_logs_keep_original_metrics(self):
        metrics = _available_phase_metrics(
            self._results(include_tpr_gap=False),
            ('group_shift', 'covariate_shift', 'label_shift'),
        )

        self.assertEqual(
            [metric for metric, _ in metrics],
            ['dp', 'eo', 'ece_gap'],
        )

    def test_method_must_be_available_for_every_shift(self):
        results = self._results(include_tpr_gap=True)
        for shift_results in results.values():
            shift_results['target_retrained'] = {'dp': [0.05]}

        self.assertTrue(
            _method_available(
                results,
                ('group_shift', 'covariate_shift', 'label_shift'),
                'target_retrained',
            )
        )

        del results['label_shift']['target_retrained']
        self.assertFalse(
            _method_available(
                results,
                ('group_shift', 'covariate_shift', 'label_shift'),
                'target_retrained',
            )
        )

    def test_schema_v5_exposes_metric_standard_deviation(self):
        shift_results = {
            'baseline': {'dp': [0.1, 0.2]},
            'metric_std': {'baseline': {'dp': [0.01, 0.02]}},
        }
        self.assertEqual(
            _metric_std_for_method(
                shift_results, 'baseline', 'dp'
            ).tolist(),
            [0.01, 0.02],
        )
        self.assertIsNone(
            _metric_std_for_method(
                {'baseline': {'dp': [0.1]}}, 'baseline', 'dp'
            )
        )


if __name__ == '__main__':
    unittest.main()
