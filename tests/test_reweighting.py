import unittest

import numpy as np

from src.data_generator import generate_data
from src.reweighting import (
    compute_importance_weights_kde,
    effective_sample_size,
)
from src.shifts import covariate_shift, group_shift, label_shift


class KDEReweightingTests(unittest.TestCase):
    def test_weights_are_normalized_bounded_and_nonconstant_under_scale_shift(self):
        X_train, _, _ = generate_data(num_samples=400, seed=1)
        X_target, _, _ = covariate_shift(
            severity=4.0, num_samples=400, seed=2
        )

        weights = compute_importance_weights_kde(
            X_train, X_target, max_fit_samples=300, random_state=3
        )

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=8)
        self.assertGreaterEqual(float(weights.min()), 0.1)
        self.assertLessEqual(float(weights.max()), 10.0)
        self.assertGreater(float(weights.std()), 0.01)
        self.assertLessEqual(effective_sample_size(weights), len(weights))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_importance_weights_kde(np.ones(3), np.ones((3, 1)))


class ShiftConfigurationTests(unittest.TestCase):
    def test_all_shifts_honor_requested_sample_count(self):
        common = {'num_samples': 137, 'num_features': 3, 'seed': 7}
        for shifted in (
            group_shift(**common),
            covariate_shift(**common),
            label_shift(**common),
        ):
            X, y, s = shifted
            self.assertEqual(X.shape, (137, 3))
            self.assertEqual(y.shape, (137,))
            self.assertEqual(s.shape, (137,))

    def test_noise_features_vary_within_a_dataset(self):
        X, _, _ = generate_data(num_samples=100, num_features=3, seed=4)
        self.assertGreater(float(X[:, 1].std()), 0.5)
        self.assertGreater(float(X[:, 2].std()), 0.5)

    def test_group_shift_changes_conditional_base_rates(self):
        _, y, s = group_shift(severity=1.0, num_samples=20000, seed=8)
        rate_a = y[s == 0].mean()
        rate_b = y[s == 1].mean()
        self.assertAlmostEqual(float(rate_a), 0.5, delta=0.03)
        self.assertAlmostEqual(float(rate_b), 0.1, delta=0.02)


if __name__ == '__main__':
    unittest.main()
