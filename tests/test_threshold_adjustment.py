import unittest

import numpy as np

from src.adjust_threshold import (
    adjust_threshold_equal_opportunity,
    adjust_threshold_equalized_odds,
    apply_group_thresholds,
)


class ThresholdAdjustmentTests(unittest.TestCase):
    def test_equal_opportunity_raises_lower_tpr_group(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        thresholds = adjust_threshold_equal_opportunity(
            y_true, scores, groups, grid_resolution=0.1
        )
        predictions = apply_group_thresholds(scores, groups, thresholds)

        for group in (0, 1):
            mask = (groups == group) & (y_true == 1)
            self.assertEqual(float(predictions[mask].mean()), 1.0)
        self.assertLess(thresholds[1], 0.5)

    def test_apply_thresholds_requires_every_group(self):
        with self.assertRaises(ValueError):
            apply_group_thresholds(
                np.array([0.2, 0.8]), np.array([0, 1]), {0: 0.5}
            )

    def test_approximate_equalized_odds_handles_single_class_group(self):
        thresholds = adjust_threshold_equalized_odds(
            np.array([1, 1, 0, 1]),
            np.array([0.9, 0.6, 0.2, 0.8]),
            np.array([0, 0, 1, 1]),
            grid_resolution=0.1,
        )
        self.assertEqual(set(thresholds), {0, 1})


if __name__ == '__main__':
    unittest.main()
