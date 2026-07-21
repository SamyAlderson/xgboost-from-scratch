import unittest
import numpy as np
from src.utils import calculate_metric

class TestUtils(unittest.TestCase):

    def test_calculate_metric(self):
        # Generate some sample data
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.8, 0.2, 0.9, 0.1, 0.7])

        # Calculate the metric
        metric = calculate_metric(y_true, y_pred)

        # Check that the metric is within expected bounds
        self.assertAlmostEqual(metric, 0.8)  # Average precision should be around 0.8

    def test_calculate_metric_empty(self):
        # Test edge case with empty arrays
        y_true = np.array([])
        y_pred = np.array([])

        # Calculate the metric
        metric = calculate_metric(y_true, y_pred)

        # Check that the metric is NaN (not a number) in this case
        self.assertTrue(np.isnan(metric))

    def test_calculate_metric_invalid_input(self):
        # Test edge case with invalid input types
        y_true = "not a number"
        y_pred = np.array([1, 0, 1, 0, 1])

        # Attempt to calculate the metric
        with self.assertRaises(TypeError):
            calculate_metric(y_true, y_pred)

if __name__ == '__main__':
    unittest.main()