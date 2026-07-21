import unittest
import numpy as np
from xgboost import XGBoost, DecisionTree, utils

class TestXGBoost(unittest.TestCase):
    def test_init(self):
        xgb = XGBoost()
        self.assertEqual(xgb.n_estimators, 100)
        self.assertEqual(xgb.learning_rate, 0.1)
        self.assertEqual(xgb.max_depth, 6)

    def test_train(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        xgb = XGBoost()
        xgb.train(X, y)
        self.assertIsInstance(xgb.tree, DecisionTree)

    def test_predict(self):
        xgb = XGBoost()
        X = np.array([[1, 2], [3, 4]])
        y_pred = xgb.predict(X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, (2,))

    def test_gradient(self):
        xgb = XGBoost()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        grad = xgb.gradient(X, y)
        self.assertIsInstance(grad, np.ndarray)
        self.assertEqual(grad.shape, (2, 2))

class TestDecisionTree(unittest.TestCase):
    def test_init(self):
        tree = DecisionTree()
        self.assertEqual(tree.n_nodes, 0)
        self.assertEqual(tree.max_depth, 6)

    def test_train(self):
        tree = DecisionTree()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        tree.train(X, y)
        self.assertEqual(tree.n_nodes, 4)

class TestUtils(unittest.TestCase):
    def test_metric(self):
        utils.metric(np.array([0, 1]), np.array([0, 1]))
        utils.metric(np.array([0, 1]), np.array([1, 0]))

if __name__ == "__main__":
    unittest.main()