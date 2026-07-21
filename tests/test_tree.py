import unittest
import numpy as np
from src import tree

class TestTree(unittest.TestCase):

    def test_gini_impurity(self):
        # Test case for Gini impurity calculation
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(tree.gini_impurity(y), 0.5)

        # Test case for Gini impurity calculation with a mix of classes
        y = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(tree.gini_impurity(y), 0.5)

    def test_entropy(self):
        # Test case for entropy calculation
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(tree.entropy(y), 1)

        # Test case for entropy calculation with a mix of classes
        y = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(tree.entropy(y), 1.5)

    def test_best_split(self):
        # Test case for best split calculation
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        self.assertEqual(tree.best_split(X, y), (1, 0))

        # Test case for best split calculation with a mix of classes
        y = np.array([0, 1, 1, 0])
        self.assertEqual(tree.best_split(X, y), (1, 1))

    def test_build_tree(self):
        # Test case for tree building
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        tree_root = tree.build_tree(X, y)
        self.assertEqual(tree_root.label, 0)

    def test_predict_tree(self):
        # Test case for tree prediction
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        tree_root = tree.build_tree(X, y)
        self.assertEqual(tree.predict(tree_root, X[0]), 0)

        # Test case for tree prediction with a mix of classes
        y = np.array([0, 1, 1, 0])
        tree_root = tree.build_tree(X, y)
        self.assertEqual(tree.predict(tree_root, X[0]), 0)

if __name__ == '__main__':
    unittest.main()