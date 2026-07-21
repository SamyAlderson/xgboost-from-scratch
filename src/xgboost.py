# XGBoost implementation from scratch
# Reference: https://xgboost.readthedocs.io/en/latest/doc/boosting.html

import numpy as np
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor

class XGBoost:
    def __init__(self, num_estimators=100, learning_rate=0.1, max_depth=6, gamma=0, subsample=1, colsample_bytree=1):
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []

    def _init_tree(self, X, y):
        tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        tree.fit(X, y)
        return tree

    def _predict(self, tree, X):
        # Use tree.predict to get the predicted values
        return tree.predict(X)

    def _calculate_hessian(self, y, y_pred):
        # Calculate the Hessian matrix of the loss function
        # This is the negative of the gradient of the loss function
        return np.array([-np.exp(-yi * y_predi) * (1 - np.exp(-yi * y_predi)) * yi for yi, y_predi in zip(y, y_pred)])

    def _update_tree(self, tree, X, y, hessian, learning_rate):
        # Update the tree parameters using the gradient and Hessian
        # This is the key step in the XGBoost algorithm
        # We use the gradient and Hessian to update the tree parameters
        tree.fit(X, y, sample_weight=hessian)
        return tree

    def fit(self, X, y):
        # Fit the XGBoost model to the training data
        for i in range(self.num_estimators):
            # Initialize a new tree
            tree = self._init_tree(X, y)
            # Predict the values using the current tree
            y_pred = self._predict(tree, X)
            # Calculate the gradient and Hessian of the loss function
            gradient = np.array([expit(-yi * y_predi) for yi, y_predi in zip(y, y_pred)])
            hessian = self._calculate_hessian(y, y_pred)
            # Update the tree parameters using the gradient and Hessian
            tree = self._update_tree(tree, X, y, hessian, self.learning_rate)
            # Add the updated tree to the list of trees
            self.trees.append(tree)
        return self

    def predict(self, X):
        # Use the final tree to make predictions on the test data
        y_pred = [0] * len(X)
        for tree in reversed(self.trees):
            y_pred = [yi + self.learning_rate * tree.predict(np.array([xi]))[0] for yi, xi in zip(y_pred, X)]
        return np.array(y_pred)

    def score(self, X, y):
        # Calculate the mean absolute error of the predictions
        return np.mean(np.abs(y - self.predict(X)))

# Usage example
if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

    xg = XGBoost()
    xg.fit(X_train, y_train)
    print(xg.score(X_test, y_test))