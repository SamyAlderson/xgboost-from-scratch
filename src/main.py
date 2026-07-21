# src/main.py

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from src.xgboost import XGBoost
from src.tree import DecisionTree
from src.utils import calculate_metrics

def load_data():
    """Load the iris dataset and split it into training and testing sets."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train an XGBoost model on the training data."""
    model = XGBoost()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the performance of the trained model on the testing data."""
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    return metrics

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print("Model performance metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
```

```python
# src/xgboost.py

import numpy as np
from src.tree import DecisionTree
from src.utils import calculate_gradient

class XGBoost:
    """Implementation of the XGBoost algorithm from scratch."""

    def __init__(self):
        self.num_trees = 10
        self.learning_rate = 0.1
        self.trees = []

    def fit(self, X, y):
        """Train the XGBoost model on the training data."""
        for _ in range(self.num_trees):
            tree = DecisionTree()
            tree.fit(X, y)
            self.trees.append(tree)
            y_pred = tree.predict(X)
            gradient = calculate_gradient(y, y_pred)
            X = X + self.learning_rate * gradient

    def predict(self, X):
        """Make predictions on the input data using the trained model."""
        predictions = np.zeros((X.shape[0],))
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions
```

```python
# src/tree.py

import numpy as np

class DecisionTree:
    """Implementation of the decision tree algorithm used by XGBoost."""

    def __init__(self):
        self.max_depth = 5
        self.splitting_feature = None
        self.splitting_value = None

    def fit(self, X, y):
        """Train the decision tree model on the training data."""
        # Not proud of this but it works
        self.splitting_feature = np.random.choice(X.shape[1])
        self.splitting_value = np.median(X[:, self.splitting_feature])
        y_left = y[X[:, self.splitting_feature] < self.splitting_value]
        y_right = y[X[:, self.splitting_feature] >= self.splitting_value]
        X_left = X[X[:, self.splitting_feature] < self.splitting_value]
        X_right = X[X[:, self.splitting_feature] >= self.splitting_value]
        if y_left.shape[0] == 0 or y_right.shape[0] == 0:
            return
        if self.max_depth == 0:
            return
        self.children = [DecisionTree(), DecisionTree()]
        self.children[0].fit(X_left, y_left)
        self.children[1].fit(X_right, y_right)

    def predict(self, X):
        """Make predictions on the input data using the trained model."""
        if self.children is None:
            return X[:, 0]
        left_mask = X[:, self.splitting_feature] < self.splitting_value
        predictions = np.zeros((X.shape[0],))
        predictions[left_mask] = self.children[0].predict(X[left_mask])
        predictions[~left_mask] = self.children[1].predict(X[~left_mask])
        return predictions
```

```python
# src/utils.py

import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate the performance metrics of the model."""
    accuracy = np.mean(y_true == y_pred)
    precision = np.mean(y_true[y_true == y_pred] == y_pred[y_true == y_pred])
    recall = np.mean(y_true[y_pred == y_true] == y_true[y_pred == y_true])
    return {"accuracy": accuracy, "precision": precision, "recall": recall}

def calculate_gradient(y, y_pred):
    """Calculate the gradient of the loss function."""
    return 2 * (y_pred - y)