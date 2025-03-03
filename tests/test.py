# test.py
# This script tests the logistic regression model implemented from scratch.
# It uses the breast cancer dataset from sklearn and evaluates the model's accuracy.
# Key steps:
# - Loads and splits the dataset into training and test sets.
# - Initializes and trains the logistic regression model.
# - Evaluates the model on the test data and prints the accuracy.
# - Reports if early stopping occurred during training.
#
# Dependencies:
# - logistic_regression (custom implementation)
# - utils (custom train_test_split function)
# - sklearn (for dataset loading)

import sys
import os

# Add the src directory to the Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from logistic_regression import LogisticRegression
from utils import train_test_split

from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

model = LogisticRegression()

X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.8)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

if __name__ == "__main__":
    print(f"Model accuracy: {score:.4f}")
    print(f"Early stopped after {model.iterations_before_stop} iterations.") if model.early_stopped else None