# utils.py
# This file contains utility functions used in the logistic regression model.
# Currently, it includes a function for splitting a dataset into training and testing sets.
#
# Key functions:
# - train_test_split: Splits the dataset into training and testing sets while optionally setting the size of the training set
#   and ensuring reproducibility with a random seed.
#
# Dependencies:
# - numpy (for numerical operations)
# - numpy.typing (for type annotations)


import numpy as np
from numpy.typing import ArrayLike

def train_test_split(X: ArrayLike, y: ArrayLike, train_size:float = 0.8, random_seed: int = None) -> tuple:
    """
    Split the dataset into training and testing sets.

    Parameters:
    X (ArrayLike): The input features.
    y (ArrayLike): The target variable.
    train_size (float): The proportion of the dataset to include in the training set. Default is 0.8.
    random_seed (int): The random seed for reproducibility. Default is None.

    Returns:
    tuple: A tuple containing the training and testing sets in the following order: X_train, y_train, X_test, y_test.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    np.random.seed(random_seed)
    
    n_samples = X.shape[0]
    n_train = int(n_samples * train_size)
    
    # Shuffle the data and split into training and testing sets
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]
    