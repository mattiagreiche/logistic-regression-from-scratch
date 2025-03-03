# logistic_regression.py
# This file contains the implementation of a custom Logistic Regression classifier from scratch.
# Key features of the implementation:
# - Uses batch gradient descent for model training.
# - Implements early stopping to prevent overfitting.
# - Includes methods for prediction, loss calculation, and model evaluation.
#
# The model is initialized with random weights and bias, which are updated iteratively using gradient descent.
# Early stopping is incorporated to halt training once the validation loss stops improving.
#
# Dependencies:
# - numpy (for numerical operations)
# - numpy.typing (for type annotations)
# - typing (for type annotations)
#
# Key Methods:
# - fit: Trains the model using the training data.
# - predict: Makes predictions on new data.
# - loss: Calculates logistic loss for the model's predictions.
# - score: Evaluates the model's accuracy.
# - gradient_descent: Performs weight updates using gradient descent.
# - train_validate_split: Splits the data into training and validation sets.


import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Optional

class LogisticRegression():
    """
    Logistic Regression classifier.

    Parameters:
    - random_seed (int): Random seed for reproducibility. Default is None.

    Attributes:
    - weights (Optional[NDArray]): Coefficients of the features.
    - bias (Optional[float]): Intercept term.
    - iterations_before_stop (int): The number of iterations the model ran during training.
    - early_stopped (bool): Flag to indicate if the model was stopped early due to lack of improvement.

    Methods:
    - fit(X, y, learning_rate=1e-6, num_iterations=100_000, tolerance=1e-6, patience=10): Fit the model to the training data.
    - predict(X): Predict class labels for samples in X.
    - loss(y_true, y_pred): Compute the logistic loss.
    - initialize_weights(n_features): Initialize the weights and bias.
    - gradient_descent(X, y): Perform gradient descent to update the weights and bias.
    - train_validate_split(X, y, train_size=0.8): Split the data into training and validation sets.
    - score(X, y): Compute the accuracy of the model on the given data.
    """

    def __init__(self, random_seed: int = None) -> None:
        """
        Initialize the LogisticRegression object.

        Parameters:
        - random_seed (int): Random seed for reproducibility. Default is None.
        
        Attributes:
        - learning_rate (float): Learning rate for gradient descent, initialized later.
        - weights (Optional[NDArray]): Coefficients of the features, initialized later.
        - bias (Optional[float]): Intercept term, initialized later.
        - iterations_before_stop (int): The number of iterations the model has run after training (initialized to None). Used to track the number of iterations if early stopping is triggered.
        - early_stopped (bool): Flag to indicate if the model was early stopped during training (initialized to False).
        """
        np.random.seed(random_seed)
        self.learning_rate: float = None
        
        self.weights: Optional[NDArray] = None # Set later
        self.bias: Optional[float] = None # Set later
        
        self.iterations_before_stop: int = 0 # Track the number of iterations before early stopping
        self.early_stopped: bool = False # Track if the model was early stopped
    
    def fit(self, X: ArrayLike, y: ArrayLike, learning_rate: float = 1e-5, num_iterations: int = 200_000, tolerance: float = 1e-7, patience: int = 50) -> None:
        """
        Fit the model to the training data using batch gradient descent and early stopping.

        Parameters:
        - X (ArrayLike): Training samples.
        - y (ArrayLike): Target values.
        - learning_rate (float): Learning rate for gradient descent. Default is 1e-5.
        - num_iterations (int): Maximum number of iterations. Default is 200,000.
        - tolerance (float): Tolerance for early stopping. Decrease value for looser tolerance. Default is 1e-7.
        - patience (int): Number of iterations to wait for improvement before early stopping. Default is 50.
        """
        self.learning_rate = learning_rate
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        X_validate, y_validate = self.train_validate_split(X, y)[2:] # Only need the validation set
        
        self.initialize_weights(X.shape[1])
        
        # Early stopping parameters
        previous_validate_loss = np.inf
        patience_counter = 0
        
        for iteration in range(num_iterations):
            self.iterations_before_stop += 1
            self.gradient_descent(X, y)
            y_validate_pred = self.predict(X_validate)
            validate_loss = self.loss(y_validate, y_validate_pred)
            
            if abs(previous_validate_loss - validate_loss) / (previous_validate_loss + 1e-8) < tolerance: # Relative change plus small value to avoid division by zero
                patience_counter += 1
            else:
                patience_counter = 0
            
            previous_validate_loss = validate_loss
            
            if patience_counter == patience:
                self.early_stopped = True
                break
    
    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict class labels for samples in X.

        Parameters:
        - X (ArrayLike): Samples to predict.

        Returns:
        - NDArray: Predicted class labels.
        """
        X = np.asarray(X)
        z = X @ self.weights + self.bias
        z = np.clip(z, -100, 100) # Avoid overflow
        y_pred = 1 / (1 + np.exp(-z))
        return np.clip(y_pred, 1e-15, 1 - 1e-15) # Avoid log(0)
    
    def loss(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Compute the logistic loss.

        Parameters:
        - y_true (ArrayLike): True class labels.
        - y_pred (ArrayLike): Predicted class probabilities.

        Returns:
        - float: Logistic loss.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def initialize_weights(self, n_features: int) -> None:
        """
        Initialize the weights and bias.

        Parameters:
        - n_features (int): Number of features.
        """
        self.weights = np.random.randn(n_features) * 0.01 # Standard normal distribution
        self.bias = 0
    
    def gradient_descent(self, X: ArrayLike, y: ArrayLike) -> None:
        """
        Perform gradient descent to update the weights and bias.

        Parameters:
        - X (ArrayLike): Training samples.
        - y (ArrayLike): Target values.
        """
        y_pred = self.predict(X)
        
        # Compute the gradient for weights and bias
        gradient_weights = X.T @ (y_pred - y) / len(y)
        gradient_bias = np.sum(y_pred - y) / len(y)
        
        # Update the weights and bias using the gradient and the learning rate
        self.weights -= self.learning_rate * gradient_weights
        self.bias -= self.learning_rate * gradient_bias

    def train_validate_split(self, X: ArrayLike, y: ArrayLike, train_size: float = 0.8) -> tuple:
        """
        Split the data into training and validation sets.

        Parameters:
        - X (ArrayLike): Samples.
        - y (ArrayLike): Target values.
        - train_size (float): Proportion of the data to use for training. Default is 0.8.

        Returns:
        - tuple: Training samples, training target values, validation samples, validation target values.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = X.shape[0]
        n_train = int(n_samples * train_size)
        
        # Shuffle the data and split into training and validation sets
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        validate_indices = indices[n_train:]
        
        return X[train_indices], y[train_indices], X[validate_indices], y[validate_indices]
    
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute the accuracy of the model on the given data.

        Parameters:
        - X (ArrayLike): Samples.
        - y (ArrayLike): Target values.

        Returns:
        - float: Accuracy of the model.
        """
        y_pred = self.predict(X)
        
        # Round the predicted probabilities to get binary class predictions
        correct_predictions = (np.round(y_pred) == y)
        
        return np.mean(correct_predictions)