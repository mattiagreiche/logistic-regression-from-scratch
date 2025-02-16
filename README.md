# Logistic Regression From Scratch

This project implements a batch logistic regression model from scratch using only NumPy. The model is trained and tested on the breast cancer dataset from Scikit-learn.

## Files

- `logistic_regression.py`: Contains the implementation of the `LogisticRegression` class,  which includes methods for model training, prediction, loss calculation, and evaluation using batch gradient descent.
- `utils.py`: Provides the `train_test_split` function for splitting datasets into training and testing sets.
- `test.py`: Tests the model on Scikit-learn's breast cancer dataset and outputs the accuracy.

## Features

- **Batch Gradient Descent**: Minimizes the logistic loss function to find optimal weights for prediction.
- **Custom Weight Initialization**: Weights are initialized randomly using a standard normal distribution and updated during training.
- **Early Stopping**: Stops training early if the validation loss doesn’t improve, preventing overfitting.
- **Reproducibility**: The model’s training process can be initialized with a specified random seed, ensuring consistent results across multiple runs.
- **Model Evaluation**: The model evaluates performance on a test dataset.
- **Custom Dataset Support**: Works with any binary classification dataset by changing the input data.

## Usage

To use the logistic regression model, follow these steps:

1. Install the required dependencies: `numpy`, `scikit-learn` (only if a sample dataset is needed).
2. Import the `LogisticRegression` class from `logistic_regression.py`.
3. Import the `train_test_split` function from `utils.py`.
4. Load your dataset.
5. Split the dataset into training and testing sets using the `train_test_split` function.
6. Create an instance of the `LogisticRegression` class.
7. Train the model using the `fit` method with your training data.
8. Evaluate the model using the `score` method with your testing data.

## Results

The model achieves an accuracy of around 92% on Scikit-learn's breast cancer dataset. Accuracy may vary slightly depending on the random seed used for initialization and data splitting.

