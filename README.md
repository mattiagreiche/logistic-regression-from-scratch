# Logistic Regression From Scratch

This project implements a batch logistic regression model from scratch using only NumPy.

## Folder Structure

```
.
├── src/
│   ├── logistic_regression.py  # Contains the LogisticRegression class for model implementation
│   └── utils.py               # Provides the train_test_split function
└── tests/
    └── test.py                # Tests the model on Scikit-learn's breast cancer dataset
```

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
7. Train the model using the `fit` method with your training data. Feel free to modify the hyperparameters.
8. Evaluate the model using the `score` method with your testing data.

## Results

The model achieves an accuracy of ~94% on Scikit-learn's breast cancer dataset. Accuracy may vary slightly depending on the random seed used for initialization and data splitting.

## Roadmap

* [x] Implement early stopping to prevent overfitting and speed up convergence.
* [x] Test on a variety of data sets to manually fine-tune default hyperparameters.
* [ ] Implement momentum in the gradient descent algorithm to speed up convergence.
* [ ] Implement L2 or L1 regularization to prevent overfitting.
