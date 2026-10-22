import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    
    m, n = X.shape
    y = y.reshape(-1, 1)
    weights = np.zeros((n, 1))

    for iteration in range(iterations):
        grad = 1/m * (X.T @ (X @ weights - y))
        weights = weights - alpha * grad

    return np.round(weights.flatten(), 4)