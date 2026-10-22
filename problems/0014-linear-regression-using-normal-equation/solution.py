import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y)
    weight = np.linalg.inv((X.T @ X)) @ X.T @ y
    return np.round(weight, 4).tolist()