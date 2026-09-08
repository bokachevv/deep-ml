import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    traced = np.trace(matrix)
    deter = np.linalg.det(matrix)
    eigenvalues = np.roots([1, -traced, deter])
    return eigenvalues