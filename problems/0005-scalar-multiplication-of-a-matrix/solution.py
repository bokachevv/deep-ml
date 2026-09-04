import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    matrix = np.array(matrix)
    for i in range(len(matrix)):
        matrix[i] = matrix[i] * scalar
    return matrix.tolist()