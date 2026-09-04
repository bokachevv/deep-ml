import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    matrix = np.array(matrix)
    shape_m = np.shape(matrix)
    means=[]
    if mode == "column":
        for i in range(shape_m[1]):
            means.append(float(np.mean(matrix[:, i])))
    else:
        for j in range(shape_m[0]):
            means.append(float(np.mean(matrix[j,:])))
    return means