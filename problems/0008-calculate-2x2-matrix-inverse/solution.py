import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    matrix=np.array(matrix)
    deter = matrix[0,0]*matrix[1,1] - matrix[0,1]*matrix[1,0]
    if deter == 0 : return None
    return np.dot(1/deter,[[matrix[1,1],-matrix[0,1]],[-matrix[1,0],matrix[0,0]]])
