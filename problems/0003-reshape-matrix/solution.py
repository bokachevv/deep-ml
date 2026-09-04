from re import L
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:

    reshaped_matrix = np.zeros((new_shape[0], new_shape[1]))
    new_a=np.array(a)
    shape_a=np.shape(a)
    if shape_a[0]*shape_a[1]!=new_shape[0]*new_shape[1]:
        return []
    listed = []
    for i in a:
        for j in i:
            listed.append(j)

    for i in range(len(listed)):
        reshaped_matrix[i//new_shape[1], i%new_shape[1]] = listed[i]
        
    return reshaped_matrix