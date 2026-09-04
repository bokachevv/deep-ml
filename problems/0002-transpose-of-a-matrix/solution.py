import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    a=np.array(a)
    rows = len(a)
    cols = len(a[0])
    new_matrix = np.zeros((cols,rows))
    for i in range(rows):
        for j in range(cols):
            new_matrix[j,i]=a[i,j]
    
    return new_matrix