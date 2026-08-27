import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    
    res = np.zeros(len(a))
    a=np.array(a)
    b=np.array(b)
    if a.shape[1]!=len(b): return -1
    for i in range(len(a)):
        res[i]=np.sum(a[i]*b)
    return list(res)