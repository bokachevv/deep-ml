import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A,b=np.array(A),np.array(b)
    sols = np.zeros(3)
    for i in range(n):
        buffer = np.zeros(3)
        for j in range(3):
            buffer[j]=(b[j]-(A[j] @ sols - (A[j,j] * sols[j])))/A[j,j]
        sols = buffer
    return np.round(sols,4)