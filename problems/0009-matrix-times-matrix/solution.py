import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    a,b=np.array(a),np.array(b)
    if a.shape[1] != b.shape[0]: return -1
    c=[]
    for i in range(a.shape[0]):
        mid_c=[]
        for j in range(b.shape[1]):
            mid_c.append(int(a[i,:] @ b[:,j]))
        c.append(mid_c)
    return c