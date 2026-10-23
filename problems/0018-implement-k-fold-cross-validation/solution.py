import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    
    indicies = np.arange(n_samples)
    
    if shuffle is True:
        np.random.shuffle(indicies)
    
    base_size = n_samples//k
    remainder = n_samples%k

    start = 0
    folds = []

    for i in range(k):

        if i < remainder:
            fold_size = base_size + 1
        else:
            fold_size = base_size

        stop = start + fold_size

        test_i = indicies[start : stop]
        train_i = np.concatenate((indicies[:start], indicies[stop:]))

        folds.append((train_i, test_i))

        start = stop

    return folds