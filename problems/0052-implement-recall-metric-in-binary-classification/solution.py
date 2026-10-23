import numpy as np
def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred==1))
    tn = np.sum((y_true == 0) & (y_pred==0))
    fp = np.sum((y_true == 0) & (y_pred==1))
    fn = np.sum((y_true == 1) & (y_pred==0))
    if tp + fn == 0: return 0
    return tp / (tp + fn)
