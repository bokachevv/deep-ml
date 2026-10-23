import numpy as np
def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred==1))
    tn = np.sum((y_true == 0) & (y_pred==0))
    fp = np.sum((y_true == 0) & (y_pred==1))
    fn = np.sum((y_true == 1) & (y_pred==0))
    
    pr = tp / (tp + fp)
    rec = tp / (tp + fn)

    if tp + fp + fn == 0: return 0

    return np.round(((1+beta**2) * (pr * rec) / ((beta**2 * pr) + rec)), 3)
