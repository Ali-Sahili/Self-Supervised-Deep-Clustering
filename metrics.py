import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

NMI = normalized_mutual_info_score
ARI = adjusted_rand_score


def ACC(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels
        y_pred: predicted labels
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
