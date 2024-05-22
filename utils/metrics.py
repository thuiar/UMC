from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import brentq, linear_sum_assignment
from scipy.interpolate import interp1d
import logging
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'fmi':round(fowlkes_mallows_score(y_true, y_pred)*100, 2),
            }