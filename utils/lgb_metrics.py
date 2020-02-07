import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from numba import jit

from config.config import *

@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


##### Custom metric #####
## qwk for integer
def eval_qwk_lgb(y_true, y_pred):
    """
    Fast kappa eval function for lgb.
    """
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'kappa', qwk(y_true, y_pred), True

## qwk for decimal
def eval_qwk_lgb_regr(y_true_ori, y_pred_ori, w=None):
    """
    Fast kappa eval function for lgb.
    """
    try:
        y_true = y_pred_ori.get_label()
        y_pred = y_true_ori.copy()
    except:
        y_true = y_true_ori.copy()
        y_pred = y_pred_ori.copy()
    
    y_pred[y_pred <= VALID_TH[0]] = 0
    y_pred[np.where(np.logical_and(y_pred > VALID_TH[0], y_pred <= VALID_TH[1]))] = 1
    y_pred[np.where(np.logical_and(y_pred > VALID_TH[1], y_pred <= VALID_TH[2]))] = 2
    y_pred[y_pred > VALID_TH[2]] = 3
    
    return 'kappa', cohen_kappa_score(y_true, y_pred, weights='quadratic', sample_weight=w), True # return 'kappa', qwk(y_true, y_pred), True

## rmse
def rmse(preds, data):
    y_true = data.get_label()
    metric = (((preds-y_true)**2/len(preds)).sum())**0.5
    return 'rmse', metric, False

## qwk for decimal with sample weight
class eval_qwk_lgb_regr_weight(object):
    def __init__(self, trn_w=None, val_w=None):
        self.trn_w = trn_w
        self.val_w = val_w
        
    def __call__(self, y_true_ori, y_pred_ori):
        """
        Fast kappa eval function for lgb.
        """
        try:
            y_true = y_pred_ori.get_label()
            y_pred = y_true_ori.copy()
        except:
            y_true = y_true_ori.copy()
            y_pred = y_pred_ori.copy()

#         ##### For log target #####
#         y_true = np.exp(y_true)-1
#         y_pred = np.exp(y_pred)-1
#         ##########################
            
        y_pred[y_pred <= VALID_TH[0]] = 0
        y_pred[np.where(np.logical_and(y_pred > VALID_TH[0], y_pred <= VALID_TH[1]))] = 1
        y_pred[np.where(np.logical_and(y_pred > VALID_TH[1], y_pred <= VALID_TH[2]))] = 2
        y_pred[y_pred > VALID_TH[2]] = 3
        
        if len(y_true_ori)==len(self.trn_w):
            w = self.trn_w
        else:
            w = self.val_w

        return 'kappa', cohen_kappa_score(y_true, y_pred, weights='quadratic', sample_weight=w), True
    

## multiple custom-metric
def lgb_metrics(preds, data):
    return [
        eval_qwk_lgb_regr(preds, data),
        rmse(preds, data),
    ]
