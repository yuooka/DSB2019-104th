import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import scipy as sp
from numba import jit
from functools import partial

from config.config import *


class OptimizedRounder(object):
    def __init__(self, method):
        self.coef_ = 0
        self.method = method
        
    def _kappa_loss(self, coef, X, y, w=None):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3

        ll = cohen_kappa_score(y, X_p, weights='quadratic', sample_weight=w)
        return -ll

    def fit(self, X, y, w=None):
        loss_partial = partial(self._kappa_loss, X=X, y=y, w=w)
        initial_coef = VALID_TH
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method=self.method)
        
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3
        return X_p

    def coefficients(self):
        return self.coef_['x']
