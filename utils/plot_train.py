import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config.config import *


def save_importances(importances_, figsize=(8, 12)):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=figsize)
    sns.barplot(x='gain', y='feature', data=importances_.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
#     plt.savefig('importances.png')

def save_iter_scores(iter_scores, figsize=(8, 12)):
    plt.figure(figsize=figsize)
    sns.lineplot(x="iteration", y="score", hue='fold', style="trn_val", data=iter_scores, palette=sns.color_palette("muted")[:FOLDS]);
    plt.tight_layout()
    plt.show();
#     plt.savefig('importances.png')

def plot_metric(self):
    """
    Plot training progress.
    Inspired by `plot_metric` from https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/plotting.html

    :return:
    """
    full_evals_results = pd.DataFrame()
    for model in self.models:
        evals_result = pd.DataFrame()
        for k in model.model.evals_result_.keys():
            evals_result[k] = model.model.evals_result_[k][self.eval_metric]
        evals_result = evals_result.reset_index().rename(columns={'index': 'iteration'})
        full_evals_results = full_evals_results.append(evals_result)

    full_evals_results = full_evals_results.melt(id_vars=['iteration']).rename(columns={'value': self.eval_metric,
                                                                                        'variable': 'dataset'})
    sns.lineplot(data=full_evals_results, x='iteration', y=self.eval_metric, hue='dataset')
    plt.title('Training progress')
    
    
def cm_show(target, pred, weight):
    cm = confusion_matrix(target, pred, sample_weight=weight)
    classes = [0,1,2,3]
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
              title=None, ylabel='True label', xlabel='Predicted label')

    fmt = '.2f'
    thresh = cm.max() / 2.
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            ax.text(k, j, format(cm[j, k], fmt), ha="center", va="center", color="white" if cm[j,k] > thresh else "black")

    fig.tight_layout()
    plt.show();
