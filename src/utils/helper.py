import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import random
import pickle
import os
from config.config import *

def seed_torch(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
#     torch.manual_seed(s)
#     torch.cuda.manual_seed(s)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
seed_torch(SEED)

def to_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj , f)

        
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def high_corr(df, th):
    a = df.corr(method='pearson')
    a = a*np.tri(len(a), k=-1)
    b = a[abs(a)>=th]
    display(b.loc[b.notnull().any(axis=1), b.notnull().any(axis=0)])
    

def create_INS_ID_LIST_TRAIN(df, seed):
    INS_ID_LIST_TRAIN = df['installation_id'].unique()
    INS_ID_LIST_TRAIN.sort()
    random.Random(seed).shuffle(INS_ID_LIST_TRAIN)
    return INS_ID_LIST_TRAIN


class norm_opt_th(object):
    def __init__(self, seed_avg_times, target, target_weight, trans_type='average'):
        self.seed_avg_times = seed_avg_times
        self.trans_type     = trans_type
        self.target         = target
        self.target_weight  = target_weight
        
        self.norm_valid_th  = [0]+VALID_TH+[3]
        self.th_lists       = np.zeros((5, seed_avg_times))
        self.min_max        = np.zeros((4, 2, seed_avg_times))
        self.opt_th         = np.zeros(3)
        self.preds          = np.zeros((17690, seed_avg_times))
        self.fit_flg        = False
        
    def fit(self, preds, th_lists):
        if preds.shape[1]!=self.seed_avg_times:
            raise ValueError(f'Input: {preds.shape[1]} preds != Expected: {self.seed_avg_times} preds')
            
        self.preds = preds
        
        for sa_id in range(self.seed_avg_times):
            ##### Threshold information #####
            oof_th = [-np.inf]+th_lists[:, sa_id].tolist()+[np.inf]
            self.th_lists[:, sa_id] = np.array(oof_th)

            ##### Normalize information #####
            min_max = []
            for i in range(4):
                min_max.append([oof_th[i], oof_th[i+1]])
            min_max[0][0], min_max[4-1][1] = preds[:, sa_id].min(), preds[:, sa_id].max()
            self.min_max[:,:,sa_id] = np.array(min_max)
            
        self.fit_flg = True
            
    def transform(self, preds, weights=None):
        ##### Raise error #####
        if preds.shape[1]!=self.seed_avg_times:
            raise ValueError(f'Input: {preds.shape[1]} preds != Expected: {self.seed_avg_times} preds')
            
        if not self.fit_flg:
            raise ValueError('Has not been fit to OOF!')
        #######################
        if weights==None:
            weights=np.ones(self.seed_avg_times)
        
        if self.trans_type=='average':
            out_preds = preds
            return np.average(out_preds, axis=1, weights=weights)
        
        elif self.trans_type=='norm_th':
            out_preds = np.zeros(preds.shape)

            for sa_id in range(self.seed_avg_times):
                oof = pd.Series(preds[:, sa_id].copy(), index=np.arange(len(preds[:, sa_id])))

                for i in range(4):
                    oof_tmp = oof[(self.th_lists[i,sa_id]<=oof) & (oof<=self.th_lists[i+1,sa_id])].values

                    oof[(self.th_lists[i,sa_id]<=oof) & (oof<=self.th_lists[i+1,sa_id])] = \
                        (oof_tmp-self.min_max[i,0,sa_id])/(self.min_max[i,1,sa_id]-self.min_max[i,0,sa_id]) * \
                        (self.norm_valid_th[i+1]-self.norm_valid_th[i]) + self.norm_valid_th[i]

                out_preds[:, sa_id] = oof.values

            return np.average(out_preds, axis=1, weights=weights)
