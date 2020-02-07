import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import gc
from config.config import *
from utils.helper import read_pickle, to_pickle

## Drop features
def create_drop_col_list(score_df_01dir, type_01='gain>0', score_df_02dir=None, type_02='gain>0', cutoff_01=None, cutoff_02=None):
    drop_col_list = []
    drop_col_list_ni01 = []
    drop_col_list_ni02 = []
    
    scores_df = pd.read_csv(score_df_01dir+'/scores_df.csv')
    if type_01=='cutoff':
        drop_col_list_ni01 = scores_df.loc[int(cutoff_01*1.2):, 'feature'].tolist()
    elif type_01=='gain>0':
        drop_col_list_ni01 = scores_df.loc[scores_df['gain_score']<=0, 'feature'].tolist()

    if score_df_02dir!=None:
        scores_df = pd.read_csv(score_df_02dir+'/scores_df.csv')
        if type_02=='cutoff':
            drop_col_list_ni02 = scores_df.loc[int(cutoff_02):, 'feature'].tolist()
        elif type_02=='gain>0':
            drop_col_list_ni02 = scores_df.loc[scores_df['gain_score']<=0, 'feature'].tolist()
            
    drop_col_list = list(set(drop_col_list+drop_col_list_ni01+drop_col_list_ni02))
        
    return drop_col_list


## Import X
FE_dir_c = '/kaggle/input/dsb2019-fe-c08'
pkl_name = 'FE_LGB_c02-07_train.pkl'

X = read_pickle(FE_dir_c+'/'+pkl_name)

print(X.head())
print(f'X.shape\t:{X.shape}')


## Drop features depending on null-importance culc. in FE-c08
score_df_01dir = '/kaggle/input/dsb2019-lgb-fs-c08-ni01-03'
score_df_02dir = '/kaggle/input/dsb2019-lgb-fs-c08-ni02-03'
cutoff_01 = 225
cutoff_02 = 180
type_01 = 'cutoff'
type_02 = 'cutoff'

drop_col_list = create_drop_col_list(score_df_01dir, type_01, score_df_02dir, type_02, cutoff_01, cutoff_02)
X.drop(columns=drop_col_list, inplace=True, errors='ignore')


## Drop features depending on adversarial validation culc. in FE-c08-01-adv
LGB_ADV_FEAT = pd.read_csv('/kaggle/input/dsb2019-lgb-c08-01-adv/scores_df.csv')
LGB_ADV_FEAT = LGB_ADV_FEAT.loc[LGB_ADV_FEAT['gain_score']>=2.0, 'feature'].tolist()

X.drop(columns=LGB_ADV_FEAT, inplace=True, errors='ignore')

print(X.head())
print(f'X.shape\t:{X.shape}')

cols_df = pd.DataFrame({'col_name':X.columns.tolist()})
cols_df.to_csv(OUT_DIR+'/cols_df.csv')

to_pickle(X, OUT_DIR+'/X.pkl')
del X; gc.collect()
