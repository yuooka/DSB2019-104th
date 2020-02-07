import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from config.config import *


def data_prep(is_train=True):
    print(f'\n{"*"*30}\n{"*"*10} Prepare X {"*"*10}\n{"*"*30}')
    ##### Import & Cast dtypes #####
    print(TRAIN_DIR)
    df = pd.read_csv(TRAIN_DIR if is_train else TEST_DIR)
    print(f'length:\n{len(df):,}')
    
    df['event_code'] = df['event_code'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f'dtypes:\n{df.dtypes}')
    print(df.head())
    

    if is_train:
        ##### Import labels #####
        train_labels_df = pd.read_csv(TRAIN_LABELS_DIR)
        session_has_labels = list(set(train_labels_df['game_session'].values))
        print(f'Length of train_labels:\n\t{len(train_labels_df):,}')
        print(f'Unique game_session in train_labels:\n\t{len(session_has_labels):,}')
        
        ##### Extract the final assessment & Delete after the final assessment #####
        max_timestamp = df[
            (df['event_count']==1) & (df['game_session'].isin(session_has_labels))
                ].groupby('installation_id')[['timestamp']].max().reset_index()
        max_timestamp.columns = ['installation_id', 'max_timestamp']
        df['flg_non_future'] = pd.merge(df[['installation_id']], max_timestamp, on='installation_id', how='left')['max_timestamp']
        
        print(f'All rows:\n\t{len(df):,}')
        df = df[(df['flg_non_future'].notnull()) & (df['timestamp']<=df['flg_non_future'])]
        print(f'After exclude future rows:\n\t{len(df):,}')
        
    else:
        df['flg_non_future'] = 0
        
    ##### Sort records along with "installation_id" & "timestamp" #####
    df.sort_values(['installation_id','timestamp'], inplace=True)
    
    
    ##### Create "history_group" #####
    if is_train:
        df['history_group'] = 0
        df.loc[(df['event_count']==1) & (df['game_session'].isin(session_has_labels)), ['history_group']] = 1

        df.sort_values(['installation_id','timestamp'], inplace=True, ascending=[True,False])
        df['history_group'] = df.groupby(['installation_id'])['history_group'].cumsum()
        
    else:
        df['history_group'] = 1
        

    df.sort_values(['installation_id','timestamp'], inplace=True)
    print(df.head())
    
    MAX_HISTORY_GROUP = df["history_group"].max()
    print(f'Max assessment in history:\t{MAX_HISTORY_GROUP}')
    
    if is_train:
        print(f'\n{"*"*30}\n{"*"*10} Prepare y {"*"*10}\n{"*"*30}')
        y = pd.merge(
            train_labels_df[['installation_id','game_session','accuracy_group']],
            df.loc[(df['event_count']==1), ['installation_id','game_session','history_group']].drop_duplicates(),
            on=['installation_id','game_session'],
            how='left')
        y = y[['installation_id','history_group','accuracy_group']]

        y['sample_weight'] = y.groupby('installation_id').transform('count')['history_group']
        y['sample_weight'] = 1/y['sample_weight']
        y = y.set_index(['installation_id','history_group']).sort_index()

        print(y.head())
        print(f'Shape of y:\n\t{y.shape}')
        
        return df, y
    else:
        return df
