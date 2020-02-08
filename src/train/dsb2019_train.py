import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pandas as pd
import gc
import time
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import lightgbm as lgb

from config.config import *
from utils.helper import read_pickle, to_pickle
from utils.helper import create_INS_ID_LIST_TRAIN
from utils.lgb_metrics import rmse, eval_qwk_lgb_regr, eval_qwk_lgb_regr_weight
from utils.metrics import OptimizedRounder

def train_cv(params, ins_id_list_train, shuffle=False, random_state=0, drop_cols=None, th_optimize=False):
    X = read_pickle(OUT_DIR+'/X.pkl')
    y = read_pickle(OUT_DIR+'/y.pkl')

    oof_valid = pd.Series(np.zeros(len(X), dtype=float), index=X.index, name='accuracy_group')

    len_df = len(ins_id_list_train)
    seq_len_df = np.arange(len_df)
    len_each = math.ceil(len_df/FOLDS)

    trn_scores = []
    val_scores = []
    best_iter = []
    clfs = []
    importances = pd.DataFrame()
    iter_scores = pd.DataFrame()

    for f in range(FOLDS):
        print(f'{"*"*5} Fold {f} {"*"*5}')
        evals_result                  = {}
        counter                       = 0
        params['lgb_params']['seed'] += 1
        
        print(f'\r                                                               \r', end='',flush=True)
        ### Split ##########
        s, e = f*(len_each), (f+1)*len_each

        trn_ = ~((seq_len_df>=s)&(seq_len_df<e))
        val_ =  ((seq_len_df>=s)&(seq_len_df<e))

        trn_x, trn_y = X.loc[ins_id_list_train[trn_], :].copy(), y.loc[ins_id_list_train[trn_], :].copy()
        val_x, val_y = X.loc[ins_id_list_train[val_], :].copy(), y.loc[ins_id_list_train[val_], :].copy()

        if drop_cols!=None:
            trn_x.drop(columns=drop_cols, inplace=True)
            val_x.drop(columns=drop_cols, inplace=True)

        if shuffle:
            trn_y['accuracy_group'] = trn_y.sample(frac=1, random_state=random_state).values
            val_y['accuracy_group'] = val_y.sample(frac=1, random_state=random_state).values
        #####################

        
#         ##### For log target #####
#         trn_y[['accuracy_group']] = np.log(trn_y[['accuracy_group']]+1)
#         val_y[['accuracy_group']] = np.log(val_y[['accuracy_group']]+1)
#         ##########################
        
        ##### LightGBM dataset #####
        dtrain = lgb.Dataset(data=trn_x, label=trn_y[['accuracy_group']], weight=trn_y['sample_weight'], free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=val_x, label=val_y[['accuracy_group']], weight=val_y['sample_weight'], free_raw_data=False, silent=True, reference=dtrain)
        #############################
        
        ##### Custom metric with weight #####
        eqlrw = eval_qwk_lgb_regr_weight(trn_y['sample_weight'].values, val_y['sample_weight'].values)
        def lgb_metrics_weight(preds, data):
            return [eqlrw(preds, data), rmse(preds, data),]
        #####################################

        print(f'Shape of trn_x:\t{trn_x.shape}')
        
#         reducelr_cb = ReduceLearningRateCallback(monitor_metric='kappa', reduce_every=40, ratio=0.5)
#         callbacks = [reducelr_cb]
    
        params['lgb_params']['seed'] +=1
        clf = lgb.train(
            params                =params['lgb_params'],
            train_set             =dtrain,
            num_boost_round       =params['n_estimators'],
            valid_sets            =[dvalid, dtrain],
            valid_names           =['valid', 'train'],
            feval                 =lgb_metrics_weight,
            init_model            =None,
            early_stopping_rounds =params['early_stopping_rounds'],
            evals_result          =evals_result,
            verbose_eval          =params['lgb_params']['verbose'],
            keep_training_booster =True,
#             callbacks             =callbacks,
        )
        clf.save_model(OUT_DIR+'/LGB_'+'{:02}_'.format(params['seed_avg_times'])+'{:02}'.format(f)+'.txt')
        
        clfs.append(clf)
        best_iter.append(clf.best_iteration)

        oof_valid_tmp = clf.predict(dvalid.data)
        
#         ##### For log target #####
#         oof_valid_tmp = np.exp(oof_valid_tmp)-1
#         ##########################
        
        oof_valid.loc[ins_id_list_train[val_], :] = oof_valid_tmp

        t = np.array(evals_result['train']['kappa'])
        v = np.array(evals_result['valid']['kappa'])
        trn_scores.append(t[v.argmax()])
        val_scores.append(v.max())

        imp_df = pd.DataFrame({
                'feature': trn_x.columns,
                'gain': clf.feature_importance(importance_type='gain'),
                'fold': [f] * len(trn_x.columns),
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False).reset_index(drop=True)

        iter_scores_tmp = pd.DataFrame()
        for k in ['valid', 'train']:
            tmp = pd.DataFrame()
            tmp['loss'] = evals_result[k]['rmse']
            tmp['score'] = evals_result[k]['kappa']
            tmp['trn_val'] = k
            iter_scores_tmp = pd.concat([iter_scores_tmp, tmp], axis=0)
        iter_scores_tmp = iter_scores_tmp.reset_index().rename(columns={'index': 'iteration'})
        iter_scores_tmp['fold'] = f
        iter_scores_tmp['id'] = 'trn_val_'+str(f)
        iter_scores = pd.concat([iter_scores, iter_scores_tmp], axis=0).reset_index(drop=True)
        print(f'\n')

    print(f'\n{"-"*10}Train{"-"*10}')
    for i in np.arange(len(trn_scores)):
        print(f'\tFold {i}:\t{trn_scores[i]:.5f}')
    print(f'Average train:\t{np.average(trn_scores):.5f}')
    print(f'\n')

    print(f'\n{"-"*10}Validation{"-"*10}')
    for i in np.arange(len(val_scores)):
        print(f'\tFold {i}:\t{val_scores[i]:.5f}')
    score_valid = eval_qwk_lgb_regr(y[['accuracy_group']].values, oof_valid.values, y['sample_weight'])[1].item()
    print(f'Average valid:\t{np.average(val_scores):.5f}')
    print(f'Score valid:\t{score_valid:.5f}')
    print(f'Threshold while training:\t{VALID_TH}')

    optR, oof_opt_valid = None, None
    if th_optimize:
        print(f'\n{"-"*10}Optimized with validation{"-"*10}')
        optR = OptimizedRounder('nelder-mead') # 'Powell'
        optR.fit(oof_valid.values, y['accuracy_group'].values, y['sample_weight'].values)
        oof_opt_valid = optR.predict(oof_valid.values, optR.coefficients())
        score_valid = cohen_kappa_score(y['accuracy_group'], oof_opt_valid, weights='quadratic', sample_weight=y['sample_weight'])
        print(f'Optimized threshold:\t{[round(coef,3) for coef in optR.coefficients().tolist()]}')
        print(f'Score valid:\t\t{score_valid:.5f}')
#         to_pickle(optR, 'optR_'+'{:02}_'.format(params['seed_avg_times'])+'{:.5f}'.format(score_valid)+'.pkl')

    print(f'params\n{params}')
    X_idx = X.index
    
    del X, y; gc.collect()
    return clfs, iter_scores, optR, importances, (trn_scores), (val_scores, score_valid, X_idx, oof_valid, oof_opt_valid)


def train_seedavg():
    drop_cols = []

    params = {
        'n_estimators':2000,
        'early_stopping_rounds': 100,
        'lgb_params':{
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'kappa',
            'first_metric_only': True,
            'subsample': 1.0,
            'subsample_freq': 1,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'max_depth': -1,
            'lambda_l1': 1,  
            'lambda_l2': 5,
            'min_data_in_leaf': 15,
        #     'min_child_weight': 0.1,
            'verbose': -1,
        #     'importance_type': 'gain', 
        #     'eval_metric': 'kappa',
            'num_threads':4,
            'seed': SEED,
        },


        'seed_avg_times':0,
    }

    train_df = read_pickle(OUT_DIR+'/train_df.pkl')

    sa_clfs, sa_iter_scores, sa_optR, sa_importances, sa_trn_tup, sa_val_tup = [], [], [], [], [], []

    start = time.time()
    for i in range(SEED_AVG_TIMES):
        print(f'{"*"*30}\n{"*"*10} Trial {i} {"*"*10}\n{"*"*30}')

        ins_id_list_train = create_INS_ID_LIST_TRAIN(train_df, SEED+i)
        params['seed_avg_times'] = i
        clfs, iter_scores, optR, importances, trn_tup, val_tup = train_cv(params, ins_id_list_train, shuffle=False, random_state=SEED, drop_cols=drop_cols, th_optimize=True)

        for sa_, r in zip([sa_clfs, sa_iter_scores, sa_optR, sa_importances, sa_trn_tup, sa_val_tup],
                          [clfs, iter_scores, optR, importances, trn_tup, val_tup]):
            sa_.append(r)

        print(f'{"*"*5} Time:\t{(time.time()-start)/60:.1f} mins {"*"*5}')
        print(f'\n')

    del train_df; gc.collect()
    
    return sa_clfs, sa_iter_scores, sa_optR, sa_importances, sa_trn_tup, sa_val_tup
