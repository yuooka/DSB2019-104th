import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from config.config import *
from utils.helper import read_pickle, to_pickle
from utils.helper import norm_opt_th
from utils.metrics import OptimizedRounder
from utils.plot_train import save_importances, save_iter_scores, cm_show
from train.dsb2019_train import train_seedavg


###################### Training ######################
sa_clfs, sa_iter_scores, sa_optR, sa_importances, sa_trn_tup, sa_val_tup = train_seedavg()



#################### Check results ###################
## Import targets
y = read_pickle(OUT_DIR+'/y.pkl')

## Coefficients
print(f'coefficients')
for i in range(len(sa_optR)):
    print(f'{i}:\t{sa_optR[i].coefficients()}')


##### Average threshold
## Prepare input
input_preds = np.zeros((17690, SEED_AVG_TIMES))
input_ths = np.zeros((3, SEED_AVG_TIMES))
tmp = 0.0
for i in range(SEED_AVG_TIMES):
    tmp += sa_val_tup[i][1]
    input_preds[:,i] = sa_val_tup[i][3]
    input_ths[:,i] = sa_optR[i].coefficients().tolist()
    print(f'Optimized score (seed {i}):\t{sa_val_tup[i][1]:.5}')

## Fit "norm_opt_th"
norm_OT = norm_opt_th(SEED_AVG_TIMES, y['accuracy_group'].values, y['sample_weight'].values, trans_type='average')
norm_OT.fit(input_preds, input_ths)

## Transform
oof_val_all = norm_OT.transform(input_preds)

## Optimize thresholding
optR = OptimizedRounder('nelder-mead')
optR.fit(oof_val_all, norm_OT.target, norm_OT.target_weight)
oof_opt_val_all = (optR.predict(oof_val_all, optR.coefficients())).astype(int)
OOF_OPT_TH = optR.coefficients().tolist()
norm_OT.opt_th = OOF_OPT_TH
to_pickle(norm_OT, OUT_DIR+'/norm_OT.pkl')

sa_score = cohen_kappa_score(norm_OT.target, oof_opt_val_all, weights='quadratic', sample_weight=norm_OT.target_weight)

print(f'{"*"*5} Average {"*"*5}')
print(f'Ave threshold:\t{[round(coef,3) for coef in np.mean(input_ths, axis=1)]}')
print(f'Ave score (average):\t{tmp/SEED_AVG_TIMES:.5}')

print(f'{"*"*5} Optimized {"*"*5}')
print(f'Opt threshold:\t{[round(coef,3) for coef in OOF_OPT_TH]}')
print(f'Opt seed average score:\t{sa_score:.5}')

cm_show(norm_OT.target, oof_opt_val_all, norm_OT.target_weight)


############## Check learning results ################
save_iter_scores(pd.concat(sa_iter_scores, axis=0).groupby(['iteration','trn_val','fold','id']).mean().reset_index(), figsize=(10, 6))
save_importances(importances_=pd.concat(sa_importances, axis=0), figsize=(10,45))
