import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process.processing_01 import data_prep
from config.config import *
from utils.helper import read_pickle, to_pickle

train_df, y = data_prep(is_train=True)

to_pickle(train_df, OUT_DIR+'/train_df.pkl')
to_pickle(y, OUT_DIR+'/y.pkl')

del train_df, y; gc.collect()
