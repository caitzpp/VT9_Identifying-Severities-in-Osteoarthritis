import pandas as pd
import os

import config

from utils.evaluation_utils import get_best_epoch

PATH_TO_RESULTS = config.PATH_TO_RESULTS
print(PATH_TO_RESULTS)

path_to_results = os.path.join(PATH_TO_RESULTS, 'ss')
metrics = ['auc', 'auc_mid', 'auc_mid2', 'auc_sev', 'spearman']


# best_epochs = 
get_best_epoch(path_to_results, last_epoch=0, metric='auc', model_prefix='mod_1')