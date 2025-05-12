import pandas as pd
import os
import json

import config

from utils.evaluation_utils import get_best_epoch

PATH_TO_RESULTS = config.PATH_TO_RESULTS
SAVE_PATH = os.path.join('outputs', 'results')
model_prefix = 'mod_1'

path_to_results = os.path.join(PATH_TO_RESULTS, 'ss')
metrics = ['auc', 'auc_mid', 'auc_mid2', 'auc_sev', 'spearman']

best_epochs_d = {}

for metric in metrics:
    best_epochs = get_best_epoch(path_to_results, last_epoch=0, metric=metric, model_prefix=model_prefix)
    best_epochs_d[metric] = best_epochs

with open(os.path.join(SAVE_PATH, f'best_epochs_{model_prefix}.json'), 'w', encoding='utf-8') as f:
    json.dump(best_epochs_d, f, ensure_ascii=False, indent=4)