from utils.evaluation_utils import calc_mean_std
import config
import os
import pandas as pd

stage = 'ss'

DATA_PATH = os.path.join(config.PATH_TO_RESULTS, stage)
SAVE_PATH = os.path.join(config.OUTPUT_PATH, "results", stage)
PATH_TO_ANOM_SCORES = os.path.join(config.PATH_TO_ANOM, stage)

TRAIN_PLATEAU_EPOCH = 400

MOD_NAME="mod_st"
metric= 'centre_mean'
seeds = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]


if __name__=="__main__":
    files_total = os.listdir(DATA_PATH)
    # #files_total = os.listdir(PATH_TO_ANOM_SCORES)

    print('---------------------------------------------------- For stage ' + stage + '----------------------------------------------------')
    print('-----------------------------RESULTS ON UNLABELLED DATA---------------------------')
    print('Warning: the results on unlabelled data includes the pseudo labels i.e. for stages that are not SSL and severe predictor, the model was trained on the psuedo labels which are also included in the unlabelled results')

    metrics_per_seed = []
    
    files = [file for file in files_total if (('epoch_' + str(TRAIN_PLATEAU_EPOCH) ) in file) & ('on_test_set' not in file ) & (MOD_NAME in file)]

    means, stds = calc_mean_std(files, DATA_PATH, metric)

    spearman = means['spearman']
    auc_mid = means['auc_mid']
    auc_sev = means['auc_sev']

    spearman_std = stds['spearman']
    auc_mid_std = stds['auc_mid']
    auc_sev_std = stds['auc_sev']
#ss_training_mod_1_bs_1_task_test_lr_1e-06_N_30_seed_34_epoch_0
    file_name = f'{stage}_training_{MOD_NAME}_mean'
    means.to_csv(os.path.join(SAVE_PATH, file_name))
    file_name = f'{stage}_training_{MOD_NAME}_std'
    stds.to_csv(os.path.join(SAVE_PATH, file_name))
    print(f"Spearman rank correlation coefficient: {spearman} with std {spearman_std}")
    print(f"OA AUC is {auc_mid}, w/ std {auc_mid_std}")
    print(f'Severe AUC is {auc_sev}, w/ std {auc_sev_std}')

    print('-----------------------------RESULTS ON TEST SET---------------------------')
    metrics_per_seed = []
    
    files = [file for file in files_total if (('epoch_' + str(TRAIN_PLATEAU_EPOCH) ) in file) & ('on_test_set' in file ) & (MOD_NAME in file)]

    means, stds = calc_mean_std(files, DATA_PATH, metric)

    spearman = means['spearman']
    auc_mid = means['auc_mid']
    auc_sev = means['auc_sev']

    spearman_std = stds['spearman']
    auc_mid_std = stds['auc_mid']
    auc_sev_std = stds['auc_sev']

    file_name = f'{stage}_training_{MOD_NAME}_on_test_set_mean'
    means.to_csv(os.path.join(SAVE_PATH, file_name))
    file_name = f'{stage}_training_{MOD_NAME}_on_test_set_std'
    stds.to_csv(os.path.join(SAVE_PATH, file_name))

    print(f"Spearman rank correlation coefficient: {spearman} with std {spearman_std}")
    print(f"OA AUC is {auc_mid}, w/ std {auc_mid_std}")
    print(f'Severe AUC is {auc_sev}, w/ std {auc_sev_std}')
