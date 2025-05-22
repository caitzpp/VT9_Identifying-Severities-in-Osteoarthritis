from utils.evaluation_utils import calc_mean_std, combine_results
import config
import os
import pandas as pd

#stage = 'ss'

# DATA_PATH = os.path.join(config.PATH_TO_RESULTS, stage)
# SAVE_PATH = os.path.join(config.OUTPUT_PATH, "results", stage)
# PATH_TO_ANOM_SCORES = os.path.join(config.PATH_TO_ANOM, stage)

DATA_PATH = config.PATH_TO_RESULTS
SAVE_PATH = config.OUTPUT_PATH
PATH_TO_ANOM_SCORES = config.PATH_TO_ANOM

stages = ['ss', 'stage2', 'stage3', 'stage_severe_pred']
# TRAIN_PLATEAU_EPOCH = 400

MOD_NAME="mod_2"

stage_dict = {
    'mod_st': {
        'ss': {
            'metric': 'centre_mean',
            'train_epoch': 400,
            'seeds': [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
        },
        'stage2': {
            'metric': 'centre_mean',
            'train_epoch': {'1001': 1390, '138647': 1090, '193': 1300, '34': 1290, '44': 1250, '71530': 1400, '875688': 1320, '8765': 1200, '985772': 1370, '244959': 1250},
            'seeds': [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]

        },
        'stage3': {
            'metric': 'w_centre',
            'train_epoch': {"1001": 2120, "138647": 1790, "193": 2060, "34": 1860, "44": 1860},
            'seeds': [ 1001, 138647, 193, 34, 44]
        },
        'stage_severe_pred': {
            'metric': 'centre_mean',
            'train_epoch': 990,
            'seeds': [1001, 138647, 193, 34, 44]
        }

    },
    'mod_2':{
     'ss': {
            'metric': 'centre_mean',
            'train_epoch': 400,
            'seeds': [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
        },
        'stage2': {
            'metric': 'centre_mean',
            'train_epoch': {"1001": 1390, "138647": 1190, "193": 800, "34": 1270, "44": 1290, "71530": 1250, "875688": 800, "8765": 800, "985772": 1320, "244959": 1380},
            'seeds': [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]

        },
        'stage3': {
            'metric': 'w_centre',
            'train_epoch': {"1001": 1990, "138647": 2110, "193": 1550, "34": 2040, "44": 2070},
            'seeds': [ 1001, 138647, 193, 34, 44]
        },
        'stage_severe_pred': {
            'metric': 'centre_mean',
            'train_epoch': 990,
            'seeds': [1001, 138647, 193, 34, 44]
        }
    }

}
# metric= 'centre_mean'
# seeds = [1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]


if __name__=="__main__":
    for stage in stages:
        data_path = os.path.join(DATA_PATH, stage)
        save_path = os.path.join(SAVE_PATH, stage)
        os.makedirs(save_path, exist_ok=True)
        path_to_anom_scores = os.path.join(PATH_TO_ANOM_SCORES, stage)

        cur_dict = stage_dict[MOD_NAME][stage]
        metric = cur_dict['metric']
        train_epoch = cur_dict['train_epoch']
        seeds = cur_dict['seeds']
        
        files_total = os.listdir(data_path)
    # #files_total = os.listdir(PATH_TO_ANOM_SCORES)

        print('---------------------------------------------------- For stage ' + stage + '----------------------------------------------------')
        print('-----------------------------RESULTS ON UNLABELLED DATA---------------------------')
        print('Warning: the results on unlabelled data includes the pseudo labels i.e. for stages that are not SSL and severe predictor, the model was trained on the psuedo labels which are also included in the unlabelled results')

        metrics_per_seed = []

        if isinstance(train_epoch, dict):
            for seed in seeds:
                files = [file for file in files_total if (('epoch_' + str(train_epoch[str(seed)]) ) in file) & ('on_test_set' not in file ) & (MOD_NAME in file) & (str(seed) in file)]
        else:
            files = [file for file in files_total if (('epoch_' + str(train_epoch) ) in file) & ('on_test_set' not in file ) & (MOD_NAME in file)]

        means, stds = calc_mean_std(files, data_path, metric)

        spearman = means['spearman']
        auc_mid = means['auc_mid']
        auc_sev = means['auc_sev']

        spearman_std = stds['spearman']
        auc_mid_std = stds['auc_mid']
        auc_sev_std = stds['auc_sev']
#ss_training_mod_1_bs_1_task_test_lr_1e-06_N_30_seed_34_epoch_0
        file_name = f'{stage}_training_{MOD_NAME}_mean'
        means.to_csv(os.path.join(save_path, file_name))
        file_name = f'{stage}_training_{MOD_NAME}_std'
        stds.to_csv(os.path.join(save_path, file_name))
        print(f"Spearman rank correlation coefficient: {spearman} with std {spearman_std}")
        print(f"OA AUC is {auc_mid}, w/ std {auc_mid_std}")
        print(f'Severe AUC is {auc_sev}, w/ std {auc_sev_std}')

        print('-----------------------------RESULTS ON TEST SET---------------------------')
        metrics_per_seed = []
        
        if isinstance(train_epoch, dict):
            for seed in seeds:
                files = [file for file in files_total if (('epoch_' + str(train_epoch[str(seed)]) ) in file) & ('on_test_set' in file ) & (MOD_NAME in file) & (str(seed) in file)]
        else:
            files = [file for file in files_total if (('epoch_' + str(train_epoch) ) in file) & ('on_test_set' in file ) & (MOD_NAME in file)]

        means, stds = calc_mean_std(files, data_path, metric)

        spearman = means['spearman']
        auc_mid = means['auc_mid']
        auc_sev = means['auc_sev']

        spearman_std = stds['spearman']
        auc_mid_std = stds['auc_mid']
        auc_sev_std = stds['auc_sev']

        file_name = f'{stage}_training_{MOD_NAME}_on_test_set_mean'
        means.to_csv(os.path.join(save_path, file_name))
        file_name = f'{stage}_training_{MOD_NAME}_on_test_set_std'
        stds.to_csv(os.path.join(save_path, file_name))

        print(f"Spearman rank correlation coefficient: {spearman} with std {spearman_std}")
        print(f"OA AUC is {auc_mid}, w/ std {auc_mid_std}")
        print(f'Severe AUC is {auc_sev}, w/ std {auc_sev_std}')
    
    stage3_path_to_anom_scores = os.path.join(PATH_TO_ANOM_SCORES, 'stage3')
    stage_severe_path_to_anom_scores = os.path.join(PATH_TO_ANOM_SCORES, 'stage_severe_pred')
    stage3_epoch = stage_dict[MOD_NAME]['stage3']['train_epoch']
    stage_severe_epoch = stage_dict[MOD_NAME]['stage_severe_pred']['train_epoch']

    combine_results(stage3_path_to_anom_scores, stage_severe_path_to_anom_scores, stage3_epoch, stage_severe_epoch, 'w_centre', model_name_prefix = MOD_NAME)
