import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, cohen_kappa_score, average_precision_score
from scipy import stats
import torch.nn.functional as F

def ensemble_results(df, stage, metric, meta_data_dir, get_oarsi_results):

    res, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df, metric)

    print('Spearman rank correlation coeffient of stage {}'.format(res))
    print('OA AUC is {}'.format(auc_mid))
    print('Severe AUC is {}'.format(auc_sev))


def print_ensemble_results(path_to_anom_scores, epoch, stage, metric, meta_data_dir, get_oarsi_results, model_name_prefix ):
    """
    Prints evaluation results for an ensemble of models based on their anomaly scores.

    Parameters:
    ----------
    path_to_anom_scores : str
        Path to the directory containing score CSVs (e.g., "outputs/dfs/ss/").
        Each CSV should contain per-image rows with fields such as:
        'id', 'label', 'norm_min', 'max_scores', 'mean_scores', 
        'mean_scores_min', 'max_scores_min', 'norm_min_max', 
        'mean_scores_max', 'norm_min_mean', 'max_scores_mean', 
        'centre_min', 'centre_max', 'centre_mean', 'binary_label'.

    epoch : int or dict
        - If `int`: evaluates all model outputs corresponding to that epoch.
        - If `dict`: maps seeds to their best epoch (e.g., `{1001: 30, 71530: 40}`),
          and selects the corresponding file per seed.

    stage : str
        Name of the training stage (e.g., "ss", "severe", "final").
        Used for display and optionally for selecting OARSI evaluation parameters.

    metric : str
        Name of the metric column to evaluate (e.g., 'centre_mean', 'w_centre').
        Common usage:
            - Stage "ss": uses 'centre_mean' (mean distance to reference centre).
            - Later stages: typically use 'w_centre' (weighted difference metric).

    meta_data_dir : str
        Path to the metadata file (e.g., 'meta/xxx.txt').
        Required if `get_oarsi_results=True`, otherwise unused.

    get_oarsi_results : bool
        Whether to compute OARSI-based AUC results using clinical metadata.
        (May be ignored or unused in some versions of this repo.)

    model_name_prefix : str
        Filters the score files by model name prefix (e.g., "SSL_KNEE").
        Ensures only relevant model outputs are included in the ensemble.

    Behavior:
    ---------
    - Gathers the appropriate score files based on epoch/seed info and model prefix.
    - Aggregates metric values across all selected files (averages per sample).
    - Prints ensemble-level performance, including:
        - Spearman rank correlation
        - AUCs for OA detection and severity
        - Optional OARSI AUC (if enabled)
    """
    print('---------------------------------------------------- For stage ' + stage + '----------------------------------------------------')
    print('-----------------------------RESULTS ON UNLABELLED DATA---------------------------')
    print('Warning: the results on unlabelled data includes the pseudo labels i.e. for stages that are not SSL and severe predictor, the model was trained on the psuedo labels which are also included in the unlabelled results')

    files_total = os.listdir(path_to_anom_scores)
    if isinstance(epoch, dict):
        files=[]
        for key in epoch.keys():
            files = files + [file for file in files_total if (('epoch_' + str(epoch[key]) ) in file) & ('on_test_set' not in file ) & ('seed_' + str(key) in file) & (model_name_prefix in file) ]

    else:
        files = [file for file in files_total if (('epoch_' + str(epoch) ) in file) & ('on_test_set' not in file ) & (model_name_prefix in file)]

    df = create_scores_dataframe(path_to_anom_scores, files, metric)
    ensemble_results(df, stage, metric, meta_data_dir, get_oarsi_results)

    print('-----------------------------RESULTS ON TEST SET---------------------------')
    if isinstance(epoch, dict):
        files=[]
        for key in epoch.keys():
            files = files + [file for file in files_total if (('epoch_' + str(epoch[key]) ) in file) & ('on_test_set' in file ) & ('seed_' + str(key) in file) & (model_name_prefix in file)]
    else:
        files = [file for file in files_total if (('epoch_' + str(epoch) ) in file) & ('on_test_set' in file ) & (model_name_prefix in file)]

    df = create_scores_dataframe(path_to_anom_scores, files, metric)
    ensemble_results(df, stage, metric, meta_data_dir, get_oarsi_results)


def create_scores_dataframe(path_to_anom_scores, files, metric):
    for i,file in enumerate(files):
        if i ==0:
            df = pd.read_csv(path_to_anom_scores + file)
            df = df.sort_values(by='id').reset_index(drop=True)[['id','label', metric]]
        else:
            sc = pd.read_csv(path_to_anom_scores + file)
            sc = sc.sort_values(by='id').reset_index(drop=True)[['id','label', metric]]
            df.iloc[:,2:] = df.iloc[:,2:] + sc.iloc[:,2:]

    df.iloc[:,2:] = df.iloc[:,2:] / len(files)
    return df

def get_metrics(df, score):

    res = stats.spearmanr(df[score].tolist(), df['label'].tolist())

    df['binary_label'] = 0
    df.loc[df['label'] > 0, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df['label'] > 1, 'binary_label'] = 1

    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df['label'] > 2, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid2 = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df['label'] == 4, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_sev = metrics.auc(fpr, tpr)



    return res[0], auc, auc_mid, auc_mid2, auc_sev