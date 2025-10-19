import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, f1_score, recall_score, average_precision_score, normalized_mutual_info_score, v_measure_score
from scipy import stats
import torch.nn.functional as F
from scipy.ndimage.filters import uniform_filter1d
from scipy.stats import entropy

def get_best_epoch(path_to_centre_dists, last_epoch, metric, model_prefix, test_data=True):
    '''
    Find the best epoch (by metric value) per seed across multiple files.
    '''
    files = os.listdir(path_to_centre_dists)
    files = [
        f for f in files
        if ('on_test_set' in f if test_data else 'on_test_set' not in f) and model_prefix in f
    ]

    best_epochs = {}
    top_results = []

    for file in files:
        df = pd.read_csv(os.path.join(path_to_centre_dists, file))
        if len(df) == 0 or metric not in df.columns:
            continue

        seed = file.split('_seed_')[1].split('_')[0]

        best_idx = df[metric].idxmax()  # or idxmin() for loss
        best_metric = df.loc[best_idx, metric]
        best_epoch_relative = df.loc[best_idx, 'epoch'] if 'epoch' in df.columns else best_idx * 10
        best_epoch_absolute = best_epoch_relative + (last_epoch[seed] if isinstance(last_epoch, dict) else last_epoch)

        # If this is the first file for the seed or has a better metric, store it
        if (seed not in best_epochs) or (best_metric > best_epochs[seed][metric]):
            best_epochs[seed] = {
                'epoch': best_epoch_absolute,
                metric: best_metric,
            }

    # Build top_results from filtered best_epochs
    for seed, result in best_epochs.items():
        top_results.append((seed, result['epoch'], result[metric]))

    # Sort top results by metric descending
    top_results.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 results:")
    for seed, epoch, value in top_results[:10]:
        print(f"Seed: {seed}, Best Epoch: {epoch}, {metric}: {value:.4f}")

    return best_epochs

def ensemble_results(df, stage, metric):

    res, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df, metric)

    print('Spearman rank correlation coeffient of stage {}'.format(res))
    print('OA AUC is {}'.format(auc_mid))
    print('Severe AUC is {}'.format(auc_sev))


def print_ensemble_results(path_to_anom_scores, epoch, stage, metric, meta_data_dir, get_oarsi_results, model_name_prefix, seed = None):
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
    elif seed is not None:
        files = [file for file in files_total if (('seed_' + str(seed) ) in file) & ('on_test_set' not in file ) & (model_name_prefix in file)]
    else:
        files = [file for file in files_total if (('epoch_' + str(epoch) ) in file) & ('on_test_set' not in file ) & (model_name_prefix in file)]

    df = create_scores_dataframe(path_to_anom_scores, files, metric)
    ensemble_results(df, stage, metric, meta_data_dir, get_oarsi_results)

    print('-----------------------------RESULTS ON TEST SET---------------------------')
    if isinstance(epoch, dict):
        files=[]
        for key in epoch.keys():
            files = files + [file for file in files_total if (('epoch_' + str(epoch[key]) ) in file) & ('on_test_set' in file ) & ('seed_' + str(key) in file) & (model_name_prefix in file)]
    elif seed is not None:
        files = [file for file in files_total if (('seed_' + str(seed) ) in file) & ('on_test_set' in file ) & (model_name_prefix in file)]
    else:
        files = [file for file in files_total if (('epoch_' + str(epoch) ) in file) & ('on_test_set' in file ) & (model_name_prefix in file)]

    df = create_scores_dataframe(path_to_anom_scores, files, metric)
    ensemble_results(df, stage, metric, meta_data_dir, get_oarsi_results)

def get_results(path_to_results, epoch, stage, metric, model_name_prefix, seed=None):
    files_total = os.listdir(path_to_results)

    if isinstance(epoch, dict):
        files=[]
        for key in epoch.keys():
            files = files + [file for file in files_total if (('epoch_' + str(epoch[key]) ) in file) & ('on_test_set' not in file ) & ('seed_' + str(key) in file) & (model_name_prefix in file) ]
    elif seed is not None:
        files = [file for file in files_total if (('seed_' + str(seed) ) in file) & ('on_test_set' not in file ) & (model_name_prefix in file)]
    else:
        files = [file for file in files_total if (('epoch_' + str(epoch) ) in file) & ('on_test_set' not in file ) & (model_name_prefix in file)]

    #TODO: Function that takes all files and calculates AUC etc + std deviation.

    means, stdv = calc_mean_std(files, path_to_results, metric)


def calc_mean_std(files, path_to_results, metric):
    df_all = []
    for i, file in enumerate(files):
        sc = pd.read_csv(os.path.join(path_to_results, file))
        filtered_df = sc[sc['Unnamed: 0']==metric]
        df_all.append(filtered_df)
    
    df = pd.concat(df_all, ignore_index=True)
    means = df.iloc[:, 1:].mean()
    stds = df.iloc[:, 1:].std()

    return means, stds

def create_scores_dataframe(path_to_anom_scores, files, metric):
    """
    Creates a DataFrame by aggregating metric values (e.g., 'centre_mean') from multiple CSV files.

    This function reads the specified CSV files, each containing anomaly scores, and extracts the
    'id', 'label', and the specified metric (e.g., 'centre_mean'). It then combines the data from
    all files, averages the metric values for each 'id' across all files, and returns the resulting
    DataFrame.

    The returned DataFrame will contain the following columns:
        - 'id': Unique identifier for each row.
        - 'label': A binary or categorical label (e.g., 0 or 1).
        - 'metric': The average value of the specified metric (e.g., 'centre_mean') across all files.

    The function ensures that the data from all files is properly aligned by sorting based on the 'id'
    column and resetting the index. After the data from each file is aggregated, the metric values are
    averaged by dividing the sum by the number of files.

    Parameters:
    ----------
    path_to_anom_scores : str
        Path to the directory containing the anomaly score CSV files.
    
    files : list of str
        A list of filenames (strings) corresponding to the CSV files to be processed.
    
    metric : str
        The name of the metric column to aggregate (e.g., 'centre_mean', 'w_centre').

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the averaged metric values across all files for each 'id'.
        The DataFrame includes the 'id', 'label', and the averaged 'metric' values.
    """
    for i,file in enumerate(files):
        if i ==0:
            df = pd.read_csv(os.path.join(path_to_anom_scores, file))
            df = df.sort_values(by='id').reset_index(drop=True)[['id','label', metric]]
        else:
            sc = pd.read_csv(os.path.join(path_to_anom_scores, file))
            sc = sc.sort_values(by='id').reset_index(drop=True)[['id','label', metric]]
            df.iloc[:,2:] = df.iloc[:,2:] + sc.iloc[:,2:]

    df.iloc[:,2:] = df.iloc[:,2:] / len(files)
    return df

def get_metrics(df, score, label = 'KL-Score'):
    #'label' is ground truth

    res = stats.spearmanr(df[score].tolist(), df[label].tolist())

    df['binary_label'] = 0
    df.loc[df[label] > 0, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df[label] > 1, 'binary_label'] = 1

    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df[label] > 2, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid2 = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df[label] == 4, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_sev = metrics.auc(fpr, tpr)



    return res[0], auc, auc_mid, auc_mid2, auc_sev

def majority_vote(df, cluster_col, feature_col):
    clusters = df[cluster_col].unique()
    clusters.sort()

    results = pd.DataFrame({cluster_col: clusters})

    for feature in feature_col:
        majority_vote = df.groupby(cluster_col)[feature].agg(lambda x: list(x.mode()))
        majority_vote = pd.DataFrame(majority_vote).reset_index()
        majority_vote = majority_vote.rename(columns={feature: f'MV_{feature}'})
        results = results.merge(majority_vote, on = cluster_col, how = 'left')

    return results.dropna(axis=0, how='all')

def handle_modes(x, id_):
    if len(x) == 1:
        try:
            return float(x[0])
        except Exception as e:
            print(f"Conversion error for id={id_}: {e}")
            return None
    else:
        return float(0.5) 

def get_metrics_external(df, externalcol, chadjusted, label = 'cluster_label'):
    results = {}
    
    maj_vote = majority_vote(df, label, externalcol)

    for i in range(len(externalcol)):
        col_name = f'MV_{externalcol[i]}'
        maj_vote[col_name] = [
            handle_modes(row[col_name], row[label])
            for _, row in maj_vote.iterrows()
        ]

        df = df.merge(maj_vote[[label, col_name]], on=label, how='left')

        col = externalcol[i] 
        
        y_true = df[col]
        y_pred = df[col_name]

        res = stats.spearmanr(df[label].tolist(), y_true.tolist())
        nmi = normalized_mutual_info_score(df[label], y_true)
        vmeasure = v_measure_score(df[label], y_true)
        precision=precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        res_mv = stats.spearmanr(y_pred, y_true)
        nmi_mv = normalized_mutual_info_score(y_pred, y_true)
        vmeasure_mv = v_measure_score(y_pred, y_true)

        col_results = {
            'spearman_clusterlabel' + '_' + str(col): res[0],
            'nmi_clusterlabel' + '_' + str(col): nmi,
            'vmeasure_clusterlabel' + '_' + str(col): vmeasure,
            'spearman_MV' + '_' + str(col): res_mv[0],
            'nmi_MV' + '_' + str(col): nmi_mv,
            'vmeasure_MV' + '_' + str(col): vmeasure_mv,
            'precision' + '_' + str(col): precision,
            'recall' + '_' + str(col): recall,
            'f1_score' + '_' + str(col): f1
        }

        results.update(col_results)

    return results


def create_scores_dataframe(path_to_anom_scores, files, metric):
    for i,file in enumerate(files):
        if i ==0:
            df = pd.read_csv(os.path.join(path_to_anom_scores, file))
            df = df.sort_values(by='id').reset_index(drop=True)[['id','label', metric]]
        else:
            sc = pd.read_csv(os.path.join(path_to_anom_scores, file))
            sc = sc.sort_values(by='id').reset_index(drop=True)[['id','label', metric]]
            df.iloc[:,2:] = df.iloc[:,2:] + sc.iloc[:,2:]

    df.iloc[:,2:] = df.iloc[:,2:] / len(files)
    return df

def create_scores_stats(path_to_anom_scores, files, metric):
    """
    Reads all files, aligns them by 'id' and 'label', and returns a DataFrame
    with columns:
      - id, label
      - mean_<metric>
      - std_<metric>
    """
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(path_to_anom_scores, file))
        df = (
            df.sort_values(by='id')
              .reset_index(drop=True)
              [['id', 'label', metric]]
        )
        dfs.append(df)

    # start from the first run, but rename its metric column to metric_0
    base = dfs[0].copy().rename(columns={ metric: f"{metric}_0" })

    # add each subsequent run's scores as its own column metric_1, metric_2, ...
    for i, df_i in enumerate(dfs[1:], start=1):
        base[f'{metric}_{i}'] = df_i[metric]

    # now all metric_X columns exist
    metric_cols = [f'{metric}_{i}' for i in range(len(dfs))]
    base[f'mean_{metric}'] = base[metric_cols].mean(axis=1)
    base[f'std_{metric}']  = base[metric_cols].std(axis=1)

    return base[['id', 'label', f'mean_{metric}', f'std_{metric}']]

def get_threshold(path_to_anom_scores_sev, epoch_sev, metric):
    files_total = os.listdir(path_to_anom_scores_sev)
    files = [
        f for f in files_total
        if f'epoch_{epoch_sev}' in f and 'on_test_set' in f
    ]

    stats = create_scores_stats(path_to_anom_scores_sev, files, metric)
    # normalize the MEAN score before thresholding
    stats[f'mean_{metric}'] = (stats[f'mean_{metric}'] + 2) / 4
    threshold = np.percentile(stats[f'mean_{metric}'], 95)
    return threshold

# def get_threshold(path_to_anom_scores_sev, epoch_sev, metric):
#     # (same as before)
#     files = [
#         f for f in os.listdir(path_to_anom_scores_sev)
#         if f'epoch_{epoch_sev}' in f and 'on_test_set' in f
#     ]
#     stats = create_scores_stats(path_to_anom_scores_sev, files, metric)
#     stats[f'mean_{metric}'] = (stats[f'mean_{metric}'] + 2) / 4
#     return np.percentile(stats[f'mean_{metric}'], 95)


def combine_results(
    path_to_anom_scores_oa,
    path_to_anom_scores_sev,
    epoch_oa,
    epoch_sev,
    metric,
    model_name_prefix
):
    # get your severity threshold from the SEV runs
    threshold = get_threshold(path_to_anom_scores_sev, epoch_sev, metric)

    # collect OA files across all seeds
    files_total = os.listdir(path_to_anom_scores_oa)
    oa_files = []
    for seed, ep in epoch_oa.items():
        oa_files += [
            f for f in files_total
            if f'epoch_{ep}' in f
           and 'on_test_set' in f
           and f'seed_{seed}' in f
           and model_name_prefix in f
        ]

    # build OA stats (mean and std)
    oa_stats = create_scores_stats(path_to_anom_scores_oa, oa_files, metric)
    # normalize OA mean
    oa_stats['comb_score'] = (oa_stats[f'mean_{metric}'] + 2) / 4

    # build SEV stats (mean and std)
    sev_files = [
        f for f in os.listdir(path_to_anom_scores_sev)
        if f'epoch_{epoch_sev}' in f and 'on_test_set' in f
    ]
    sev_stats = create_scores_stats(path_to_anom_scores_sev, sev_files, metric)
    sev_stats['comb_score'] = (sev_stats[f'mean_{metric}'] + 2) / 4

    # align and apply threshold: wherever sev > thresh, bump OA score
    oa_stats = oa_stats.sort_values('id').reset_index(drop=True)
    sev_stats = sev_stats.sort_values('id').reset_index(drop=True)

    mask = sev_stats['comb_score'] > threshold
    oa_stats.loc[mask, 'comb_score'] = 1 + sev_stats.loc[mask, 'comb_score']

    # for clarity, rename the std columns
    oa_stats.rename(
        columns={f'std_{metric}': 'std_oa_' + metric},
        inplace=True
    )
    sev_stats.rename(
        columns={f'std_{metric}': 'std_sev_' + metric},
        inplace=True
    )

    # merge OA and SEV stds into one table if you like
    combined = oa_stats.merge(
        sev_stats[['id', 'std_sev_' + metric]],
        on='id', how='left'
    )

    print('----------------------------------------------------')
    print('    RESULTS ON TEST SET — Final, combined')
    print('----------------------------------------------------')
    # this assumes ensemble_results knows to look for 'comb_score'
    ensemble_results(combined, 'Final, combined', 'comb_score')

    # and if you want to inspect the stds:
    print('\nPer‐ID standard deviations:')
    print(combined[['id', 'std_oa_' + metric, 'std_sev_' + metric]].head())

    return combined


def combine_results_with_std(
    path_to_anom_scores_oa,
    path_to_anom_scores_sev,
    epoch_oa,
    epoch_sev,
    metric,
    model_name_prefix
):
    # 1) compute the severity threshold once
    threshold = get_threshold(path_to_anom_scores_sev, epoch_sev, metric)

    # 2) pre-compute the SEV-side “bump” (it's the same for all OA seeds)
    sev_files = [
        f for f in os.listdir(path_to_anom_scores_sev)
        if f'epoch_{epoch_sev}' in f and 'on_test_set' in f
    ]
    sev_stats = create_scores_stats(path_to_anom_scores_sev, sev_files, metric)
    sev_stats['comb_score_sev'] = (sev_stats[f'mean_{metric}'] + 2) / 4
    sev_stats = sev_stats.sort_values('id').reset_index(drop=True)

    # 3) now loop over each OA seed and compute its metrics
    results = []
    for seed, ep in epoch_oa.items():
        # pick only that seed’s OA files
        files_total = os.listdir(path_to_anom_scores_oa)
        oa_files = [
            f for f in files_total
            if f'epoch_{ep}' in f
            and 'on_test_set' in f
            and f'seed_{seed}' in f
            and model_name_prefix in f
        ]
        oa_stats = create_scores_stats(path_to_anom_scores_oa, oa_files, metric)
        oa_stats['comb_score'] = (oa_stats[f'mean_{metric}'] + 2) / 4

        # apply the severity bump wherever sev > threshold
        oa_stats = oa_stats.sort_values('id').reset_index(drop=True)
        mask = sev_stats['comb_score_sev'] > threshold
        oa_stats.loc[mask, 'comb_score'] = 1 + sev_stats.loc[mask, 'comb_score_sev']

        # now compute your three metrics:
        # — Spearman between the true label (0/1/2) and comb_score
        # — OA‐vs‐normal AUC: label>0
        # — severe‐vs‐nonsevere AUC: label==2
        y_true = oa_stats['label']
        y_oa   = (y_true > 0).astype(int)
        y_sev  = (y_true == 2).astype(int)
        cs     = oa_stats['comb_score']

        sp     = stats.spearmanr(y_true, cs).correlation
        auc_oa = roc_auc_score(y_oa, cs)
        auc_sev= roc_auc_score(y_sev, cs)

        results.append({
            'seed': seed,
            'spearman': sp,
            'oa_auc':   auc_oa,
            'sev_auc':  auc_sev
        })

        #print(f"Seed {seed:<2} →  Spearman: {sp:.3f}   OA AUC: {auc_oa:.3f}   Sev AUC: {auc_sev:.3f}")

    # 4) aggregate across seeds
    dfm = pd.DataFrame(results).set_index('seed')
    means = dfm.mean()
    stds  = dfm.std()

    print('\nOverall performance across seeds:')
    print(f" • Spearman rank: {means['spearman']:.3f}  ±  {stds['spearman']:.3f}")
    print(f" • OA AUC:         {means['oa_auc']:.3f}  ±  {stds['oa_auc']:.3f}")
    print(f" • Severe AUC:     {means['sev_auc']:.3f}  ±  {stds['sev_auc']:.3f}")

    return dfm, means, stds



def normalized_entropy(p):
    p = np.array(p)
    p = p / p.sum()  # normalize to probability distribution
    raw_entropy = entropy(p, base=2)
    max_entropy = np.log2(len(p)) if len(p) > 1 else 1  # avoid log2(1) = 0
    return raw_entropy / max_entropy


def ch_externalval_score(CHadjusted, auc):
    if auc > 0.5:
        return CHadjusted * (auc - 0.5) * 2
    else:
        metric = 0
        return metric