
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, cohen_kappa_score, average_precision_score, precision_recall_curve
import pandas as pd
import numpy as np
from scipy import stats



def get_metrics(df, score, label_name = 'KL-Score'):

    res = stats.spearmanr(df[score].tolist(), df[label_name].tolist())

    df['binary_label'] = 0
    df.loc[df[label_name] > 0, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df[label_name] > 1, 'binary_label'] = 1

    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df[label_name] > 2, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid2 = metrics.auc(fpr, tpr)


    df['binary_label'] = 0
    df.loc[df[label_name] == 4, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_sev = metrics.auc(fpr, tpr)



    return res[0], auc, auc_mid, auc_mid2, auc_sev

def get_metrics_pr(df, score, label_name='KL-Score'):
    """
    Computes Spearman correlation + PR AUCs + max F1 for each severity threshold.
    """

    # Spearman correlation
    spearman = stats.spearmanr(df[score].tolist(), df[label_name].tolist())[0]

    # Store AUC-PR and F1-max scores
    results = {}

    # Definitions for your four binary splits
    thresholds = {
        "auc_pr":  lambda x: x > 0,   # KL > 0
        "auc_mid": lambda x: x > 1,   # KL > 1
        "auc_mid2": lambda x: x > 2,  # KL > 2
        "auc_sev": lambda x: x == 4   # KL == 4
    }

    for key, condition in thresholds.items():
        # Binary label
        y_true = (condition(df[label_name])).astype(int)
        y_score = np.array(df[score])

        # Precision-recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)

        # AUC-PR
        auc_pr = auc(recall, precision)

        # Compute F1 for all thresholds
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0)

        f1_max = max(f1_scores)

        results[key] = (auc_pr, f1_max)

    # Flatten output to match your expected return structure
    return (
        spearman,
        results["auc_pr"][0],   # AUC-PR for KL>0
        results["auc_mid"][0],  # AUC-PR for KL>1
        results["auc_mid2"][0], # AUC-PR for KL>2
        results["auc_sev"][0]   # AUC-PR for KL==4
    )

def evaluate_all_as(df, as_cols, label_name='KL-Score'):

    results = {}
    for col in as_cols:
        results[col] = get_metrics(df, col, label_name=label_name)

    # Create DataFrame
    results_df = pd.DataFrame(results).T
    results_df.columns = ['spearmanr', 'auc', 'auc_mid', 'auc_mid2', 'auc_sev']

    # Calculate mean and std for each metric
    for metric in ['spearmanr', 'auc', 'auc_mid', 'auc_mid2', 'auc_sev']:
        results_df[f'{metric}_mean'] = results_df[metric].mean()
        results_df[f'{metric}_std'] = results_df[metric].std()

    return results_df

def evaluate_all_as_pr(df, as_cols, label_name='KL-Score'):

    results = {}
    for col in as_cols:
        results[col] = get_metrics_pr(df, col, label_name=label_name)

    # Create DataFrame
    results_df = pd.DataFrame(results).T
    results_df.columns = ['spearmanr', 'auc_pr', 'auc_mid', 'auc_mid2', 'auc_sev']

    # Calculate mean and std for each metric
    for metric in ['spearmanr', 'auc_pr', 'auc_mid', 'auc_mid2', 'auc_sev']:
        results_df[f'{metric}_mean'] = results_df[metric].mean()
        results_df[f'{metric}_std'] = results_df[metric].std()

    return results_df

def ensemble_results(df):
    print(f'SRC mean: {np.round(df['spearmanr_mean'].iloc[0], 3)} with std {np.round(df['spearmanr_std'].iloc[0], 3)}')
    print(f'AUC mean: {np.round((df['auc_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_std'].iloc[0])*100, 1)}')
    print(f'AUC Mid mean: {np.round((df['auc_mid_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_mid_std'].iloc[0])*100, 1)}')
    print(f'AUC Mid2 mean: {np.round((df['auc_mid2_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_mid2_std'].iloc[0])*100, 1)}')
    print(f'AUC Sev mean: {np.round((df['auc_sev_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_sev_std'].iloc[0])*100, 1)}')

def ensemble_results_pr(df):

   print(f'SRC mean: {np.round(df['spearmanr_mean'].iloc[0], 3)} with std {np.round(df['spearmanr_std'].iloc[0], 3)}')
   print(f'AUC-PR mean: {np.round((df['auc_pr_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_pr_std'].iloc[0])*100, 1)}')
   print(f'AUC-PR Mid mean: {np.round((df['auc_mid_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_mid_std'].iloc[0])*100, 1)}')
   print(f'AUC-PR Mid2 mean: {np.round((df['auc_mid2_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_mid2_std'].iloc[0])*100, 1)}')
   print(f'AUC-PR Sev mean: {np.round((df['auc_sev_mean'].iloc[0])*100, 1)} with std {np.round((df['auc_sev_std'].iloc[0])*100, 1)}')