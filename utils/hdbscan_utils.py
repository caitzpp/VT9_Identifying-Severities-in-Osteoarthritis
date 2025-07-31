import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy

from utils.evaluation_utils import normalized_entropy, get_metrics
from utils.load_utils import fix_id


def get_unique_filepath(base_path):
    """If file exists, append _2, _3, etc. until unique."""
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    counter = 2
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path

def save_results(df, clusterer, params, scaler, save_dir, filename, id = 'id', comment = None, use_wandb= False, fold = None):
    if fold is not None:
        df_filename = f"{filename}_fold{fold}.csv"
        results_df = pd.DataFrame({
                        'id': df[id],
                        'cluster_label': clusterer.labels_,
                        'probability': clusterer.probabilities_,
                    })
        df_savepath = get_unique_filepath(os.path.join(save_dir, df_filename))
        results_df.to_csv(df_savepath, index=False)

        model_info = {
                'df_savepath': df_savepath,
                'params': params,
                'scaler': scaler.__class__.__name__,
                'n_clusters': len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0),
                'centroids': clusterer.centroids_.tolist(),
                'comment': comment,
            }
        
        if use_wandb:
            wandb.log({
                'fold': fold, #is this how we log the fold?
                'n_clusters': model_info['n_clusters'],
                #'centroids': model_info['centroids'],
                # 'comment': comment
            })

        model_info_filename = f"{filename}_fold{fold}_model_info.json"
        model_info_savepath = get_unique_filepath(os.path.join(save_dir, model_info_filename))
        with open(model_info_savepath, 'w') as f:
            json.dump(model_info, f, indent=4)
        return os.path.basename(df_savepath).split('.')[0], results_df
    elif fold is None:
        df_filename = f"{filename}.csv"
        results_df = pd.DataFrame({
                        'id': df[id],
                        'cluster_label': clusterer.labels_,
                        'probability': clusterer.probabilities_,
                    })
        df_savepath = get_unique_filepath(os.path.join(save_dir, df_filename))
        results_df.to_csv(df_savepath, index=False)

        model_info = {
                'df_savepath': df_savepath,
                'params': params,
                'scaler': scaler.__class__.__name__,
                'n_clusters': len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0),
                'centroids': clusterer.centroids_.tolist(),
                'comment': comment,
            }
        
        if use_wandb:
            wandb.log({
                # 'df_savepath': df_savepath,
                # 'params': params, #TODO log params one by one not within the wandb log
                'n_clusters': model_info['n_clusters'],
                #'centroids': model_info['centroids'],
                # 'comment': comment
            })

        model_info_filename = f"{filename}_model_info.json"
        model_info_savepath = get_unique_filepath(os.path.join(save_dir, model_info_filename))
        with open(model_info_savepath, 'w') as f:
            json.dump(model_info, f, indent=4)
        return os.path.basename(df_savepath).split('.')[0], results_df

def plot_hdbscan(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None, save_path = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
            label = "Noise"
            marker_style = "x"
        else:
            label = f"Cluster {k}"
            marker_style = "o"

        class_index = (labels == k).nonzero()[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                marker_style,
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
                label=label if ci == class_index[0] else None  # Only label once per cluster
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    ax.legend(title="Cluster Labels", loc="best", fontsize='small')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
       # plt.show()

def prep_data(df, scaler, umap_params = None, wUMAP = True, id_col='name', y_value = 'KL-Score'):
        df_scaled = df.copy()
        y = df_scaled[y_value]
        X = df_scaled.drop(columns=[id_col, y_value])
        X_scaled = scaler.fit_transform(X)

        if wUMAP:
            X_umap = UMAP(**umap_params).fit_transform(X_scaled)
        else:
            X_umap = X_scaled
        
        return X_umap, y, df_scaled

def train_fold(fold, train_idx, test_idx, X, y, df, hdbscan_params, umap_params, scaler,filename, save_dir_temp):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    clusterer = HDBSCAN(**hdbscan_params).fit(X_train)

    base_name, results_df = save_results(df = df_train, clusterer=clusterer, params={
        'umap': umap_params,
        'hdbscan': hdbscan_params
    }, scaler=scaler, save_dir=save_dir_temp, filename=filename, id = 'name', use_wandb=True, fold=fold)

    return base_name, results_df, clusterer, X_train, df_train, X_test, df_test

def get_metrics_hdbscan(results_df, kl_df, save_dir_temp, base_name, score='cluster_label', label='KL-Score', use_wandb=True, fold = None):
    if fold is not None:
        results_df['id'] = results_df['id'].apply(fix_id)

        noise_count = (results_df[score]==-1).sum()

        df_filtered = results_df[results_df[score] != -1]
        avg_probs = df_filtered.groupby(score)['probability'].mean().sort_values(ascending=False)
        avg_probs.to_csv(os.path.join(save_dir_temp, f"{base_name}_fold{fold}_avg_probs_per_cluster.csv"))
                
        p_dist = df_filtered['probability'] / np.sum(df_filtered['probability'])
        membership_entropy = entropy(p_dist, base=2)
        H_max = np.log2(len(p_dist))
                
        entropy_per_cluster = df_filtered.groupby(score)['probability'].apply(
                        normalized_entropy
                    ).sort_values()
        entropy_per_cluster.to_csv(os.path.join(save_dir_temp, f"{base_name}_fold{fold}_entropy_per_cluster.csv"))
        df_merged = df_filtered.merge(kl_df, left_on='id', right_on='name', how='left', validate='one_to_one')

        df_merged.to_csv(os.path.join(save_dir_temp, f"{base_name}_fold{fold}_wKL.csv"), index=False)
        df_merged = df_merged.dropna(subset=[label])
        spr, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df_merged, score=score, label=label)
        nmi = normalized_mutual_info_score(df_merged[label], df_merged[score])

        if use_wandb:
            wandb.log({
                "fold": fold,
                "noise_count": noise_count,
                "avg_probs": avg_probs.mean(),
                "entropy": membership_entropy,
                "normalized_entropy": membership_entropy / H_max,
                "missing_kl_scores": len(df_merged[df_merged[label].isna()]),
                "spearman_correlation": spr,
                "auc": auc,
                "auc_mid": auc_mid,
                "auc_mid2": auc_mid2,
                "auc_sev": auc_sev,
                "nmi": nmi
            })
        return noise_count, avg_probs, membership_entropy, normalized_entropy, spr, auc, auc_mid, auc_mid2, auc_sev, nmi
    if fold is None:
        results_df['id'] = results_df['id'].apply(fix_id)

        noise_count = (results_df[score]==-1).sum()

        df_filtered = results_df[results_df[score] != -1]
        avg_probs = df_filtered.groupby(score)['probability'].mean().sort_values(ascending=False)
        avg_probs.to_csv(os.path.join(save_dir_temp, f"{base_name}_avg_probs_per_cluster.csv"))
                
        p_dist = df_filtered['probability'] / np.sum(df_filtered['probability'])
        membership_entropy = entropy(p_dist, base=2)
        H_max = np.log2(len(p_dist))
                
        entropy_per_cluster = df_filtered.groupby(score)['probability'].apply(
                        normalized_entropy
                    ).sort_values()
        entropy_per_cluster.to_csv(os.path.join(save_dir_temp, f"{base_name}_entropy_per_cluster.csv"))
        df_merged = df_filtered.merge(kl_df, left_on = 'id', right_on='name', how='left', validate='one_to_one')

        df_merged.to_csv(os.path.join(save_dir_temp, f"{base_name}_wKL.csv"), index=False)
        df_merged = df_merged.dropna(subset=[label])
        spr, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df_merged, score = score, label = label)
        nmi = normalized_mutual_info_score(df_merged[label], df_merged[score])

        if use_wandb:
            wandb.log({
                "noise_count": noise_count,
                "avg_probs": avg_probs.mean(),
                "entropy": membership_entropy,
                "normalized_entropy": membership_entropy / H_max,
                "missing_kl_scores": len(df_merged[df_merged[label].isna()]),
                "spearman_correlation": spr,
                "auc": auc,
                "auc_mid": auc_mid,
                "auc_mid2": auc_mid2,
                "auc_sev": auc_sev,
                "nmi": nmi
            })