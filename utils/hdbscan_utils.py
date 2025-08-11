import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score
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

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def plot_hdbscan(
    X,
    labels,
    probabilities=None,
    parameters=None,
    ground_truth=False,
    ax=None,
    save_path=None,
    size_min=8,
    size_max=80,
    use_first_three_dims=True,
):
    """
    Auto-plots 2D or 3D depending on X shape. If X has >=3 features, uses 3D.
    - sizes scale with `probabilities` (in [0,1]); noise gets fixed small size.
    - noise label is -1 (black X markers).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    n, d = X.shape

    # choose 2D vs 3D
    is_3d = d >= 3
    if is_3d and use_first_three_dims:
        Xp = X[:, :3]
    else:
        if d < 2:
            raise ValueError("X must have at least 2 features to plot.")
        Xp = X[:, :2]

    # probabilities -> sizes
    if probabilities is None:
        probabilities = np.ones(n, dtype=float)
    else:
        probabilities = np.asarray(probabilities, dtype=float)
        # make sure it's in a sane range
        pmin, pmax = probabilities.min(), probabilities.max()
        if pmax > 1.0 or pmin < 0.0:
            # normalize to 0..1 if needed
            probabilities = (probabilities - pmin) / (pmax - pmin + 1e-12)

    sizes = size_min + (size_max - size_min) * probabilities

    # figure / axes
    if ax is None:
        fig = plt.figure(figsize=(9, 5))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    unique_labels = np.unique(labels)
    # color map per cluster (exclude noise for color count)
    n_colors = len(unique_labels) - (1 if -1 in unique_labels else 0)
    # fall back to at least 1 color to avoid linspace errors
    n_colors = max(n_colors, 1)
    color_list = [plt.cm.Spectral(t) for t in np.linspace(0, 1, n_colors)]

    # build a deterministic color map for non-noise clusters
    non_noise = [lab for lab in unique_labels if lab != -1]
    color_map = {lab: color_list[i % len(color_list)] for i, lab in enumerate(sorted(non_noise))}

    # plot each cluster once (vectorized scatter)
    handles = []
    labels_for_legend = []

    for k in sorted(unique_labels, key=lambda x: (x == -1, x)):
        mask = labels == k
        if not np.any(mask):
            continue

        if k == -1:
            # noise: black 'x', fixed size
            if is_3d:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1], Xp[mask, 2],
                               marker='x', c='k', s=size_min, linewidths=0.8, alpha=0.9)
            else:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1],
                               marker='x', c='k', s=size_min, linewidths=0.8, alpha=0.9)
            handles.append(h); labels_for_legend.append("Noise")
        else:
            col = color_map[k]
            if is_3d:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1], Xp[mask, 2],
                               marker='o', c=[col], s=sizes[mask], edgecolors='k', linewidths=0.2, alpha=0.9)
            else:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1],
                               marker='o', c=[col], s=sizes[mask], edgecolors='k', linewidths=0.2, alpha=0.9)
            handles.append(h); labels_for_legend.append(f"Cluster {k}")

    # title
    n_clusters_ = len(non_noise)
    pre = "True" if ground_truth else "Estimated"
    title = f"{pre} number of clusters: {n_clusters_}"
    if parameters is not None and isinstance(parameters, dict) and len(parameters):
        param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {param_str}"
    ax.set_title(title)

    # axes labels
    if is_3d:
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1"); ax.set_zlabel("dim 2")
        # a gentle view angle
        ax.view_init(elev=18, azim=35)
    else:
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")

    # legend (avoid too many items)
    if len(handles) <= 20:
        ax.legend(handles, labels_for_legend, title="Cluster Labels", fontsize='small', loc="best")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    # return ax so caller can further tweak
    #return ax


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
    X_train, X_test = X[train_idx], X[test_idx]
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    clusterer = HDBSCAN(**hdbscan_params).fit(X_train)

    ch_score = calinski_harabasz_score(X_train, clusterer.labels_)

    # wandb.log({
    #     'calinski_harabasz_score': ch_score
    #         }
    #     )


    base_name, results_df = save_results(df = df_train, clusterer=clusterer, params={
        'umap': umap_params,
        'hdbscan': hdbscan_params
    }, scaler=scaler, save_dir=save_dir_temp, filename=filename, id = 'name', use_wandb=True, fold=fold)

    return base_name, results_df, clusterer, X_train, df_train, X_test, df_test, ch_score

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

        n_entropy = membership_entropy / H_max

        if use_wandb:
            wandb.log({
                "fold": fold,
                "noise_count": noise_count,
                "avg_probs": avg_probs.mean(),
                "entropy": membership_entropy,
                "normalized_entropy": n_entropy,
                "missing_kl_scores": len(df_merged[df_merged[label].isna()]),
                "spearman_correlation": spr,
                "auc": auc,
                "auc_mid": auc_mid,
                "auc_mid2": auc_mid2,
                "auc_sev": auc_sev,
                "nmi": nmi
            })
        return noise_count, avg_probs, membership_entropy, n_entropy, spr, auc, auc_mid, auc_mid2, auc_sev, nmi
    if fold is None:
        results_df['id'] = results_df['id'].apply(fix_id)

        noise_count = (results_df[score]==-1).sum()

        df_filtered = results_df[results_df[score] != -1].copy()
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

        n_entropy = membership_entropy / H_max

        if use_wandb:
            wandb.log({
                "noise_count": noise_count,
                "avg_probs": avg_probs.mean(),
                "entropy": membership_entropy,
                "normalized_entropy": n_entropy,
                "missing_kl_scores": len(df_merged[df_merged[label].isna()]),
                "spearman_correlation": spr,
                "auc": auc,
                "auc_mid": auc_mid,
                "auc_mid2": auc_mid2,
                "auc_sev": auc_sev,
                "nmi": nmi
            })

def get_metrics_hdbscan_radiographic(results_df, save_dir_temp, base_name, score='cluster_label', label='label', use_wandb=True, fold = None):
    if fold is None:
        results_df['id'] = results_df['id'].apply(fix_id)

        noise_count = (results_df[score]==-1).sum()

        df_filtered = results_df[results_df[score] != -1].copy()
        avg_probs = df_filtered.groupby(score)['probability'].mean().sort_values(ascending=False)
        avg_probs.to_csv(os.path.join(save_dir_temp, f"{base_name}_avg_probs_per_cluster.csv"))
                
        p_dist = df_filtered['probability'] / np.sum(df_filtered['probability'])
        membership_entropy = entropy(p_dist, base=2)
        H_max = np.log2(len(p_dist))
                
        entropy_per_cluster = df_filtered.groupby(score)['probability'].apply(
                        normalized_entropy
                    ).sort_values()
        entropy_per_cluster.to_csv(os.path.join(save_dir_temp, f"{base_name}_entropy_per_cluster.csv"))
        # df_filtered.to_csv(os.path.join(save_dir_temp, f"{base_name}_wKL.csv"), index=False)
        df_filtered = df_filtered.dropna(subset=[label])
        spr, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df_filtered, score = score, label = label)
        nmi = normalized_mutual_info_score(df_filtered[label], df_filtered[score])

        n_entropy = membership_entropy / H_max

        if use_wandb:
            wandb.log({
                "noise_count": noise_count,
                "avg_probs": avg_probs.mean(),
                "entropy": membership_entropy,
                "normalized_entropy": n_entropy,
                "missing_kl_scores": len(df_filtered[df_filtered[label].isna()]),
                "spearman_correlation": spr,
                "auc": auc,
                "auc_mid": auc_mid,
                "auc_mid2": auc_mid2,
                "auc_sev": auc_sev,
                "nmi": nmi
            })