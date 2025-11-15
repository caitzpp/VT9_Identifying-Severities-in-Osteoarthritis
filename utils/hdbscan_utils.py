import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import smote_variants as sv
import joblib
import numpy as np
import wandb
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score, mean_squared_error
from scipy.stats import entropy, combine_pvalues
import scikit_posthocs as sp

from utils.evaluation_utils import normalized_entropy, get_metrics, get_metrics_external
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

def save_results(df, 
                 ypred, 
                 strengths,
                 clusterer, 
                 params, 
                 scaler, 
                 save_dir, 
                 filename, 
                 id = 'id',
                 artifacts = None, 
                 comment = None, 
                 use_wandb= False, 
                 fold = None, 
                 smote=False):
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
                # 'centroids': clusterer.centroids_.tolist(),
                'files': artifacts,
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
        if smote:
            results_df = pd.DataFrame({
                        'id': df[id],
                        'cluster_label': ypred,
                        'probability': strengths
                    })
        else:
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
                # 'centroids': clusterer.centroids_.tolist(),
                'files': artifacts,
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
    
def save_results_SMOTE(df,ypred, clusterer, params, scaler, save_dir, filename, artifacts = None,  id = 'name', comment=None,use_wandb=False):
    df_filename = f"{filename}.csv"

    results_df = pd.DataFrame({
                        'id': df[id],
                        'cluster_label': ypred
                    })
    df_savepath = get_unique_filepath(os.path.join(save_dir, df_filename))
    results_df.to_csv(df_savepath, index=False)

    model_info = {
                'df_savepath': df_savepath,
                'params': params,
                'scaler': scaler.__class__.__name__,
                'n_clusters': len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0),
                # 'centroids': clusterer.centroids_.tolist(),
                'files': artifacts,
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

def make_cluster_color_map(labels, cmap=plt.cm.tab20):
    unique_labels = sorted(set(labels) - {-1})   # exclude noise
    n_colors = len(unique_labels) + 2
    n_colors = max(n_colors, 1)
    color_map = {}
    color_list = [plt.cm.Spectral(t) for t in np.linspace(0, 1, n_colors)]
    color_map = {lab: color_list[i % len(color_list)] for i, lab in enumerate(sorted(unique_labels))}
    color_map[-1] = (0, 0, 0, 1)  # black for noise
    return color_map

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
    
def add_jitter(X, scale=0.02):
    """Add Gaussian noise to spread out overlapping points."""
    return X + np.random.normal(0, scale, X.shape)

def plot_hdbscan_highlight_kl(
    X,
    labels,
    y_kl,                 # array-like of KL-scores per point
    focus_kl,             # the KL value to highlight (e.g., 0,1,2,3,4)
    probabilities=None,
    parameters=None,
    ground_truth=False,
    ax=None,
    save_path=None,
    size_min=8,
    size_max=80,
    use_first_three_dims=True,
    gray_alpha=0.75,      # transparency for non-focused points
    gray_size_factor=1,  # size multiplier for gray points
    color_alpha = 0.3,
    global_color_map=None  # if provided, use this color map for clusters
):
    """
    Plots clusters but highlights only points with y_kl == focus_kl in color.
    All other points are rendered in light gray. Noise is still 'x' markers.

    - Keeps your 2D/3D auto logic.
    - Sizes scale with `probabilities` for focused points; gray points use reduced size.
    """
    X = np.asarray(X)
    X = add_jitter(X, scale = 0.05)
    labels = np.asarray(labels)
    y_kl = np.asarray(y_kl)
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
        pmin, pmax = probabilities.min(), probabilities.max()
        if pmax > 1.0 or pmin < 0.0:
            probabilities = (probabilities - pmin) / (pmax - pmin + 1e-12)
    sizes = size_min + (size_max - size_min) * probabilities

    # figure / axes
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(9, 5))
        created_fig = True
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    # masks
    focus_mask = (y_kl == focus_kl)
    other_mask = ~focus_mask

    # --- 1) plot NON-focused points in uniform light gray (behind)
    if np.any(other_mask):
        gray_sizes = (size_min + (size_max - size_min) * 0.3) * gray_size_factor
        if is_3d:
            ax.scatter(Xp[other_mask, 0], Xp[other_mask, 1], Xp[other_mask, 2],
                       marker='o', c='lightgray', s=gray_sizes, alpha=gray_alpha)
        else:
            ax.scatter(Xp[other_mask, 0], Xp[other_mask, 1],
                       marker='o', c='lightgray', s=gray_sizes, alpha=gray_alpha)

    # --- 2) plot FOCUSED points with the original cluster coloring
    unique_labels = np.unique(labels[focus_mask]) if np.any(focus_mask) else np.array([])
    non_noise = [lab for lab in unique_labels if lab != -1]

    # build color map for focused clusters
    if global_color_map is None:
        n_colors = len(non_noise)
        n_colors = max(n_colors, 1)
        color_list = [plt.cm.Spectral(t) for t in np.linspace(0, 1, n_colors)]
        color_map = {lab: color_list[i % len(color_list)] for i, lab in enumerate(sorted(non_noise))}
    else:
        color_map = global_color_map

    handles, labels_for_legend = [], []

    # plot focused clusters (and noise)
    for k in sorted(set(unique_labels), key=lambda x: (x == -1, x)):
        mask = focus_mask & (labels == k)
        if not np.any(mask):
            continue

        if k == -1:
            # noise: black 'x'
            if is_3d:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1], Xp[mask, 2],
                               marker='x', c='k', s=size_min, linewidths=0.8, alpha=0.9)
            else:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1],
                               marker='x', c='k', s=size_min, linewidths=0.8, alpha=0.9)
            handles.append(h); labels_for_legend.append(f"Noise (KL={focus_kl})")
        else:
            col = color_map[k]
            if is_3d:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1], Xp[mask, 2],
                               marker='o', c=[col], s=sizes[mask], edgecolors='k', linewidths=0.2, alpha=color_alpha)
            else:
                h = ax.scatter(Xp[mask, 0], Xp[mask, 1],
                               marker='o', c=[col], s=sizes[mask], edgecolors='k', linewidths=0.2, alpha=color_alpha)
            handles.append(h); labels_for_legend.append(f"Cluster {k} (KL={focus_kl})")

    # title
    pre = "True" if ground_truth else "Estimated"
    n_clusters_ = len(non_noise)
    title = f"{pre} clusters in KL={focus_kl}: {n_clusters_}"
    if parameters and isinstance(parameters, dict) and len(parameters):
        param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {param_str}"
    ax.set_title(title)

    # axes labels
    if is_3d:
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1"); ax.set_zlabel("dim 2")
        ax.view_init(elev=18, azim=35)
    else:
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")

    if len(handles) <= 20 and len(handles) > 0:
        ax.legend(handles, labels_for_legend, title="Focused clusters", fontsize='small', loc="best")

    plt.tight_layout()
    if save_path is not None and created_fig:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return ax

def prep_data(df, scaler, Xcol = None, umap_params = None, wUMAP = True, id_col='name', y_value = 'KL-Score', save_path = None):
        df_scaled = df.copy()
        y = df_scaled[y_value]
        ids = df_scaled[id_col]
        if Xcol is None:
            X = df_scaled.drop(columns=[id_col, y_value])
        else:
            X = df_scaled[Xcol]
        X_scaled = scaler.fit_transform(X)

        if wUMAP:
            reducer = UMAP(**umap_params)
            X_umap = reducer.fit_transform(X_scaled)
            emb_name = "X_umap"
            if save_path is not None:
                artifacts = {}
                umap_path = os.path.join(save_path, "umap_model.pkl")
                joblib.dump(reducer, umap_path)
                artifacts["umap_model"] = umap_path
                artifacts["ids"] = list(ids)

        else:
            X_umap = X_scaled
            emb_name = "X_scaled"
        
        if save_path is not None:
            #emb_cols = [f"umap{i+1}" for i in range(X_umap.shape[1])]

            #save scaler
            scaler_path = os.path.join(save_path, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            artifacts['scaler'] = scaler_path

            #save embeddings
            embeddings_path = os.path.join(save_path, f"{emb_name}_embeddings.npy")
            np.save(embeddings_path, X_umap)
            artifacts['embeddings'] = embeddings_path
        return X_umap, y, df_scaled, artifacts

def smote_data_preparation(df, scaler, umap_params, wUMAP, id_col = 'name', y_value='KL-Score', oversample_method='SMOTE', save_path=None):
    df_scaled = df.copy()
    y = df_scaled[y_value]
    ids = df_scaled[id_col]

    X = df_scaled.drop(columns=[id_col, y_value])

    X_scaled = scaler.fit_transform(X)

    oversampler = sv.MulticlassOversampling(oversampler=oversample_method,oversampler_params={'random_state': 5})
    X_samp, y_samp = oversampler.sample(X_scaled, y)
    
    X_gen = scaler.inverse_transform(X_samp)
    df_gen = pd.DataFrame(X_gen, columns=X.columns)
    df_gen['KL-Score'] = y_samp

    #to_csv(os.path.join(PROCESSED_DATA_PATH, folder, f'smote_oversampled_data_{oversample_method}.csv'), index=False)


    if wUMAP:
        reducer = UMAP(**umap_params)
        X_samp_umap = reducer.fit_transform(X_samp)
        X_umap = reducer.transform(X_scaled)
        emb_name = "X_umap"
        if save_path is not None:
            artifacts = {}
            umap_path = os.path.join(save_path, "umap_model.pkl")
            joblib.dump(reducer, umap_path)
            artifacts["umap_model"] = umap_path
            artifacts["ids"] = list(ids)

    else:
        X_umap = X_scaled
        X_samp_umap = X_samp
        emb_name = "X_scaled"
    
    if save_path is not None:
            #emb_cols = [f"umap{i+1}" for i in range(X_umap.shape[1])]

            #save scaler
            scaler_path = os.path.join(save_path, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            artifacts['scaler'] = scaler_path

            #save embeddings
            embeddings_path = os.path.join(save_path, f"{emb_name}_embeddings.npy")
            embeddings_samp_path = os.path.join(save_path, f"{emb_name}_samp_embeddings.npy")
            np.save(embeddings_path, X_umap)
            np.save(embeddings_samp_path, X_samp_umap)
            artifacts['embeddings'] = embeddings_path
            artifacts['embeddings_samp'] = embeddings_samp_path
    
    return X_umap, y, df_scaled, X_samp_umap, y_samp, df_gen, artifacts

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

def get_metrics_hdbscan(results_df, kl_df, save_dir_temp, base_name, clusterer, score='cluster_label', label='KL-Score', 
                        use_wandb=True, fold = None, smote=False):
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

        if smote:
            results_df_samp = pd.DataFrame({
                'cluster_label': clusterer.labels_,
                'probability': clusterer.probabilities_,
            })

            df_filtered_samp = results_df_samp[results_df_samp[score] != -1].copy()
            avg_probs = df_filtered_samp.groupby(score)['probability'].mean().sort_values(ascending=False)
            avg_probs.to_csv(os.path.join(save_dir_temp, f"{base_name}_avg_probs_per_cluster_orgsmote.csv"))

            p_dist = df_filtered_samp['probability'] / np.sum(df_filtered_samp['probability'])
            membership_entropy = entropy(p_dist, base=2)
            H_max = np.log2(len(p_dist))
            entropy_per_cluster = df_filtered_samp.groupby(score)['probability'].apply(
                            normalized_entropy
                        ).sort_values()
            entropy_per_cluster.to_csv(os.path.join(save_dir_temp, f"{base_name}_entropy_per_cluster_orgsmote.csv"))
        if smote==False:
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

        df_merged2 = results_df.merge(kl_df, left_on = 'id', right_on='name', how='left', validate='one_to_one')
        df_merged2.to_csv(os.path.join(save_dir_temp, f"{base_name}_allpoints_wKL.csv"), index=False)
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
                "normalized_entropy_kl_clusterlabel": n_entropy,
                "missing_kl_scores": len(df_merged[df_merged[label].isna()]),
                "spearman_correlation_klscore": spr,
                "auc_kl": auc,
                "auc_mid_kl": auc_mid,
                "auc_mid2_kl": auc_mid2,
                "auc_sev_kl": auc_sev,
                "nmi_kl": nmi
            })
        return df_merged, df_filtered

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

def save_umap_true_plot(X_umap, y, out_path, umap_params, title_suffix=""):
    """Save UMAP plot colored by true labels."""
    cmap5 = plt.cm.get_cmap('tab10', 5)
    if umap_params['n_components'] == 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        sc = ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=y, cmap=cmap5, s=30, alpha=0.7, edgecolor='k', linewidth=0.5
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
            c=y, cmap=cmap5, s=30, alpha=0.7, edgecolor='k', linewidth=0.5
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")

    handles, labels = sc.legend_elements(prop="colors", num=4)
    ax.legend(
        handles, labels, title="Label",
        loc="upper left", bbox_to_anchor=(1.02, 1),
        borderaxespad=0., frameon=False
    )
    ax.set_title(f"True Labels{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def external_validation(df, externaldf, chadjustd, label = 'cluster_label', external_cols = ['mri_cart_yn', 'mri_osteo_yn'], leftid_col = 'id', rightid_col = 'id', use_wandb = False):
    df['id'] = df['id'].apply(fix_id)

    #noise_count = (results_df[score]==-1).sum()

    df_filtered = df[df[label] != -1].copy()
    df_ev = df_filtered.merge(externaldf, left_on=leftid_col, right_on=rightid_col, how='left', validate='one_to_one')
    print(f"External validation: {len(df_ev)} out of {len(df_filtered)} clustered points have external labels.")
    df_ev.dropna(subset=external_cols, inplace=True)

    results = get_metrics_external(df = df_ev, chadjusted=chadjustd, externalcol = external_cols, label = label)

    if use_wandb:
        wandb.log(results)

    return results

def get_pvalues(df):
    pval = df.where(np.triu(np.ones(df.shape), k=1).astype(bool)).stack().values
    _, global_p = combine_pvalues(pval, method='stouffer')
    return pval, global_p
def conover_test(df, val_column, group_column):
    conover = sp.posthoc_conover(df, val_col = val_column, group_col = group_column, p_adjust = 'holm')
    comb_pvalues = get_pvalues(conover)[1]
    return conover, comb_pvalues

def external_validation_2(df, combined, val_column, cluster_col = 'cluster_label', use_wandb = False):
    df['id'] = df['id'].apply(fix_id)
    df_filtered = df[df[cluster_col] != -1].copy()
    dfc = combined.merge(df_filtered, on='id', how = 'inner')
    lendfc = len(dfc)

    clusters = dfc[cluster_col].unique()
    d = {}
    for cluster in clusters:
        mean_cluster_value = dfc[dfc[cluster_col] == cluster][val_column].mean()
        #log
        d[cluster] = mean_cluster_value

    mse = mean_squared_error(dfc[val_column], dfc[cluster_col].map(d))

    if use_wandb:
        wandb.log({
            f"len_dfc": lendfc,
            f"mse": mse
        })
    return lendfc, mse




def get_hdbscan_umap_defaults():
    umap_keys = ['n_neighbors', 'min_dist', 'n_components', 'metric']
    # hdbscan_keys = ['min_cluster_size', 'min_samples', 'cluster_selection_method', 
    #                 'metric', 'metric_params', 'max_cluster_size',
    #                 'cluster_selection_epsilon', 'algorithm', 'leaf_size',
    #                 'store_centers', 'alpha']
    hdbscan_keys = [
            'min_cluster_size',
            'min_samples',
            'metric',
            'p',
            'alpha',
           # 'cluster_selection_epsilon',
            'algorithm',
            #'approx_min_span_tree',
            'cluster_selection_method',
            'prediction_data'
        ]

    umap_defaults = {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'n_components': 2,
        'metric': 'euclidean',
        'random_state': 42
    }

    hdbscan_defaults = {
        'min_cluster_size': 5,
        'min_samples': None,
        'metric': 'euclidean',
        'p': None,
        'alpha': 1.0,
        'cluster_selection_epsilon': 0.0,
        'algorithm': 'best',
        'leaf_size': 40,
        'approx_min_span_tree': True,
        'gen_min_span_tree': False,
        'core_dist_n_jobs': 4,
        'cluster_selection_method': 'eom',
        'allow_single_cluster': False,
        'prediction_data': True,
        'match_reference_implementation': False
    }

    return umap_keys, umap_defaults, hdbscan_keys, hdbscan_defaults