import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb

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

def save_results(df, clusterer, params, scaler, save_dir, filename, comment = None, wandb= False):
    df_filename = f"{filename}.csv"
    results_df = pd.DataFrame({
                    'id': df['id'],
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
    
    if wandb:
        wandb.log({
            'df_savepath': df_savepath,
            'params': params, #TODO log params one by one not within the wandb log
            'n_clusters': model_info['n_clusters'],
            #'centroids': model_info['centroids'],
            'comment': comment
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

def merge_klscores(kl_df, df):

