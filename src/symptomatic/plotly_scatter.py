import config
import os
import pandas as pd
import numpy as np

from utils.plot_utils import plotly_hdbscan_highlight_kl, plotly_hdbscan_highlight_kl2, plotly_hdbscan_highlight

proc_dir = config.PROC_DATA_PATH

folder = "2025-08-23_hdbscan"
run="run2"
# folder = "2025-09-25_hdbscan"
folder_date = folder.split('_')[0]
# run = "run42"
# run = "run74"

cluster_num = None
reversed = True

filepath = os.path.join(proc_dir, folder, "questionnaire", run)
embeddings_path = os.path.join(filepath, "X_umap_embeddings.npy")

if __name__=="__main__":
    df = pd.read_csv(os.path.join(filepath, f'questionnaire_{run}_umap_hdbscan_scaled_wKL_v2.csv'))
    X_umap = np.load(embeddings_path)

    if reversed==True:
        label = 'cluster_label'
        score = 'KL-Score'
        legendby = 'kl'
        buttonby = 'cluster'
    else:
        label = 'cluster_label'
        score = 'KL-Score'
        legendby = 'cluster'
        buttonby = 'kl'

    if cluster_num is not None:
        index_ids = df[df[label]==cluster_num].index
        x_umap = X_umap[index_ids, :]
        df_temp = df.iloc[index_ids]
        fig = plotly_hdbscan_highlight(
                    X=x_umap,
                    labels=df_temp[label],
                    y_kl = df_temp[score],
                    id_names= df_temp['id'],
                    probabilities=None,
                    dim=3,
                    title="UMAP + HDBSCAN (highlight)",
                    # legend / buttons / colors
                    legend_by=legendby,     # "cluster" or "kl"
                    buttons_by=buttonby,         # "cluster" or "kl"
                    color_by="legend",       # "legend" or explicitly "cluster"/"kl"
                    # gray base & sizes
                    base_gray_opacity=0.2,
                    base_gray_size=6,
                    size_min=8,
                    size_max=20,
                    # compact controls
                    compact=False,
                    q_clip=0.01,
                    pad_frac=0.03,
                    equal_aspect=True,
                    zoom_by=None,            # None, "kl", or "cluster"
                )
    else:
        fig = plotly_hdbscan_highlight(
                    X=X_umap,
                    labels=df[label],
                    y_kl = df[score],
                    id_names= df['id'],
                    probabilities=None,
                    dim=3,
                    title="UMAP + HDBSCAN (highlight)",
                    # legend / buttons / colors
                    legend_by=legendby,     # "cluster" or "kl"
                    buttons_by=buttonby,         # "cluster" or "kl"
                    color_by="legend",       # "legend" or explicitly "cluster"/"kl"
                    # gray base & sizes
                    base_gray_opacity=0.2,
                    base_gray_size=6,
                    size_min=8,
                    size_max=20,
                    # compact controls
                    compact=False,
                    q_clip=0.01,
                    pad_frac=0.03,
                    equal_aspect=True,
                    zoom_by=None,            # None, "kl", or "cluster"
                )
        
    fig.show(renderer = "browser")