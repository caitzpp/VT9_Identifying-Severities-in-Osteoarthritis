import config
import os
import pandas as pd
import numpy as np

from utils.plot_utils import plotly_hdbscan_highlight_kl, plotly_hdbscan_highlight_kl2, plotly_hdbscan_highlight

proc_dir = config.PROC_DATA_PATH

folder = "2025-08-23_hdbscan"
folder_date = folder.split('_')[0]
run = "run2"

cluster_num = None
reversed = True

filepath = os.path.join(proc_dir, folder, "questionnaire", run)
embeddings_path = os.path.join(filepath, "X_umap_embeddings.npy")

if __name__=="__main__":
    df = pd.read_csv(os.path.join(filepath, f'questionnaire_{run}_umap_hdbscan_scaled_wKL_v2.csv'))
    X_umap = np.load(embeddings_path)

    if reversed==True:
        label = 'KL-Score'
        score = 'cluster_label'
    else:
        label = 'cluster_label'
        score = 'KL-Score'

    if cluster_num is not None:
        index_ids = df[df[label]==cluster_num].index
        x_umap = X_umap[index_ids, :]
        df_temp = df.iloc[index_ids]
        fig = plotly_hdbscan_highlight_kl2(
            X= x_umap,
            labels = df_temp[label],
            y_kl=df_temp[score],
            dim = 3,
            title = " ",
            compact=False, q_clip=0.01, pad_frac=0.03,
            zoom_by_kl=False
        )
    else:
        fig = plotly_hdbscan_highlight_kl2(
            X= X_umap,
            labels = df[label],
            y_kl=df[score],
            dim = 3,
            title = " ",
            compact=False, q_clip=0.01, pad_frac=0.03,
        zoom_by_kl=False
        )
        
    fig.show(renderer = "browser")