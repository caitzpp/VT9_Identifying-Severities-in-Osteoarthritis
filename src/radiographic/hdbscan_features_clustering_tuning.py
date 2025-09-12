import wandb

import config
import os
import joblib
import pandas as pd
import json
import numpy as np
from IPython.display import display
from itertools import product
import datetime
from utils.hdbscan_utils import get_unique_filepath, plot_hdbscan,  get_metrics_hdbscan_radiographic

from sklearn.preprocessing import StandardScaler
#import hdbscan
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedKFold
from utils.load_utils import fix_id, get_next_run_folder
from utils.load_utils import load_npy_folder_as_array


today = datetime.date.today()

proc_dir = config.PROC_DATA_PATH

STAGE = "average_mod_smallimg2" #'ss2' #had to redownload it, this should be the correct one
NEPOCH = '400'
MODEL_NAME = 'mod_st'
seed = '34'
on_test_set = False
DATA_PATH = config.FEATURE_PATH
# DATA_TYPE = "images" #"features" #"chenetal_train"

folder = None

random_state=42
# stratify_on = 'KL-Score'

if folder is not None:
    save_dir = os.path.join(proc_dir, 'radiographic_features', folder)
else:
    save_dir = os.path.join(proc_dir,'radiographic_features', f"{today}_hdbscan")
# img_savepath = os.path.join(save_dir, 'img')
os.makedirs(save_dir, exist_ok=True)

wandb.login(key=config.HDBSCAN_SYMP_WANDBAPI_KEY)

if __name__=="__main__":
    run_folder_name, run_path = get_next_run_folder(save_dir)
    run = wandb.init(
        name = f"{os.path.basename(save_dir)}_{run_folder_name}"
        )
    save_folder = run_folder_name
    filename = f"{run.name}_umap_hdbscan_scaled"

    save_dir_temp = os.path.join(save_dir, save_folder)
    os.makedirs(save_dir_temp, exist_ok=True)
    filename = f"{run.name}_umap_hdbscan_scaled"
    df_savepath = get_unique_filepath(os.path.join(save_dir_temp, filename + '.csv'))

    img_savepath = os.path.join(save_dir_temp, 'img')
    os.makedirs(img_savepath, exist_ok=True)

    wandb_config = wandb.config

    wUMAP = wandb_config.get('wUMAP', True)
    k = wandb_config.get('k-cv', None)

    umap_keys = ['n_neighbors', 'min_dist', 'n_components', 'metric']
    hdbscan_keys = ['min_cluster_size', 'min_samples', 'cluster_selection_method', 
                    'metric', 'metric_params', 'max_cluster_size',
                    'cluster_selection_epsilon', 'algorithm', 'leaf_size',
                    'store_centers', 'alpha']

    umap_defaults = {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'n_components': 2,
        'metric': 'euclidean'
    }

    hdbscan_defaults = {
        'min_cluster_size': 10,
        'min_samples': None,
        'cluster_selection_method': 'eom',
        'metric': 'euclidean',
        'metric_params': None,
        'max_cluster_size': None,
        'cluster_selection_epsilon': 0.0,
        'algorithm': 'auto',
        'leaf_size': 40,
        'store_centers': 'centroid',
        'alpha': 1.0
    }

    if wUMAP:
        umap_params = {k: wandb_config.get(f'umap.{k}', umap_defaults[k]) for k in umap_keys}
    else:
        umap_params = "woUMAP"
    hdbscan_params = {k: wandb_config.get(f'hdbscan.{k}', hdbscan_defaults[k]) for k in hdbscan_keys}

    if hdbscan_params['min_samples'] == -1:
        hdbscan_params['min_samples'] = None

    files = os.listdir(os.path.join(DATA_PATH, STAGE))

    DATA_PATH = os.path.join(DATA_PATH, STAGE)
    if on_test_set:
        feature_dir = os.path.join(DATA_PATH, 'test')
    else:
        feature_dir = os.path.join(DATA_PATH, 'train')

    # file = [f for f in files if (MODEL_NAME in f) & (NEPOCH in f) & (seed in f) & ('on_test_set' not in f)]

    # if len(file) == 1:
    #     folder_dir = file[0]
    #     feature_dir = os.path.join(DATA_PATH, STAGE, folder_dir)

    #     if on_test_set:
    #         feature_dir = os.path.join(feature_dir, 'test')
    #     else:
    #         feature_dir = os.path.join(feature_dir, 'train')
    # else:
    #     print("more than one folder found: ", file)
    #     raise
    
    scaler = StandardScaler()

    X, y, names = load_npy_folder_as_array(feature_dir)
    X = scaler.fit_transform(X)
    reducer = UMAP(**umap_params)
    X_umap = reducer.fit_transform(X)
    
    umap_path = os.path.join(save_dir_temp, "umap_model.pkl")
    joblib.dump(reducer, umap_path)
    scaler_path = os.path.join(save_dir_temp, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    embeddings_path = os.path.join(save_dir_temp, "embeddings.npz")
    np.savez(embeddings_path, X_umap=X_umap, names=np.array(names))

    clusterer = HDBSCAN(**hdbscan_params).fit(X_umap)
    clusterer_path = os.path.join(save_dir_temp, f"{filename}_clusterer.pkl")
    joblib.dump(clusterer, clusterer_path)

    ch_score = calinski_harabasz_score(X_umap, clusterer.labels_)
    
    wandb.log({
        'calinski_harabasz_score': ch_score})
    
    results_df = pd.DataFrame({
        'id': names,
        'label': y,
        'cluster_label': clusterer.labels_,
        'probability': clusterer.probabilities_,
    })

   
    results_df.to_csv(df_savepath, index=False)

    model_info = {
                'df_savepath': df_savepath,
                'params': {'umap': umap_params, 'hdbscan': hdbscan_params},
                'scaler': scaler.__class__.__name__,
                'n_clusters': len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0),
                'centroids': clusterer.centroids_.tolist()
            }

    wandb.log({
        'n_clusters': model_info['n_clusters']
    })

    model_info_filename = f"{filename}_model_info.json"
    model_info_savepath = get_unique_filepath(os.path.join(save_dir_temp, model_info_filename))
    with open(model_info_savepath, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    b_name=os.path.basename(df_savepath).split('.')[0]
    get_metrics_hdbscan_radiographic(results_df=results_df, save_dir_temp=save_dir_temp, 
                                     base_name=b_name, score='cluster_label', 
                                     label='label', use_wandb=True)
    
    plot_hdbscan(X_umap, clusterer.labels_,
                    probabilities=clusterer.probabilities_,
                    save_path=os.path.join(save_dir_temp, f"{b_name}_plot.png"))
    wandb.finish()