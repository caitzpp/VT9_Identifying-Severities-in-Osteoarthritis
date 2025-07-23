import wandb

import config
import os
import pandas as pd
import numpy as np
from IPython.display import display
from itertools import product
import datetime
from utils.hdbscan_utils import get_unique_filepath, save_results, plot_hdbscan

from sklearn.preprocessing import StandardScaler
#import hdbscan
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy
from utils.load_utils import fix_id
from utils.evaluation_utils import normalized_entropy, get_metrics

import sys

def get_next_run_folder(base_path):
    i = 1
    while True:
        folder_name = f"run{i}"
        full_path = os.path.join(base_path, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return folder_name, full_path
        i += 1

today = datetime.date.today()

base_dir = config.RAW_DATA_PATH
proc_dir = config.PROC_DATA_PATH

#folder = '2025-07-18_hdbscan'
folder = None

if folder is not None:
    save_dir = os.path.join(proc_dir, folder)
else:
    save_dir = os.path.join(proc_dir, f"{today}_hdbscan")
os.makedirs(save_dir, exist_ok=True)

folder = "2025-07-14_data_exploration"
kl_file = "inmodi_data_personalinformation_kl.csv"
kl_filepath = os.path.join(proc_dir, folder, kl_file)
unpivoted = True

#Choose relevant columns
cols = ['id',
    #'record_id', # id column
            #'visit', 'side', 
            'pain', 
            'age', 
            # 'ce_height', 
            # 'ce_weight',
        'ce_bmi', 
        'ce_fm', 
        'gender', 
        'OKS_score', 
        'UCLA_score', 
        'FJS_score',
        'KOOS_pain', 
        'KOOS_symptoms', 
        'KOOS_sport', 
        'KOOS_adl', 
        'KOOS_qol'
    ]

wandb.login(key=config.HDBSCAN_SYMP_WANDBAPI_KEY)

if __name__ == "__main__":
    run_folder_name, run_path = get_next_run_folder(save_dir)

    run = wandb.init(
        project="HDBSCAN_SymptomaticData",
        # config={
        #     "umap": {
        #         'n_neighbors': 5,
        #         'min_dist': 0.1,
        #         'n_components': 2,
        #         'metric': 'euclidean'
        #     },
        #     "hdbscan": {
        #             'min_cluster_size': 5,
        #             'min_samples': None,
        #             'cluster_selection_method': 'eom',
        #             'metric': 'euclidean',
        #             'metric_params': None,
        #             'max_cluster_size': None,
        #             'cluster_selection_epsilon': 0.0,
        #             'algorithm': 'auto',
        #             'leaf_size': 40,
        #             'store_centers': 'centroid',
        #             'alpha': 1.0
        #         }
        # }
        name = f"{os.path.basename(save_dir)}_{run_folder_name}"
        )
    wandb_config = wandb.config
    
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

    umap_params = {k: wandb_config.get(f'umap.{k}', umap_defaults[k]) for k in umap_keys}
    hdbscan_params = {k: wandb_config.get(f'hdbscan.{k}', hdbscan_defaults[k]) for k in hdbscan_keys}

    if hdbscan_params['min_samples'] == -1:
        hdbscan_params['min_samples'] = None

    if unpivoted:
        df = pd.read_csv(os.path.join(proc_dir, folder, "inmodi_data_personalinformation_unpivoted.csv"))
    else:
        df = pd.read_csv(os.path.join(proc_dir, folder, "inmodi_data_personalinformation.csv"))
    
    df['id'] = (df['record_id'].astype(str) + "_" + df['visit'].astype(str) + "_" 
                + df['side'].map({'l': 'left', 'r': 'right'}))

    df2 = df[cols].copy()
    print("Dataframe before dropping NaN values: ", df2.shape)
    df2 = df2.dropna(axis=0, how='any')

    print()
    print("Dataframe after dropping NaN values: ", df2.shape)

    # 'gender' convert to int
    df2['is_male'] = df['gender'].apply(lambda x: 1 if x=='male' else 0)
    df2 = df2.drop(columns= 'gender')

    df2_scaled = df2.copy()
    scaler = StandardScaler()
    X = df2_scaled.drop(columns=['id'])
    X_scaled = scaler.fit_transform(X)

    X_umap = UMAP(**umap_params).fit_transform(X_scaled)

    clusterer = HDBSCAN(**hdbscan_params).fit(X_umap)
    
    #need to get a better name here
    save_folder = run_folder_name
    filename = f"{run.name}_umap_hdbscan_scaled"

    save_dir_temp = os.path.join(save_dir, save_folder)
    os.makedirs(save_dir_temp, exist_ok=True)
    base_name, results_df = save_results(df2, clusterer, {
        'umap': umap_params,
        'hdbscan': hdbscan_params
    }, scaler, save_dir_temp, filename, use_wandb=True)

    results_df['id'] = results_df['id'].apply(fix_id)

    noise_count = (results_df['cluster_label']==-1).sum()

    df_filtered = results_df[results_df['cluster_label'] != -1]
    avg_probs = df_filtered.groupby('cluster_label')['probability'].mean().sort_values(ascending=False)
    wandb.log({"avg_probs": avg_probs.mean()})
    avg_probs.to_csv(os.path.join(save_dir_temp, f"{base_name}_avg_probs_per_cluster.csv"))
            
    p_dist = df_filtered['probability'] / np.sum(df_filtered['probability'])
    membership_entropy = entropy(p_dist, base=2)
    H_max = np.log2(len(p_dist))

    wandb.log({"entropy": membership_entropy,
                "normalized_entropy": membership_entropy / H_max})
            
    entropy_per_cluster = df_filtered.groupby('cluster_label')['probability'].apply(
                    normalized_entropy
                ).sort_values()
    entropy_per_cluster.to_csv(os.path.join(save_dir_temp, f"{base_name}_entropy_per_cluster.csv"))

    kl_df = pd.read_csv(kl_filepath)

    df_merged = df_filtered.merge(kl_df, left_on='id', right_on='name', how='left', validate='one_to_one')
    wandb.log({"missing_kl_scores": len(df_merged[df_merged['KL-Score'].isna()])})

    df_merged.to_csv(os.path.join(save_dir_temp, f"{base_name}_wKL.csv"), index=False)
    df_merged = df_merged.dropna(subset=['KL-Score'])
    spr, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df_merged, score = 'cluster_label', label = 'KL-Score')
    nmi = normalized_mutual_info_score(df_merged['KL-Score'], df_merged['cluster_label'])

    wandb.log({
        "spearman_correlation": spr,
        "auc": auc,
        "auc_mid": auc_mid,
        "auc_mid2": auc_mid2,
        "auc_sev": auc_sev,
        "nmi": nmi
    })

    plot_hdbscan(X_umap, clusterer.labels_, 
                probabilities=clusterer.probabilities_, 
                save_path=os.path.join(save_dir, f"{base_name}_plot.png"))
    
    wandb.finish()
