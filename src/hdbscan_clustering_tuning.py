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

# umap_params_grid = {
#     'n_neighbors': [5, 15],
#     'min_dist': [0.1, 0.5],
#     'n_components': [2],
#     'metric': ['euclidean']
# }

# hdbscan_params_grid = {
#     'min_cluster_size': [5, 10, 20],
#     'min_samples': [None, 5, 10],
#     'cluster_selection_method': ['eom', 'leaf'],
#     'metric': ['euclidean'],
#     'metric_params': [None],  # Default is None, can be adjusted for performance
#     'max_cluster_size': [None], # None means no limit
#     'cluster_selection_epsilon': [0.0],
#     'algorithm': ['auto'],
#     'leaf_size': [40],  # Default is 40, can be adjusted for performance
#     'store_centers': ['centroid'],  # Not default, but want to keep
#     'alpha': [1.0]  # Default is 1.0, can be adjusted for performance
# }


wandb.login(key=config.HDBSCAN_SYMP_WANDBAPI_KEY)

if __name__ == "__main__":
    wandb.init(
        project="HDBSCAN_SymptomaticData",
        config={
            "umap": {
                'n_neighbors': 5,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean'
            },
            "hdbscan": {
                    'min_cluster_size': 5,
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
        })
    wandb_config = wandb.config

    umap_params = wandb_config.umap
    hdbscan_params = wandb_config.hdbscan

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

    # umap_combos = list(product(*umap_params_grid.values()))
    # hdbscan_combos = list(product(*hdbscan_params_grid.values()))

    run_id = 0

    # for umap_vals in umap_combos:
        # umap_params = dict(zip(umap_params_grid.keys(), umap_vals))
        # #print(umap_params)

    X_umap = UMAP(**umap_params).fit_transform(X_scaled)

        # for hdb_vals in hdbscan_combos:
        #     hdbscan_params = dict(zip(hdbscan_params_grid.keys(), hdb_vals))
        #     #print(hdbscan_params)
    clusterer = HDBSCAN(**hdbscan_params).fit(X_umap)

            # print(f"Run ID: {run_id}, UMAP Params: {umap_params}, HDBSCAN Params: {hdbscan_params}")

            # # Save results
            # run_id += 1
            # save_folder = f"run_{run_id}_umap_{umap_vals}_hdbscan_{hdb_vals}".replace(" ", "")
            # filename = f"run_{run_id}_umap_hdbscan_scaled"
    
    save_folder = f"test_umap_hdbscan".replace(" ", "")
    filename = f"test_umap_hdbscan_scaled"

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
