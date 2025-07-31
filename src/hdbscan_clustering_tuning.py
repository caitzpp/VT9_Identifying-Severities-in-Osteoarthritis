import wandb

import config
import os
import pandas as pd
import numpy as np
from IPython.display import display
from itertools import product
import datetime
from utils.hdbscan_utils import get_unique_filepath, save_results, plot_hdbscan, prep_data, get_metrics_hdbscan, train_fold

from sklearn.preprocessing import StandardScaler
#import hdbscan
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from scipy.stats import entropy
from utils.load_utils import fix_id, get_next_run_folder
#from utils.evaluation_utils import normalized_entropy, get_metrics

import sys

today = datetime.date.today()

base_dir = config.RAW_DATA_PATH
proc_dir = config.PROC_DATA_PATH

#folder = '2025-07-18_hdbscan'
folder = None

random_state=42
stratify_on = 'KL-Score'

if folder is not None:
    save_dir = os.path.join(proc_dir, folder)
else:
    save_dir = os.path.join(proc_dir, f"{today}_hdbscan")
os.makedirs(save_dir, exist_ok=True)

folder = "2025-07-28_data_exploration"
# kl_file = "inmodi_data_personalinformation_kl.csv"
# kl_filepath = os.path.join(proc_dir, folder, kl_file)

#Choose relevant columns
cols = ['name',
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
        'KOOS_qol',
        'KL-Score'
    ]

wandb.login(key=config.HDBSCAN_SYMP_WANDBAPI_KEY)

if __name__ == "__main__":
    run_folder_name, run_path = get_next_run_folder(save_dir)

    run = wandb.init(
        name = f"{os.path.basename(save_dir)}_{run_folder_name}"
        )
    wandb_config = wandb.config

    wUMAP = wandb_config.get('wUMAP', True)
    replace_NanValues = wandb_config.get('replace_NanValues', False)
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

    df = pd.read_csv(os.path.join(proc_dir, folder, "inmodi_data_personalinformation_kl_woSC.csv"))

    df2 = df[cols].copy()

    if replace_NanValues==False:
        print("Dataframe before dropping NaN values: ", df2.shape)
        df2 = df2.dropna(axis=0, how='any')

        print()
        print("Dataframe after dropping NaN values: ", df2.shape)

    # 'gender' convert to int
    df2['is_male'] = df['gender'].apply(lambda x: 1 if x=='male' else 0)
    df2 = df2.drop(columns= 'gender')

    scaler = StandardScaler()
    
    X_umap, y, df2_scaled = prep_data(df2, scaler, umap_params = umap_params, wUMAP=wUMAP, id_col='name', y_value='KL-Score')

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state) if k is not None else None

    save_folder = run_folder_name
    filename = f"{run.name}_umap_hdbscan_scaled"

    save_dir_temp = os.path.join(save_dir, save_folder)
    os.makedirs(save_dir_temp, exist_ok=True)

    # if kf is not None:
    #     for fold, (train_idx, test_idx) in enumerate(kf.split(X_umap, y)):
    #         print(f"\n--- Fold {fold+1} ---")
       
    #         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    #         df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]


    #need to get a better name here
    # save_folder = run_folder_name
    # filename = f"{run.name}_umap_hdbscan_scaled"

    # save_dir_temp = os.path.join(save_dir, save_folder)
    # os.makedirs(save_dir_temp, exist_ok=True)

    if kf is None:
        print("\n--- No cross-validation, training on full dataset ---")
        clusterer = HDBSCAN(**hdbscan_params).fit(X_umap)

        base_name, results_df = save_results(df=df2_scaled, clusterer=clusterer, params={
            'umap': umap_params,
            'hdbscan': hdbscan_params
        }, scaler=scaler, save_dir=save_dir_temp, filename=filename, id='name', use_wandb=True)
    
        get_metrics_hdbscan(results_df, df, save_dir_temp, base_name, score='cluster_label', label='KL-Score', use_wandb=True)

        # Plot the results
        plot_hdbscan(X_umap, clusterer.labels_,
                    probabilities=clusterer.probabilities_,
                    save_path=os.path.join(save_dir_temp, f"{base_name}_plot.png"))
    elif kf is not None:
        print("\n--- Cross-validation ---")

        l_noise_count = []
        l_avg_probs = []
        l_entropy = []
        l_normalized_entropy = []
        l_spearman_correlation = []
        l_auc = []
        l_auc_mid = []
        l_auc_mid2 = []
        l_auc_sev = []
        l_nmi = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_umap, y)):
            print(f"\n--- Fold {fold+1} ---")
            base_name, results_df, clusterer, X_train, df_train, _, _ = train_fold(fold=fold, 
                                                                           train_idx=train_idx, 
                                                                           test_idx=test_idx, 
                                                                           X=X_umap, y=y, 
                                                                           df=df2_scaled,
                                                                           hdbscan_params=hdbscan_params,
                                                                           umap_params=umap_params,
                                                                           scaler=scaler,
                                                                           filename=filename,
                                                                           save_dir_temp=save_dir_temp)
            noise_count, avg_probs, membership_entropy, normalized_entropy, spr, auc, auc_mid, auc_mid2, auc_sev, nmi = get_metrics_hdbscan(results_df, df, save_dir_temp, base_name, score='cluster_label', label='KL-Score', use_wandb=True, fold=fold)
            l_noise_count.append(noise_count)
            mean_avg_probs = avg_probs.mean()
            l_avg_probs.append(mean_avg_probs)
            l_entropy.append(membership_entropy)
            l_normalized_entropy.append(normalized_entropy)
            l_spearman_correlation.append(spr)
            l_auc.append(auc)
            l_auc_mid.append(auc_mid)
            l_auc_mid2.append(auc_mid2)
            l_auc_sev.append(auc_sev)
            l_nmi.append(nmi)

            plot_hdbscan(X_train, clusterer.labels_,
                    probabilities=clusterer.probabilities_,
                    save_path=os.path.join(save_dir_temp, f"{base_name}__{fold}_plot.png"))
        
        wandb.log({
            "average_noise_count": np.mean(l_noise_count),
            "average_avg_probs": np.mean(l_avg_probs),
            "average_entropy": np.mean(l_entropy),
            "average_normalized_entropy": np.mean(l_normalized_entropy),
            "average_spearman_correlation": np.mean(l_spearman_correlation),
            "average_auc": np.mean(l_auc),
            "average_auc_mid": np.mean(l_auc_mid),
            "average_auc_mid2": np.mean(l_auc_mid2),
            "average_auc_sev": np.mean(l_auc_sev),
            "average_nmi": np.mean(l_nmi),
            "std_noise_count": np.std(l_noise_count),
            "std_avg_probs": np.std(l_avg_probs),
            "std_entropy": np.std(l_entropy),
            "std_normalized_entropy": np.std(l_normalized_entropy),
            "std_spearman_correlation": np.std(l_spearman_correlation),
            "std_auc": np.std(l_auc),
            "std_auc_mid": np.std(l_auc_mid),
            "std_auc_mid2": np.std(l_auc_mid2),
            "std_auc_sev": np.std(l_auc_sev),
            "std_nmi": np.std(l_nmi)
        })

    #     'umap': umap_params,
    #     'hdbscan': hdbscan_params
    # }, scaler=scaler, save_dir=save_dir_temp, filename=filename, id = 'name', use_wandb=True)

    # results_df['id'] = results_df['id'].apply(fix_id)

    # noise_count = (results_df['cluster_label']==-1).sum()
    # wandb.log({"noise_count": noise_count})

    # df_filtered = results_df[results_df['cluster_label'] != -1]
    # avg_probs = df_filtered.groupby('cluster_label')['probability'].mean().sort_values(ascending=False)
    # wandb.log({"avg_probs": avg_probs.mean()})
    # avg_probs.to_csv(os.path.join(save_dir_temp, f"{base_name}_avg_probs_per_cluster.csv"))
            
    # p_dist = df_filtered['probability'] / np.sum(df_filtered['probability'])
    # membership_entropy = entropy(p_dist, base=2)
    # H_max = np.log2(len(p_dist))

    # wandb.log({"entropy": membership_entropy,
    #             "normalized_entropy": membership_entropy / H_max})
            
    # entropy_per_cluster = df_filtered.groupby('cluster_label')['probability'].apply(
    #                 normalized_entropy
    #             ).sort_values()
    # entropy_per_cluster.to_csv(os.path.join(save_dir_temp, f"{base_name}_entropy_per_cluster.csv"))

    # # kl_df = pd.read_csv(kl_filepath)

    # df_merged = df_filtered.merge(df, left_on = 'id', right_on='name', how='left', validate='one_to_one')
    # wandb.log({"missing_kl_scores": len(df_merged[df_merged['KL-Score'].isna()])})

    # df_merged.to_csv(os.path.join(save_dir_temp, f"{base_name}_wKL.csv"), index=False)
    # df_merged = df_merged.dropna(subset=['KL-Score'])
    # spr, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df_merged, score = 'cluster_label', label = 'KL-Score')
    # nmi = normalized_mutual_info_score(df_merged['KL-Score'], df_merged['cluster_label'])

    # wandb.log({
    #     "spearman_correlation": spr,
    #     "auc": auc,
    #     "auc_mid": auc_mid,
    #     "auc_mid2": auc_mid2,
    #     "auc_sev": auc_sev,
    #     "nmi": nmi
    # })

    # plot_hdbscan(X_umap, clusterer.labels_, 
    #             probabilities=clusterer.probabilities_, 
    #             save_path=os.path.join(save_dir, f"{base_name}_plot.png"))
    
    wandb.finish()
