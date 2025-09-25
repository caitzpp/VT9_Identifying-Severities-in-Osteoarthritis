import wandb

import config
import os
import pandas as pd
import numpy as np
from IPython.display import display
from itertools import product
import joblib
import datetime
from utils.hdbscan_utils import get_unique_filepath, save_results, plot_hdbscan, prep_data, get_metrics_hdbscan, train_fold, external_validation

from sklearn.preprocessing import StandardScaler
#import hdbscan
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score
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

project_name = 'symptomatic_rawq_hdbscan_pipeline_dev'

random_state=42
stratify_on = 'KL-Score'

if folder is not None:
    save_dir = os.path.join(proc_dir, folder)
else:
    save_dir = os.path.join(proc_dir, f"{today}_hdbscan", 'questionnaire')
os.makedirs(save_dir, exist_ok=True)

# folder = "2025-07-28_data_exploration"
# df_filename = "inmodi_data_personalinformation_kl_woSC.csv"
folder = "2025-08-11_data_exploration"
df_filename = "inmodi_data_questionnaire_kl_woSC.csv"

#TODO: bring in MRI data, eval on jsn and osteophyte data specifically
externaldf_path = os.path.join(base_dir, '2025-09-25_mrismall.csv')
externalcols = ['mri_cart_yn', 'mri_osteo_yn']

cols = [
    # 'record_id', 'visit', 'side', 
    # 'pain', 'age', #'ce_height', 'ce_weight',
    #    'ce_bmi', 'ce_fm', 
        'gender',  #keep because I believe it will be relevant
       'name', 
       'KL-Score',  
       'oks_q1', 'oks_q2', 'oks_q3', 'oks_q4',
       'oks_q5', 'oks_q6', 'oks_q7', 'oks_q8', 'oks_q9', 'oks_q10', 'oks_q11',
       'oks_q12', 'koos_s1', 
       'koos_s2', 'koos_s3', 'koos_s4', 'koos_s5', 'koos_s6',
       'koos_s7', 
       'koos_p1', 'koos_p2', 'koos_p3', 'koos_p4', 'koos_p5',
       'koos_p6', 'koos_p7', 'koos_p8', 'koos_p9', 
       'koos_a1', 'koos_a2',
       'koos_a3', 'koos_a4', 'koos_a5', 'koos_a6', 'koos_a7', 'koos_a8',
       'koos_a9', 'koos_a10', 'koos_a11', 'koos_a12', 'koos_a13', 'koos_a14',
       'koos_a15', 'koos_a16', 'koos_a17', 
       'koos_sp1', 'koos_sp2', 'koos_sp3',
       'koos_sp4', 'koos_sp5', 
       'koos_q1', 'koos_q2', 'koos_q3', 'koos_q4'
       ]

umap_keys = ['n_neighbors', 'min_dist', 'n_components', 'metric']
hdbscan_keys = ['min_cluster_size', 'min_samples', 'cluster_selection_method', 
                'metric', 'metric_params', 'max_cluster_size',
                'cluster_selection_epsilon', 'algorithm', 'leaf_size',
                'store_centers', 'alpha']

umap_defaults = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42
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
        'alpha': 1.0,
        'approx_min_span_tree': False,
    }
wandb.login(key=config.HDBSCAN_SYMP_WANDBAPI_KEY)

config_w = {

}

if __name__ == "__main__":
    np.random.seed(random_state)
    run_folder_name, run_path = get_next_run_folder(save_dir)

    externaldf = pd.read_csv(externaldf_path)
    df = pd.read_csv(os.path.join(proc_dir, folder, df_filename))
    df2 = df[cols].copy()
    df2['is_male'] = df2['gender'].apply(lambda x: 1 if x=='male' else 0)
    df2 = df2.drop(columns= 'gender')

    run = wandb.init(
        project=project_name,
        name = f"{os.path.basename(save_dir)}_{run_folder_name}"
        , config = config_w
        )
    wandb_config = run.config
  
    wUMAP = wandb_config.get('wUMAP', True)
    replace_NanValues = wandb_config.get('replace_NanValues', False)
    k = wandb_config.get('k-cv', None)

    if wUMAP:
        umap_params = {k: wandb_config.get(f'umap.{k}', umap_defaults[k]) for k in umap_keys}
    else:
        umap_params = "woUMAP"
    hdbscan_params = {k: wandb_config.get(f'hdbscan.{k}', hdbscan_defaults[k]) for k in hdbscan_keys}

    if hdbscan_params['min_samples'] == -1:
        hdbscan_params['min_samples'] = None
    

    if replace_NanValues==False:
        print("Dataframe before dropping NaN values: ", df2.shape)
        df2 = df2.dropna(axis=0, how='any')

        print()
        print("Dataframe after dropping NaN values: ", df2.shape)

    save_folder = run_folder_name
    filename = f"{run.name}_umap_hdbscan_scaled"

    save_dir_temp = os.path.join(save_dir, save_folder)
    os.makedirs(save_dir_temp, exist_ok=True)

    scaler = StandardScaler()
    
    X_umap, y, df2_scaled, artifacts = prep_data(df=df2, scaler=scaler, umap_params = umap_params, wUMAP=wUMAP, id_col='name', y_value='KL-Score', save_path=save_dir_temp)

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state) if k is not None else None

    if kf is None:
        print("\n--- No cross-validation, training on full dataset ---")
        clusterer = HDBSCAN(**hdbscan_params).fit(X_umap)
        clusterer_path = os.path.join(save_dir_temp, f"{filename}_clusterer.pkl")
        joblib.dump(clusterer, clusterer_path)
        artifacts['clusterer'] = clusterer_path

        ch_score = calinski_harabasz_score(X_umap, clusterer.labels_)

        wandb.log(
            {'calinski_harabasz_score': ch_score,
             'calinski_harabasz_scorev2': int(ch_score)}
        )

        base_name, results_df = save_results(df=df2_scaled, clusterer=clusterer, params={
            'umap': umap_params,
            'hdbscan': hdbscan_params
        }, scaler=scaler, save_dir=save_dir_temp, artifacts = artifacts, filename=filename, id='name', use_wandb=True)
    
        _, _ = get_metrics_hdbscan(results_df, df, save_dir_temp, base_name, score='cluster_label', label='KL-Score', use_wandb=True)
        
        results = external_validation(results_df, externaldf, label = 'cluster_label', external_cols = externalcols, leftid_col = 'id', rightid_col='id', use_wandb=True)



        # Plot the results
        # plot_hdbscan(X_umap, clusterer.labels_,
        #             probabilities=clusterer.probabilities_,
        #             save_path=os.path.join(save_dir_temp, f"{base_name}_plot.png"))
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
        l_chscore = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_umap, y)):
            print(f"\n--- Fold {fold+1} ---")
            base_name, results_df, clusterer, X_train, df_train, _, _, ch_score = train_fold(fold=fold, 
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
            l_chscore.append(ch_score)

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
            "std_nmi": np.std(l_nmi),
            "calinski_harabasz_score": ch_score
        })
    
    wandb.finish()
