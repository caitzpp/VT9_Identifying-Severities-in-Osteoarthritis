import wandb
import config

import os
# os.environ["WANDB_MODE"] = "disabled"
import pandas as pd
import numpy as np
import json
import joblib
import datetime
from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import HDBSCAN
from hdbscan import HDBSCAN
import hdbscan
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score, silhouette_score, davies_bouldin_score

from utils.load_utils import fix_id, get_next_run_folder
from utils.hdbscan_utils import  save_results, plot_hdbscan,  get_metrics_hdbscan, external_validation, get_hdbscan_umap_defaults, external_validation_2, smote_data_preparation, save_results_SMOTE, get_test_train_lists, get_train_test_dfs

import sys

today = datetime.date.today()

base_dir = config.RAW_DATA_PATH
proc_dir = config.PROC_DATA_PATH
output_dir = config.OUTPUT_PATH

folder = None
random_state=42

STAGE = 'ss'
#MOD_PREFIX = "mod_2"
MOD_PREFIX = "mod_smallimg3"
NEPOCH = 400

n = 90
min_n_clusters = 3
sil_threshold = 0

if folder is not None:
    save_dir = os.path.join(proc_dir, folder)
else:
    save_dir = os.path.join(proc_dir, f"{today}_hdbscan", 'comb_modalities')

os.makedirs(save_dir, exist_ok=True)

folder = "2025-09-11_data_exploration"
df_filename = "inmodi_data_questionnaire_kl_woSC.csv"

umap_folder = '2026-01-16_umap_scaler_values'
# umap_path = os.path.join(proc_dir, umap_folder, 'pipeline')

smote_type = 'Borderline_SMOTE2'

externaldf_path = os.path.join(base_dir, '2025-09-25_mrismall.csv')
externalcols = ['mri_cart_yn', 'mri_osteo_yn']

outputs = os.path.join(output_dir, 'outputs', 'dfs', STAGE)
anomalyscore_metric = "centre_mean"

df_aggscore_filename = f"{MOD_PREFIX}_{STAGE}_aggregated_scores.csv"
df_aggscore_path = os.path.join(outputs, df_aggscore_filename)

feature_groups = {
    "pi": ['pain', 'age', 'ce_bmi', 'ce_fm'],
    "koos": [f"koos_s{i}" for i in range(1, 8)] +
             [f"koos_p{i}" for i in range(1, 10)] +
             [f"koos_a{i}" for i in range(1, 18)] +
             [f"koos_sp{i}" for i in range(1, 6)] +
             [f"koos_q{i}" for i in range(1, 5)],
    "oks": [f"oks_q{i}" for i in range(1, 13)],
    "gender": ['gender']
}

wandb.login(key=config.HDBSCAN_SYMP_WANDBAPI_KEY)

if __name__ == "__main__":
    np.random.seed(random_state)
    run_folder_name, run_path = get_next_run_folder(save_dir)

    run = wandb.init(
        name = f"{os.path.basename(save_dir)}_{run_folder_name}"
        )
    wandb_config = wandb.config

    wandb.log({'ssfewsome_model': MOD_PREFIX})

    # Combine selected columns
    flags = {"pi": wandb_config.get("pi", True), "koos": wandb_config.get("koos", True), 
             "oks": wandb_config.get("oks", True), "gender": wandb_config.get("gender", True)}
    cols = [col for key, active in flags.items() if active for col in feature_groups[key]]
    cols += ['name', 'KL-Score']

    wUMAP = wandb_config.get('wUMAP', True)
    replace_NanValues = wandb_config.get('replace_NanValues', False)
    smote = wandb_config.get('smote', True)

    umap_path = os.path.join(proc_dir, umap_folder, 'comb_modalities')

    umap_keys, umap_defaults, hdbscan_keys, hdbscan_defaults = get_hdbscan_umap_defaults()
    
    if wUMAP:
        umap_params = {k: wandb_config.get(f'umap.{k}', umap_defaults[k]) for k in umap_keys}
    else:
        umap_params = "woUMAP"
    hdbscan_params = {k: wandb_config.get(f'hdbscan.{k}', hdbscan_defaults[k]) for k in hdbscan_keys}

    if hdbscan_params['min_samples'] == -1:
        hdbscan_params['min_samples'] = None

    df = pd.read_csv(os.path.join(proc_dir, folder, df_filename))
    externaldf = pd.read_csv(externaldf_path)

    df2 = df[cols].copy()
    if replace_NanValues==False:
        print("Dataframe before dropping NaN values: ", df2.shape)
        df2 = df2.dropna(axis=0, how='any')

        print()
        print("Dataframe after dropping NaN values: ", df2.shape)

    # 'gender' convert to int
    gender = wandb_config.get("gender", True)
    if gender:
        df2['is_male'] = df['gender'].apply(lambda x: 1 if x=='male' else 0)
        df2 = df2.drop(columns= 'gender')
    save_folder = run_folder_name
    filename = f"{run.name}_umap_hdbscan_scaled"

    agg_df = pd.read_csv(df_aggscore_path)
    agg_df['name'] = agg_df['id'].str.split('/').str[-1].str.replace('.png', '', regex=False)
    df2 = df2.merge(agg_df[['name', 'mean']], on='name', how='left')

    save_dir_temp = os.path.join(save_dir, save_folder)
    os.makedirs(save_dir_temp, exist_ok=True)
     
    # umap_folder = f'nneigh5_mindist{umap_params["min_dist"]}_metric{umap_params["metric"]}'
    umap_folder = f'nneigh{umap_params["n_neighbors"]}_mindist{umap_params["min_dist"]}_metric{umap_params["metric"]}'
    umap_path = os.path.join(umap_path, umap_folder)

    #load X_umap
    X_umap_train = np.load(os.path.join(umap_path, "X_umap_embeddings.npy"))
    X_samp_umap_train = np.load(os.path.join(umap_path, "X_umap_samp_embeddings.npy"))

    scaler = joblib.load(os.path.join(umap_path, "scaler.pkl"))
    umap = joblib.load(os.path.join(umap_path, "umap_model.pkl"))

    if smote:
        clusterer = HDBSCAN(**hdbscan_params).fit(X_samp_umap_train)
        
    else:
        clusterer = HDBSCAN(**hdbscan_params).fit(X_umap_train)


    y_pred_train, strengths_train = hdbscan.approximate_predict(clusterer, X_umap_train)
    df2_train, df2_test, ids_train, ids_test, y_test, y_pred_test, strengths_test, X_umap_test = get_train_test_dfs(df2, clusterer, umap, scaler, umap_path, base_dir, id_col='name', y_col='KL-Score', save_path=save_dir_temp)

    clusterer_path = os.path.join(save_dir_temp, f"{filename}_clusterer.pkl")
    joblib.dump(clusterer, clusterer_path)

    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    if n_clusters>1:
        ch_score = calinski_harabasz_score(X_umap_train, y_pred_train)
        adj_chscore = ch_score / n_clusters if n_clusters > 1 else 0
        sil_score = silhouette_score(X_umap_train, y_pred_train)
        db_score = davies_bouldin_score(X_umap_train, y_pred_train)

        ch_score_test = calinski_harabasz_score(X_umap_test, y_pred_test)
        adj_chscore_test = ch_score_test / n_clusters if n_clusters > 1 else 0
        sil_score_test = silhouette_score(X_umap_test, y_pred_test)
        db_score_test = davies_bouldin_score(X_umap_test, y_pred_test)
    else:
        print("Stopping due to only one cluster found")
        sys.exit(1)

    noise_count = np.sum(clusterer.labels_ == -1)

    cont_pipeline = False  # default

    if noise_count >= n:
        print(f"Too much noise found: {noise_count} >= {n}")

    elif n_clusters < min_n_clusters:
        print(f"Not enough clusters found: {n_clusters} < {min_n_clusters}")

    elif sil_score_test <= sil_threshold:
        print(f"Silhouette score too low: {sil_score_test:.3f} < {sil_threshold}")

    else:
        cont_pipeline = True
    
    if cont_pipeline:
        base_name, results_df_train = save_results(df=df2_train, ypred = y_pred_train, strengths = strengths_train,  clusterer=clusterer, params={
            'umap': umap_params,
            'hdbscan': hdbscan_params
        }, scaler=scaler, save_dir=save_dir_temp, artifacts=None, filename=filename, id='name', use_wandb=True,
        smote=smote)

        df_filename_test = filename + "_test_results.csv"
        results_df_test = pd.DataFrame({'id':df2_test['name'],
                                        'cluster_label': y_pred_test,
                                        'probability': strengths_test})
        results_df_test.to_csv(os.path.join(save_dir_temp, df_filename_test), index=False)

        get_metrics_hdbscan(results_df = results_df_train, kl_df = df, 
                            save_dir_temp = save_dir_temp, base_name = base_name, 
                            clusterer = clusterer, score = 'cluster_label', label = 'KL-Score', 
                            use_wandb = True, smote = smote, test=False)
        get_metrics_hdbscan(results_df = results_df_test, kl_df = df, 
                            save_dir_temp = save_dir_temp, base_name = base_name + "_test", 
                            clusterer = clusterer, score = 'cluster_label', label = 'KL-Score', 
                            use_wandb = True, smote = smote, test=True)

        results = external_validation(results_df_train, externaldf, 
                                      chadjustd= adj_chscore, label = 'cluster_label', 
                                      external_cols = externalcols, leftid_col = 'id', 
                                      rightid_col='id', use_wandb=True, test=False)
        results_test = external_validation(results_df_test, externaldf, 
                                      chadjustd= adj_chscore_test, label = 'cluster_label', 
                                      external_cols = externalcols, leftid_col = 'id', 
                                      rightid_col='id', use_wandb=True, test=True)
        
        combined = pd.read_csv(df_aggscore_path)
        combined['filepath'] = combined['id']
        combined['id'] = combined['id'].apply(lambda x: x.split('/')[-1].replace('.png', ''))

        _ = external_validation_2(results_df_train, combined, val_column='mean', 
                                  cluster_col = 'cluster_label', 
                                  use_wandb = True, test=False)
        _ = external_validation_2(results_df_test, combined, val_column='mean', 
                                  cluster_col = 'cluster_label', 
                                  use_wandb = True, test=True)
        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        if smote: 
            plot_hdbscan(X_umap_train, y_pred_train, 
                         save_path = os.path.join(save_dir_temp, f"{base_name}_plot_train.png"),
                         n_clusters=n_clusters)
            plot_hdbscan(X_umap_test, y_pred_test, 
                         save_path = os.path.join(save_dir_temp, f"{base_name}_plot_test.png"),
                         n_clusters=n_clusters)
            plot_hdbscan(X_samp_umap_train, clusterer.labels_, 
                            save_path = os.path.join(save_dir_temp, f"{base_name}_plot_smotedata.png"),
                            n_clusters=n_clusters)
        else:
            plot_hdbscan(X_umap_train, y_pred_train, 
                         save_path = os.path.join(save_dir_temp, f"{base_name}_plot_train.png"),
                         n_clusters=n_clusters)
            plot_hdbscan(X_umap_test, y_pred_test, 
                         save_path = os.path.join(save_dir_temp, f"{base_name}_plot_test.png"),
                         n_clusters=n_clusters)
            
        wandb.log(
                {'calinski_harabasz_score': np.round(ch_score, 3),
                'calinski_harabasz_score_adjusted': np.round(adj_chscore, 3),
                #  'calinski_harabasz_score_adjusted_v2': np.round(adj_chscore2, 3)
                'silhouette_score': np.round(sil_score, 3),
                'davies_bouldin_score': np.round(db_score, 3),
                'n_clusters': n_clusters,
                    'noise_count': noise_count,
                'calinski_harabasz_score_test': np.round(ch_score_test, 3),
                'calinski_harabasz_score_adjusted_test': np.round(adj_chscore_test, 3),
                'silhouette_score_test': np.round(sil_score_test, 3),
                'davies_bouldin_score_test': np.round(db_score_test, 3)
            })
        wandb.finish()
    else:
        wandb.log(
                {'calinski_harabasz_score': np.round(ch_score, 3),
                'calinski_harabasz_score_adjusted': np.round(adj_chscore, 3),
                #  'calinski_harabasz_score_adjusted_v2': np.round(adj_chscore2, 3)
                'silhouette_score': np.round(sil_score, 3),
                'davies_bouldin_score': np.round(db_score, 3),
                'n_clusters': n_clusters,
                    'noise_count': noise_count,
                    'calinski_harabasz_score_test': np.round(ch_score_test, 3),
                'calinski_harabasz_score_adjusted_test': np.round(adj_chscore_test, 3),
                'silhouette_score_test': np.round(sil_score_test, 3),
                'davies_bouldin_score_test': np.round(db_score_test, 3)
            })
        wandb.finish()
        sys.exit("Stopping pipeline due to not meeting criteria")
    

