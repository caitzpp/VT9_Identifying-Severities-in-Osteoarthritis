import wandb
import config

import os
os.environ["WANDB_MODE"] = "disabled"
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import HDBSCAN
from hdbscan import HDBSCAN
import hdbscan
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score, silhouette_score, davies_bouldin_score

from utils.load_utils import fix_id, get_next_run_folder
from utils.hdbscan_utils import  get_hdbscan_umap_defaults, smote_data_preparation, get_test_train_lists

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

# n = 90
# min_n_clusters = 3
# sil_threshold = 0.3
use_aggscore = True

if folder is not None:
    save_dir = os.path.join(proc_dir, folder)
elif use_aggscore:
    save_dir = os.path.join(proc_dir, f"{today}_umap_scaler_values", 'comb_modalities')
else:
    save_dir = os.path.join(proc_dir, f"{today}_umap_scaler_values", 'pipeline')

os.makedirs(save_dir, exist_ok=True)

folder = "2025-09-11_data_exploration"
df_filename = "inmodi_data_questionnaire_kl_woSC.csv"

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
     # Combine selected columns
    flags = {"pi": wandb_config.get("pi", True), "koos": wandb_config.get("koos", True), 
             "oks": wandb_config.get("oks", True), "gender": wandb_config.get("gender", True)}
    cols = [col for key, active in flags.items() if active for col in feature_groups[key]]
    cols += ['name', 'KL-Score']

    wUMAP = wandb_config.get('wUMAP', True)
    replace_NanValues = wandb_config.get('replace_NanValues', False)
    smote = wandb_config.get('smote', False)

    umap_keys, umap_defaults, hdbscan_keys, hdbscan_defaults = get_hdbscan_umap_defaults()
    
    if wUMAP:
        umap_params = {k: wandb_config.get(f'umap.{k}', umap_defaults[k]) for k in umap_keys}
    else:
        umap_params = "woUMAP"

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

    if use_aggscore:
        agg_df = pd.read_csv(df_aggscore_path)
        agg_df['name'] = agg_df['id'].str.split('/').str[-1].str.replace('.png', '', regex=False)
        df2 = df2.merge(agg_df[['name', 'mean']], on='name', how='left')
        trainl, testl = get_test_train_lists(base_dir)

    save_folder = f'nneigh{umap_params["n_neighbors"]}_mindist{umap_params["min_dist"]}_metric{umap_params["metric"]}'

    save_dir_temp = os.path.join(save_dir, save_folder)
    os.makedirs(save_dir_temp, exist_ok=True)

    scaler = StandardScaler()
    
    if smote:
        if use_aggscore:
            df2_train = df2[df2['name'].isin(trainl)].copy()
            X_umap, y, df_scaled, X_samp_umap, y_samp, df_gen, artifacts = smote_data_preparation(df=df2_train, scaler=scaler, 
                               umap_params=umap_params, wUMAP=wUMAP, id_col='name', y_value='KL-Score', 
                               oversample_method=smote_type, save_path=save_dir_temp)
            df_gen.to_csv(os.path.join(save_dir_temp, f"SMOTE_generated_samples.csv"), index=False)
        else:
            X_umap, y, df_scaled, X_samp_umap, y_samp, df_gen, artifacts = smote_data_preparation(df=df2, scaler=scaler, 
                                umap_params=umap_params, wUMAP=wUMAP, id_col='name', y_value='KL-Score', 
                                oversample_method=smote_type, save_path=save_dir_temp)
            df_gen.to_csv(os.path.join(save_dir_temp, f"SMOTE_generated_samples.csv"), index=False)
       

