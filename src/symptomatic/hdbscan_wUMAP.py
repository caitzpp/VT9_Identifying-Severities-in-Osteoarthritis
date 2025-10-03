import os
import datetime
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import HDBSCAN
from umap import UMAP

import config
from utils.hdbscan_utils import save_results, plot_hdbscan, prep_data, get_metrics_hdbscan, train_fold, save_umap_true_plot

today = datetime.date.today()



base_dir = config.RAW_DATA_PATH
proc_dir = config.PROC_DATA_PATH

run = "run150"
save_dir = os.path.join(proc_dir, "symptomatic","aggr",  run)
os.makedirs(save_dir, exist_ok=True)

folder = "2025-07-28_data_exploration"
df_filename = "inmodi_data_personalinformation_kl_woSC.csv"
# folder = "2025-08-11_data_exploration"
# df_filename = "inmodi_data_questionnaire_kl_woSC.csv"

if "personalinformation" in df_filename:
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
elif "questionnaire" in df_filename:
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

random_state=42
replace_NanValues = False
wUMAP = True
k=None
use_wandb = False

umap_params = {
    'n_neighbors': 15,
    'min_dist': 0.001,
    'n_components': 3,
    'metric': 'euclidean',
    #'random_state': random_state
}

hdbscan_params = {
    "min_cluster_size": 10,
            "min_samples": 5,
            "cluster_selection_method": "eom",
            "metric": "euclidean",
            "cluster_selection_epsilon": 0.0,
            "algorithm": "auto",
            "metric_params": None,
            "max_cluster_size": None,
            "leaf_size": 40,
            "store_centers": "centroid",
            "alpha": 2
}


if __name__=="__main__":
    df = pd.read_csv(os.path.join(proc_dir, folder, df_filename))

    df2 = df[cols].copy()

    if replace_NanValues==False:
        print("Dataframe before dropping NaN values: ", df2.shape)
        df2 = df2.dropna(axis=0, how='any')

        print()
        print("Dataframe after dropping NaN values: ", df2.shape)

    # 'gender' convert to int
    if "gender" in df2.columns:
        df2['is_male'] = df2['gender'].apply(lambda x: 1 if x=='male' else 0)
        df2 = df2.drop(columns= 'gender')

    scaler = StandardScaler()
    
    X_umap, y, df2_scaled = prep_data(df2, scaler, umap_params = umap_params, wUMAP=wUMAP, id_col='name', y_value='KL-Score')

    filename = f"{today}_umap_hdbscan_scaled"

    if wUMAP:
        np.savez_compressed(os.path.join(save_dir, f"{today}_umap_embedding.npz"), X_umap=X_umap)
        save_umap_true_plot(X_umap, y, out_path = os.path.join(save_dir, f"{today}_umap.png"), 
                            umap_params=umap_params)
    # save_dir_temp = os.path.join(save_dir, save_folder)
    # os.makedirs(save_dir_temp, exist_ok=True)

    if k is None:
        print("\n--- No cross-validation, training on full dataset ---")
        clusterer = HDBSCAN(**hdbscan_params).fit(X_umap)

        ch_score = calinski_harabasz_score(X_umap, clusterer.labels_)

        base_name, results_df = save_results(df=df2_scaled, clusterer=clusterer, params={
            'umap': umap_params,
            'hdbscan': hdbscan_params
        }, scaler=scaler, save_dir=save_dir, filename=filename, id='name', use_wandb=use_wandb)
    
        get_metrics_hdbscan(results_df, df, save_dir, base_name, score='cluster_label', label='KL-Score', use_wandb=use_wandb)

        # Plot the results
        plot_hdbscan(X_umap, clusterer.labels_,
                    probabilities=clusterer.probabilities_,
                    save_path=os.path.join(save_dir, f"{base_name}_plot.png"))
    elif k is not None:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state) if k is not None else None


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
                                                                           save_dir_temp=save_dir)
            noise_count, avg_probs, membership_entropy, normalized_entropy, spr, auc, auc_mid, auc_mid2, auc_sev, nmi = get_metrics_hdbscan(results_df, df, save_dir, base_name, score='cluster_label', label='KL-Score', use_wandb=use_wandb, fold=fold)
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
                    save_path=os.path.join(save_dir, f"{base_name}__{fold}_plot.png"))