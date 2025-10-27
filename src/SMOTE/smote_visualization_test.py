import smote_variants as sv
import sklearn.datasets as datasets
from utils.load_utils import DataLoader
import config
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROCESSED_DATA_PATH = config.PROC_DATA_PATH
RAW_DATA_PATH = config.RAW_DATA_PATH

modelfolder = "2025-10-19_hdbscan"
run = "run27"

folder = "2025-08-11_data_exploration"
df_filename = "inmodi_data_questionnaire_kl_woSC.csv"

cols = [
    # 'record_id', 'visit', 'side', 
    'pain', 'age', #'ce_height', 'ce_weight',
    'ce_bmi', 'ce_fm', 
       'KL-Score',  'koos_s1', 
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
       'koos_q1', 'koos_q2', 'koos_q3', 'koos_q4',
       'oks_q1', 'oks_q2', 'oks_q3', 'oks_q4',
       'oks_q5', 'oks_q6', 'oks_q7', 'oks_q8', 'oks_q9', 'oks_q10', 'oks_q11',
       'oks_q12', 
       'gender',  #keep because I believe it will be relevant
       'name', 
       'KL-Score'
       ]

if __name__ == "__main__":
    dataloader = DataLoader(os.path.join(PROCESSED_DATA_PATH, folder))
    df = dataloader.load_csv(df_filename)

    df2 = df[cols].copy()
    df2['is_male'] = df2['gender'].apply(lambda x: 1 if x=='male' else 0)
    df2 = df2.drop(columns= 'gender')

    #umap_model.pkl
    model = joblib.load(os.path.join(PROCESSED_DATA_PATH, modelfolder,'pipeline', run, 'umap_model.pkl'))
    scaler = joblib.load(os.path.join(PROCESSED_DATA_PATH, modelfolder, 'pipeline', run, 'scaler.pkl'))
    print(scaler.feature_names_in_)
    cols2 = cols.copy()
    cols2.remove('KL-Score')
    cols2.remove('name')
    cols2.remove('gender')
    cols2.append('is_male')
    df2 = df2.dropna(axis=0, how='any', subset= scaler.feature_names_in_)

    #show if any nan values
    print(df2[scaler.feature_names_in_].isna().sum())
    X, y= df2[scaler.feature_names_in_], df2['KL-Score'].iloc[:, 0]

    X_scaled = scaler.transform(X)
    # print(y)

    oversampler= sv.MulticlassOversampling(oversampler='distance_SMOTE',
                                        oversampler_params={'random_state': 5})

    # X_samp and y_samp contain the oversampled dataset
    X_samp, y_samp= oversampler.sample(X_scaled, y)

    X_gen = scaler.inverse_transform(X_samp)

    df_samp = pd.DataFrame(X_gen, columns=scaler.feature_names_in_)
    df_samp['KL-Score'] = y_samp

    df_samp.to_csv(os.path.join(PROCESSED_DATA_PATH, modelfolder,'pipeline', run, 'smote_oversampled_data.csv'), index=False)

    X_umap_samp = model.transform(X_samp)
    X_umap_real = model.transform(X_scaled)

    print("UMAP embedding shapes:")
    print("Real data:", X_umap_real.shape)
    print("Generated data:", X_umap_samp.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X_umap_real[:, 0],
        X_umap_real[:, 1],
        X_umap_real[:, 2],
        label="Real Data",
        alpha=0.6,
        s=40,
        c="tab:blue"
    )

    ax.scatter(
        X_umap_samp[:, 0],
        X_umap_samp[:, 1],
        X_umap_samp[:, 2],
        label="Synthetic Data",
        alpha=0.6,
        s=40,
        c="tab:orange"
    )

    # -------------------------------------------------
    # 4. Customize & show
    # -------------------------------------------------
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend()
    plt.tight_layout()
    plt.show()
