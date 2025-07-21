import config
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from IPython.display import display
import datetime
from utils.hdbscan_utils import get_unique_filepath, save_results, plot_hdbscan

from sklearn.preprocessing import StandardScaler
#import hdbscan
from umap import UMAP
from sklearn.cluster import HDBSCAN

today = datetime.date.today()

base_dir = config.RAW_DATA_PATH
proc_dir = config.PROC_DATA_PATH

folder = '2025-07-18_hdbscan'

if folder is not None:
    save_dir = os.path.join(proc_dir, folder)
else:
    save_dir = os.path.join(proc_dir, f"{today}_hdbscan")
os.makedirs(save_dir, exist_ok=True)

folder = "2025-07-14_data_exploration"
unpivoted = True

if unpivoted:
    df = pd.read_csv(os.path.join(proc_dir, folder, "inmodi_data_personalinformation_unpivoted.csv"))
else:
    df = pd.read_csv(os.path.join(proc_dir, folder, "inmodi_data_personalinformation.csv"))

#Choose relevant columns
cols = ['record_id', # id column
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

params = {}

if __name__ == "__main__":
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
    X = df2_scaled.drop(columns=['record_id'])
    X_scaled = scaler.fit_transform(X)

    X_umap = UMAP().fit_transform(X_scaled)

    clusterer = HDBSCAN(**params)
    clusterer = clusterer.fit(X_umap)

    base_name = save_results(df2, clusterer, params, scaler, save_dir, 'hdbscan_scaled_umap')
    plot_hdbscan(X_scaled, clusterer.labels_, 
                probabilities=clusterer.probabilities_, 
                #parameters={'parameters': 'default'},
                save_path = os.path.join(save_dir, f"{base_name}_plot.png"))
