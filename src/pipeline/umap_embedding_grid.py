import os
import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from utils.load_utils import get_next_run_folder
from utils.hdbscan_utils import smote_data_preparation, plot_umap, plot_umap_3d
import config


today = datetime.date.today()

base_dir = config.RAW_DATA_PATH
proc_dir = config.PROC_DATA_PATH
output_dir = config.OUTPUT_PATH

random_state = 42

STAGE = "ss"
MOD_PREFIX = "mod_smallimg3"

folder = "2025-08-11_data_exploration"
df_filename = "inmodi_data_questionnaire_kl_woSC.csv"

smote_type = "Borderline_SMOTE2"
use_smote = True

n_comp = 3

# Optional: path to an external dataset (you can remove this if unused)
externaldf_path = os.path.join(base_dir, "2025-09-25_mrismall.csv")

save_dir = os.path.join(proc_dir, f"{today}_umap")
os.makedirs(save_dir, exist_ok=True)

feature_groups = {
    "pi": ["pain", "age", "ce_bmi", "ce_fm"],
    "koos": [f"koos_s{i}" for i in range(1, 8)]
    + [f"koos_p{i}" for i in range(1, 10)]
    + [f"koos_a{i}" for i in range(1, 18)]
    + [f"koos_sp{i}" for i in range(1, 6)]
    + [f"koos_q{i}" for i in range(1, 5)],
    "oks": [f"oks_q{i}" for i in range(1, 13)],
    "gender": ["gender"],
}


from umap import UMAP
n_neighbors_list = [5, 20, 40, 80]
min_dist_list = [0.0125, 0.05, 0.2, 0.8]

umap_params = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": n_comp,
    "metric": "manhattan",
    "random_state": random_state,
}


df = pd.read_csv(os.path.join(proc_dir, folder, df_filename))
df2 = df.copy()

# Select columns manually (all feature groups active)
cols = []
for key in ["pi", "koos", "oks", "gender"]:
    cols += feature_groups[key]

cols += ["name", "KL-Score"]

df2 = df2[cols]

# Drop missing rows
df2 = df2.dropna(axis=0, how="any")

# Convert gender
df2["is_male"] = df2["gender"].apply(lambda x: 1 if x == "male" else 0)
df2 = df2.drop(columns="gender")

run_counter = 100
for n_neighbors in n_neighbors_list:
    for min_dist in min_dist_list:

        print(f"\n=== Running UMAP: n_neighbors={n_neighbors}, min_dist={min_dist} ===")

        # Prepare per-run folder
        run_folder_name = f"run_{run_counter:03d}_nn{n_neighbors}_md{min_dist}"
        save_dir_temp = os.path.join(save_dir, run_folder_name)
        os.makedirs(save_dir_temp, exist_ok=True)

      
        umap_params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_comp,
            "metric": "manhattan",
            "random_state": random_state,
        }

   
        scaler = StandardScaler()

 
        if use_smote:

            X_umap, y, df_scaled, X_samp_umap, y_samp, df_gen, artifacts = smote_data_preparation(
                df=df2,
                scaler=scaler,
                umap_params=umap_params,
                wUMAP=True,
                id_col="name",
                y_value="KL-Score",
                oversample_method=smote_type,
                save_path=save_dir_temp
            )

    
            if n_comp==2:
                plot_umap(
                embeddings=X_umap,
                labels=y,
                title=f"UMAP Original (nn={n_neighbors}, md={min_dist})",
                save_path=os.path.join(save_dir_temp, "umap_original.png")
            )

        
                plot_umap(
                    embeddings=X_samp_umap,
                    labels=y_samp,
                    title=f"UMAP SMOTE (nn={n_neighbors}, md={min_dist})",
                    save_path=os.path.join(save_dir_temp, "umap_smote.png")
                )
            elif n_comp==3:
                plot_umap_3d(
                    embeddings=X_umap,
                    labels=y,
                    title=f"UMAP Original (nn={n_neighbors}, md={min_dist})",
                    save_path=os.path.join(save_dir_temp, "umap_original.png")
                )

                plot_umap_3d(
                    embeddings=X_samp_umap,
                    labels=y_samp,
                    title=f"UMAP SMOTE (nn={n_neighbors}, md={min_dist})",
                    save_path=os.path.join(save_dir_temp, "umap_smote.png")
                )

     
        else:
            y = df2["KL-Score"]
            ids = df2["name"]
            X = df2.drop(columns=["name", "KL-Score"])

            X_scaled = scaler.fit_transform(X)

            reducer = UMAP(**umap_params)
            X_umap = reducer.fit_transform(X_scaled)

            # Save artifacts
            joblib.dump(scaler, os.path.join(save_dir_temp, "scaler.pkl"))
            joblib.dump(reducer, os.path.join(save_dir_temp, "umap_model.pkl"))
            np.save(os.path.join(save_dir_temp, "X_umap_embeddings.npy"), X_umap)

            # Plot
            plot_umap(
                embeddings=X_umap,
                labels=y,
                title=f"UMAP (nn={n_neighbors}, md={min_dist})",
                save_path=os.path.join(save_dir_temp, "umap_original.png")
            )


        run_counter += 1

print("\n✓ Completed all UMAP gridsearch runs.")

