import config
import os
import pandas as pd
import numpy as np
import json
from scipy.stats import entropy
from utils.evaluation_utils import normalized_entropy, get_metrics
from IPython.display import display
from utils.load_utils import fix_id, convert_numpy
from sklearn.metrics import normalized_mutual_info_score

proc_dir = config.PROC_DATA_PATH

folder = "2025-07-22_hdbscan"

kl_file = "inmodi_data_personalinformation_kl.csv"
kl_folder = "2025-07-14_data_exploration"


if __name__ == "__main__":
    # Load KL Scores
    filepath = os.path.join(proc_dir, kl_folder, kl_file)

    kl_df = pd.read_csv(filepath)

    filepath = os.path.join(proc_dir, folder)
    for subfolder in os.listdir(filepath):
        if not os.path.isdir(os.path.join(filepath, subfolder)):
            continue
        else:
            temp_filepath = os.path.join(filepath, subfolder)
            print(temp_filepath)
            for file in os.listdir(temp_filepath):
                if file.endswith("scaled.csv"):
                    base_name = file.split('.')[0]
                    json_filename = base_name + "_model_info.json"

                    df = pd.read_csv(os.path.join(temp_filepath, file))

                    df['id'] = df['id'].apply(fix_id)

                    # Count rows where df['cluster_label'] == -1
                    noise_count = (df['cluster_label'] == -1).sum()
                    #print(f"Noise points: {noise_count}")

                    df_filtered = df[df['cluster_label'] != -1]

                    avg_probs = df_filtered.groupby('cluster_label')['probability'].mean().sort_values(ascending=False)
                    # print("Average membership probability per cluster:")
                    # print(avg_probs)
                    # print(f"Total Average Probability: {avg_probs.mean():.4f}")

                    p_dist = df_filtered['probability'] / np.sum(df_filtered['probability'])
                    membership_entropy = entropy(p_dist, base=2)
                    H_max = np.log2(len(p_dist))
                    # print(f"Entropy of membership probabilities: {membership_entropy:.4f}")
                    # print(f"Normalized Entropy: {membership_entropy / H_max:.4f}")
                    entropy_per_cluster = df_filtered.groupby('cluster_label')['probability'].apply(
                           normalized_entropy
                        ).sort_values()

                    # print("Entropy per cluster:")
                    # print(entropy_per_cluster)

                    df_merged = df_filtered.merge(kl_df, left_on='id', right_on='name', how='left', validate='one_to_one')
                    display(df_merged)
                    print(f"Missing KL-Scores: {len(df_merged[df_merged['KL-Score'].isna()])}")

                    df_merged.to_csv(os.path.join(temp_filepath, f"{base_name}_wKL.csv"), index=False)

                    df_merged = df_merged.dropna(subset=['KL-Score'])

                    spr, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df_merged, score = 'cluster_label', label = 'KL-Score')

                    print(f"Spearman Correlation: {spr:.4f}")
                    print(f"Overall AUC: {auc:.4f}")
                    print(f"Mid AUC: {auc_mid:.4f}")
                    print(f"Mid2 AUC: {auc_mid2:.4f}")
                    print(f"Severe AUC: {auc_sev:.4f}")

                    nmi = normalized_mutual_info_score(df_merged['KL-Score'], df_merged['cluster_label'])
                    print(f"NMI: {nmi:.4f}")


                    model_info = {
                        "noise_count": noise_count,
                        "total_average_probability": avg_probs.mean(),
                        "average_probabilities_per_cluster": avg_probs,
                        "total_entropy": membership_entropy,
                        "normalized_total_entropy": membership_entropy / H_max,
                        "entropy_per_cluster": entropy_per_cluster.to_dict(),
                        "missing_kl_scores": len(df_merged[df_merged['KL-Score'].isna()]),
                        "spearman_correlation": spr,
                        "overall_auc": auc,
                        "mid_auc": auc_mid,
                        "mid2_auc": auc_mid2,
                        "severe_auc": auc_sev,
                        "nmi": nmi
                    }
                    # for keys, values in model_info.items():
                    #     print(f"Key {keys} has value type {type(values)}")

                    model_info_json_ready = {k: convert_numpy(v) for k, v in model_info.items()}

                    with open(os.path.join(temp_filepath, json_filename), 'r') as f:
                        data = json.load(f)
                        data['eval'] = model_info_json_ready

                    with open(os.path.join(temp_filepath, json_filename), 'w') as f:
                        json.dump(data, f, indent=4)
                        

