import config
import os
import pandas as pd
import numpy as np
import json
from scipy.stats import entropy
from utils.evaluation_utils import normalized_entropy

proc_dir = config.PROC_DATA_PATH

folder = "2025-07-22_hdbscan"


# need to get a way to get each folder within here and to get the correct file
# record_id cluster_label
# probability

if __name__ == "__main__":
    filepath = os.path.join(proc_dir, folder)
    for subfolder in os.listdir(filepath):
        if not os.path.isdir(os.path.join(filepath, subfolder)):
            continue
        else:
            temp_filepath = os.path.join(filepath, subfolder)
            for file in os.listdir(temp_filepath):
                if file.endswith(".csv"):
                    base_name = file.split('.')[0]
                    json_filename = base_name + "_model_info.json"
                    
                    df = pd.read_csv(os.path.join(temp_filepath, file))

                    # Count rows where df['cluster_label'] == -1
                    noise_count = (df['cluster_label'] == -1).sum()
                    print(f"Noise points: {noise_count}")

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

                    



                    model_info = {
                        "noise_count": noise_count,
                        "total_average_probability": avg_probs.mean(),
                        "average_probabilities_per_cluster": avg_probs,
                        "total_entropy": membership_entropy,
                        "normalized_total_entropy": membership_entropy / H_max,
                        "entropy_per_cluster": entropy_per_cluster.to_dict(),
                    }



                    

        break


#               import numpy as np
# import pandas as pd

# # Create a DataFrame
# df = pd.DataFrame({
#     'label': clusterer.labels_,
#     'probability': clusterer.probabilities_
# })

# # Filter out noise points (label = -1)
# df = df[df['label'] != -1]

# # Group by cluster label and compute average probability
# avg_probs = df.groupby('label')['probability'].mean().sort_values(ascending=False)

# print("Average membership probability per cluster:")
# print(avg_probs)

# from scipy.stats import entropy

# # Optionally remove noise
# probabilities = clusterer.probabilities_[clusterer.labels_ != -1]

# # Normalize to sum to 1 (needed for entropy)
# p_dist = probabilities / np.sum(probabilities)

# # Compute Shannon entropy
# membership_entropy = entropy(p_dist, base=2)

# # print(f"Entropy of membership probabilities: {membership_entropy:.4f}")
# entropy_per_cluster = df.groupby('label')['probability'].apply(
#     lambda p: entropy(p / p.sum(), base=2)
# ).sort_values()

# print("Entropy per cluster:")
# print(entropy_per_cluster)
