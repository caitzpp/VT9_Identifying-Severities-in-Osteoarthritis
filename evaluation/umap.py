import matplotlib.pyplot as plt
import os

# Dimension reduction and clustering libraries
import umap
import hdbscan

from utils.load_utils import load_image_folder_as_array
import config

DATA_TYPE = "chenetal_train"
DATA_PATH = os.path.join(config.CHENETAL_DATAPATH, "train")
SAVE_PATH = os.path.join(config.OUTPUT_PATH, "UMAP", "img")

X, y = load_image_folder_as_array(DATA_PATH)

X_umap = umap.UMAP().fit_transform(X)
labels = hdbscan.HDBSCAN().fit_predict(X_umap)

# Plot by cluster
plt.figure(figsize=(10, 8))
unique_labels = set(labels)
for label in unique_labels:
    mask = labels == label
    plt.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1],
        s=5,
        label=f"Cluster {label}" if label != -1 else "Noise"
    )

plt.legend(title="HDBSCAN Clusters")
plt.title("UMAP Projection with HDBSCAN Clusters")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, DATA_TYPE + '.png'))