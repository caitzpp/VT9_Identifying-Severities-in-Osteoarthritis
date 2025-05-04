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
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# HDBSCAN Clusters
for label in set(labels):
    mask = labels == label
    axs[0].scatter(X_umap[mask, 0], X_umap[mask, 1], s=5, label=f"Cluster {label}")
axs[0].set_title("HDBSCAN Clusters")
axs[0].legend()

# Ground Truth Labels
scatter = axs[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10", s=5)
axs[1].set_title("True Labels")
fig.colorbar(scatter, ax=axs[1], label="Label")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, DATA_TYPE + '_comparison.png'))
