from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os

# Dimension reduction and clustering libraries
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from utils.load_utils import load_image_folder_as_array
import config

DATA_PATH = os.path.join(config.CHENETAL_DATAPATH, "train")
SAVE_PATH = os.path.join(config.OUTPUT_PATH, )

X, y = load_image_folder_as_array(DATA_PATH)

X_umap = umap.UMAP().fit_transform(X)
labels = hdbscan.HDBSCAN().fit_predict(X_umap)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=5, cmap="Spectral")
plt.savefig()
