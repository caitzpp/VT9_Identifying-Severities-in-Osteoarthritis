import matplotlib.pyplot as plt
import os
import config

from umap import UMAP
import hdbscan

from utils.load_utils import load_image_folder_as_array, load_npy_folder_as_array

STAGE = 'ss'
NEPOCH = '400'
MODEL_NAME = 'mod_st'
seed = '34'
on_test_set = False
DATA_PATH = config.FEATURE_PATH
DATA_TYPE = "features" #"chenetal_train"

SAVE_PATH = os.path.join(config.OUTPUT_PATH, "UMAP", "img")


if __name__=="__main__":
    files = os.listdir(os.path.join(DATA_PATH, STAGE))

    if on_test_set:
        file = [f for f in files if (MODEL_NAME in f) & (NEPOCH in f) & (seed in f) & ('on_test_set' in f)]
        
    else:
        file = [f for f in files if (MODEL_NAME in f) & (NEPOCH in f) & (seed in f) & ('on_test_set' not in f)]

    if len(file) == 1:
        folder_dir = file[0]
        feature_dir = os.path.join(DATA_PATH, STAGE, folder_dir)

        if on_test_set:
            feature_dir = os.path.join(feature_dir, 'test')
        else:
            feature_dir = os.path.join(feature_dir, 'train')
    else:
        print("more than one folder found: ", file)
        raise

    X, y = load_npy_folder_as_array(feature_dir)
    X_umap = UMAP().fit_transform(X)
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
    plt.savefig(os.path.join(SAVE_PATH, MODEL_NAME + '_' + DATA_TYPE + '_' + STAGE + '_' + NEPOCH + '_' + seed +'_comparison.png'))
    # file_paths = []
    # for dirs in os.listdir(feature_dir):
    #     for files in os.listdir(os.path.join(feature_dir, dirs)):
    #         file_paths.append(os.path.join(feature_dir, dirs, files))
    #         file_path = os.path.join(feature_dir, dirs, files)


    

    
   

# X, y = load_image_folder_as_array(DATA_PATH)

# X_umap = umap.UMAP().fit_transform(X)
# labels = hdbscan.HDBSCAN().fit_predict(X_umap)

# # Plot by cluster
# fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# # HDBSCAN Clusters
# for label in set(labels):
#     mask = labels == label
#     axs[0].scatter(X_umap[mask, 0], X_umap[mask, 1], s=5, label=f"Cluster {label}")
# axs[0].set_title("HDBSCAN Clusters")
# axs[0].legend()

# # Ground Truth Labels
# scatter = axs[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10", s=5)
# axs[1].set_title("True Labels")
# fig.colorbar(scatter, ax=axs[1], label="Label")

# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH, DATA_TYPE + '_comparison.png'))
