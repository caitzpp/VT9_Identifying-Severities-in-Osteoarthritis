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
on_test_set = True
DATA_PATH = config.FEATURE_PATH
DATA_TYPE = "features" #"chenetal_train"

SAVE_PATH = os.path.join(config.OUTPUT_PATH, "UMAP", "img")

UMAP_PARAMS = {
    'n_neighbors': 5,
    'metric': 'cosine',
    'min_dist': 0.01,
    'random_state': int(seed)
}


if __name__=="__main__":
    files = os.listdir(os.path.join(DATA_PATH, STAGE))

    # if on_test_set:
    #     file = [f for f in files if (MODEL_NAME in f) & (NEPOCH in f) & (seed in f) & ('on_test_set' in f)]
        
    # else:
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
    # print(feature_dir)
    # print(os.listdir(feature_dir))
    # files = []
    # for i in os.listdir(feature_dir):
    #     files+=os.listdir(os.path.join(feature_dir, i))
    # print(len(files))

    X, y = load_npy_folder_as_array(feature_dir)
    X_umap = UMAP(**UMAP_PARAMS).fit_transform(X)
    labels = hdbscan.HDBSCAN().fit_predict(X_umap)

    # # Plot by cluster
    # fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # # HDBSCAN Clusters
    # for label in set(labels):
    #     mask = labels == label
    #     axs[0].scatter(X_umap[mask, 0], X_umap[mask, 1], s=5, label=f"Cluster {label}")
    # axs[0].set_title("HDBSCAN Clusters")
    # axs[0].legend()
    cmap5 = plt.cm.get_cmap('tab10', 5)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=y,
        cmap=cmap5,
        s=5
    )
    ax.set_title("True Labels")

    # 5) add a colorbar tied to that scatter
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(5))
    cbar.set_label("Label")

    plt.tight_layout()

    if on_test_set:
        plt.savefig(os.path.join(SAVE_PATH, MODEL_NAME + '_' + DATA_TYPE + '_' + STAGE + '_' + NEPOCH + '_' + seed + '_' + 'n_neighbors' + '_' + str(UMAP_PARAMS['n_neighbors']) +'_' + 'min_dist' +'_' + str(UMAP_PARAMS['min_dist'])
                             +'_' + str(UMAP_PARAMS['metric']) +'_' + 'testset' +'_comparison.png'))
    else:
        plt.savefig(os.path.join(SAVE_PATH, MODEL_NAME + '_' + DATA_TYPE + '_' + STAGE + '_' + NEPOCH + '_' + seed +'_' + 'n_neighbors' +'_' + str(UMAP_PARAMS['n_neighbors']) + '_' +'min_dist' +'_' + str(UMAP_PARAMS['min_dist'])
                                +'_' + str(UMAP_PARAMS['metric']) +'_comparison.png'))
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
