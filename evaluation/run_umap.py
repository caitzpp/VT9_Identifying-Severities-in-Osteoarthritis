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

UMAP_PARAMS = {
    'n_neighbors': 15,
    'metric': 'euclidean',
    'min_dist': 0.1,
    'learning_rate': 1.0,
    'random_state': int(seed),
    'n_components': 3,
}


if __name__=="__main__":
    files = os.listdir(os.path.join(DATA_PATH, STAGE))

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
    X_umap = UMAP(**UMAP_PARAMS).fit_transform(X)
    labels = hdbscan.HDBSCAN().fit_predict(X_umap)

    cmap5 = plt.cm.get_cmap('tab10', 5)
    
    if UMAP_PARAMS['n_components'] == 2: 
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=y,
            cmap=cmap5,
            s=30,
            alpha= 0.7,
            edgecolor='k',
            linewidth=0.5,
        )
    elif UMAP_PARAMS['n_components'] == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            X_umap[:, 2],
            c=y,
            cmap=cmap5,
            s=30,
            alpha=0.7,
            edgecolor='k',
            linewidth=0.5,
        )

    handles, labels = scatter.legend_elements(
        prop="colors",
        num=4
    )

    ax.legend(
        handles, labels,
        title="Label",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        frameon=False
    )

    ax.set_title("True Labels")

    plt.tight_layout()

    if on_test_set:
        plt.savefig(os.path.join(SAVE_PATH, MODEL_NAME + '_' + DATA_TYPE + '_' + STAGE + '_' + NEPOCH + '_' + seed + '_' + 'n_neighbors' + '_' + str(UMAP_PARAMS['n_neighbors']) +'_' + 'min_dist' +'_' + str(UMAP_PARAMS['min_dist'])
                             +'_' + "lr_" + str(UMAP_PARAMS['learning_rate']) + '_' + 'ncomp' + str(UMAP_PARAMS['n_components']) + '_' + str(UMAP_PARAMS['metric']) +'_' + 'testset' +'_comparison.png'))
    else:
        plt.savefig(os.path.join(SAVE_PATH, MODEL_NAME + '_' + DATA_TYPE + '_' + STAGE + '_' + NEPOCH + '_' + seed +'_' + 'n_neighbors' +'_' + str(UMAP_PARAMS['n_neighbors']) + '_' +'min_dist' +'_' + str(UMAP_PARAMS['min_dist'])
                                +'_' +"lr_" + str(UMAP_PARAMS['learning_rate']) + '_' + 'ncomp' + str(UMAP_PARAMS['n_components']) + '_'+ str(UMAP_PARAMS['metric']) +'_comparison.png'))