import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from umap import UMAP
import config
from utils.load_utils import load_npy_folder_as_array
import random
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
STAGE       = 'ss'
NEPOCH      = '400'
MODELS      = ['mod_st', 'mod_2']
SEED        = '34'
AVERAGE = True
wScaler = True
SCALER = StandardScaler()
DATA_PATH   = config.FEATURE_PATH
DATA_PATH = os.path.dirname(DATA_PATH)
DATA_PATH = os.path.join(DATA_PATH, "features_woNorm")
SAVE_PATH   = os.path.join(config.OUTPUT_PATH, "UMAP", "img_woNorm")
n_neighbors = [5]
min_dist = [0.01, 0.05]
metric = 'cosine' 
# UMAP_PARAMS = {
#     'n_neighbors': 50,
#     'metric':      'cosine',
#     'min_dist':    0.1,
#     'random_state': int(SEED)
# }
colors = [
    '#1f77b4',  # tab:blue
    '#ff7f0e',  # tab:orange
    '#2ca02c',  # tab:green
    '#d62728',  # tab:red
    '#9467bd',  # tab:purple
]

cmap5 = ListedColormap(colors)

# -------------------------------------------------------------------
# HELPER
# -------------------------------------------------------------------
def load_features(model_name, on_test_set: bool, average = False):
    """Find the one folder matching model_name, epoch, seed, then load train/test npy."""
    base_dir = os.path.join(DATA_PATH, STAGE)
    all_dirs = os.listdir(base_dir)
    # pick the folder
    if average:
        folder = [d for d in all_dirs
            if 'average' in d and model_name in d]
    else:
        folder = [
            d for d in all_dirs
            if model_name in d
            and f'epoch_{NEPOCH}' in d
            and f'seed_{SEED}' in d
        ]
    if len(folder) != 1:
        raise ValueError(f"Found {folder} for {model_name}")
    feat_dir = os.path.join(base_dir, folder[0], 'test' if on_test_set else 'train')
    X, y = load_npy_folder_as_array(feat_dir)
    return X, y

if __name__ == "__main__":
    min_dist_copy = min_dist[:]
    n_neighbors_copy = n_neighbors[:]
    random.shuffle(n_neighbors_copy)
    random.shuffle(min_dist_copy)

    pairings = list(zip(n_neighbors_copy, min_dist_copy))

    for i, (n_neighbors, min_dist) in enumerate(pairings):
        UMAP_PARAMS = {
            'n_neighbors': n_neighbors,
            'metric':      metric,
            'min_dist':    min_dist,
            'random_state': int(SEED)
        }
        print(f"UMAP_PARAMS: {UMAP_PARAMS}")
       
        embeddings = {}
        min_x = float('inf')
        max_x = -float('inf')
        min_y = float('inf')
        max_y = -float('inf')

        fig, axes = plt.subplots(
            nrows=len(MODELS),
            ncols=2,
            figsize=(14, 12),
            sharex=True,
            sharey=True
        )
        fig.suptitle("UMAP embeddings: train vs test for two models", fontsize=18, y=0.92)

        for i, model_name in enumerate(MODELS):
            for j, on_test in enumerate((False, True)):
                ax = axes[i, j]
                X, y = load_features(model_name, on_test, average=AVERAGE)
                if wScaler:
                    X = SCALER.fit_transform(X)

                if UMAP_PARAMS['metric'] == 'euclidean':
                    X = StandardScaler().fit_transform(X)
                X_umap = UMAP(**UMAP_PARAMS).fit_transform(X)
                embeddings[(model_name, on_test)] = (X_umap, y)

                # update global bounds
                min_x = min(min_x, X_umap[:,0].min())
                max_x = max(max_x, X_umap[:,0].max())
                min_y = min(min_y, X_umap[:,1].min())
                max_y = max(max_y, X_umap[:,1].max())


        fig, axes = plt.subplots(
            nrows=len(MODELS),
            ncols=2,
            figsize=(14, 12)
        )
        fig.suptitle("UMAP embeddings: Train vs Test for Two Models", fontsize=18, y=0.92)

        for i, model_name in enumerate(MODELS):
            for j, on_test in enumerate((False, True)):
                ax = axes[i, j]
                X_umap, y = embeddings[(model_name, on_test)]

                # plot each true label separately so we get 0,1,2,3,4 in the legend
                for lab in sorted(set(y)):
                    mask = (y == lab)
                    ax.scatter(
                        X_umap[mask, 0],
                        X_umap[mask, 1],
                        s=25,
                        alpha=0.7,
                        edgecolor='k',
                        linewidth=0.3,
                        color=cmap5(int(lab)),
                        label=str(int(lab))
                    )

                # enforce the same limits everywhere
                ax.set_xlim(min_x-1, max_x+1)
                ax.set_ylim(min_y-1, max_y+1)

                # keep axes, ticks, spines visible
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.tick_params(direction='in', length=6, width=1)

                ax.set_title(f"{model_name} — {'Test' if on_test else 'Train'}")

                # only draw one legend (top‐right subplot)
                if i == 0 and j == 1:
                    ax.legend(
                        title="KL - Score",
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1),
                        frameon=False
                    )

        plt.tight_layout(rect=[0, 0, 0.95, 0.90])

        # save out
        if wScaler:
            scaler_name = SCALER.__class__.__name__
            b_outname = "compare_models_train_test" + '_' + scaler_name + '_' + 'n_neighbors' + '_' + str(UMAP_PARAMS['n_neighbors']) +'_' + 'min_dist' +'_' + str(UMAP_PARAMS['min_dist'])  +  '_' + str(UMAP_PARAMS['metric']) + f"{STAGE}_{NEPOCH}"
        else:
            b_outname = "compare_models_train_test" + '_' + 'n_neighbors' + '_' + str(UMAP_PARAMS['n_neighbors']) +'_' + 'min_dist' +'_' + str(UMAP_PARAMS['min_dist'])  +  '_' + str(UMAP_PARAMS['metric']) + f"{STAGE}_{NEPOCH}"

        if AVERAGE:
            outname = b_outname + "_average.png"
        else:
            outname = b_outname + f"_{SEED}.png"
        plt.savefig(os.path.join(SAVE_PATH, outname), dpi=150)
        plt.close(fig)