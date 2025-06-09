import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from umap import UMAP
import config
from utils.load_utils import load_npy_folder_as_array
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
STAGE       = 'ss'
NEPOCH      = '400'
MODELS      = ['mod_st', 
               'mod_2'
               ]
SEED        = '34'
AVERAGE = True
wScaler = True
SCALER = StandardScaler()
DATA_PATH   = config.FEATURE_PATH
DATA_PATH = os.path.dirname(DATA_PATH)
DATA_PATH = os.path.join(DATA_PATH, "features_woNorm")

SAVE_PATH   = os.path.join(config.OUTPUT_PATH, "UMAP", "img_woNorm")
os.makedirs(SAVE_PATH, exist_ok=True)
UMAP_PARAMS = {
    'n_neighbors': 5,
    'metric':      'cosine',
    'min_dist':    0.001,
    'random_state': int(SEED)
}
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

    labels = [0, 1, 2, 3, 4]

    for lab in labels:
        fig, axes = plt.subplots(
            nrows=len(MODELS),
            ncols=2,
            figsize=(14, 12),
            sharex=True,
            sharey=True
        )
        fig.suptitle(f"UMAP – KL-Score {lab}", fontsize=18, y=0.92)
        
        for i, model_name in enumerate(MODELS):
            for j, on_test in enumerate((False, True)):
                ax = axes[i, j]
                X_umap, y = embeddings[(model_name, on_test)]

                ax.scatter(
                    X_umap[:, 0],
                    X_umap[:, 1],
                    s=20,
                    color='lightgray',
                    alpha=0.4,
                    edgecolor='none'
                )

                # only plot the points of this class
                mask = (y == lab)
                ax.scatter(
                    X_umap[mask, 0],
                    X_umap[mask, 1],
                    s=30,
                    alpha=0.8,
                    edgecolor='k',
                    linewidth=0.3,
                    color=cmap5(int(lab)),
                    label=f"KL Score {lab}"
                )

                # set identical limits
                ax.set_xlim(min_x-1, max_x+1)
                ax.set_ylim(min_y-1, max_y+1)

                # keep axes visible
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.tick_params(direction='in', length=6, width=1)
                ax.set_title(f"{model_name} — {'Test' if on_test else 'Train'}")

        # put a single legend in the top-right panel
        axes[0, 1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title="KL-Score",
            frameon=False
        )

        plt.tight_layout(rect=[0, 0, 0.95, 0.90])

        # save one file per label
        if AVERAGE:
            out_file = os.path.join(SAVE_PATH, "umap_label" '_' + 'n_neighbors' + '_' + str(UMAP_PARAMS['n_neighbors']) +'_' + 'min_dist' +'_' + str(UMAP_PARAMS['min_dist']) + '_' + str(UMAP_PARAMS['metric']) + f"_{STAGE}_{NEPOCH}_average_{lab}.png")
        else:
            out_file = os.path.join(SAVE_PATH, "umap_label" '_' + 'n_neighbors' + '_' + str(UMAP_PARAMS['n_neighbors']) +'_' + 'min_dist' +'_' + str(UMAP_PARAMS['min_dist']) + '_' + str(UMAP_PARAMS['metric']) + f"_{STAGE}_{NEPOCH}_{SEED}_{lab}.png")
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

        # for (model_name, on_test), (X_umap, y) in embeddings.items():
        #     split_name = 'test' if on_test else 'train'
        #     out_dir   = os.path.join(SAVE_PATH, "per_label")
        #     os.makedirs(out_dir, exist_ok=True)

        # for lab in sorted(set(y)):
        #     fig, ax = plt.subplots(figsize=(6,6))

        #     # 1) plot everything in light gray
        #     ax.scatter(
        #         X_umap[:,0],
        #         X_umap[:,1],
        #         s=10,
        #         color='lightgray',
        #         alpha=0.4,
        #         edgecolor='none'
        #     )

        #     # 2) overplot only this label in its true color
        #     mask = (y == lab)
        #     ax.scatter(
        #         X_umap[mask,0],
        #         X_umap[mask,1],
        #         s=30,
        #         color=cmap5(int(lab)),
        #         edgecolor='k',
        #         linewidth=0.4,
        #         alpha=0.8,
        #         label=f"Label {lab}"
        #     )

            # ax.set_title(f"{model_name} – {split_name} – label {lab}", fontsize=14)
            # ax.set_xlabel("UMAP 1");  ax.set_ylabel("UMAP 2")
            # ax.legend(loc='upper right', frameon=False)

            # # optional: keep same x/y limits across all labels
            # ax.set_xlim(min_x, max_x)
            # ax.set_ylim(min_y, max_y)

            # plt.tight_layout()
            # fname = os.path.join(out_dir, f"{model_name}_{split_name}_label_{lab}.png")
            # fig.savefig(fname, dpi=150)
            # plt.close(fig)