import os
import matplotlib.pyplot as plt
from umap import UMAP
import config
from utils.load_utils import load_npy_folder_as_array

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
STAGE       = 'ss'
NEPOCH      = '400'
MODELS      = ['mod_st', 'mod_2']
SEED        = '34'
DATA_PATH   = config.FEATURE_PATH
SAVE_PATH   = os.path.join(config.OUTPUT_PATH, "UMAP", "img")
UMAP_PARAMS = {
    'n_neighbors': 5,
    'metric':      'cosine',
    'min_dist':    0.01,
    'random_state': int(SEED)
}
cmap5 = plt.cm.get_cmap('tab10', 5)

# -------------------------------------------------------------------
# HELPER
# -------------------------------------------------------------------
def load_features(model_name, on_test_set: bool):
    """Find the one folder matching model_name, epoch, seed, then load train/test npy."""
    base_dir = os.path.join(DATA_PATH, STAGE)
    all_dirs = os.listdir(base_dir)
    # pick the folder
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
            X, y = load_features(model_name, on_test)
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
    outname = f"compare_models_train_test_{STAGE}_{NEPOCH}_{SEED}.png"
    plt.savefig(os.path.join(SAVE_PATH, outname), dpi=150)
    plt.close(fig)

    for (model_name, on_test), (X_umap, y) in embeddings.items():
        split_name = 'test' if on_test else 'train'
        out_dir   = os.path.join(SAVE_PATH, 'per_label', f"{model_name}_{split_name}")
        os.makedirs(out_dir, exist_ok=True)

        for lab in sorted(set(y)):
            fig, ax = plt.subplots(figsize=(6,6))

            # 1) plot everything in light gray
            ax.scatter(
                X_umap[:,0],
                X_umap[:,1],
                s=10,
                color='lightgray',
                alpha=0.4,
                edgecolor='none'
            )

            # 2) overplot only this label in its true color
            mask = (y == lab)
            ax.scatter(
                X_umap[mask,0],
                X_umap[mask,1],
                s=30,
                color=cmap5(int(lab)),
                edgecolor='k',
                linewidth=0.4,
                alpha=0.8,
                label=f"Label {lab}"
            )

            ax.set_title(f"{model_name} – {split_name} – label {lab}", fontsize=14)
            ax.set_xlabel("UMAP 1");  ax.set_ylabel("UMAP 2")
            ax.legend(loc='upper right', frameon=False)

            # optional: keep same x/y limits across all labels
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

            plt.tight_layout()
            fname = os.path.join(out_dir, f"{model_name}_{split_name}_label_{lab}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)