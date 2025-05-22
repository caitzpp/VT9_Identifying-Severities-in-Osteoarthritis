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

# -------------------------------------------------------------------
# PLOT
# -------------------------------------------------------------------
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

        # scatter by true label
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

        ax.set_title(f"{model_name} — {'Test' if on_test else 'Train'}")
        ax.set_xticks([]); ax.set_yticks([])
        # for spine in ax.spines.values():
        #     spine.set_visible(False)

        # only add a legend on the top-right subplot
        if i == 0 and j == 1:
            ax.legend(
                title="Label",
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False
            )

plt.tight_layout(rect=[0, 0, 0.95, 0.90])
plt.savefig(os.path.join(
    SAVE_PATH,
    f"compare_models_train_test_{STAGE}_{NEPOCH}_{SEED}.png"
))

