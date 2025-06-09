import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from umap import UMAP
from sklearn.decomposition import PCA
import config
from utils.load_utils import load_npy_folder_as_array
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
STAGE        = 'ss'
NEPOCH       = '400'
MODELS       = ['mod_st', 'mod_2']
SEED         = '34'
AVERAGE      = True
PCA_N_COMPONENTS = 50       # <-- how many PCs before UMAP
wScaler = True
SCALER = StandardScaler()
DATA_PATH   = config.FEATURE_PATH
DATA_PATH = os.path.dirname(DATA_PATH)
DATA_PATH = os.path.join(DATA_PATH, "features_woNorm")
SAVE_PATH    = os.path.join(config.OUTPUT_PATH, "UMAP", "img_woNorm", "pca")
os.makedirs(SAVE_PATH, exist_ok=True)
UMAP_PARAMS  = {
    'n_neighbors':  15,
    'metric':       'cosine',
    'min_dist':     0.01,
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
# HELPER TO LOAD FEATURES
# -------------------------------------------------------------------
def load_features(model_name, on_test_set: bool, average=False):
    """Locate the correct folder and load its train/test feature arrays."""
    base_dir = os.path.join(DATA_PATH, STAGE)
    all_dirs = os.listdir(base_dir)

    if average:
        dirs = [d for d in all_dirs if 'average' in d and model_name in d]
    else:
        dirs = [
            d for d in all_dirs
            if model_name in d
            and f'epoch_{NEPOCH}' in d
            and f'seed_{SEED}' in d
        ]

    if len(dirs) != 1:
        raise ValueError(f"Expected 1 folder for {model_name}, got {dirs}")

    feat_dir = os.path.join(base_dir, dirs[0], 'test' if on_test_set else 'train')
    X, y = load_npy_folder_as_array(feat_dir)
    return X, y

# -------------------------------------------------------------------
# 1) Compute PCA + UMAP embeddings & track global bounds
# -------------------------------------------------------------------
embeddings = {}
min_x, min_y = float('inf'), float('inf')
max_x, max_y = -float('inf'), -float('inf')

for model_name in MODELS:
    for on_test in (False, True):
        # load raw features
        X, y = load_features(model_name, on_test, average=AVERAGE)
        if wScaler:
            # apply standard scaling
            X = SCALER.fit_transform(X)

        # PCA reduction
        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=int(SEED))
        X_pca = pca.fit_transform(X)

        # UMAP embedding
        X_umap = UMAP(**UMAP_PARAMS).fit_transform(X_pca)

        # store for later plotting
        embeddings[(model_name, on_test)] = (X_umap, y)

        # update global axis bounds
        min_x = min(min_x, X_umap[:,0].min())
        max_x = max(max_x, X_umap[:,0].max())
        min_y = min(min_y, X_umap[:,1].min())
        max_y = max(max_y, X_umap[:,1].max())

# -------------------------------------------------------------------
# 2) Plot the 2×2 grid of embeddings
# -------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=len(MODELS),
    ncols=2,
    figsize=(14, 12),
    sharex=True,
    sharey=True
)
fig.suptitle("PCA → UMAP embeddings: Train vs Test for Two Models", fontsize=18, y=0.92)

for i, model_name in enumerate(MODELS):
    for j, on_test in enumerate((False, True)):
        ax = axes[i, j]
        X_umap, y = embeddings[(model_name, on_test)]

        # plot each class separately
        for lab in sorted(set(y)):
            mask = (y == lab)
            ax.scatter(
                X_umap[mask,0],
                X_umap[mask,1],
                s=25,
                alpha=0.7,
                edgecolor='k',
                linewidth=0.3,
                color=cmap5(int(lab)),
                label=str(int(lab))
            )

        # enforce identical x/y limits
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)

        # show axes
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(direction='in', length=6, width=1)
        ax.set_title(f"{model_name} — {'Test' if on_test else 'Train'}")

        # only one legend on the top-right
        if i == 0 and j == 1:
            ax.legend(
                title="KL – Score",
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False
            )

plt.tight_layout(rect=[0, 0, 0.95, 0.90])

# -------------------------------------------------------------------
# 3) Save figure
# -------------------------------------------------------------------
suffix = "average" if AVERAGE else SEED
if wScaler:
    suffix += f"_scaler_{SCALER.__class__.__name__}"
outname = (
    f"compare_models_train_test_pca{PCA_N_COMPONENTS}"
    f"_nn{UMAP_PARAMS['n_neighbors']}"
    f"_md{UMAP_PARAMS['min_dist']}"
    f"_{UMAP_PARAMS['metric']}_{STAGE}_{NEPOCH}_{suffix}.png"
)
plt.savefig(os.path.join(SAVE_PATH, outname), dpi=150)
plt.close(fig)
