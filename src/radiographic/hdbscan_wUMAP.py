import matplotlib.pyplot as plt
import os
import config
import wandb

from umap import UMAP
import hdbscan

from utils.load_utils import load_image_folder_as_array, load_npy_folder_as_array, get_next_run_folder
from utils.hdbscan_utils import save_results



DATA_PATH = config.FEATURE_PATH

SAVE_PATH = os.path.join(config.OUTPUT_PATH, "UMAP", "img")

#TODO: get kl file and kl filepath


umap_keys = ['n_neighbors', 'min_dist', 'n_components', 'metric']
hdbscan_keys = ['min_cluster_size', 'min_samples', 'cluster_selection_method', 
                'metric', 'metric_params', 'max_cluster_size',
                'cluster_selection_epsilon', 'algorithm', 'leaf_size',
                'store_centers', 'alpha']

umap_defaults = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean'
}

hdbscan_defaults = {
    'min_cluster_size': 10,
    'min_samples': None,
    'cluster_selection_method': 'eom',
    'metric': 'euclidean',
    'metric_params': None,
    'max_cluster_size': None,
    'cluster_selection_epsilon': 0.0,
    'algorithm': 'auto',
    'leaf_size': 40,
    'store_centers': 'centroid',
    'alpha': 1.0
}


if __name__=="__main__":
    run_folder_name, run_path = get_next_run_folder(save_dir)

    run = wandb.init(
        name = f"{os.path.basename(save_dir)}_{run_folder_name}"
        )
    wandb_config = wandb.config

    STAGE = wandb_config.get('stage', 'ss')
    NEPOCH = wandb_config.get('n_epochs', 400)
    MODEL_NAME = wandb_config.get('model_name', 'mod_st')
    seed = wandb_config.get('seed', '34')
    on_test_set = wandb_config.get('on_test_set', False)
    DATA_TYPE = wandb_config.get('data_type', 'features')

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

    UMAP_PARAMS = {k: wandb_config.get(f'umap.{k}', umap_defaults[k]) for k in umap_keys}
    HDBSCAN_PARAMS = {k: wandb_config.get(f'hdbscan.{k}', hdbscan_defaults[k]) for k in hdbscan_keys}

    if HDBSCAN_PARAMS['min_samples'] == -1:
        HDBSCAN_PARAMS['min_samples'] = None

    #TODO: get correct y values, so get labels such as KL for later

    X, y = load_npy_folder_as_array(feature_dir)
    X_umap = UMAP(**UMAP_PARAMS).fit_transform(X)
    labels = hdbscan.HDBSCAN(**HDBSCAN_PARAMS).fit_predict(X_umap)

    save_folder = run_folder_name
    filename = f"{run.name}_umap_hdbscan"

    save_dir_temp = os.path.join(save_dir, save_folder)
    os.makedirs(save_dir_temp, exist_ok=True)

    base_name, results_df = save_results(df2, clusterer, {
        'umap': UMAP_PARAMS,
        'hdbscan': HDBSCAN_PARAMS
    }, scaler, save_dir_temp, filename, use_wandb=True)




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