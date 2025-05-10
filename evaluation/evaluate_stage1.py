from utils.evaluation_utils import calc_mean_std, create_scores_dataframe, get_metrics
import config
import os
import pandas as pd

DATA_PATH = os.path.join(config.PATH_TO_RESULTS, 'ss')
SAVE_PATH = os.path.join(config.OUTPUT_PATH, "results", "ss")
PATH_TO_ANOM_SCORES = os.path.join(config.PATH_TO_ANOM, 'ss')

TRAIN_PLATEAU_EPOCH = 400

MOD_NAME="mod_2"
metric= 'centre_mean'

if __name__=="__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)

    files_total = os.listdir(DATA_PATH)
    #files_total = os.listdir(PATH_TO_ANOM_SCORES)
    metrics_per_seed = []
    
    files = [file for file in files_total if (('epoch_' + str(TRAIN_PLATEAU_EPOCH) ) in file) & ('on_test_set' in file ) & (MOD_NAME in file)]

    df_all = []
    for file in files:
    #     df = pd.read_csv(os.path.join(DATA_PATH, file))
    #     df = df[df["Unnamed: 0"]==metric]
    #     df_all.append(df)
    
    # results = pd.concat(df_all, ignore_index=True)
    # #print(results)

    # means = results.iloc[:, 1:].mean()
    # stds = results.iloc[:, 1:].std()

    # print(means)
    # print(stds)


        df = create_scores_dataframe(PATH_TO_ANOM_SCORES, [file], metric)
        res, auc, auc_mid, auc_mid2, auc_sev = get_metrics(df, metric)
        metrics_per_seed.append([res, auc, auc_mid, auc_mid2, auc_sev])

    metrics_df = pd.DataFrame(metrics_per_seed, columns=["Spearman", "AUC", "AUC_mid", "AUC_mid2", "AUC_sev"])

    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    print(f"Mean Metrics: \n{mean_metrics}")
    print(f"Standard Deviation of Metrics: \n{std_metrics}")
